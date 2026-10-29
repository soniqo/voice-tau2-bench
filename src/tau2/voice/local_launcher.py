"""Reproduce the pinned local runtime and execute a smoke check or voice task."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tau2.voice.baseline import repository_root, verify_baseline


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(source):
    names = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "Sources",
            "Package.swift",
            "Package.resolved",
        ],
        cwd=source,
        text=True,
    ).splitlines()
    value = hashlib.sha256()
    for name in sorted(set(names)):
        value.update(name.encode() + b"\0" + (source / name).read_bytes())
    return value.hexdigest()


def runtime_lock(root):
    lock = json.loads((root / "runtime/lock.json").read_text())
    patch = root / "runtime" / lock["patch"]
    if (
        patch.parent.resolve() != (root / "runtime").resolve()
        or digest(patch) != lock["patch_sha256"]
    ):
        raise ValueError("Bundled runtime patch does not match runtime/lock.json")
    dependencies = root / "runtime" / lock["dependencies"]
    if (
        dependencies.parent.resolve() != (root / "runtime").resolve()
        or digest(dependencies) != lock["dependencies_sha256"]
    ):
        raise ValueError("Bundled runtime dependencies do not match runtime/lock.json")
    return lock, patch


def check_toolchain():
    missing = [name for name in ("git", "swift", "xcrun") if shutil.which(name) is None]
    if missing:
        raise ValueError(
            f"Missing build tools: {', '.join(missing)}. Install Xcode and its command-line tools."
        )
    metal = subprocess.run(["xcrun", "--find", "metal"], capture_output=True, text=True)
    if metal.returncode:
        raise ValueError(
            "Metal toolchain is missing. Install it with xcodebuild -downloadComponent MetalToolchain, then retry."
        )


def busy_jobs():
    """Exclude our process tree and read-only Swift metadata commands."""
    rows = []
    for line in subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,comm="], text=True
    ).splitlines():
        pid, parent, command = line.strip().split(None, 2)
        rows.append((int(pid), int(parent), Path(command).name))
    owned = {os.getpid()}
    while True:
        added = {pid for pid, parent, _ in rows if parent in owned} - owned
        if not added:
            break
        owned.update(added)
    heavy = {
        "swift-build",
        "swift-test",
        "swift-frontend",
        "xctest",
        "speech-server",
        "speech",
        "audio",
        "asr-bench",
        "diarization-bench",
        "vad-bench",
        "FishS2Speak",
        "VoxCPM2Batch",
        "InstructTTSBatch",
    }
    jobs = []
    for pid, _, name in rows:
        if pid in owned or name not in heavy:
            continue
        args = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True
        ).stdout
        if name in {"swift-build", "swift-test"} and any(
            token in args.split()
            for token in ("--show-bin-path", "--help", "--version")
        ):
            continue
        jobs.append({"pid": pid, "program": name})
    return jobs


def stop(process):
    if process is not None:
        # The group may still own the native child after its Python guard exits.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class LocalRun:
    def __init__(self, root, output):
        self.root, self.output = root, output
        self.guard = None
        self.env = dict(os.environ, PYTHONPATH=str(root / "src"), UV_FROZEN="1")

    def check(self):
        jobs = busy_jobs()
        if jobs:
            raise RuntimeError(
                f"Another heavy local job is active: {jobs}. Retry when it finishes; no other job was stopped."
            )
        if self.guard is not None and self.guard.poll() is not None:
            raise RuntimeError(
                "The guarded native runtime exited; this run is unscored"
            )

    def step(self, name, command, *, cwd=None, env=None, heavy=False, timeout=None):
        print(f"{name}…", flush=True)
        if heavy:
            self.check()
        child = None
        try:
            with (self.output / (name + ".log")).open("w") as log:
                child = subprocess.Popen(
                    command,
                    cwd=cwd or self.root,
                    env=env or self.env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                started = last = time.monotonic()
                while child.poll() is None:
                    if heavy:
                        self.check()
                    now = time.monotonic()
                    if timeout and now - started > timeout:
                        raise RuntimeError(
                            f"{name} exceeded its {timeout:g}s limit; see {self.output / (name + '.log')}"
                        )
                    if now - last >= 30:
                        print(
                            f"{name}: {now - started:.0f}s; log: {self.output / (name + '.log')}",
                            flush=True,
                        )
                        last = now
                    time.sleep(0.5)
            if child.returncode:
                raise RuntimeError(
                    f"{name} failed; see {self.output / (name + '.log')}"
                )
        finally:
            stop(child)

    def prepare_source(self, lock, patch, source=None):
        managed = source is None
        source = source or self.root / ".cache/voice-runtime" / lock["revision"]
        if not source.exists():
            if not managed:
                raise ValueError(f"Native source does not exist: {source}")
            source.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="runtime-", dir=source.parent
            ) as temporary:
                checkout = Path(temporary) / "source"
                self.step(
                    "runtime-clone",
                    [
                        "git",
                        "clone",
                        "--no-checkout",
                        lock["repository"],
                        str(checkout),
                    ],
                )
                self.step(
                    "runtime-checkout",
                    ["git", "checkout", "--detach", lock["revision"]],
                    cwd=checkout,
                )
                self.step(
                    "runtime-patch",
                    ["git", "apply", "--check", str(patch)],
                    cwd=checkout,
                )
                self.step("runtime-apply", ["git", "apply", str(patch)], cwd=checkout)
                if source_digest(checkout) != lock["source_sha256"]:
                    raise ValueError(
                        "Fetched runtime source does not match the tested source hash"
                    )
                checkout.rename(source)
        if source_digest(source) != lock["source_sha256"]:
            raise ValueError(
                "Native source differs from the pinned tested runtime. Use the managed checkout or an exact matching source; existing checkouts are never patched."
            )
        return source

    def prepare_dependencies(self, lock, source, *, managed):
        """Install the tested Swift pins only in the launcher-owned checkout."""
        bundled = self.root / "runtime" / lock["dependencies"]
        target = source / "Package.resolved"
        if not target.exists() and managed:
            target.write_bytes(bundled.read_bytes())
        if not target.is_file() or digest(target) != lock["dependencies_sha256"]:
            raise ValueError(
                "Native Package.resolved differs from the pinned tested dependencies. "
                "Use the managed checkout; existing dependency files are never overwritten."
            )

    def serve(self, server, manifest, port, profile):
        from tau2.voice.speech_swift import SpeechSwiftBackend

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
        log = (self.output / "native.log").open("w")
        try:
            self.check()
            self.guard = subprocess.Popen(
                [
                    sys.executable,
                    str(self.root / "scripts/run_speech_swift_guarded.py"),
                    "--server",
                    str(server),
                    "--manifest",
                    str(manifest),
                    "--metrics",
                    str(self.output / "memory.jsonl"),
                    "--port",
                    str(port),
                ],
                cwd=self.root,
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
        backend = SpeechSwiftBackend(f"http://127.0.0.1:{port}", timeout=2)
        try:
            print("native-load…", flush=True)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                self.check()
                try:
                    native = backend.provenance()
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise RuntimeError("Native startup timed out; inspect native.log")
            if os.getpgid(native["pid"]) != self.guard.pid:
                raise RuntimeError("The local service is not owned by this launch")
            profile.check_runtime(backend.capabilities())
        finally:
            backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime-source",
        type=Path,
        help="Exact matching native source; otherwise fetch the pinned managed checkout",
    )
    parser.add_argument(
        "--server",
        type=Path,
        help="Use an already built server instead of building; requires --runtime-source",
    )
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--domain", choices=["airline", "retail", "telecom"])
    parser.add_argument("--task-id")
    args = parser.parse_args(argv)
    if args.server and not args.runtime_source:
        parser.error("--server requires --runtime-source for source provenance")
    if args.task_id and not args.domain:
        parser.error("--task-id requires --domain")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        parser.error("Use Python 3.12 or 3.13")
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or int(platform.mac_ver()[0].split(".")[0]) < 15
    ):
        parser.error("The local runtime requires Apple Silicon and macOS 15+")
    root = repository_root()
    try:
        check_toolchain()
        lock, patch = runtime_lock(root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"Invalid pinned runtime package: {error}")
    output = args.output.resolve()
    if output.exists():
        parser.error(
            "--output must be a fresh directory; previous evidence is never overwritten"
        )
    output.mkdir(parents=True, mode=0o700)
    run = LocalRun(root, output)
    status = {
        "status": "preparing",
        "mode": "task" if args.domain else "controlled_smoke",
        "api_cost_usd": 0,
    }
    try:
        verify_baseline()
        source = run.prepare_source(
            lock, patch, args.runtime_source.resolve() if args.runtime_source else None
        )
        if args.server:
            server = args.server.resolve()
        else:
            run.prepare_dependencies(lock, source, managed=args.runtime_source is None)
            run.step(
                "runtime-build",
                [
                    "swift",
                    "build",
                    "-c",
                    "release",
                    "--disable-sandbox",
                    "--force-resolved-versions",
                    "--product",
                    "speech-server",
                ],
                cwd=source,
                heavy=True,
                timeout=3600,
            )
            run.step(
                "runtime-shaders",
                [str(source / "scripts/build_mlx_metallib.sh"), "release"],
                cwd=source,
                heavy=True,
            )
            server = source / ".build/release/speech-server"
        if source_digest(source) != lock["source_sha256"]:
            raise ValueError(
                "Runtime sources changed during build; refusing to stage a mismatched runtime"
            )
        if not server.is_file():
            raise ValueError(f"Native server not found: {server}")
        status.update(
            runtime_source_sha256=lock["source_sha256"],
            runtime_dependencies_sha256=lock["dependencies_sha256"],
            runtime_binary_sha256=digest(server),
            profile=lock["profile"],
        )
        profile_path = root / lock["profile"]
        from tau2.voice.runtime_profile import load_profile

        profile = load_profile(profile_path)
        manifest = output / "models.json"
        run.step(
            "models-stage",
            [
                sys.executable,
                str(root / "scripts/prepare_speech_swift_models.py"),
                "--runtime-source",
                str(source),
                "--agent-model",
                "gemma4-12b-it",
                "--failure-capture-directory",
                str(output / "native-failures"),
                "--output",
                str(manifest),
            ],
        )
        run.serve(server, manifest, args.port, profile)
        base = f"http://127.0.0.1:{args.port}"
        qualification = output / "qualification.json"
        command = [sys.executable, "-m", "tau2.voice.cli"]
        run.step(
            "caller-calibration",
            [
                *command,
                "qualify",
                "--profile",
                str(profile_path),
                "--base-url",
                base,
                "--output",
                str(qualification),
            ],
            heavy=True,
            timeout=1800,
        )
        if args.domain:
            task = output / "task"
            task_args = ["--task-id", args.task_id] if args.task_id else []
            run.step(
                "voice-task",
                [
                    *command,
                    "run",
                    "--domain",
                    args.domain,
                    *task_args,
                    "--profile",
                    str(profile_path),
                    "--qualification",
                    str(qualification),
                    "--base-url",
                    base,
                    "--seed",
                    "70",
                    "--complexity",
                    "control",
                    "--output",
                    str(task),
                ],
                heavy=True,
                timeout=profile.execution.wall_timeout_seconds + 180,
            )
            simulation = json.loads((task / "simulation.json").read_text())
            status.update(
                status="completed",
                task_reward=simulation["reward_info"]["reward"],
                termination=simulation["termination_reason"],
            )
        else:
            smoke_env = dict(
                run.env,
                SPEECH_SWIFT_INFERENCE_URL=base,
                TAU_VOICE_PROFILE=str(profile_path),
                TAU_VOICE_FULL_E2E="1",
                TAU_VOICE_E2E_OUTPUT=str(output / "smoke"),
            )
            run.step(
                "voice-smoke",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_voice_local/test_real_models.py",
                    "-k",
                    "full_voice_task",
                ],
                env=smoke_env,
                heavy=True,
                timeout=1800,
            )
            status.update(status="passed", domain_score=False)
    except (Exception, KeyboardInterrupt) as error:
        status.update(
            status="failed_unscored", error=str(error) or type(error).__name__
        )
        print(f"Local setup/run failed; inspect {output}. {error}", file=sys.stderr)
        return 1
    finally:
        stop(run.guard)
        (output / "result.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
