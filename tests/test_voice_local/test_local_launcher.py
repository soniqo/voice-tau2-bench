"""Packaging must preserve pinned source and distinguish task failure from setup failure."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.voice import local_launcher as launcher
from tau2.voice.baseline import repository_root


def test_bundled_patch_is_pinned_and_has_no_machine_paths():
    lock, patch = launcher.runtime_lock(repository_root())
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == lock["patch_sha256"]
    assert "/Users/" not in patch.read_text()
    assert lock["profile"].endswith("gemma12b-v4.json")
    dependencies = repository_root() / "runtime" / lock["dependencies"]
    assert launcher.digest(dependencies) == lock["dependencies_sha256"]
    assert "/Users/" not in dependencies.read_text()


def test_modified_dependency_pins_are_rejected(tmp_path):
    folder = tmp_path / "runtime"
    folder.mkdir()
    (folder / "speech-swift.patch").write_text("patch")
    (folder / "dependencies.json").write_text("changed")
    (folder / "lock.json").write_text(
        json.dumps(
            {
                "patch": "speech-swift.patch",
                "patch_sha256": launcher.digest(folder / "speech-swift.patch"),
                "dependencies": "dependencies.json",
                "dependencies_sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="dependencies do not match"):
        launcher.runtime_lock(tmp_path)


def test_managed_checkout_gets_exact_dependencies_without_overwriting(tmp_path):
    root = repository_root()
    lock, _ = launcher.runtime_lock(root)
    source = tmp_path / "source"
    source.mkdir()
    run = launcher.LocalRun(root, tmp_path)
    run.prepare_dependencies(lock, source, managed=True)
    assert launcher.digest(source / "Package.resolved") == lock["dependencies_sha256"]
    (source / "Package.resolved").write_text("other")
    with pytest.raises(ValueError, match="never overwritten"):
        run.prepare_dependencies(lock, source, managed=True)
    assert (source / "Package.resolved").read_text() == "other"


def test_existing_source_requires_matching_dependencies_without_mutation(tmp_path):
    root = repository_root()
    lock, _ = launcher.runtime_lock(root)
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="never overwritten"):
        launcher.LocalRun(root, tmp_path).prepare_dependencies(
            lock, source, managed=False
        )
    assert not (source / "Package.resolved").exists()


def test_modified_patch_is_rejected_before_any_subprocess(tmp_path):
    folder = tmp_path / "runtime"
    folder.mkdir()
    (folder / "speech-swift.patch").write_text("changed")
    (folder / "lock.json").write_text(
        json.dumps({"patch": "speech-swift.patch", "patch_sha256": "0" * 64})
    )
    with pytest.raises(ValueError, match="does not match"):
        launcher.runtime_lock(tmp_path)


def test_matching_existing_source_is_never_edited(monkeypatch, tmp_path):
    source = tmp_path / "existing"
    source.mkdir()
    monkeypatch.setattr(launcher, "source_digest", lambda _: "expected")
    run = launcher.LocalRun(tmp_path, tmp_path)
    monkeypatch.setattr(
        run,
        "step",
        lambda *_args, **_kwargs: pytest.fail("Existing source was mutated"),
    )
    assert (
        run.prepare_source({"source_sha256": "expected"}, tmp_path / "patch", source)
        == source
    )


def test_wrong_existing_source_is_rejected(monkeypatch, tmp_path):
    source = tmp_path / "existing"
    source.mkdir()
    monkeypatch.setattr(launcher, "source_digest", lambda _: "other")
    with pytest.raises(ValueError, match="differs"):
        launcher.LocalRun(tmp_path, tmp_path).prepare_source(
            {"source_sha256": "expected"}, tmp_path / "patch", source
        )


def test_readonly_swift_queries_do_not_block_but_builds_do(monkeypatch):
    pid = os.getpid()
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *_args,
        **_kwargs: f"{pid} 1 Python\n99 1 /tmp/swift-build\n100 1 /tmp/swift-test\n101 {pid} /tmp/speech-server\n",
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, **_: SimpleNamespace(
            stdout="swift-build --show-bin-path"
            if command[2] == "99"
            else "swift-test --no-parallel"
        ),
    )
    assert launcher.busy_jobs() == [{"pid": 100, "program": "swift-test"}]


@pytest.mark.parametrize(
    "domain,reward", [(None, None), ("airline", 0), ("airline", 1)]
)
def test_launch_workflow_and_task_zero_is_completed(
    monkeypatch, tmp_path, domain, reward
):
    root = repository_root()
    source = tmp_path / "source"
    source.mkdir()
    server = source / "speech-server"
    server.write_text("fixture")
    output = tmp_path / "run"
    steps = []
    closed = []
    monkeypatch.setattr(launcher, "check_toolchain", lambda: None)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(launcher.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(launcher.platform, "mac_ver", lambda: ("15.0", (), ""))
    monkeypatch.setattr(launcher.LocalRun, "prepare_source", lambda *_args: source)
    lock, _ = launcher.runtime_lock(root)
    monkeypatch.setattr(launcher, "source_digest", lambda _: lock["source_sha256"])
    monkeypatch.setattr(launcher, "stop", lambda process: closed.append(process))
    monkeypatch.setattr(
        launcher.LocalRun, "serve", lambda self, *_: setattr(self, "guard", "owned")
    )

    def step(self, name, command, **kwargs):
        steps.append((name, command))
        if name == "voice-task":
            (output / "task").mkdir()
            (output / "task/simulation.json").write_text(
                json.dumps(
                    {
                        "reward_info": {"reward": reward},
                        "termination_reason": "user_stop",
                    }
                )
            )

    monkeypatch.setattr(launcher.LocalRun, "step", step)
    args = [
        "--output",
        str(output),
        "--runtime-source",
        str(source),
        "--server",
        str(server),
    ]
    if domain:
        args += ["--domain", domain, "--task-id", "12"]
    assert launcher.main(args) == 0
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == ("passed" if domain is None else "completed")
    assert report.get("task_reward") == reward
    assert [name for name, _ in steps] == [
        "models-stage",
        "caller-calibration",
        "voice-task" if domain else "voice-smoke",
    ]
    assert closed == ["owned"]


def test_failed_stage_is_unscored_and_cleans_runtime(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    server = source / "server"
    server.touch()
    monkeypatch.setattr(launcher, "check_toolchain", lambda: None)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(launcher.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(launcher.platform, "mac_ver", lambda: ("15.0", (), ""))
    monkeypatch.setattr(launcher.LocalRun, "prepare_source", lambda *_args: source)
    lock, _ = launcher.runtime_lock(repository_root())
    monkeypatch.setattr(launcher, "source_digest", lambda _: lock["source_sha256"])
    closed = []
    monkeypatch.setattr(launcher, "stop", lambda proc: closed.append(proc))

    def fail(self, *_args, **_kwargs):
        self.guard = "owned"
        raise RuntimeError("inference failed")

    monkeypatch.setattr(launcher.LocalRun, "step", fail)
    output = tmp_path / "failed"
    assert (
        launcher.main(
            [
                "--output",
                str(output),
                "--runtime-source",
                str(source),
                "--server",
                str(server),
            ]
        )
        == 1
    )
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "failed_unscored" and "task_reward" not in report
    assert closed == ["owned"]


def test_patch_applies_to_clean_pinned_source(tmp_path):
    """Git-level replay uses existing objects only; no models, build or download."""
    configured = os.environ.get("SPEECH_SWIFT_TEST_SOURCE")
    if not configured:
        pytest.skip(
            "Set SPEECH_SWIFT_TEST_SOURCE for the offline native-patch integration check"
        )
    original = Path(configured)
    lock, patch = launcher.runtime_lock(repository_root())
    source = tmp_path / "source"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(original),
            str(source),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", lock["revision"]],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    assert launcher.source_digest(source) == lock["source_sha256"]


@pytest.mark.local_models
def test_real_packaged_launcher_builds_and_passes_smoke(tmp_path):
    """Explicit opt-in: this test owns the single native service, including cleanup."""
    if os.environ.get("TAU_VOICE_PACKAGING_E2E") != "1" or not os.environ.get(
        "SPEECH_SWIFT_INFERENCE_URL"
    ):
        pytest.skip(
            "Set TAU_VOICE_PACKAGING_E2E=1 and SPEECH_SWIFT_INFERENCE_URL; no native service may already be running"
        )
    command = [
        os.sys.executable,
        str(repository_root() / "scripts/run_local_voice.py"),
        "--output",
        str(tmp_path / "packaged"),
    ]
    source = os.environ.get("SPEECH_SWIFT_TEST_SOURCE")
    server = os.environ.get("SPEECH_SWIFT_TEST_SERVER")
    if source:
        command += ["--runtime-source", source]
    if server:
        command += ["--server", server]
    result = subprocess.run(command, cwd=repository_root(), timeout=7200)
    report = json.loads((tmp_path / "packaged/result.json").read_text())
    assert result.returncode == 0, report
    assert report["status"] == "passed" and report["domain_score"] is False
    assert report["api_cost_usd"] == 0
    assert len(report["runtime_source_sha256"]) == 64


def test_missing_build_tools_are_reported_before_setup(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="Missing build tools"):
        launcher.check_toolchain()


def test_missing_metal_toolchain_has_actionable_error(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/tools/" + name)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(ValueError, match="downloadComponent MetalToolchain"):
        launcher.check_toolchain()


def test_cleanup_stops_owned_group_even_after_guard_exits(monkeypatch):
    calls = []
    process = SimpleNamespace(pid=12345, poll=lambda: 1, wait=lambda **_: 1)
    monkeypatch.setattr(
        launcher.os, "killpg", lambda pid, sig: calls.append((pid, sig))
    )
    launcher.stop(process)
    assert calls == [(12345, launcher.signal.SIGTERM)]
