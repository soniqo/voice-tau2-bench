#!/usr/bin/env python3
"""Run the local server, record macOS physical footprint, and bound resource use."""

import argparse
import ctypes
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


def validate_assets(manifest_path):
    """Check pinned local files before allocating native model memory."""
    manifest = json.loads(manifest_path.read_text())
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Model manifest has no pinned assets; stage models first")
    errors = []
    required = {"gemma", "vox", "whisper", "vad", "tokenizer"}
    if manifest.get("agentDirectory"):
        required.add("agent")
    errors.extend(
        f"{role}: missing asset pins" for role in sorted(required - assets.keys())
    )
    for role, pin in assets.items():
        directory = manifest.get(f"{role}Directory")
        files = pin.get("files") if isinstance(pin, dict) else None
        if (
            not isinstance(directory, str)
            or not directory
            or not isinstance(files, list)
            or not files
        ):
            errors.append(f"{role}: missing directory or file pins")
            continue
        for record in files:
            if not isinstance(record, dict):
                errors.append(f"{role}: invalid file pin")
                continue
            name = record.get("path", "")
            expected = record.get("sha256", "")
            if (
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected)
            ):
                errors.append(f"{role}: invalid file pin")
                continue
            path = Path(directory) / name
            label = f"{role}/{name}"
            if not path.is_file():
                errors.append(f"{label}: missing")
                continue
            if "size" in record and path.stat().st_size != record["size"]:
                errors.append(f"{label}: size mismatch")
                continue
            value = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != expected:
                errors.append(f"{label}: checksum mismatch")
    if errors:
        raise ValueError(
            "Local model assets are not ready:\n"
            + "\n".join(errors)
            + "\nRestage with scripts/prepare_speech_swift_models.py before starting."
        )


class RUsage(ctypes.Structure):
    # Darwin's rusage_info_v2 includes GPU allocations in ri_phys_footprint.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--memory-gb", type=float, default=24)
    parser.add_argument("--min-free-gb", type=float, default=2)
    args = parser.parse_args()
    try:
        validate_assets(args.manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libproc.proc_pid_rusage.restype = ctypes.c_int
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(
        [
            str(args.server.resolve()),
            "--port",
            str(args.port),
            "--inference-manifest",
            str(args.manifest.resolve()),
        ]
    )
    started = time.perf_counter()
    try:
        with args.metrics.open("w") as metrics:
            while child.poll() is None:
                usage = RUsage()
                if libproc.proc_pid_rusage(child.pid, 2, ctypes.byref(usage)) != 0:
                    time.sleep(0.2)
                    continue
                free = shutil.disk_usage(args.metrics.parent).free
                row = {
                    "seconds": time.perf_counter() - started,
                    "pid": child.pid,
                    "physical_footprint_bytes": usage.phys_footprint,
                    "rss_bytes": usage.resident_size,
                    "free_disk_bytes": free,
                }
                if (
                    usage.phys_footprint > args.memory_gb * 1024**3
                    or free < args.min_free_gb * 1024**3
                ):
                    row["resource_limit_exceeded"] = True
                    child.kill()
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                time.sleep(1)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        status = child.wait()
        print(f"Native server exited: {status}", flush=True)
        if status:
            raise SystemExit(status if status > 0 else 128 - status)


if __name__ == "__main__":
    main()
