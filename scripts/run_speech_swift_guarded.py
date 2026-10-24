#!/usr/bin/env python3
"""Run the local server, record macOS physical footprint, and bound resource use."""

import argparse
import ctypes
import json
import shutil
import subprocess
import time
from pathlib import Path


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
