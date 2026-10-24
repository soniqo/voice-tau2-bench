"""Stage sharded models without downloading alternate copies of existing models."""

import hashlib
import importlib.util
import io
import json
import sys
from types import SimpleNamespace

import pytest

from tau2.voice.baseline import repository_root
from tau2.voice.runtime_profile import load_profile


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("vox", {"config.json", "model.safetensors"}),
        (
            "agent12b",
            {
                "config.json",
                "model.safetensors",
                "model-00001.safetensors",
                "model.safetensors.index.json",
            },
        ),
    ],
)
def test_staging_selects_shards_only_for_the_sharded_model(
    monkeypatch, tmp_path, kind, expected
):
    path = repository_root() / "scripts/prepare_speech_swift_models.py"
    spec = importlib.util.spec_from_file_location("model_staging", path)
    staging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staging)
    files = {
        name: name.encode()
        for name in (
            "config.json",
            "model.safetensors",
            "model-00001.safetensors",
            "model.safetensors.index.json",
        )
    }
    inventory = {
        "siblings": [
            {
                "rfilename": name,
                "size": len(data),
                "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
            }
            for name, data in files.items()
        ]
    }
    requested = []

    def download(url, **_):
        if "/api/models/" in url:
            return io.BytesIO(json.dumps(inventory).encode())
        name = url.rsplit("/", 1)[-1]
        requested.append(name)
        return io.BytesIO(files[name])

    monkeypatch.setattr(staging.urllib.request, "urlopen", download)
    monkeypatch.setattr(
        staging.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 1024**3)
    )
    folder, provenance = staging.stage(kind, tmp_path)
    assert set(requested) == expected
    assert {item["path"] for item in provenance["files"]} == expected
    assert {item.name for item in folder.iterdir()} == expected


def test_twelve_b_profile_changes_only_the_agent():
    original = load_profile()
    extended = load_profile(
        repository_root() / "configs/voice/soniqo-local-gemma12b-v1.json"
    )
    assert extended.roles["agent"].model_id == "gemma4-12b-it"
    assert extended.roles["agent"].precision == "int4"
    for role in original.roles.keys() - {"agent"}:
        assert extended.roles[role] == original.roles[role]


@pytest.mark.parametrize("capture", [False, True])
def test_staging_failure_capture_is_explicit_and_records_absolute_directory(
    monkeypatch, tmp_path, capture
):
    spec = importlib.util.spec_from_file_location(
        "model_staging_capture",
        repository_root() / "scripts/prepare_speech_swift_models.py",
    )
    staging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staging)
    source = tmp_path / "source"
    (source / "Sources").mkdir(parents=True)
    (source / "Sources/local.swift").write_text("// fixture\n")

    def stage(kind, root):
        folder = root / kind
        folder.mkdir(parents=True)
        if kind == "tokenizer":
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (folder / name).write_text("{}")
        return folder, {"files": []}

    monkeypatch.setattr(staging, "stage", stage)
    monkeypatch.setattr(
        staging.subprocess,
        "check_output",
        lambda command, **_: "a" * 40
        if command == ["git", "rev-parse", "HEAD"]
        else "Sources/local.swift\n",
    )
    output = tmp_path / "models.json"
    arguments = [
        "stage",
        "--cache",
        str(tmp_path / "cache"),
        "--runtime-source",
        str(source),
        "--output",
        str(output),
    ]
    if capture:
        arguments += ["--failure-capture-directory", str(tmp_path / "failures")]
    monkeypatch.setattr(sys, "argv", arguments)
    staging.main()
    manifest = json.loads(output.read_text())
    if capture:
        assert manifest["failureCaptureDirectory"] == str(
            (tmp_path / "failures").resolve()
        )
    else:
        assert "failureCaptureDirectory" not in manifest


@pytest.mark.parametrize("status,expected", [(0, 0), (1, 1), (-9, 137)])
def test_guard_propagates_native_startup_and_resource_failures(
    monkeypatch, tmp_path, status, expected
):
    spec = importlib.util.spec_from_file_location(
        "runtime_guard", repository_root() / "scripts/run_speech_swift_guarded.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    def rusage(*_):
        return 0

    monkeypatch.setattr(
        guard.ctypes, "CDLL", lambda *_, **__: SimpleNamespace(proc_pid_rusage=rusage)
    )
    child = SimpleNamespace(poll=lambda: status, wait=lambda **_: status)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *_: child)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guard",
            "--server",
            str(tmp_path / "server"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--metrics",
            str(tmp_path / "metrics.jsonl"),
        ],
    )
    if expected:
        with pytest.raises(SystemExit) as error:
            guard.main()
        assert error.value.code == expected
    else:
        guard.main()
