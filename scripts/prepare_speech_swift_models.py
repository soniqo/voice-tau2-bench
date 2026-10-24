#!/usr/bin/env python3
"""Stage pinned local model assets; inference itself never needs the network."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

PINS = {
    "gemma": (
        "aufklarer/gemma-4-E4B-it-MLX-4bit",
        "561d32b215dea1fd2747e5367b094581813588b3",
    ),
    "agent12b": (
        "mlx-community/gemma-4-12B-it-4bit",
        "73bcf09092aa277861d5a191b989b666f7f32e8f",
    ),
    "vox": ("aufklarer/VoxCPM2-MLX-bf16", "6ab24e43c00cb54e129ef93b77edbb8e33190a65"),
    "whisper": (
        "argmaxinc/whisperkit-coreml",
        "0f63a7800b00dd0226abd051b906c246e1907482",
    ),
    "vad": (
        "aufklarer/Silero-VAD-v6.2.1-CoreML",
        "523876545a57961474fee9df913e833e130560b8",
    ),
    "tokenizer": (
        "openai/whisper-large-v3",
        "06f233fe06e710322aca913c1bc4249a0d71fce1",
    ),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stage(kind, root):
    repo, revision = PINS[kind]
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    ) as response:
        info = json.load(response)
    folder = root / repo
    if kind == "whisper":
        folder = root / "argmaxinc/whisper-large-v3-coreml"
    records = []
    for entry in info["siblings"]:
        name = entry["rfilename"]
        relative = name
        if kind == "whisper":
            prefix = "openai_whisper-large-v3/"
            if not name.startswith(prefix):
                continue
            relative = name[len(prefix) :]
        elif kind == "vad":
            if not (name == "config.json" or name.startswith("silero_vad.mlmodelc/")):
                continue
        elif kind == "tokenizer":
            if name not in {"tokenizer.json", "tokenizer_config.json"}:
                continue
        elif name not in {
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        } and not (
            kind == "agent12b"
            and (
                name.endswith(".safetensors") or name == "model.safetensors.index.json"
            )
        ):
            continue
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size = entry.get("size", 0)
        expected_sha = (entry.get("lfs") or {}).get("sha256")
        if not target.exists() or target.stat().st_size != size:
            remaining = size - (target.stat().st_size if target.exists() else 0)
            if shutil.disk_usage(root).free - remaining < 7 * 1024**3:
                raise RuntimeError(f"Insufficient disk space to stage {repo}/{name}")
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}"
            partial = target.with_suffix(target.suffix + ".partial")
            print(
                f"Downloading {kind}: {relative} ({size / 1024**2:.1f} MiB)", flush=True
            )
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                partial.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            if partial.stat().st_size != size:
                raise RuntimeError(f"Incomplete download: {name}")
            actual_sha = digest(partial)
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(f"SHA256 mismatch: {name}")
            partial.replace(target)
        else:
            actual_sha = digest(target)
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(f"Corrupt cached asset: {target}")
        if not expected_sha and entry.get("blobId"):
            contents = target.read_bytes()
            git_blob = hashlib.sha1(
                f"blob {len(contents)}\0".encode() + contents
            ).hexdigest()
            if git_blob != entry["blobId"]:
                raise RuntimeError(f"Git blob mismatch: {target}")
        records.append({"path": relative, "size": size, "sha256": actual_sha})
    return folder, {"repository": repo, "revision": revision, "files": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / "Library/Caches/qwen3-speech/models"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument(
        "--failure-capture-directory",
        type=Path,
        help="Opt in to private native chat failure records, including raw 12B output",
    )
    parser.add_argument(
        "--agent-model",
        choices=["gemma4-e4b-it", "gemma4-12b-it"],
        default="gemma4-e4b-it",
    )
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    config = {"assets": {}}
    if args.failure_capture_directory is not None:
        config["failureCaptureDirectory"] = str(
            args.failure_capture_directory.resolve()
        )
    for kind in ("gemma", "vox", "whisper", "vad", "tokenizer"):
        folder, provenance = stage(kind, args.cache)
        config[kind + "Directory"] = str(folder)
        config["assets"][kind] = provenance
    if args.agent_model == "gemma4-12b-it":
        folder, provenance = stage("agent12b", args.cache)
        config["agentDirectory"] = str(folder)
        config["assets"]["agent"] = provenance
    tokenizer = Path(config["tokenizerDirectory"])
    for name in ("tokenizer.json", "tokenizer_config.json"):
        source = tokenizer / name
        if not source.is_file():
            raise RuntimeError(f"Missing staged Large-v3 tokenizer: {source}")
        target = Path(config["whisperDirectory"]) / name
        shutil.copyfile(source, target)
        config["assets"]["whisper"]["files"].append(
            {"path": name, "sha256": digest(target), "copied_from": str(source)}
        )
    source_root = args.runtime_source.resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    source_files = subprocess.check_output(
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
        cwd=source_root,
        text=True,
    ).splitlines()
    source_digest = hashlib.sha256()
    for name in sorted(set(source_files)):
        source_digest.update(name.encode() + b"\0" + (source_root / name).read_bytes())
    config["runtimeRevision"] = revision + "+source-sha256:" + source_digest.hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Ready: {args.output}", flush=True)


if __name__ == "__main__":
    main()
