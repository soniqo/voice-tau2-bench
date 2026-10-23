"""Integrity checks for the English frame used by the local voice extraction."""

import hashlib
import json
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def verify_baseline(root: Path | None = None) -> dict:
    """Verify pinned task, policy, database, language, and domain-tool inputs."""
    if root is None:
        from tau2.utils.utils import DATA_DIR

        data_root = DATA_DIR
    else:
        data_root = root / "data"
    root = root or repository_root()
    manifest_path = root / "data/tau2/voice_local/soniqo-local-v1.json"
    manifest = json.loads(manifest_path.read_text())
    mismatches = []
    for relative, expected in manifest["inputs"].items():
        relative_path = Path(relative)
        path = (
            data_root / relative_path.relative_to("data")
            if relative_path.parts[0] == "data"
            else root / relative_path
        )
        actual = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(
            "Pinned voice inputs changed or missing: " + ", ".join(mismatches)
        )
    return manifest
