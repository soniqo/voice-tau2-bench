"""Content fingerprints for local runs, including uncommitted runtime changes."""

import hashlib
import json
import subprocess

from tau2.voice.baseline import repository_root, verify_baseline


def benchmark_sources() -> dict:
    """Hash executable inputs rather than relying on a possibly dirty Git HEAD."""
    root = repository_root()
    paths = [*root.glob("src/**/*.py"), *root.glob("scripts/*.py")]
    paths += [root / name for name in ("pyproject.toml", "uv.lock")]
    paths += list(root.glob("configs/voice/*.json"))
    inputs = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
    )
    return {
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "inputs": inputs,
        "sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest(),
    }


def run_provenance(backend) -> dict:
    """Record frozen data, native asset pins and both working source revisions."""
    native = backend.provenance()
    if not native.get("configuration", {}).get("assets"):
        raise ValueError("Native inference must report staged model asset pins")
    from tau2.utils.utils import DATA_DIR

    acoustic_assets = {
        str(path.relative_to(DATA_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((DATA_DIR / "voice").rglob("*"))
        if path.is_file()
    }
    return {
        "benchmark": benchmark_sources(),
        "baseline": verify_baseline(),
        "native": native,
        "acoustic_assets": acoustic_assets,
    }


def stable_provenance(provenance: dict) -> dict:
    """Exclude process counters so a restarted identical runtime can resume."""
    native = provenance["native"]
    return {
        "benchmark": provenance["benchmark"],
        "baseline": provenance["baseline"],
        "acoustic_assets": provenance.get("acoustic_assets", {}),
        "native": {
            "configuration": native["configuration"],
            "voxcpm2_parameter_elements": native["voxcpm2_parameter_elements"],
        },
    }
