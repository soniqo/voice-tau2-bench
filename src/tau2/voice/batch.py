"""Sequential, resumable trials for the standalone English voice frame."""

import hashlib
import json
import math
from pathlib import Path

from tau2.voice.baseline import repository_root, verify_baseline
from tau2.voice.local_run import run_local_task, voice_tasks
from tau2.voice.provenance import run_provenance, stable_provenance
from tau2.voice.runtime_profile import LocalVoiceProfile
from tau2.voice.speech_swift import SpeechSwiftBackend

DOMAINS = ("airline", "retail", "telecom")


def write_json(path: Path, value) -> None:
    """Publish a complete checkpoint, including when replacing an earlier one."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def plan_jobs(domain: str, num_tasks: int | None, trials: int, seed: int) -> list[dict]:
    """Use frozen order, with an independent reproducible seed for every trial."""
    if trials < 1 or (num_tasks is not None and num_tasks < 1):
        raise ValueError("trials and num_tasks must be positive")
    if domain not in (*DOMAINS, "all", "mock"):
        raise ValueError(f"Unsupported voice domain: {domain}")
    jobs = []
    for name in DOMAINS if domain == "all" else (domain,):
        tasks, _ = voice_tasks(name)
        for task in tasks[:num_tasks]:
            task_key = hashlib.sha256(task.id.encode()).hexdigest()[:16]
            for trial in range(trials):
                jobs.append(
                    {
                        "domain": name,
                        "task_id": task.id,
                        "trial": trial,
                        "seed": seed + trial,
                        "case": f"{name}/{task_key}/trial-{trial:03d}",
                    }
                )
    return jobs


def summarize(jobs: list[dict], results: list[dict], trials: int) -> dict:
    """Report pass^k only for tasks with all requested trials completed."""
    task_results = {}
    for result in results:
        key = (result["domain"], result["task_id"])
        task_results.setdefault(key, []).append(result)
    complete = {key: rows for key, rows in task_results.items() if len(rows) == trials}

    def scores(groups):
        values = {}
        for k in range(1, trials + 1):
            reliability = [
                math.comb(sum(row["reward"] == 1 for row in rows), k)
                / math.comb(trials, k)
                for rows in groups
            ]
            values[str(k)] = (
                sum(reliability) / len(reliability) if reliability else None
            )
        return {"tasks": len(groups), "pass^k": values}

    return {
        "standard_leaderboard": False,
        "planned_trials": len(jobs),
        "completed_trials": len(results),
        "complete": len(results) == len(jobs),
        "overall": scores(list(complete.values())),
        "domains": {
            domain: scores(
                [rows for (name, _), rows in complete.items() if name == domain]
            )
            for domain in sorted({job["domain"] for job in jobs})
        },
    }


def artifact_hashes(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"result.json", "result.json.tmp"}
    }


def run_batch(
    *,
    domain: str,
    num_tasks: int | None,
    trials: int,
    seed: int,
    base_url: str,
    output: Path,
    max_seconds: float = 600,
    complexity: str = "control",
    resume: bool = False,
) -> dict:
    """Run one task at a time; infrastructure errors stop without receiving a score."""
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    verify_baseline()
    jobs = plan_jobs(domain, num_tasks, trials, seed)
    profile = LocalVoiceProfile.model_validate_json(
        (repository_root() / "configs/voice/soniqo-local-v1.json").read_text()
    )
    backend = SpeechSwiftBackend(base_url)
    try:
        capabilities = backend.capabilities()
        profile.check_runtime(capabilities)
        provenance = run_provenance(backend)
    finally:
        backend.close()
    manifest = {
        "schema_version": 1,
        "profile": profile.model_dump(mode="json"),
        "provenance": stable_provenance(provenance),
        "capabilities": capabilities.model_dump(mode="json"),
        "max_seconds": max_seconds,
        "complexity": complexity,
        "jobs": jobs,
    }
    manifest_path = output / "batch.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError(
                "Batch already exists; use --resume after reviewing its artifacts"
            )
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Cannot resume: task plan, source, models or settings changed"
            )
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("A new batch requires an empty output directory")
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)
    results = []
    for job in jobs:
        case_output = output / job["case"]
        result_path = case_output / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if any(result.get(key) != value for key, value in job.items()):
                raise ValueError(f"Saved result does not match job {job['case']}")
            if result.get("artifact_hashes") != artifact_hashes(case_output):
                raise ValueError(f"Saved artifacts changed for {job['case']}")
        else:
            if case_output.exists() and any(case_output.iterdir()):
                archive = output / "interrupted"
                archive.mkdir(exist_ok=True)
                index = 0
                while (archive / f"{job['case'].replace('/', '-')}-{index}").exists():
                    index += 1
                case_output.rename(archive / f"{job['case'].replace('/', '-')}-{index}")
            result = run_local_task(
                domain=job["domain"],
                task_id=job["task_id"],
                seed=job["seed"],
                base_url=base_url,
                output=case_output,
                max_seconds=max_seconds,
                complexity=complexity,
                expected_provenance=manifest["provenance"],
            )
            result.update(job)
            result["artifact_hashes"] = artifact_hashes(case_output)
            write_json(result_path, result)
        results.append(result)
        write_json(output / "summary.json", summarize(jobs, results, trials))
    return summarize(jobs, results, trials)
