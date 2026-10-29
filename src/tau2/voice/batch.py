"""Sequential, resumable trials for the standalone English voice frame."""

import hashlib
import json
import math
from pathlib import Path

from tau2.voice.baseline import verify_baseline
from tau2.voice.local_run import run_local_task, voice_tasks
from tau2.voice.provenance import run_provenance, stable_provenance
from tau2.voice.qualification import require_qualification
from tau2.voice.runtime_profile import load_profile
from tau2.voice.speech_swift import SpeechSwiftBackend

DOMAINS = ("airline", "retail", "telecom")


def write_json(path: Path, value) -> None:
    """Publish a complete checkpoint, including when replacing an earlier one."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def plan_jobs(
    domain: str,
    num_tasks: int | None,
    trials: int,
    seed: int,
    task_plan: dict | None = None,
) -> list[dict]:
    """Use frozen order, with an independent reproducible seed for every trial."""
    if trials < 1 or (num_tasks is not None and num_tasks < 1):
        raise ValueError("trials and num_tasks must be positive")
    if domain not in (*DOMAINS, "all", "mock"):
        raise ValueError(f"Unsupported voice domain: {domain}")
    selected = None
    if task_plan is not None:
        if num_tasks is not None:
            raise ValueError("Use either an explicit task plan or num_tasks")
        if (
            not isinstance(task_plan, dict)
            or task_plan.get("schema_version") != 1
            or not isinstance(task_plan.get("tasks"), list)
            or not task_plan["tasks"]
        ):
            raise ValueError(
                "Task plan requires schema_version 1 and a nonempty tasks list"
            )
        selected = {}
        for entry in task_plan["tasks"]:
            if not isinstance(entry, dict) or not all(
                isinstance(entry.get(key), str) and entry[key]
                for key in ("domain", "task_id")
            ):
                raise ValueError(
                    "Every task-plan entry requires nonempty domain and task_id strings"
                )
            name, task_id = entry["domain"], entry["task_id"]
            if name not in DOMAINS or (domain != "all" and name != domain):
                raise ValueError("Task plan domains must match the requested domain")
            ids = selected.setdefault(name, [])
            if task_id in ids:
                raise ValueError("Task plan contains duplicate task IDs")
            ids.append(task_id)
    jobs = []
    for name in DOMAINS if domain == "all" else (domain,):
        tasks, _ = voice_tasks(name)
        if selected is not None:
            index = {task.id: task for task in tasks}
            ids = selected.get(name, [])
            if any(task_id not in index for task_id in ids):
                raise ValueError(
                    f"Task plan includes an ID outside the frozen {name} voice frame"
                )
            tasks = [index[task_id] for task_id in ids]
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
        "unscored_trials": len(jobs) - len(results),
        "scored_coverage": len(results) / len(jobs) if jobs else 0,
        "terminations": {
            reason: sum(row.get("termination") == reason for row in results)
            for reason in sorted(
                {row["termination"] for row in results if "termination" in row}
            )
        },
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
    max_seconds: float | None = None,
    wall_timeout: float | None = None,
    complexity: str = "control",
    resume: bool = False,
    profile_path: Path | None = None,
    qualification_path: Path | None = None,
    task_plan_path: Path | None = None,
    agent_factory=None,
    external_agent: dict | None = None,
) -> dict:
    """Run one task at a time; infrastructure errors stop without receiving a score."""
    verify_baseline()
    task_plan = json.loads(task_plan_path.read_text()) if task_plan_path else None
    jobs = plan_jobs(domain, num_tasks, trials, seed, task_plan)
    profile = load_profile(profile_path)
    max_seconds, wall_timeout = profile.budgets(max_seconds, wall_timeout)
    backend = SpeechSwiftBackend(base_url)
    try:
        capabilities = backend.capabilities()
        profile.check_runtime(
            capabilities,
            **(
                {"roles": {"caller", "listener", "judge", "caller_tts"}}
                if external_agent is not None
                else {}
            ),
        )
        provenance = run_provenance(backend)
        qualification = require_qualification(qualification_path, profile, provenance)
    finally:
        backend.close()
    manifest = {
        "schema_version": 1,
        "profile": profile.model_dump(mode="json"),
        "provenance": stable_provenance(provenance),
        "capabilities": capabilities.model_dump(mode="json"),
        "max_seconds": max_seconds,
        "wall_timeout_seconds": wall_timeout,
        "qualification": qualification,
        "task_plan": task_plan,
        "complexity": complexity,
        "jobs": jobs,
        **({"external_agent": external_agent} if external_agent is not None else {}),
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
            try:
                result = run_local_task(
                    domain=job["domain"],
                    task_id=job["task_id"],
                    seed=job["seed"],
                    base_url=base_url,
                    output=case_output,
                    max_seconds=max_seconds,
                    wall_timeout=wall_timeout,
                    complexity=complexity,
                    expected_provenance=manifest["provenance"],
                    profile_path=profile_path,
                    qualification_path=qualification_path,
                    **(
                        {
                            "agent_factory": agent_factory,
                            "external_agent": external_agent,
                        }
                        if agent_factory is not None
                        else {}
                    ),
                )
            except Exception as error:
                summary = summarize(jobs, results, trials)
                summary["failed_trial"] = {
                    **job,
                    "error": f"{type(error).__name__}: {error}",
                    "scored": False,
                }
                write_json(output / "summary.json", summary)
                raise
            result.update(job)
            result["artifact_hashes"] = artifact_hashes(case_output)
            write_json(result_path, result)
        results.append(result)
        write_json(output / "summary.json", summarize(jobs, results, trials))
    return summarize(jobs, results, trials)
