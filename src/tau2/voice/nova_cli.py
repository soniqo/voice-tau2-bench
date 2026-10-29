"""Explicit optional Nova voice benchmark; local entry points stay offline."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from tau2.voice.baseline import repository_root


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tau-voice-nova", description=__doc__)
    parser.add_argument(
        "--domain",
        choices=["all", "mock", "airline", "retail", "telecom"],
    )
    parser.add_argument("--task-id")
    parser.add_argument("--task-plan", type=Path)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--qualify-only",
        action="store_true",
        help="Calibrate the local caller/judge; performs no AWS inference",
    )
    parser.add_argument("--aws-profile")
    parser.add_argument("--voice", default="tiffany")
    parser.add_argument("--max-seconds", type=float, default=240)
    parser.add_argument("--budget-usd", type=float, default=1)
    parser.add_argument("--stop-usd", type=float, default=0.25)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument(
        "--complexity", choices=["control", "regular"], default="control"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.qualify_only and (
        args.domain is None or args.qualification is None or args.aws_profile is None
    ):
        parser.error("A run requires --domain, --qualification and --aws-profile")
    if not 1 <= args.max_seconds <= 420:
        parser.error("Calls must be 1–420 seconds, within one bounded Nova connection")
    if args.task_id and (
        args.domain == "all"
        or args.task_plan
        or args.num_tasks
        or args.trials != 1
        or args.resume
    ):
        parser.error("A single task ID requires one domain/trial and a fresh output")
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    from tau2.voice.batch import run_batch
    from tau2.voice.local_run import run_local_task
    from tau2.voice.nova import NovaSettings
    from tau2.voice.nova_run import NovaBudget, NovaFactory

    try:
        if args.qualify_only:
            if args.output.exists():
                raise ValueError("Qualification output already exists")
            from tau2.voice.provenance import run_provenance
            from tau2.voice.qualification import qualify
            from tau2.voice.runtime_profile import load_profile
            from tau2.voice.speech_swift import SpeechSwiftBackend

            profile = load_profile(
                repository_root() / "configs/voice/soniqo-local-gemma12b-v6.json"
            )
            backend = SpeechSwiftBackend(args.base_url)
            try:
                profile.check_runtime(
                    backend.capabilities(),
                    roles={"caller", "listener", "judge", "caller_tts"},
                )
                report = qualify(backend, profile, run_provenance(backend))
            finally:
                backend.close()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            if not report["passed"]:
                raise ValueError("Caller/judge qualification failed")
            print(json.dumps({"passed": True, "cases": len(report["cases"])}, indent=2))
            return 0
        settings = NovaSettings(
            profile=args.aws_profile,
            voice=args.voice,
            stop_usd=args.stop_usd,
            max_seconds=args.max_seconds + 20,
        )
        ledger = args.budget_ledger or args.output.with_name(
            args.output.name + "-budget.json"
        )
        factory = NovaFactory(
            settings,
            NovaBudget(ledger, args.budget_usd, recover_interrupted=args.resume),
        )
        common = dict(
            base_url=args.base_url,
            output=args.output,
            seed=args.seed,
            max_seconds=args.max_seconds,
            wall_timeout=args.max_seconds,
            complexity=args.complexity,
            qualification_path=args.qualification,
            profile_path=repository_root()
            / "configs/voice/soniqo-local-gemma12b-v6.json",
            agent_factory=factory,
            external_agent=factory.descriptor,
        )
        if args.task_id or args.domain == "mock":
            result = run_local_task(domain=args.domain, task_id=args.task_id, **common)
        else:
            result = run_batch(
                domain=args.domain,
                num_tasks=args.num_tasks,
                trials=args.trials,
                task_plan_path=args.task_plan,
                resume=args.resume,
                **common,
            )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        parser.exit(1, f"tau-voice-nova: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
