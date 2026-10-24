"""Voice-only development entry point; never initializes a cloud provider."""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from tau2.voice.baseline import repository_root, verify_baseline


def main():
    parser = argparse.ArgumentParser(prog="tau-voice")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="ERROR"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Verify the pinned English 278-task voice frame")
    replay = commands.add_parser(
        "replay", help="Replay synthetic audio and real mock-domain tools offline"
    )
    replay.add_argument(
        "--output", type=Path, help="Directory for complete replay artifacts"
    )
    check = commands.add_parser(
        "check-runtime",
        help="Validate a runtime's reported capabilities against the requested models",
    )
    check.add_argument("--capabilities", type=Path, required=True)
    check.add_argument(
        "--profile",
        type=Path,
        default=repository_root() / "configs/voice/soniqo-local-v1.json",
    )
    run = commands.add_parser(
        "run", help="Run a voice task with the local speech-swift runtime"
    )
    run.add_argument(
        "--domain", choices=["mock", "airline", "retail", "telecom"], required=True
    )
    run.add_argument("--task-id")
    run.add_argument("--profile", type=Path, help="Explicit local model-role profile")
    run.add_argument("--base-url", default="http://127.0.0.1:8080")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-seconds", type=float, default=600)
    run.add_argument("--complexity", choices=["control", "regular"], default="control")
    batch = commands.add_parser("batch", help="Run sequential resumable voice trials")
    batch.add_argument(
        "--domain",
        choices=["all", "mock", "airline", "retail", "telecom"],
        default="all",
    )
    batch.add_argument("--num-tasks", type=int, help="Limit tasks per domain")
    batch.add_argument("--trials", type=int, default=1)
    batch.add_argument("--seed", type=int, default=42)
    batch.add_argument("--base-url", default="http://127.0.0.1:8080")
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--max-seconds", type=float, default=600)
    batch.add_argument(
        "--complexity", choices=["control", "regular"], default="control"
    )
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--profile", type=Path, help="Explicit local model-role profile")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    try:
        manifest = verify_baseline()
        if args.command == "verify":
            result = {
                "profile": manifest["profile"],
                "tasks": manifest["frame"]["total_tasks"],
                "verified_inputs": len(manifest["inputs"]),
            }
        elif args.command == "check-runtime":
            from tau2.voice.runtime_profile import (
                LocalVoiceProfile,
                RuntimeCapabilities,
            )

            profile = LocalVoiceProfile.model_validate_json(args.profile.read_text())
            capabilities = RuntimeCapabilities.model_validate_json(
                args.capabilities.read_text()
            )
            profile.check_runtime(capabilities)
            result = {
                "profile": profile.profile,
                "runtime": capabilities.runtime,
                "revision": capabilities.revision,
                "compatible": True,
            }
        elif args.command == "batch":
            from tau2.voice.batch import run_batch

            result = run_batch(
                domain=args.domain,
                num_tasks=args.num_tasks,
                trials=args.trials,
                seed=args.seed,
                base_url=args.base_url,
                output=args.output,
                max_seconds=args.max_seconds,
                complexity=args.complexity,
                resume=args.resume,
                profile_path=args.profile,
            )
        elif args.command == "run":
            from tau2.voice.local_run import run_local_task

            result = run_local_task(
                domain=args.domain,
                task_id=args.task_id,
                base_url=args.base_url,
                output=args.output,
                seed=args.seed,
                max_seconds=args.max_seconds,
                complexity=args.complexity,
                profile_path=args.profile,
            )
        else:
            from tau2.voice.replay import canonical_trace, run_replay

            cases = []
            for name in ("voice-create-task", "voice-tool-flow"):
                fixture = repository_root() / f"data/tau2/replays/{name}.json"
                simulation, deliveries = run_replay(fixture)
                expected = json.loads(fixture.with_suffix(".expected.json").read_text())
                if canonical_trace(simulation, deliveries) != expected:
                    raise ValueError(
                        f"Replay {name} differs from the pinned upstream trace"
                    )
                if args.output:
                    args.output.mkdir(parents=True, exist_ok=True)
                    (args.output / f"{name}.json").write_text(
                        simulation.model_dump_json(indent=2) + "\n"
                    )
                cases.append(
                    {
                        "name": name,
                        "ticks": len(simulation.ticks),
                        "reward": simulation.reward_info.reward,
                    }
                )
            result = {"replay_only": True, "matches_upstream": True, "cases": cases}
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, StopIteration) as error:
        parser.exit(1, f"tau-voice: {error}\n")


if __name__ == "__main__":
    main()
