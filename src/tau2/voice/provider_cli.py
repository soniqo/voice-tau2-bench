"""Explicit optional voice adapters; local and Nova commands keep their defaults."""

import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path

from loguru import logger

from tau2.voice.baseline import repository_root


def session_window(call_seconds, requested=None, *, eleven=False):
    if not math.isfinite(call_seconds) or call_seconds < 1:
        raise ValueError("Call limit must be finite and at least 1 second")
    minimum = max(call_seconds + 15, 60 if eleven else 1)
    value = minimum if requested is None else requested
    if not math.isfinite(value) or value < minimum or value > 420:
        raise ValueError(
            "Provider window must cover the call plus 15 s, be at most 420 s, and at least 60 s for ElevenLabs"
        )
    return value


def public_domain(domain, seed):
    from tau2.multilingual.runtime import (
        agent_language_guidance,
        language_environment_kwargs,
    )
    from tau2.registry import registry
    from tau2.voice.local_run import task_context

    _task, loaded, language = task_context(domain, None, seed)
    environment = registry.get_env_constructor(domain)(
        **(language_environment_kwargs(loaded) if loaded else {})
    )
    policy = environment.get_policy()
    if language:
        policy += "\n\n" + agent_language_guidance(language)
    return policy, environment.get_tools()


async def inspect_eleven(agent_id, key_env):
    import httpx

    key = os.environ.get(key_env)
    if not key or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", agent_id or ""):
        raise ValueError(
            "Inspection requires an agent ID and credential environment variable"
        )
    async with httpx.AsyncClient(
        headers={"xi-api-key": key}, timeout=15, trust_env=False
    ) as client:

        async def get(path):
            response = await client.get("https://api.elevenlabs.io/v1/convai/" + path)
            response.raise_for_status()
            return response.json()

        preset = await get("agents/" + agent_id)
        config = preset["conversation_config"]
        prompt = config.get("agent", {}).get("prompt", {})
        from tau2.voice.providers.elevenlabs import preset_fingerprint, resolve_tools

        await resolve_tools(preset, get)
        return {
            "preset_sha256": preset_fingerprint(preset),
            "model": prompt.get("llm"),
            "voice": config.get("tts", {}).get("voice_id"),
        }


def main(argv=None):
    from tau2.voice.providers.base import PROVIDERS

    parser = argparse.ArgumentParser(prog="tau-voice-provider", description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--model")
    parser.add_argument("--voice")
    parser.add_argument("--backend-model", help="Named Responses model for GPT-Live")
    parser.add_argument(
        "--vertex-project",
        help="Gemini Vertex project; use an OAuth token instead of an API key",
    )
    parser.add_argument(
        "--vertex-location",
        help="Gemini Vertex location (default us-central1 with --vertex-project)",
    )
    parser.add_argument(
        "--api-key-env",
        help="Credential variable name; never put the key on the command line",
    )
    parser.add_argument(
        "--allow-cloud",
        action="store_true",
        help="Explicit opt-in to paid provider inference",
    )
    parser.add_argument(
        "--domain", choices=["all", "mock", "airline", "retail", "telecom"]
    )
    parser.add_argument("--task-id")
    parser.add_argument("--task-plan", type=Path)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=70)
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=240)
    parser.add_argument(
        "--session-seconds",
        type=float,
        help="Provider window; defaults to the call limit plus 15 s (at least 60 s for ElevenLabs)",
    )
    parser.add_argument("--budget-usd", type=float, default=1)
    parser.add_argument("--stop-usd", type=float, default=0.25)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument(
        "--complexity", choices=["control", "regular"], default="control"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--agent-id", help="Configured ElevenLabs agent")
    parser.add_argument(
        "--preset-sha256", help="Verified ElevenLabs configuration fingerprint"
    )
    parser.add_argument("--usd-per-credit", type=float)
    parser.add_argument("--max-usd-per-minute", type=float)
    parser.add_argument(
        "--export-eleven-config",
        type=Path,
        help="Write public domain policy/client tools locally; no API call",
    )
    parser.add_argument(
        "--inspect-eleven-agent",
        action="store_true",
        help="Read configuration fingerprint; does not start a conversation",
    )
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    try:
        if (
            sum(
                bool(x)
                for x in (
                    args.qualify_only,
                    args.export_eleven_config,
                    args.inspect_eleven_agent,
                )
            )
            > 1
        ):
            parser.error("Select only one setup operation")
        if args.qualify_only:
            if args.output is None:
                parser.error("Qualification requires --output")
            from tau2.voice.nova_cli import main as qualify_main

            return qualify_main(
                [
                    "--qualify-only",
                    "--base-url",
                    args.base_url,
                    "--output",
                    str(args.output),
                ]
            )
        if args.inspect_eleven_agent:
            print(
                json.dumps(
                    asyncio.run(
                        inspect_eleven(
                            args.agent_id, args.api_key_env or "ELEVENLABS_API_KEY"
                        )
                    ),
                    indent=2,
                )
            )
            return 0
        if args.export_eleven_config:
            if (
                args.domain not in {"mock", "airline", "retail", "telecom"}
                or not args.model
                or not args.voice
            ):
                parser.error("Export requires one --domain, --model and --voice")
            if args.export_eleven_config.exists():
                raise ValueError("Export output already exists")
            from tau2.voice.providers.base import agent_instructions
            from tau2.voice.providers.elevenlabs import eleven_config

            policy, tools = public_domain(args.domain, args.seed)
            config = eleven_config(
                agent_instructions(policy),
                tools,
                args.model,
                args.voice,
                session_window(args.max_seconds, args.session_seconds, eleven=True),
            )
            args.export_eleven_config.parent.mkdir(parents=True, exist_ok=True)
            args.export_eleven_config.write_text(
                json.dumps({"conversation_config": config}, indent=2) + "\n"
            )
            print(json.dumps({"exported": True, "provider_inference": False}))
            return 0
        if not args.allow_cloud:
            parser.error("Provider inference requires explicit --allow-cloud")
        if (
            not args.provider
            or not args.model
            or not args.domain
            or args.output is None
            or args.qualification is None
        ):
            parser.error(
                "A run requires --provider, --model, --domain, --output and --qualification"
            )
        if args.task_id and (
            args.domain == "all"
            or args.task_plan
            or args.num_tasks is not None
            or args.trials != 1
            or args.resume
        ):
            parser.error(
                "A single task ID requires one domain/trial and a fresh output"
            )
        if args.domain == "mock" and (
            args.task_plan
            or args.num_tasks is not None
            or args.trials != 1
            or args.resume
        ):
            parser.error("The mock control requires one fresh task/trial")
        if args.provider == "elevenlabs" and args.domain == "all":
            parser.error(
                "Use a separate verified ElevenLabs preset and run for each domain"
            )
        if args.provider == "elevenlabs" and not args.voice:
            parser.error(
                "ElevenLabs requires an explicit --voice ID matching its preset"
            )
        from tau2.voice.batch import run_batch
        from tau2.voice.local_run import run_local_task
        from tau2.voice.provider_run import ProviderBudget, ProviderFactory
        from tau2.voice.providers.base import ProviderSettings

        voice = args.voice or ("Kore" if args.provider == "gemini-live" else "marin")
        key_env = args.api_key_env or (
            "GOOGLE_OAUTH_ACCESS_TOKEN"
            if args.vertex_project
            else "GEMINI_API_KEY"
            if args.provider == "gemini-live"
            else "ELEVENLABS_API_KEY"
            if args.provider == "elevenlabs"
            else "OPENAI_API_KEY"
        )
        settings = ProviderSettings(
            args.provider,
            args.model,
            voice,
            key_env,
            max_seconds=session_window(
                args.max_seconds,
                args.session_seconds,
                eleven=args.provider == "elevenlabs",
            ),
            stop_usd=args.stop_usd,
            backend_model=args.backend_model,
            agent_id=args.agent_id,
            usd_per_credit=args.usd_per_credit,
            max_usd_per_minute=args.max_usd_per_minute,
            preset_sha256=args.preset_sha256,
            vertex_project=args.vertex_project,
            vertex_location=(args.vertex_location or "us-central1")
            if args.vertex_project
            else args.vertex_location,
        )
        ledger = args.budget_ledger or args.output.with_name(
            args.output.name + "-budget.json"
        )
        factory = ProviderFactory(
            settings,
            ProviderBudget(ledger, args.budget_usd, recover_interrupted=args.resume),
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
        if args.task_id is not None or args.domain == "mock":
            result = run_local_task(domain=args.domain, task_id=args.task_id, **common)
            (args.output / "result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
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
        parser.exit(1, f"tau-voice-provider: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
