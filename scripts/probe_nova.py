"""Bounded audio-and-tool integration check; never a domain benchmark score."""

import argparse
import asyncio
import audioop
import base64
import hashlib
import json
import time
import wave
from pathlib import Path

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.registry import registry
from tau2.voice.nova import NovaSession, NovaSettings


def load_audio(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Input must be a mono PCM16 WAV")
        pcm, _ = audioop.ratecv(
            wav.readframes(wav.getnframes()), 2, 1, wav.getframerate(), 16000, None
        )
    if not pcm or len(pcm) > 30 * 32000:
        raise ValueError("Supply a nonempty spoken request of at most 30 seconds")
    return pcm


async def probe(args):
    pcm = load_audio(args.input)
    settings = NovaSettings(
        profile=args.profile, max_seconds=args.max_seconds, stop_usd=args.stop_usd
    )
    args.output.mkdir(parents=True, exist_ok=False)
    env = registry.get_env_constructor("mock")()
    tools = env.get_tools()
    declared = {t.name: t for t in tools}
    session = NovaSession(settings)
    report = {
        "model": settings.model_id,
        "region": settings.region,
        "functional_check_only": True,
        "standard_leaderboard": False,
        "domain_score": None,
        "input_sha256": hashlib.sha256(pcm).hexdigest(),
        "input_seconds": len(pcm) / 32000,
        "stop_usd": settings.stop_usd,
        "events": [],
        "tools": [],
        "success": False,
        "source_sha256": {
            str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).resolve().parents[1] / "src/tau2/voice/nova.py",
            )
        },
    }
    audio = bytearray()
    trajectory = []
    done = asyncio.Event()

    async def publish():
        started = time.monotonic()
        offset = 0
        while not done.is_set():
            frame = pcm[offset : offset + 1600] if offset < len(pcm) else bytes(1600)
            offset += len(frame)
            await session.audio(frame)
            await asyncio.sleep(max(0, started + offset / 32000 - time.monotonic()))

    async def receive():
        pending = {}
        final_text = set()
        while True:
            event = await session.receive()
            saved = json.loads(json.dumps(event))
            if "audioOutput" in saved:
                chunk = base64.b64decode(saved["audioOutput"]["content"], validate=True)
                audio.extend(chunk)
                saved["audioOutput"]["content"] = {"pcm_bytes": len(chunk)}
            report["events"].append(saved)
            if "contentStart" in event:
                start = event["contentStart"]
                fields = json.loads(start.get("additionalModelFields", "{}"))
                if (
                    start.get("role") == "ASSISTANT"
                    and fields.get("generationStage") == "FINAL"
                ):
                    final_text.add(start["contentId"])
            if "toolUse" in event:
                call = event["toolUse"]
                pending[call["contentId"]] = call
            if "contentEnd" in event:
                call = pending.pop(event["contentEnd"]["contentId"], None)
                if call:
                    if len(report["tools"]) >= 8:
                        raise ValueError("Smoke tool-call limit reached")
                    tool = declared.get(call["toolName"])
                    if tool is None:
                        raise ValueError("Nova requested an undeclared tool")
                    arguments = json.loads(call["content"])
                    tool.params.model_validate_json(json.dumps(arguments), strict=True)
                    result = tool(**arguments)
                    serial = (
                        result.model_dump(mode="json")
                        if hasattr(result, "model_dump")
                        else result
                    )
                    report["tools"].append(
                        {"name": tool.name, "arguments": arguments, "result": serial}
                    )
                    trajectory.extend(
                        [
                            AssistantMessage(
                                role="assistant",
                                tool_calls=[
                                    ToolCall(
                                        id=call["toolUseId"],
                                        name=tool.name,
                                        arguments=arguments,
                                    )
                                ],
                            ),
                            ToolMessage(
                                role="tool",
                                requestor="assistant",
                                id=call["toolUseId"],
                                content=json.dumps(serial),
                            ),
                        ]
                    )
                    await session.tool_result(call["toolUseId"], serial)
            end = event.get("contentEnd", {})
            ended = (
                end.get("contentId") in final_text
                and end.get("stopReason") == "END_TURN"
            )
            if (ended or "completionEnd" in event) and report["tools"] and audio:
                task = next(
                    t
                    for t in registry.get_tasks_loader("mock")()
                    if t.id == "create_task_1"
                )
                reward = EnvironmentEvaluator.calculate_reward(
                    registry.get_env_constructor("mock"),
                    task,
                    trajectory,
                    strict_replay=False,
                )
                report["database_control"] = reward.model_dump(mode="json")
                wrong = [m.model_copy(deep=True) for m in trajectory]
                for message in wrong:
                    for call in getattr(message, "tool_calls", None) or []:
                        if call.name == "create_task":
                            call.arguments["title"] = "Deliberately wrong title"
                negative = EnvironmentEvaluator.calculate_reward(
                    registry.get_env_constructor("mock"),
                    task,
                    wrong,
                    strict_replay=False,
                )
                report["negative_control"] = negative.model_dump(mode="json")
                report["success"] = reward.reward == 1 and negative.reward == 0
                return

    started = time.monotonic()
    try:
        async with asyncio.timeout(settings.max_seconds):
            await session.connect(
                "You are a customer service agent. Follow this policy and use the declared tools. Speak briefly.\n\n"
                + env.get_policy(),
                tools,
            )
            publisher = asyncio.create_task(publish())
            try:
                await receive()
            finally:
                done.set()
                publisher.cancel()
                await asyncio.gather(publisher, return_exceptions=True)
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["http_status"] = getattr(
            getattr(error, "response_metadata", None), "http_status_code", None
        )
        report["error_category"] = next(
            (
                code
                for code in (
                    "AccessDeniedException",
                    "ValidationException",
                    "UnrecognizedClientException",
                    "ExpiredTokenException",
                    "InvalidSignatureException",
                    "ResourceNotFoundException",
                    "ThrottlingException",
                    "ServiceUnavailableException",
                )
                if code in str(error)
            ),
            "unclassified",
        )
        # Exception messages may contain signed HTTP request details; keep them out of artifacts.
    finally:
        try:
            await asyncio.wait_for(session.close(), timeout=5)
        except Exception as error:
            report["close_error_type"] = type(error).__name__
        report.update(
            wall_seconds=time.monotonic() - started,
            usage=session.usage.totals or None,
            usd=float(session.usage.usd) if session.usage.usd is not None else None,
            cost_status="metered_estimate" if session.usage.events else "unknown",
            output_seconds=len(audio) / 48000,
            usage_finalized=getattr(session, "usage_finalized", False),
        )
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        with wave.open(str(args.output / "agent.wav"), "wb") as wav:
            wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            wav.writeframes(audio)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("events", "tools")}, indent=2
        )
    )
    return 0 if report["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--stop-usd", type=float, default=0.25)
    parser.add_argument("--max-seconds", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 120:
        parser.error(
            "The functional probe supports at most 120 seconds; use tau-voice-nova for domain calls"
        )
    raise SystemExit(asyncio.run(probe(args)))


if __name__ == "__main__":
    main()
