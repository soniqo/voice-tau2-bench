"""One standalone local voice task using the preserved tools and evaluator."""

import audioop
import json
import time
import wave
from pathlib import Path

from tau2.agent.speech_swift_agent import SpeechSwiftAgent
from tau2.backchannel import default_backchannel_settings
from tau2.config import (
    DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS,
)
from tau2.data_model.audio import PCM_SAMPLE_RATE
from tau2.data_model.language import CallContext, TaskSelection
from tau2.data_model.semantic_communication import (
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.voice import (
    RuntimeTTSConfig,
    SynthesisConfig,
    VoiceSettings,
)
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.inference import InferenceBackends, use_inference_backends
from tau2.multilingual.runtime import (
    agent_language_guidance,
    language_environment_kwargs,
    load_language,
    resolve_task,
)
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.modes import CommunicationMode
from tau2.registry import registry
from tau2.task_subsets.store import load_subset
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator
from tau2.user_simulation_voice_presets import sample_voice_config
from tau2.voice.baseline import repository_root, verify_baseline
from tau2.voice.caller_policy import (
    LOCAL_CALLER_GUIDELINES,
    CallerPolicyError,
    QualifiedCallerBackend,
)
from tau2.voice.diagnostics import inference_diagnostics
from tau2.voice.execution import InferenceLane, SimulationClock
from tau2.voice.local_orchestrator import LocalV2Orchestrator
from tau2.voice.provenance import run_provenance, stable_provenance
from tau2.voice.qualification import require_qualification
from tau2.voice.runtime_profile import load_profile
from tau2.voice.speech_swift import SpeechSwiftBackend


def voice_tasks(domain: str):
    """Load the exact English voice-v1 membership and canonical scenarios."""
    if domain == "mock":
        tasks = registry.get_tasks_loader(domain)()
        loaded = None
    else:
        loaded = load_language(
            "en",
            domain,
            mode="voice",
            localize_db=False,
            canonical_tasks=True,
            selection=TaskSelection(task_split_name="base"),
        )
        frame = json.loads(
            (repository_root() / "data/tau2/benchmarks/voice-v1.json").read_text()
        )
        selected_ids = load_subset(frame["task_subsets"][domain]["name"]).task_ids
        index = {task.id: task for task in loaded.tasks}
        tasks = [index[task_id] for task_id in selected_ids]
    return tasks, loaded


def task_context(domain: str, task_id: str | None, seed: int):
    """Select a task without exposing its private caller goal to the agent."""
    tasks, loaded = voice_tasks(domain)
    context = None
    task = next((task for task in tasks if task_id is None or task.id == task_id), None)
    if task is None:
        raise ValueError(
            f"Task {task_id!r} is not in the selected {domain} voice frame"
        )
    if loaded is not None:
        context = resolve_task(loaded, task, seed)
    return task, loaded, context


def run_local_task(
    *,
    domain: str,
    task_id: str | None,
    base_url: str,
    output: Path,
    seed: int = 42,
    max_seconds: float | None = None,
    wall_timeout: float | None = None,
    complexity: str = "control",
    expected_provenance: dict | None = None,
    profile_path: Path | None = None,
    qualification_path: Path | None = None,
) -> dict:
    if (output / "simulation.json").exists():
        raise ValueError("Task output already contains a completed simulation")
    verify_baseline()
    task, loaded, language = task_context(domain, task_id, seed)
    profile = load_profile(profile_path)
    max_seconds, wall_timeout = profile.budgets(max_seconds, wall_timeout)
    backend = SpeechSwiftBackend(base_url)
    try:
        capabilities = backend.capabilities()
        profile.check_runtime(capabilities)
        provenance = run_provenance(backend)
        qualification = require_qualification(qualification_path, profile, provenance)
        if (
            expected_provenance is not None
            and stable_provenance(provenance) != expected_provenance
        ):
            raise ValueError("Batch source or runtime changed before the next task")
        env_kwargs = language_environment_kwargs(loaded) if loaded else {}
        environment = registry.get_env_constructor(domain)(**env_kwargs)
        scoring = CommunicateScoring(
            scorer="semantic",
            judge=SemanticCommunicateSettings(
                model=profile.roles["judge"].model_id,
                model_args={"temperature": 0, "max_tokens": 1024},
            ),
        )
        synthesis = SynthesisConfig(
            provider="runtime",
            provider_config=RuntimeTTSConfig(
                model_id="voxcpm2",
                voice_id="A natural adult male voice speaking American English.",
                seed=seed,
            ),
        )
        sampled = sample_voice_config(seed, synthesis, complexity=complexity)
        synthesis.channel_effects_config = sampled.channel_effects_config
        synthesis.source_effects_config = sampled.source_effects_config
        synthesis.speech_effects_config = sampled.speech_effects_config
        speech_env = sampled.to_speech_environment(seed).model_copy(
            update={"voice_id": synthesis.provider_config.voice_id}
        )
        # Reject requested acoustic assets that are absent, before any inference.
        if (
            sampled.source_effects_config.enable_background_noise
            and not speech_env.background_noise_file
        ):
            raise ValueError(
                "The regular acoustic profile requires the upstream background-noise assets"
            )
        policy = environment.get_policy()
        if language is not None:
            policy += "\n\n" + agent_language_guidance(language)
        clock = SimulationClock() if profile.execution.mode == "simulation-v2" else None
        lane = InferenceLane(backend, clock) if clock else None
        caller_backend = lane.for_role("caller") if lane else backend
        if profile.execution.caller_policy == "terminal-v2":
            caller_backend = QualifiedCallerBackend(
                caller_backend,
                str(task.user_scenario),
                device_grounding=profile.execution.device_grounding
                and domain == "telecom",
            )
        agent = SpeechSwiftAgent(
            environment.get_tools(),
            policy,
            backend=lane.for_role("agent") if lane else backend,
            model_id=profile.roles["agent"].model_id,
            seed=seed,
            clock=clock,
            tool_guidance=profile.execution.tool_guard,
        )
        user = VoiceStreamingUserSimulator(
            tools=environment.get_user_tools(include=task.user_tools)
            if environment.user_tools is not None
            else None,
            instructions=str(task.user_scenario),
            llm=profile.roles["caller"].model_id,
            llm_args={"temperature": 0, "max_tokens": 1024},
            decision_model=profile.roles["listener"].model_id,
            decision_model_args={"temperature": 0, "max_tokens": 16},
            voice_settings=VoiceSettings(
                synthesis_config=synthesis,
                transcription_config=None,
                speech_environment=speech_env,
            ),
            persona_config=language.persona
            if language and language.persona
            else sampled.persona_config,
            backchannel=default_backchannel_settings("en"),
            realtime_generation=clock is None,
            global_guidelines_override=LOCAL_CALLER_GUIDELINES if clock else None,
            chunk_size=round(PCM_SAMPLE_RATE * 0.05),
            tick_duration_seconds=0.05,
            wait_to_respond_threshold_other=round(
                DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS / 0.05
            ),
            wait_to_respond_threshold_self=round(
                profile.execution.caller_idle_seconds / 0.05
            ),
            use_llm_backchannel=sampled.use_llm_backchannel,
            backchannel_min_threshold=int(sampled.backchannel_min_threshold / 0.05)
            if sampled.backchannel_min_threshold
            else None,
            interruption_check_interval=40,
        )
        orchestrator_type = LocalV2Orchestrator if clock else FullDuplexOrchestrator
        orchestrator = orchestrator_type(
            **({"clock": clock} if clock else {}),
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            seed=seed,
            max_steps=round(max_seconds / 0.05) - (1 if clock else 0),
            tick_duration_seconds=0.05,
            timeout=wall_timeout,
            call_context=CallContext(language=language, communicate_scoring=scoring)
            if language
            else None,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "runtime.json").write_text(
            capabilities.model_dump_json(indent=2) + "\n"
        )
        (output / "profile.json").write_text(profile.model_dump_json(indent=2) + "\n")
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (output / "settings.json").write_text(
            json.dumps(
                {
                    "domain": domain,
                    "profile": profile.profile,
                    "agent_model": profile.roles["agent"].model_id,
                    "task_id": task.id,
                    "seed": seed,
                    "max_seconds": max_seconds,
                    "wall_timeout_seconds": wall_timeout,
                    "clock_mode": profile.execution.mode,
                    "caller_idle_seconds": profile.execution.caller_idle_seconds,
                    "device_grounding": profile.execution.device_grounding,
                    "qualification": qualification,
                    "complexity": complexity,
                    "tick_seconds": 0.05,
                    "agent_system_prompt": agent.get_init_state().messages[0].content,
                    "agent_model_args": {"temperature": 0, "max_tokens": 1024},
                    "agent_tts_seed": seed,
                    "caller_model_args": {"temperature": 0, "max_tokens": 1024},
                    "decision_model_args": {"temperature": 0, "max_tokens": 16},
                    "semantic_judge_model_args": {"temperature": 0, "max_tokens": 1024},
                },
                indent=2,
            )
            + "\n"
        )
        started = time.perf_counter()
        with use_inference_backends(
            InferenceBackends(
                chat=caller_backend,
                synthesis=caller_backend,
                transcription=caller_backend,
            )
        ):
            try:
                simulation = orchestrator.run()
                if clock and any(
                    event["execution_status"] == "rejected"
                    for event in orchestrator.tool_guards["user"].events
                ):
                    raise CallerPolicyError(
                        "Caller attempted an undeclared or repeated failing tool call; trial is unscored"
                    )
                simulation.reward_info = evaluate_simulation(
                    simulation=simulation,
                    task=task,
                    evaluation_type=EvaluationType.ALL,
                    solo_mode=False,
                    domain=domain,
                    mode=CommunicationMode.FULL_DUPLEX,
                    env_kwargs=env_kwargs,
                    communicate_scoring=scoring,
                )
            except (Exception, KeyboardInterrupt) as error:
                (output / "failed-run.json").write_text(
                    json.dumps(
                        {
                            "error": str(error),
                            "error_type": type(error).__name__,
                            "domain": domain,
                            "task_id": task.id,
                            "ticks": [
                                tick.model_dump(mode="json")
                                for tick in orchestrator.ticks
                            ],
                            "agent_metrics": agent.metrics,
                            "inference": backend.events,
                            "inference_accounting": lane.events if lane else [],
                            "caller_decisions": getattr(
                                caller_backend, "decisions", []
                            ),
                            "clock_mode": profile.execution.mode,
                            "audio_seconds": len(orchestrator.ticks) * 0.05,
                            "scored": False,
                            "diagnostics": inference_diagnostics(
                                backend.events,
                                agent.metrics["cancellations"],
                                provenance["native"]["configuration"].get(
                                    "failureCaptureDirectory"
                                ),
                            ),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                raise
        simulation.info = {
            **(simulation.info or {}),
            "local_profile": profile.profile,
            "complexity": complexity,
            "local_metrics": agent.metrics,
            "caller_stale_generations": user.stale_generation_count,
            "standard_leaderboard": False,
            "clock_mode": profile.execution.mode,
            "inference_accounting": lane.events if lane else [],
            "caller_decisions": getattr(caller_backend, "decisions", []),
            "tool_metrics": {
                role: guard.events for role, guard in orchestrator.tool_guards.items()
            }
            if clock
            else {},
            "qualification": qualification,
            "diagnostics": inference_diagnostics(
                backend.events,
                agent.metrics["cancellations"],
                provenance["native"]["configuration"].get("failureCaptureDirectory"),
            ),
        }
        (output / "simulation.json").write_text(
            simulation.model_dump_json(indent=2) + "\n"
        )
        (output / "failed-run.json").unlink(missing_ok=True)
        (output / "inference.json").write_text(
            json.dumps(backend.events, indent=2) + "\n"
        )
        for role in ("agent", "user"):
            with wave.open(str(output / f"{role}.wav"), "wb") as wav:
                wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                for tick in simulation.ticks:
                    chunk = getattr(tick, f"{role}_chunk")
                    data = (
                        chunk.get_audio_bytes()
                        if chunk and chunk.audio_content
                        else b"\xff" * 400
                    )
                    wav.writeframes(audioop.ulaw2lin(data, 2))
        return {
            "domain": domain,
            "task_id": task.id,
            "termination": simulation.termination_reason.value,
            "reward": simulation.reward_info.reward,
            "wall_seconds": time.perf_counter() - started,
            "audio_seconds": len(simulation.ticks) * 0.05,
            "clock_mode": profile.execution.mode,
            "scored": True,
            "artifacts": str(output.resolve()),
            "standard_leaderboard": False,
        }
    finally:
        backend.close()
