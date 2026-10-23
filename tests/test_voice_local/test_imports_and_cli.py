import json
import os
import subprocess
import sys
import textwrap

# Treat even an attempted SDK import as a failure, including optional probes.
BLOCKER = """
import importlib.abc
import sys
BLOCKED = (
    "litellm", "openai", "anthropic", "elevenlabs", "deepgram", "boto3",
    "google.genai", "google.cloud.aiplatform", "vertexai", "livekit", "pyaudio",
    "aws_sdk_bedrock_runtime",
)
class BlockProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if any(name == module or name.startswith(module + ".") for module in BLOCKED):
            raise AssertionError("Unexpected provider SDK import: " + name)
sys.meta_path.insert(0, BlockProviders())
"""


def run(body, *args):
    env = {
        key: value for key, value in os.environ.items() if not key.endswith("API_KEY")
    }
    env["PYTHON_DOTENV_DISABLED"] = "1"
    return subprocess.run(
        [sys.executable, "-c", BLOCKER + textwrap.dedent(body), *args],
        text=True,
        capture_output=True,
        timeout=30,
        env=env,
    )


def test_voice_core_imports_without_text_agents_or_cloud_sdks():
    result = run("""
        import tau2
        from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
        from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator
        from tau2.evaluator.evaluator import evaluate_simulation
        from tau2.registry import registry
        from tau2.inference import generate
        assert 'tau2.orchestrator.orchestrator' not in sys.modules
        assert 'tau2.agent.llm_agent' not in sys.modules
        assert 'tau2.user.user_simulator' not in sys.modules
        assert 'tau2.domains.banking_knowledge.environment' not in sys.modules
        assert 'tau2.domains.intake.environment' not in sys.modules
        for domain in ('airline', 'retail', 'telecom'):
            assert registry.get_tasks_loader(domain)()
            assert registry.get_env_constructor(domain)().get_tools()
        assert registry.get_user_constructor('voice_streaming_user_simulator') is VoiceStreamingUserSimulator
        print('isolated')
    """)
    assert result.returncode == 0, result.stderr
    assert "isolated" in result.stdout


def test_offline_cli_replays_real_files_without_provider_sdks(tmp_path):
    result = run(
        """
        from tau2.voice.cli import main
        sys.argv = ['tau-voice', 'replay', '--output', sys.argv[1]]
        main()
    """,
        str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["matches_upstream"] is True
    assert [case["reward"] for case in report["cases"]] == [1.0, 0.0]
    artifact = json.loads((tmp_path / "voice-create-task.json").read_text())
    assert artifact["info"]["synthetic_audio"] is True
    assert artifact["ticks"][1]["agent_tool_calls"][0]["name"] == "create_task"


def test_cli_rejects_an_incompatible_runtime(tmp_path):
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(
        json.dumps({"runtime": "fixture", "revision": "test", "models": {}})
    )
    result = run(
        """
        from tau2.voice.cli import main
        sys.argv = ['tau-voice', 'check-runtime', '--capabilities', sys.argv[1]]
        main()
    """,
        str(capabilities),
    )
    assert result.returncode == 1
    assert "whisper-large-v3" in result.stderr
    assert "voxcpm2" in result.stderr
    assert "compatible" not in result.stdout


def test_lazy_public_exports_keep_legacy_class_identity():
    result = run("""
        import warnings
        import tau2
        from tau2.orchestrator.base import BaseOrchestrator
        from tau2.orchestrator.orchestrator import BaseOrchestrator as LegacyBase
        from tau2.user.prompts import build_user_system_prompt
        from tau2.user.user_simulator import build_user_system_prompt as LegacyPrompt
        from tau2.registry import registry
        assert tau2.registry is registry
        assert tau2.Registry is type(registry)
        assert BaseOrchestrator is LegacyBase
        assert build_user_system_prompt is LegacyPrompt
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            assert tau2.BaseAgent is tau2.HalfDuplexAgent
            assert seen[-1].category is DeprecationWarning
    """)
    assert result.returncode == 0, result.stderr
