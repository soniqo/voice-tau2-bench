# Repository instructions

This repository is the standalone English τ-Voice extraction. Keep airline, retail, telecom, their tools/data/policies, and the full-duplex caller/evaluator. Use `tau-voice` for local runs; text execution, retrieval/intake domains, and web services are outside this package. The optional `tau-voice-nova` runner and functional probe require explicit opt-in and the `nova` dependency extra. They never change local defaults. Nova task runs use the separate real-time v6 condition, the original tools/grader and a qualified local caller.

Work in a separate Git worktree. Never commit, push, or post GitHub comments without explicit user confirmation. Preserve commit history; never amend or force-push without a specific request. Use neutral project wording in commits and documentation, with no tool attribution or co-author trailers.

The frozen 64-input manifest is `data/tau2/voice_local/soniqo-local-v1.json`. Do not rewrite frozen files or update their pins as a way to make tests pass. Changes to the experimental profile must be explicit and kept separate from upstream inputs. Never give the agent private caller goals, reference actions, or gold transcripts.

Use Python 3.12/3.13. Install with `uv sync --frozen --extra dev`; run `make test`, `make lint`, and `make verify`. Set UV_FROZEN=1 for routine uv commands. Only update uv.lock intentionally against the public PyPI index when dependency changes require it.

Local inference is explicitly bound through tau2.inference and the loopback speech-swift adapter. No cloud fallback, SDK probes on local paths, or hidden downloads during inference. The optional Nova probe requires explicit invocation, an AWS profile, bounded execution and spending thresholds, and raw usage accounting. Exceptions propagate and remain unscored. Preserve database/communication reward rules and capture working source and model hashes.

Run heavy model operations sequentially. Actual-model tests require SPEECH_SWIFT_INFERENCE_URL and the local_models marker; the default suite needs no model weights. Keep runtime documentation, CLI help, tests, and all README translations consistent.
