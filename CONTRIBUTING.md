# Contributing

Use a separate branch and worktree. Install with `uv sync --frozen --extra dev`, then run `make test`, `make lint`, and `make verify`. Native integration tests are opt-in and must run sequentially.

Keep the frozen English frame unchanged. Describe changes to the local profile explicitly and preserve source/model fingerprints. Add focused tests for transport, tool history, audio behavior, cancellation, or batch checkpoints as applicable.

Open changes for review with their behavior, validation, and limitations. Do not publish benchmark scores without identifying the custom local caller, voices, judge, seeds, and acoustic profile.
