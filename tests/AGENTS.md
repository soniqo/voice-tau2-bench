# Test instructions

`make test` runs all offline tests without native models or API keys. `make verify` checks the frozen 278-task frame and both upstream replay traces. Keep domain-specific checks under test_domains and local transport/agent/batch tests under test_voice_local.

Mark every actual-model test `local_models` and require SPEECH_SWIFT_INFERENCE_URL. Run those only with the single native service loaded, one test process at a time. Never load another multi-gigabyte model alongside it.

Test failure propagation, real audio conversion, tool/result history, cancellation, and resume provenance. Use fixtures for offline inference. Do not relax frozen hashes, reward checks, or provider-import guards to make a change pass.
