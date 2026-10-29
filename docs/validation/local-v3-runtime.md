# Local v3 validation — 2026-10-28

The local voice pipeline completed airline task 12 with Gemma 4 12B INT4 as the agent, E4B INT4 as caller/listener/judge, full Whisper Large-v3 CoreML FP16, VoxCPM2 FP16 network weights with FP32 AudioVAE, and Silero VAD. Hardware was an Apple M5 Pro with 48 GB unified memory on a shared workstation. Benchmark inference used no paid providers.

Validation passed 886 offline tests, 15 actual-model voice checks, v3 caller/judge calibration 91/91, and legacy v2 calibration 84/84. The native codec passed 26 Foundation-only tests; the full Swift package test suite was not run. Seven captured native generation failures returned valid bounded responses. The controlled spoken mock task earned 1 for the expected database write and 0 for a deliberately wrong write. The mock control contributes no score to the frozen 278-task frame.

## Airline diagnostic

The fresh call used task 12, seed 70, clean control acoustics, 600 simulated seconds and a 1,200-second wall watchdog. It ended normally with `user_stop` after 228.45 simulated seconds. The CLI process took 372.2 wall seconds, including its preflight. There were no failed inference requests or cancellations. Earlier attempts interrupted by unrelated local builds remain unscored.

The task reward was **0/1**, with DB reward 0 and communication reward 1. This is a valid model failure in one selected development task, not a model ranking or an overall accuracy estimate:

- The agent skipped the fare searches and calculation, and quoted an unsupported $350 upgrade within the caller's $650 limit.
- The caller approved that heard price and the two requested bags.
- The agent then executed a $1,200 upgrade and supplied `nonfree_baggages=2`, adding a $100 baggage charge while claiming the bags were free.
- The frozen task requires leaving the cabin unchanged and adding two bags with `nonfree_baggages=0`. Strict environment replay therefore rejected the final database state.

These amounts are simulated benchmark transactions. Paid-provider inference cost was **$0**. No private caller goals, reference actions or hidden service state were supplied to the agent. The caller acted on heard statements; it could not inspect the agent's actual charges.

## Reproduction

Use the [single-domain v3 commands](../local-v2.md#single-domain-validation) after rebuilding the matching native runtime, staging its pinned assets, and qualifying the current source. V3's price and partial-refusal controls are a separate experimental condition from v2.

The validated candidate was uncommitted. Its executable benchmark-source SHA-256 was `6813aede6670ce4d805ba07f2e0541671989ea515dcbc5f379a28c24c388e6bd`. Native source was based on `9ce78eaec038b4dc8b519a8b01ecb4059549cfec` with source SHA-256 `1986c2739d7246f857c45bb8467622d705a65bc22e44862d31c93db56a6af6ab`. The archived native binary SHA-256 was `408a42698f68720887598d03d6f67711d0867df572709748bc478bb16d85987d`.

The earlier caller-defective development call remains excluded from comparison. Its captured refusal loop is retained as regression evidence; the corrected caller now asks for the outstanding independent request in that full replay.
