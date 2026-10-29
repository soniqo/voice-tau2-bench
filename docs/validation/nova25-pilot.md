# Optional Amazon Nova 2.5 Sonic probe

This explicit experimental probe checks native audio input, tool calls, returned
audio, usage accounting and database grading. It is separate from the local
`tau-voice` CLI and does not initialize a cloud SDK on local execution paths.
This probe does not run domain tasks. Use the [full Nova runner](../nova.md)
for airline, retail and telecom calls.

```bash
uv sync --frozen --extra dev --extra nova
uv run --frozen --extra nova python scripts/probe_nova.py \
  --profile benchmark \
  --input request.wav --output data/simulations/nova-probe \
  --max-seconds 45 --stop-usd 0.25
```

Supply a mono PCM16 recording of at most 30 seconds and an explicitly configured
local AWS profile with Bedrock streaming access. Credentials are resolved at
connection time and never written to artifacts. Empty session tokens are omitted
from signing. The model is `amazon.nova-2-5-sonic` in `us-east-1`, with 16 kHz
PCM16 input and 24 kHz output. Sessions have a maximum 120-second duration, no
SDK retries and bounded tool calls. Exceptions stay failures with no fallback.

The controlled recording must request creation of a task for `user_1` titled
`Important Meeting`. Speak punctuation as necessary. Reference actions and the
expected title are used only by subsequent grading, not added to the agent prompt.

On 2026-10-29 the live probe received the request, called `create_task` with the
correct user/title and returned spoken confirmation. The preserved environment
evaluator scored the database control **1**, and a replay with a deliberately
wrong title scored **0**. A second functional run verified source-fingerprint
recording with the same positive/negative results. The runs took **18.49** and
**18.83 seconds**, including the **14.20-second** recording and shutdown, and
each returned **3.08 seconds** of audio. These are functional checks, not a domain
pass rate or response-latency metric. The [recorded controls](../results/nova25-functional-controls.json)
include usage, timing and fingerprints.

Final usage was 503 input speech tokens, 956 input text tokens, 77 output speech
tokens and 95 output text tokens: **$0.00300973** at the recorded rates. The
stream closed and final usage was drained. Initial development probes remain
separate, including authentication/transport errors and an adapter timeout after
a correct tool call; they are not task scores.

The second functional run recorded **$0.00300489**. Some earlier failed probes
returned no usage meter; their cost remains unknown rather than being reported
as zero.

The [AWS regional price list](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json),
published 2026-10-28, lists $3/$12 per million input/output speech tokens and
$0.33/$2.75 per million input/output text tokens. The probe counts cumulative
modality meters once, rejects missing/decreasing meters, and preserves raw usage.
USD is a metered estimate, not an invoice. A spending threshold closes the stream
when observed usage reaches it; delayed usage can overshoot, so it is not a
provider-enforced dollar ceiling. Unknown usage remains unknown, never zero.

The [full Nova task runner](../nova.md) now connects benchmark audio frames,
atomic tools/results, delivered-speech transcripts, barge-in and session teardown.
It requires a qualified local caller and a declared task/time/budget plan.
Its separate [two-turn production-runner control](../results/nova25-full-pipeline-control.json)
also passed positive and negative frozen-evaluator checks with final usage.
These earlier probe controls remain functional evidence rather than domain scores.
Tasks, tools, policies and grading remain frozen; adding the provider does not
sync new upstream tasks or evaluators.
