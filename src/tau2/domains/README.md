# τ-Voice domains

The benchmark includes three customer-service domains from the frozen English `voice-v1` frame:

| Domain | Tasks | Agent work | Caller tools |
| --- | ---: | --- | --- |
| `airline` | 50 | Flight reservations, changes, cancellations, and support | None |
| `retail` | 114 | Orders, returns, exchanges, and product support | None |
| `telecom` | 114 | Account support and device/network troubleshooting | Available when the task allows them |

`mock` provides small development fixtures and is excluded from the 278-task benchmark score.

Each domain supplies a database model, policy, tools, environment constructor, and task loader. Data lives in `data/tau2/domains/<domain>/`; frozen membership is resolved through `data/tau2/benchmarks/voice-v1.json` and its three task subsets. The English language pack provides the selected caller scenarios.

`evaluation_criteria.actions` supplies one reference trajectory for deriving the expected database state. A different sequence that reaches the same required state can pass. Task-selected communication facts are graded separately. See the [evaluation guide](../../../docs/evaluation.md).

Run from the repository root:

```bash
uv run --frozen tau-voice verify
uv run --frozen pytest tests/test_domains/test_airline
uv run --frozen pytest tests/test_domains/test_retail
uv run --frozen pytest tests/test_domains/test_telecom
```

Treat the domain source, policies, databases, and task files as frozen benchmark inputs. Verify their pins rather than updating them to accommodate an implementation change.
