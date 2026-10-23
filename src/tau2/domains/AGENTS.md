# AGENTS.md — src/tau2/domains/

> See `README.md` for full architecture and data storage details.

## Rules for Working in This Directory

### Required Files per Domain

Every domain in `src/tau2/domains/<name>/` MUST have:

- `data_model.py` — `DB` subclass (e.g., `FlightDB(DB)`)
- `tools.py` — `ToolKitBase` subclass with `@is_tool(ToolType.READ|WRITE|GENERIC|THINK)` decorated methods
- `environment.py` — exports `get_environment()`, `get_tasks()`, and optionally `get_tasks_split()`
- `utils.py` — data paths pointing to `data/tau2/domains/<name>/`
- `__init__.py`

Optional: `user_tools.py`, `user_data_model.py`.

### Registration Is Mandatory

After creating or renaming a domain, register it in `src/tau2/registry.py`:

```python
registry.register_domain(get_environment, "domain_name")
registry.register_tasks(get_tasks, "domain_name", get_task_splits=get_tasks_split)
```

If you forget this step, the domain will not be usable via CLI.

### Task Split Rules

- The split file MUST include a `base` split — this is the default for evaluation.
- Split file naming follows `split_<tasks_file_stem>.json` (e.g., `split_tasks.json` for `tasks.json`).
- `train`/`test` splits are for RL experiments only.

### Tool Decorator Types

Use the correct `ToolType` — this affects evaluation:

- `ToolType.READ` — read-only queries (e.g., look up a flight)
- `ToolType.WRITE` — state-mutating operations (e.g., cancel a reservation)
- `ToolType.GENERIC` — general-purpose
- `ToolType.THINK` — internal reasoning (no side effects)

### Solo Mode

If the domain does NOT support solo mode, raise `ValueError("Solo mode not supported for <domain>")` in `get_environment()`. Do not silently ignore it.

### Data Files

Domain data lives in `data/tau2/domains/<name>/`. Be careful modifying `db.json`/`db.toml` and `tasks.json` — the framework depends on these at runtime. Run `tau-voice verify` to check the frozen voice inputs. Do not modify their pins to accommodate code changes.

### Tool Arguments

Some tools accept `dict` arguments and convert them internally to Pydantic models. When adding new tools, prefer typed Pydantic model arguments. If accepting `dict`, convert explicitly (e.g., `FlightInfo(**flight)`).

### Tests

Tests go in `tests/test_domains/test_<domain_name>/`:
- `test_tools_<domain_name>.py` — test tools via `environment.get_response(ToolCall(...))`
- `test_user_tools_<domain_name>.py` — if user tools exist

Run: `pytest tests/test_domains/test_<domain_name>/`
