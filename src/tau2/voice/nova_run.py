"""Explicit Nova task execution with persistent spending reservations."""

import fcntl
import hashlib
import json
import math
import os
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from tau2.agent.nova_agent import NovaAgent
from tau2.voice.nova import NovaSettings


class NovaBudget:
    """Reserve before connecting; unknown usage retains its reservation."""

    def __init__(self, path: Path, limit_usd: float, *, recover_interrupted=False):
        if not math.isfinite(limit_usd) or not 0 < limit_usd <= 5:
            raise ValueError("Pilot budget must be positive and at most $5")
        self.path, self.limit = path, limit_usd
        self.recover_interrupted = recover_interrupted

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {"schema_version": 1, "limit_usd": self.limit, "reservations": []}
            )
            if data.get("schema_version") != 1 or data.get("limit_usd") != self.limit:
                raise ValueError("Existing Nova ledger uses a different budget")
            yield data
            temp = self.path.with_suffix(self.path.suffix + ".tmp")
            temp.write_text(json.dumps(data, indent=2) + "\n")
            temp.replace(self.path)

    def reserve(self, stop_usd):
        with self.locked() as data:
            if self.recover_interrupted:
                for row in data["reservations"]:
                    if row["status"] != "active" or not row.get("owner_pid"):
                        continue
                    try:
                        os.kill(row["owner_pid"], 0)
                    except ProcessLookupError:
                        row.update(
                            status="unknown",
                            recovery="Owning client exited; reservation retained",
                        )
                    except PermissionError:
                        pass
            if any(r["status"] == "active" for r in data["reservations"]):
                raise ValueError(
                    "A Nova session already owns this budget; review it before resuming"
                )
            used = sum(r["charged_usd"] for r in data["reservations"])
            # Retain room for delayed metering and an in-flight response.
            reserved = stop_usd + 0.10
            if self.limit - used < reserved:
                raise ValueError(
                    "Insufficient remaining Nova budget; no session started"
                )
            key = uuid.uuid4().hex
            data["reservations"].append(
                {
                    "id": key,
                    "status": "active",
                    "stop_usd": stop_usd,
                    "reserved_usd": reserved,
                    "charged_usd": reserved,
                    "owner_pid": os.getpid(),
                }
            )
            return key

    def finish(self, key, agent):
        with self.locked() as data:
            row = next(r for r in data["reservations"] if r["id"] == key)
            if row["status"] != "active":
                return
            usage = agent.metrics.get("metered_usd")
            finalized = agent.metrics.get("usage_finalized", False)
            if agent._thread is None:
                usage, finalized = 0, True  # No SDK connection was started.
            row.update(
                status="metered" if finalized and usage is not None else "unknown",
                metered_usd=usage,
                usage_finalized=finalized,
                charged_usd=usage
                if finalized and usage is not None
                else max(row["reserved_usd"], usage or 0),
            )


class NovaFactory:
    def __init__(self, settings: NovaSettings, budget: NovaBudget):
        self.settings, self.budget = settings, budget

    @property
    def descriptor(self):
        return {
            "provider": "nova",
            "model_id": self.settings.model_id,
            "region": self.settings.region,
            "voice": self.settings.voice,
            "model_args": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1024},
            "session_seconds": self.settings.max_seconds,
            "stop_usd": self.settings.stop_usd,
            "budget_usd": self.budget.limit,
            "budget_ledger_path_sha256": hashlib.sha256(
                str(self.budget.path.resolve()).encode()
            ).hexdigest(),
            "native_roles": ["caller", "listener", "judge", "caller_tts"],
            "pricing_revision": "AWS us-east-1 2026-10-28",
            "fixed_opener": "speech-swift/voxcpm2",
            "conversation": "Nova native audio; no local agent ASR/LLM/TTS",
        }

    def __call__(self, tools, policy, **kwargs):
        key = self.budget.reserve(self.settings.stop_usd)
        budget = self.budget

        class BudgetedNovaAgent(NovaAgent):
            def stop(self, *args, **kw):
                try:
                    return super().stop(*args, **kw)
                finally:
                    budget.finish(key, self)

        try:
            return BudgetedNovaAgent(
                tools, policy, settings=replace(self.settings), **kwargs
            )
        except BaseException:
            # Constructor performs no network operation.
            from types import SimpleNamespace

            budget.finish(key, SimpleNamespace(_thread=None, metrics={}))
            raise
