"""Attribute native captures to a trial by request ID, including cancellations."""

import json
from pathlib import Path


def inference_diagnostics(events, cancellations=(), native_directory=None):
    request_ids = {event["request_id"] for event in events}
    requested = {item["request_id"] for item in cancellations}
    captures = []
    read_errors = []
    if native_directory:
        for path in sorted(Path(native_directory).glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                read_errors.append({"capture": path.name, "error": str(error)})
                continue
            if record.get("requestID") in request_ids:
                captures.append(
                    {
                        "request_id": record["requestID"],
                        "error": record.get("error", ""),
                        "capture": path.name,
                    }
                )
    native_cancellations = [
        row for row in captures if row["error"] == "CancellationError()"
    ]
    native_failures = [row for row in captures if row["error"] != "CancellationError()"]
    failed = [event for event in events if not event["success"]]
    return {
        "requests": len(events),
        "capture_read_errors": read_errors,
        "cancel_requested": sorted(request_ids & requested),
        "cancelled_requests": [
            event["request_id"] for event in failed if event["request_id"] in requested
        ],
        "failed_requests_without_cancellation": [
            event["request_id"]
            for event in failed
            if event["request_id"] not in requested
        ],
        "native_chat_failures": native_failures,
        "native_cancellations": native_cancellations,
        "unexpected_native_cancellations": [
            row for row in native_cancellations if row["request_id"] not in requested
        ],
    }
