import json

from tau2.voice.diagnostics import inference_diagnostics


def test_failures_are_attributed_to_exact_trial_and_cancellation_intent(tmp_path):
    for id, error in (
        ("other-trial", "bad parse"),
        ("barge-in", "CancellationError()"),
        ("bad", "malformed output"),
        ("unexpected", "CancellationError()"),
    ):
        (tmp_path / f"{id}.json").write_text(
            json.dumps({"requestID": id, "error": error})
        )
    events = [
        {"request_id": id, "success": False} for id in ("barge-in", "bad", "unexpected")
    ]
    report = inference_diagnostics(events, [{"request_id": "barge-in"}], tmp_path)
    assert [row["request_id"] for row in report["native_chat_failures"]] == ["bad"]
    assert report["cancelled_requests"] == ["barge-in"]
    assert report["failed_requests_without_cancellation"] == ["bad", "unexpected"]
    assert [row["request_id"] for row in report["unexpected_native_cancellations"]] == [
        "unexpected"
    ]


def test_unrelated_directory_files_do_not_inflate_trial_failures(tmp_path):
    (tmp_path / "later-cleanup.json").write_text(
        json.dumps({"requestID": "unrelated", "error": "CancellationError()"})
    )
    report = inference_diagnostics(
        [{"request_id": "this-trial", "success": True}], native_directory=tmp_path
    )
    assert report["native_chat_failures"] == report["native_cancellations"] == []
