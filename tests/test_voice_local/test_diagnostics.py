import json

import pytest

from tau2.voice.diagnostics import (
    InferenceRunError,
    inference_diagnostics,
    require_successful_inference,
)


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
    assert report["confirmed_cancellations"] == ["barge-in"]
    assert report["unresolved_failed_requests"] == ["bad", "unexpected"]
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


@pytest.mark.parametrize("requested", [False, True])
def test_cancel_intent_cannot_hide_a_native_parser_failure(tmp_path, requested):
    (tmp_path / "bad.json").write_text(
        json.dumps({"requestID": "bad", "error": "malformed output"})
    )
    events = [{"request_id": "bad", "success": False}]
    cancellations = [{"request_id": "bad"}] if requested else []
    with pytest.raises(InferenceRunError, match="Trial is unscored.*bad"):
        require_successful_inference(events, cancellations, tmp_path)


def test_cancel_intent_without_acknowledged_cancellation_remains_unscored():
    with pytest.raises(InferenceRunError):
        require_successful_inference(
            [{"request_id": "bad", "success": False}], [{"request_id": "bad"}]
        )


@pytest.mark.parametrize("requested", [False, True])
def test_verified_native_cancellation_requires_matching_intent(tmp_path, requested):
    (tmp_path / "cancel.json").write_text(
        json.dumps({"requestID": "cancel", "error": "CancellationError()"})
    )
    events = [{"request_id": "cancel", "success": False}]
    cancellations = [{"request_id": "cancel"}] if requested else []
    if requested:
        assert require_successful_inference(events, cancellations, tmp_path)[
            "confirmed_cancellations"
        ] == ["cancel"]
    else:
        with pytest.raises(InferenceRunError):
            require_successful_inference(events, cancellations, tmp_path)


@pytest.mark.parametrize("requested", [False, True])
def test_closed_stream_requires_matching_cancel_request(requested):
    events = [{"request_id": "stream", "success": False, "cancelled": True}]
    cancellations = [{"request_id": "stream"}] if requested else []
    if requested:
        assert not require_successful_inference(events, cancellations)[
            "unresolved_failed_requests"
        ]
    else:
        with pytest.raises(InferenceRunError):
            require_successful_inference(events, cancellations)
