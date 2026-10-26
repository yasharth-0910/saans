import json

import cv2
import httpx2
import numpy as np
import openai
import pytest
from conftest import (
    NO_MATCH, NO_MATCH_EMPTY_LANE, FakeClient, call, face_at, finding, reasoning, response, sam_events, smoke_track,
)

import sam
from analyzer import LAST_TURN_PROMPT, MetaAnalyzer, _AgentRun
from media import save_evidence, trim_video, video_duration


def outputs_sent(client, muse_call_index):
    """function_call_output payloads sent with the given Muse request, keyed by call_id."""
    items = client.muse_calls[muse_call_index]["input"]
    return {i["call_id"]: i["output"] for i in items if i.get("type") == "function_call_output"}


def happy_client():
    return FakeClient(
        muse_script=[
            response("r1", reasoning("Looks like smoke near a garbage heap."),
                     call("get_grap_rules", {}, "c1"),
                     call("segment", {"prompt": "burning garbage"}, "c2")),
            response("r2", call("segment", {"prompt": "smoke"}, "c3")),  # retry with a different phrase
            response("r3", call("submit_findings", {
                "findings": [finding("s2:1"), finding("s2:7")],  # s2:7 never existed
                "summary": "Open burning visible."}, "c4")),
            response("r4", call("submit_findings", {"findings": [finding("s2:1")],
                                                    "summary": "Open burning visible."}, "c5")),
        ],
        sam_results={"smoke": smoke_track(), "face": face_at(6)},  # anything else: zero matches
    )


def test_agent_retries_rejects_and_finishes(video):
    client = happy_client()
    result = MetaAnalyzer(client=client).analyze(video)

    # The retry happened and the bad finding was bounced back to the model.
    assert client.sam_prompts == ["burning garbage", "smoke", "face", "license plate"]
    assert "No matches" in outputs_sent(client, 1)["c2"]
    rejection = json.loads(outputs_sent(client, 3)["c4"])
    assert rejection["status"] == "rejected" and rejection["accepted"] == 1
    assert "never returned" in rejection["rejected"][0]["reason"]

    # Turns are chained and instructions resent every time.
    assert [c.get("previous_response_id") for c in client.muse_calls] == [None, "r1", "r2", "r3"]
    assert all(c["instructions"] for c in client.muse_calls)
    assert client.muse_calls[0]["input"][0]["content"][1] == {"type": "input_file", "file_id": "file-1"}

    # The model sees a compact summary, never mask payloads.
    seg = json.loads(outputs_sent(client, 2)["c3"])
    assert seg["segment_id"] == "s2"
    assert seg["objects"][0] == {"object_id": "1", "frames": "0-9", "frame_count": 10,
                                 "mean_confidence": seg["objects"][0]["mean_confidence"],
                                 "best_frame": 6, "box_at_best_frame": [40, 30, 120, 90]}
    assert "payload" not in outputs_sent(client, 2)["c3"]

    # One verified finding at its best frame, with a mask inside its box.
    [f] = result.findings
    assert (f.sam_object, f.frame_index, f.box, f.rule_id) == ("s2:1", 6, [40, 30, 120, 90], "S1-OPEN-BURNING")
    ys, xs = np.nonzero(result.masks[0])
    assert result.masks[0].shape == (120, 160) and xs.min() >= 40 and xs.max() < 120 and ys.min() >= 30
    assert "Note:" not in result.summary

    # Privacy pass found the face even though the agent never asked for it.
    assert result.redactions == {6: [[7, 7, 33, 33]]}

    # Overlay data keeps every tracked frame with its encoded mask; trace has every step.
    assert len(result.overlay["objects"][0]["frames"]) == 10
    assert result.overlay["objects"][0]["frames"][0]["mask"]["payload"].startswith("!")
    kinds = [s["type"] for s in result.trace]
    assert {"reasoning", "tool_call", "tool_result", "privacy", "final"} <= set(kinds)
    assert client.deleted == ["file-1"]


def test_privacy_blur_applied_before_drawing(video, tmp_path):
    result = MetaAnalyzer(client=happy_client()).analyze(video)
    findings = [vars(f) for f in result.findings]
    [name] = save_evidence(video, findings, result.redactions, tmp_path, result.masks)
    saved = cv2.imread(str(tmp_path / name))
    cap = cv2.VideoCapture(str(video))
    ok, original = cap.read()
    assert ok
    face = (slice(10, 30), slice(10, 30))
    assert saved[face].std() < original[face].std() / 4, "face region was not blurred"
    away = (slice(95, 115), slice(130, 155))  # no box, no blur here
    assert abs(float(saved[away].std()) - float(original[away].std())) < 20


def test_turn_limit_returns_verified_so_far(video, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_TURNS", "3")
    client = FakeClient(
        muse_script=[
            response("r1", call("segment", {"prompt": "smoke"}, "c1")),
            response("r2", call("submit_findings", {"findings": [finding("s1:1"), finding("s9:1")],
                                                    "summary": "Burning seen."}, "c2")),
            response("r3", call("segment", {"prompt": "dust"}, "c3")),
        ],
        sam_results={"smoke": smoke_track()},
    )
    result = MetaAnalyzer(client=client).analyze(video)
    assert len(client.muse_calls) == 3
    assert {"role": "user", "content": LAST_TURN_PROMPT} in client.muse_calls[2]["input"]
    assert [f.sam_object for f in result.findings] == ["s1:1"]
    assert "Note:" in result.summary and "3 turns" in result.summary
    assert client.sam_prompts[-2:] == ["face", "license plate"]  # privacy still ran


def test_segment_limit_and_repeat_cache(video, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_SEGMENT_CALLS", "1")
    client = FakeClient(
        muse_script=[
            response("r1", call("segment", {"prompt": "smoke"}, "c1")),
            response("r2", call("segment", {"prompt": " Smoke"}, "c2"), call("segment", {"prompt": "fire"}, "c3")),
            response("r3", call("submit_findings", {"findings": [finding("s1:1")], "summary": "Burning."}, "c4")),
        ],
        sam_results={"smoke": smoke_track()},
    )
    result = MetaAnalyzer(client=client).analyze(video)
    sent = outputs_sent(client, 2)
    assert "Already segmented" in sent["c2"]
    assert "Segment limit" in sent["c3"]
    assert client.sam_prompts == ["smoke", "face", "license plate"]
    assert len(result.findings) == 1


def test_model_timeout_stops_cleanly(video):
    timeout = openai.APITimeoutError(request=httpx2.Request("POST", "https://api.meta.ai/v1/responses"))
    client = FakeClient(muse_script=[timeout], sam_results={})
    result = MetaAnalyzer(client=client).analyze(video)
    assert result.findings == [] and "model error" in result.summary
    assert client.sam_prompts == ["face", "license plate"]  # videos always get the privacy pass
    assert client.deleted == ["file-1"]


def test_model_text_without_tool_call_gets_nudged(video):
    client = FakeClient(
        muse_script=[
            response("r1", reasoning("Nothing obvious.")),
            response("r2", call("submit_findings", {"findings": [], "summary": "No violation visible."}, "c1")),
        ],
        sam_results={},
    )
    result = MetaAnalyzer(client=client).analyze(video)
    assert client.muse_calls[1]["input"][0]["role"] == "user"
    assert result.findings == [] and result.summary == "No violation visible."


@pytest.mark.parametrize("events", [NO_MATCH, NO_MATCH_EMPTY_LANE])
def test_zero_matches_both_wire_forms(events):
    client = FakeClient([], {"smoke": events})
    assert sam.segment(client, {"type": "input_video", "file_id": "f"}, "smoke", True, 10) == {}


def test_image_segment_parses_frame_zero():
    client = FakeClient([], {"smoke": sam_events({0: [("3", (1, 2, 50, 60), 0.7)]})})
    objects = sam.segment(client, {"type": "input_image", "image_url": "data:"}, "smoke", False, 10)
    assert objects["3"][0]["box"] == [1, 2, 51, 61] and objects["3"][0]["confidence"] == 0.7


@pytest.mark.parametrize("bad, reason", [
    ({**finding("s1:1"), "rule_id": "nope"}, "Unknown rule_id"),
    ({**finding("s1:1"), "violation_type": "construction_dust"}, "does not cover"),
    ({**finding("s1:1"), "confidence": 1.5}, "0 to 1"),
    ({**finding("s1:1"), "confidence": True}, "0 to 1"),
    ({**finding("s1:1"), "extra": 1}, "exactly the fields"),
    (finding("s1:2"), "never returned"),
])
def test_validate_rejects(bad, reason):
    run = _AgentRun(MetaAnalyzer(client=FakeClient([], {})), "file-1", {}, True)
    run.segments = {"s1": {"prompt": "smoke", "objects": {"1": {0: {}}}}}
    assert run.validate(finding("s1:1")) is None
    assert reason in run.validate(bad)


def test_frame_ranges():
    assert sam.frame_ranges([5, 0, 1, 2, 7, 8]) == "0-2, 5, 7-8"
    assert sam.frame_ranges([]) == ""


def test_trim_video(tmp_path):
    from conftest import make_video
    long = make_video(tmp_path / "long.mp4", seconds=25, fps=5, size=(32, 24))
    trimmed = trim_video(long, 20)
    assert trimmed != long and abs(video_duration(trimmed) - 20) < 0.5
    short = make_video(tmp_path / "short.mp4", seconds=2, fps=5, size=(32, 24))
    assert trim_video(short, 20) == short
