"""Offline fakes: an OpenAI-shaped client, Muse responses and SAM streams in the real wire format."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster  # only to build valid mask payloads

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H = 160, 120  # test video size


# --- SAM wire format ---------------------------------------------------------

def _disc_payload(size=8):
    yy, xx = np.mgrid[:size, :size]
    disc = ((xx - (size - 1) / 2) ** 2 + (yy - (size - 1) / 2) ** 2 <= (size / 2) ** 2).astype(np.uint8)
    return _encode_raster(disc.ravel().tolist(), size, size)


def sam_events(frames):
    """frames: {frame_index: [(object_id, (x1, y1, x2, y2) inclusive, confidence)]} -> stream events."""
    payload = _disc_payload()
    lines = []
    for idx in sorted(frames):
        records = ",".join(
            f"{oid}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={W};h={H};c={c}|>"
            f"<|mask;x=0;y=0;c={c};data=8,8,{payload}|>"  # c sits between y and data on the wire
            for oid, (x1, y1, x2, y2), c in frames[idx]
        )
        lines.append(f"<{idx}f>{records}\n")
    lane = {"item_id": "msg_sam", "output_index": 0, "content_index": 0}
    events = [NS(type="response.output_text.delta", delta=line, **lane) for line in lines]
    events.append(NS(type="response.output_text.done", text="".join(lines), **lane))
    events.append(NS(type="response.completed"))
    return events


NO_MATCH = [NS(type="response.completed")]  # zero matches, no output_text lane at all
NO_MATCH_EMPTY_LANE = [
    NS(type="response.output_text.done", text="", item_id="msg_sam", output_index=0, content_index=0),
    NS(type="response.completed"),
]


def smoke_track():
    """Object 1 on frames 0-9, most confident (0.91) on frame 6."""
    conf = {i: 0.6 + 0.03 * i for i in range(10)}
    conf[6] = 0.91
    return sam_events({i: [("1", (40, 30, 119, 89), round(conf[i], 2))] for i in range(10)})


def face_at(frame_index):
    return sam_events({frame_index: [("1", (10, 10, 29, 29), 0.8)]})


# --- Muse responses ----------------------------------------------------------

def call(name, args, call_id):
    return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=call_id, status="completed")


def reasoning(text):
    return NS(type="reasoning", summary=[NS(type="summary_text", text=text)])


def response(rid, *items):
    return NS(id=rid, output=list(items))


def finding(sam_object, violation_type="garbage_burning", rule_id="S1-OPEN-BURNING", confidence=0.8):
    return {"violation_type": violation_type, "rule_id": rule_id, "sam_object": sam_object,
            "confidence": confidence, "description": "Smoke rising from a garbage pile."}


class FakeClient:
    def __init__(self, muse_script, sam_results):
        self.muse_script = list(muse_script)
        self.sam_results = sam_results
        self.muse_calls, self.sam_prompts, self.deleted = [], [], []
        self.files = NS(create=self._upload, delete=self.deleted.append)
        self.responses = NS(create=self._create)

    def with_options(self, **_):
        return self

    def _upload(self, purpose, file):
        assert purpose == "user_data"
        return NS(id="file-1")

    def _create(self, **kw):
        if kw["model"] == "sam-3.1":
            assert kw["stream"] is True and kw["metadata"] == {"include_confidence": "true"}
            prompt = kw["input"][0]["content"][0]["text"]
            self.sam_prompts.append(prompt)
            return iter(self.sam_results.get(prompt, NO_MATCH))
        assert kw["model"] == "muse-spark-1.3"
        self.muse_calls.append(kw)
        step = self.muse_script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


# --- media -------------------------------------------------------------------

def make_video(path, seconds=1.0, fps=10, size=(W, H)):
    w, h = size
    yy, xx = np.mgrid[:h, :w]
    checker = (((xx // 2) + (yy // 2)) % 2 * 255).astype(np.uint8)  # sharp texture, so blur is measurable
    frame = cv2.merge([checker, checker, checker])
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int(seconds * fps)):
        out.write(frame)
    out.release()
    return path


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path / "clip.mp4")
