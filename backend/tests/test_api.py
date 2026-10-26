import asyncio
import json
import subprocess

import cv2
import numpy as np
import pytest
from conftest import FakeClient, call, face_at, finding, make_video, response, sam_events, smoke_track
from fastapi.testclient import TestClient
from meta_sam_parser import SegmentationBoxRecord, parse_responses_stream, video_segmentation_format

import db
import main
from analyzer import PRIVACY_WINDOW, MetaAnalyzer
from media import blur_video


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_MOCK", "1")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "UPLOADS_DIR", tmp_path / "uploads")
    return TestClient(main.app)


def post(api, path, name, mime):
    with open(path, "rb") as f:
        res = api.post("/reports", files={"file": (name, f, mime)}, data={"lat": "28.6", "lng": "77.4"})
    assert res.status_code == 201, res.text
    return res.json()["id"]


def test_image_report_trace_and_media(api, tmp_path):
    img = tmp_path / "photo.jpg"
    cv2.imwrite(str(img), np.full((120, 160, 3), 128, np.uint8))
    report_id = post(api, img, "photo.jpg", "image/jpeg")

    steps = api.get(f"/reports/{report_id}/trace").json()["steps"]
    kinds = [s["type"] for s in steps]
    assert kinds[0] == "reasoning" and kinds[-1] == "final"
    assert any(s["type"] == "tool_result" and '"rejected"' in s["result"] for s in steps)
    assert api.get("/reports/nope/trace").status_code == 404
    assert api.get(f"/reports/{report_id}/overlay").status_code == 404  # images have no video overlay
    assert api.get(f"/media/{report_id}/original.jpg").status_code == 404


def test_video_report_serves_only_blurred_copy(api, tmp_path):
    clip = make_video(tmp_path / "clip.mp4", seconds=2, fps=10)
    report_id = post(api, clip, "clip.mp4", "video/mp4")
    folder = db.UPLOADS_DIR / report_id

    assert api.get(f"/media/{report_id}/evidence.mp4").status_code == 200
    # The upload and anything else on disk stay private, however the path is spelled.
    assert (folder / "original.mp4").is_file()
    for name in ["original.mp4", "trace.json", "overlay.json", "original.MP4", "evidence.mp4.bak",
                 "..%2Foriginal.mp4", "%2E%2E/original.mp4"]:
        assert api.get(f"/media/{report_id}/{name}").status_code == 404, name
    assert api.get(f"/media/..%2F{report_id}/original.mp4").status_code == 404

    overlay = api.get(f"/reports/{report_id}/overlay").json()
    assert overlay["video"] == f"/media/{report_id}/evidence.mp4"
    assert [t["track_id"] for t in overlay["tracks"]] == ["1", "2"]

    # The wire lines parse with the real parser into one track per finding.
    lane = {"item_id": "m", "output_index": 0, "content_index": 0}
    events = [{"type": "response.output_text.delta", "delta": line + "\n", **lane}
              for line in overlay["wire"].splitlines()]
    events += [{"type": "response.output_text.done", "text": overlay["wire"], **lane}, {"type": "response.completed"}]

    async def parse():
        async def source():
            for e in events:
                yield e
        return await parse_responses_stream(source(), video_segmentation_format()).final_result()

    result = asyncio.run(parse())
    assert not result.diagnostics
    boxes = [r for r in result.records if isinstance(r, SegmentationBoxRecord)]
    assert {b.object_id for b in boxes} == {"1", "2"} and len(boxes) == 2 * 20


def ffprobe(path, entry):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          f"stream={entry}", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_blurred_video_pipeline(tmp_path):
    clip = make_video(tmp_path / "clip.mp4", seconds=3, fps=10)  # 30 frames
    client = FakeClient(
        muse_script=[
            response("r1", call("segment", {"prompt": "smoke"}, "c1")),
            response("r2", call("submit_findings", {"findings": [finding("s1:1")], "summary": "Burning."}, "c2")),
        ],
        sam_results={"smoke": smoke_track(), "face": face_at(20),
                     "license plate": sam_events({2: [("4", (120, 90, 149, 109), 0.7)]})},
    )
    result = MetaAnalyzer(client=client).analyze(clip)
    assert set(result.privacy_boxes) == {20, 2}  # every frame SAM saw a face or plate, not just evidence frames

    out = tmp_path / "evidence.mp4"
    assert blur_video(clip, out, result.privacy_boxes, PRIVACY_WINDOW) == 30
    assert ffprobe(out, "codec_name") == "h264" and ffprobe(out, "nb_read_frames") == "30"

    cap_in, cap_out = cv2.VideoCapture(str(clip)), cv2.VideoCapture(str(out))
    frames_in = [cap_in.read()[1] for _ in range(30)]
    frames_out = [cap_out.read()[1] for _ in range(30)]
    face, plate = (slice(12, 28), slice(12, 28)), (slice(92, 108), slice(122, 148))

    def blurred(i, region):
        return frames_out[i][region].std() < frames_in[i][region].std() / 4

    assert blurred(20, face) and blurred(20 - PRIVACY_WINDOW, face) and blurred(20 + PRIVACY_WINDOW, face)
    assert not blurred(20 - PRIVACY_WINDOW - 1, face)  # outside the safety window, untouched
    assert blurred(2, plate) and blurred(0, plate)
    assert not blurred(0, face)
    assert json.dumps(result.overlay)  # overlay stays JSON-safe


def test_startup_seeds_empty_db_with_known_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "UPLOADS_DIR", tmp_path / "uploads")
    with TestClient(main.app) as api:  # `with` runs the startup hook
        reports = api.get("/reports").json()
    assert reports and all(f["rule"] for r in reports for f in r["findings"])
