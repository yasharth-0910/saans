"""Saans API. Run from backend/: uvicorn main:app --reload"""

import asyncio
import json
import os
import re
import shutil
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import db  # noqa: E402  (env must be loaded before modules read it)
from analyzer import PRIVACY_WINDOW, get_analyzer  # noqa: E402
from hotspots import find_hotspots  # noqa: E402
from media import IMAGE_EXTS, VIDEO_EXTS, blur_video, read_frame, save_evidence  # noqa: E402
from sam import overlay_wire  # noqa: E402
from scripts.seed import main as seed_demo  # noqa: E402

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
RULES = {r["id"]: r for r in db.load_rules()}


@asynccontextmanager
async def lifespan(_app):
    # Hugging Face Spaces wipes local storage on restart, so an empty database gets the demo reports.
    # In a thread: the seed runs the scripted agent, whose SAM parsing calls asyncio.run.
    if not db.list_reports():
        await asyncio.to_thread(seed_demo)
    yield


app = FastAPI(title="Saans", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def public(report):
    """Attach GRAP rules and turn frame names into URLs. The original upload is never exposed."""
    return {
        **report,
        "findings": [{**f, "rule": RULES.get(f["rule_id"])} for f in report["findings"]],
        "frames": [f"/media/{report['id']}/{name}" for name in report["frames"]],
    }


@app.post("/reports", status_code=201)
def create_report(file: UploadFile, lat: Annotated[float, Form()], lng: Annotated[float, Form()]):
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise HTTPException(400, "lat/lng out of range")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type. Use one of {sorted(IMAGE_EXTS | VIDEO_EXTS)}")

    report_id = uuid4().hex[:12]
    out_dir = db.UPLOADS_DIR / report_id
    out_dir.mkdir(parents=True)
    media_path = out_dir / f"original{ext}"
    try:
        size = 0
        with open(media_path, "wb") as dst:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File too large (max 200 MB)")
                dst.write(chunk)
        if read_frame(media_path) is None:
            raise HTTPException(400, "Could not read the image or video")

        result = get_analyzer().analyze(media_path)
        findings = [asdict(f) for f in result.findings]
        evidence_src = result.media_path or media_path  # e.g. the trimmed video the agent saw
        frames = save_evidence(evidence_src, findings, result.redactions, out_dir, result.masks)
        if ext in VIDEO_EXTS:
            # The only video ever served: faces and plates blurred on every frame.
            blur_video(evidence_src, out_dir / "evidence.mp4", result.privacy_boxes, PRIVACY_WINDOW)
        (out_dir / "trace.json").write_text(json.dumps(result.trace, indent=1))
        if result.overlay is not None:
            (out_dir / "overlay.json").write_text(json.dumps(result.overlay))
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    report = {
        "id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lat": lat,
        "lng": lng,
        "media_type": "video" if ext in VIDEO_EXTS else "image",
        "summary": result.summary,
        "findings": findings,
        "frames": frames,
    }
    db.save_report(report)
    return public(report)


@app.get("/reports")
def list_reports():
    return [public(r) for r in db.list_reports()]


@app.get("/reports/{report_id}")
def get_report(report_id: str):
    report = db.get_report(report_id)
    if not report:
        raise HTTPException(404, "Report not found")
    return public(report)


@app.get("/reports/{report_id}/trace")
def get_trace(report_id: str):
    if not re.fullmatch(r"[a-z0-9-]+", report_id) or not db.get_report(report_id):
        raise HTTPException(404, "Report not found")
    path = db.UPLOADS_DIR / report_id / "trace.json"
    return {"report_id": report_id, "steps": json.loads(path.read_text()) if path.is_file() else []}


@app.get("/reports/{report_id}/overlay")
def get_overlay(report_id: str):
    """Blurred video URL plus SAM wire lines (one track per finding) for @meta-sam/parser."""
    if not re.fullmatch(r"[a-z0-9-]+", report_id) or not db.get_report(report_id):
        raise HTTPException(404, "Report not found")
    path = db.UPLOADS_DIR / report_id / "overlay.json"
    video = db.UPLOADS_DIR / report_id / "evidence.mp4"
    if not path.is_file() or not video.is_file():
        raise HTTPException(404, "No video overlay for this report")
    overlay = json.loads(path.read_text())
    return {
        "video": f"/media/{report_id}/evidence.mp4",
        "frame_width": overlay["frame_width"],
        "frame_height": overlay["frame_height"],
        "tracks": [{"track_id": str(o["finding"] + 1), "finding": o["finding"], "prompt": o["prompt"]}
                   for o in overlay["objects"]],
        "wire": overlay_wire(overlay),
    }


@app.get("/hotspots")
def hotspots():
    return find_hotspots(db.list_reports())


@app.get("/media/{report_id}/{name}")
def media(report_id: str, name: str):
    # Only rendered, blurred evidence is served: frame JPEGs and the blurred video. Never the upload.
    if not re.fullmatch(r"[a-z0-9-]+", report_id) or not re.fullmatch(r"frame_\d{5}\.jpg|evidence\.mp4", name):
        raise HTTPException(404)
    path = db.UPLOADS_DIR / report_id / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
