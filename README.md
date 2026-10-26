# Saans

Evidence reports for air-pollution violations in Delhi NCR, built for the Meta Global AI Developer Hackathon.

## Setup

```bash
cp .env.example .env   # then fill in MODEL_API_KEY
python3.11 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
cp frontend/.env.example frontend/.env.local   # NEXT_PUBLIC_API_URL, required
```

## Running the app

```bash
backend/.venv/bin/python backend/scripts/seed.py              # demo reports, safe to re-run
cd backend && .venv/bin/uvicorn main:app --reload --port 8000  # API on :8000
cd frontend && npm run dev                                      # UI on :3000
```

`USE_MOCK=1` (default) makes no Meta API calls: it runs the real agent code against a scripted client (`backend/mock_agent.py`), so traces, findings and overlays have the real shapes. `SAMPLE_FPS` (default 1) sets the frame sampling rate. The frontend reads the API URL from `NEXT_PUBLIC_API_URL` (no default), and the API allows the origins in `FRONTEND_ORIGIN` (comma-separated, default `http://localhost:3000`). The API seeds the demo reports itself on startup when the database is empty. Deployment: see [DEPLOY.md](DEPLOY.md).

Self-checks: `cd backend && .venv/bin/python media.py && .venv/bin/python hotspots.py`

Tests (offline, fake Meta client): `cd backend && .venv/bin/python -m pytest tests -q`

### Real analyzer (`USE_MOCK=0`)

Muse Spark runs an agent loop: it segments suspected violations with SAM 3.1, retries with other phrases, and submits findings that must point at a SAM object. Faces and number plates are then segmented and blurred in every evidence frame. Each report folder in `data/uploads/<id>/` gets `trace.json` (served at `GET /reports/{id}/trace`) and `overlay.json` (SAM tracks, served as wire lines at `GET /reports/{id}/overlay`). Video reports also get `evidence.mp4`, an H.264 copy with faces and plates blurred on every frame; `/media` serves only that and the evidence JPEGs, never the upload.

| Env var | Default | Meaning |
|---|---|---|
| `AGENT_MAX_TURNS` | 8 | Muse turns before stopping with what is verified |
| `AGENT_MAX_SEGMENT_CALLS` | 8 | SAM calls the agent may make (privacy pass not counted) |
| `MUSE_TIMEOUT_S` / `SAM_TIMEOUT_S` | 120 / 180 | Request timeouts |
| `MAX_VIDEO_SECONDS` | 20 | Longer videos are trimmed to this before upload |

## Test scripts

Both scripts read `MODEL_API_KEY` from `.env` at the repo root. Local videos are uploaded through the Files API and deleted again after the request.

**SAM 3.1** — segment one object type in an image or video (MP4/MOV, H.264/H.265/MPEG-4):

```bash
backend/.venv/bin/python backend/scripts/test_sam.py samples/<clip>.mp4 "smoke"
```

Prints object id, box `(left, top, right, bottom)` and confidence for every frame, then saves the frame with the most detections, with boxes and mask outlines, to `data/debug/sam_annotated.png` (change with `--out`). One short noun phrase per run.

**Muse Spark 1.3** — describe visible violations in a video:

```bash
backend/.venv/bin/python backend/scripts/test_muse.py samples/<clip>.mp4
```

Streams the model's answer to the terminal.

## Sample sources

| File | Source | License |
|------|--------|---------|
| | | |
