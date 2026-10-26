# Saans

Hackathon project for the Meta Global AI Developer Hackathon (Oct 26 to Nov 4 2026). Solo, indie developer track, theme "Build for Your World".

## What it does

Delhi NCR air quality collapses every winter and GRAP restrictions kick in, but violations that make it worse keep happening: open garbage burning, construction sites without dust control, uncovered debris piles, uncovered trucks carrying debris. Saans takes a photo or short video of a suspected violation and produces an evidence report.

Flow:
1. User uploads a photo or short clip and drops a pin on the map for its location.
2. A Muse Spark agent looks at the media and decides what to look for.
3. The agent calls SAM 3.1 (through a tool) to detect, segment and track the evidence, checks that detections hold across frames, and retries with different prompts when unsure.
4. Faces and number plates are found with SAM and blurred before anything is saved or shown.
5. The agent matches what it found to a rule in `data/grap_rules.json` and writes a short report with marked-up evidence frames.
6. Reports appear on a map. Reports close to each other in location get grouped so repeat hotspots stand out.

## Stack

- Backend: Python 3.11, FastAPI, `openai` SDK, `meta-sam-parser`, OpenCV, ffmpeg, SQLite
- Frontend: Next.js (TypeScript, App Router), `@meta-sam/parser`, `@meta-sam/graphics`, `@meta-sam/video`, `@meta-sam/react` for mask overlays on video, Leaflet + OpenStreetMap for the map
- Deploy: Vercel for frontend, Hugging Face Spaces (Docker) for backend

## Meta Model API

- OpenAI SDK compatible. Base URL `https://api.meta.ai/v1`. Key from `MODEL_API_KEY` env var. The OpenAI SDK does not auto-read this variable, so always pass `api_key` explicitly.
- `muse-spark-1.3`: reasoning, tool calling, accepts text, image, pdf and video input. Use the Responses API.
- `sam-3.1`: segmentation and video tracking over the Responses API. One short noun phrase per request ("dump truck", "smoke"); send several requests for several object types. Max 16 objects per prompt. Output is a special-token text format, always parse it with `meta_sam_parser` (Python) or `@meta-sam/parser` (TS), never by hand. Zero matches returns empty output, handle that. Video must be MP4/MOV with H.264, H.265 or MPEG-4.
- Read docs/META_API_NOTES.md first. Full pages are in .meta-docs/. Only fetch dev.meta.ai online if something isn't covered there. Never invent fields or parameters. If the docs don't answer it, stop and ask.

## Layout

```
backend/     FastAPI app, agent, tools, scripts/
frontend/    Next.js app
data/        grap_rules.json, sqlite db, uploads (uploads and db are gitignored)
samples/     test clips (licensed stock footage, see README for sources)
```

## Hard rules

- Never run git commit, git add or any git command that changes history. Yasharth handles all commits himself.
- Everything in this repo must be written during the hackathon. Don't copy in code from other projects.
- Only use media and data we have the rights to (Pexels, Pixabay, Creative Commons, datasets with a permissive license). Record the source of every sample in the README.
- Keep it non-political. Saans helps people document violations, it does not criticize governments or authorities.
- No medical or health advice anywhere in the app or reports.
- Keep secrets in `.env`, never hardcode keys, never print the key.
- Blur faces and number plates before any frame is stored or returned to the frontend.

## Judging (what to optimize for)

Four equal criteria: impact, creative use of Meta's AI tooling (go beyond a basic integration), technical depth and execution, and the demo (3-minute video, live product must hold up). "Simple image analyzers" are on the organizers' anti-project list, so the agent loop, video tracking and the hotspot map are the core of the project, not extras.

## Working style

- Keep code simple and readable. Prefer a few clear modules over clever abstractions.
- After each step, summarize what changed and what to run to test it.
- If a step is getting large, say so and suggest splitting it.