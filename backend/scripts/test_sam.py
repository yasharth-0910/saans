"""Send a local image or video to sam-3.1 and print what it finds.

Usage: python scripts/test_sam.py <image-or-video> "<noun phrase>" [--out path.png]
"""

import argparse
import asyncio
import base64
import mimetypes
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import (
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTS = {".mp4", ".mov"}


async def run_sam(client, path, prompt):
    is_video = path.suffix.lower() in VIDEO_EXTS
    uploaded = None
    if is_video:
        # Docs: upload local video via the Files API, then reference by file_id.
        with open(path, "rb") as f:
            uploaded = await client.files.create(purpose="user_data", file=f)
        media = {"type": "input_video", "file_id": uploaded.id}
    else:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        media = {"type": "input_image", "image_url": f"data:{mime};base64,{data}"}

    try:
        stream = await client.responses.create(
            model="sam-3.1",
            input=[{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}, media],
            }],
            stream=True,
            metadata={"include_confidence": "true"},  # a string, not a bool
        )
        fmt = video_segmentation_format() if is_video else image_segmentation_format()
        result = await parse_responses_stream(stream, fmt).final_result()
    finally:
        if uploaded:
            await client.files.delete(uploaded.id)
    return is_video, result


def frame_index(record):
    return record.frame.frame_index if record.frame else 0


def read_frame(path, is_video, index):
    if not is_video:
        return cv2.imread(str(path))  # applies EXIF rotation, matching what SAM sees
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw(frame, boxes, masks):
    for i, box in enumerate(boxes):
        color = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0)][i % 4]
        cv2.rectangle(frame, (int(box.left), int(box.top)),
                      (int(box.right) - 1, int(box.bottom) - 1), color, 2)
        label = box.object_id if box.confidence is None else f"{box.object_id} {box.confidence:.2f}"
        cv2.putText(frame, label, (int(box.left), max(int(box.top) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        mask = masks.get(box.object_id)
        if mask is None:
            continue
        # The mask raster is box-local; stretch it over the box bounds.
        raster = np.frombuffer(decode_mask_to_raster(mask.mask), dtype=np.uint8)
        raster = raster.reshape(mask.mask.height, mask.mask.width)
        b = mask.bounds
        w, h = int(b.right - b.left), int(b.bottom - b.top)
        local = cv2.resize(raster, (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2, offset=(int(b.left), int(b.top)))
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/debug/sam_annotated.png")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    client = AsyncOpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
    is_video, result = asyncio.run(run_sam(client, args.media, args.prompt))

    for d in result.diagnostics:
        print(f"[{d.severity}] {d.code}: {d.message}")
    if result.outcome.status != "completed":
        print(f"Stream incomplete: {result.outcome.reason} {result.outcome.detail or ''}")

    boxes, masks = defaultdict(list), defaultdict(dict)
    for r in result.records:
        if isinstance(r, SegmentationBoxRecord):
            boxes[frame_index(r)].append(r)
        elif isinstance(r, SegmentationMaskRecord):
            masks[frame_index(r)][r.object_id] = r  # later revisions win

    if not boxes:
        print(f'No matches for "{args.prompt}".')
        return

    for idx in sorted(boxes):
        print(f"frame {idx}:")
        for b in boxes[idx]:
            conf = "n/a" if b.confidence is None else f"{b.confidence:.3f}"
            print(f"  id={b.object_id} box=({b.left:.0f},{b.top:.0f},{b.right:.0f},{b.bottom:.0f}) conf={conf}")

    best = max(boxes, key=lambda i: len(boxes[i]))  # frame with the most objects
    frame = read_frame(args.media, is_video, best)
    if frame is None:
        print(f"Could not read frame {best} to annotate.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), draw(frame, boxes[best], masks[best]))
    print(f"Saved annotated frame {best} to {args.out}")


if __name__ == "__main__":
    main()
