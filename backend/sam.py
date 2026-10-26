"""sam-3.1 calls, parsing and the compact summaries the agent sees."""

import asyncio

import cv2
import numpy as np
from meta_sam_parser import (
    ResponsesStreamLaneError,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

SAM_MODEL = "sam-3.1"


def _event_type(event):
    return event["type"] if isinstance(event, dict) else event.type


async def _parse(events, is_video):
    seen_text = False

    async def source():
        nonlocal seen_text
        for event in events:  # the sync SDK stream, adapted for the async parser
            seen_text = seen_text or _event_type(event).startswith("response.output_text")
            yield event

    fmt = video_segmentation_format() if is_video else image_segmentation_format()
    try:
        return await parse_responses_stream(source(), fmt).final_result()
    except ResponsesStreamLaneError:
        # Zero matches can complete with no output_text lane at all. Anything else is a real error.
        if seen_text:
            raise
        return None


def segment(client, media_part, prompt, is_video, timeout):
    """Run one noun-phrase prompt. Returns {object_id: {frame_index: entry}} where an entry
    holds "box" ([x1, y1, x2, y2], half-open pixels), "confidence" and "mask" (the parser's
    SegmentationMaskRecord). Empty dict when nothing matched."""
    stream = client.with_options(timeout=timeout).responses.create(
        model=SAM_MODEL,
        input=[{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": prompt}, media_part],
        }],
        stream=True,
        metadata={"include_confidence": "true"},
    )
    result = asyncio.run(_parse(stream, is_video))
    objects = {}
    for r in result.records if result else ():
        idx = r.frame.frame_index if getattr(r, "frame", None) else 0
        if isinstance(r, SegmentationBoxRecord):
            entry = objects.setdefault(r.object_id, {}).setdefault(idx, {})
            entry["box"] = [int(r.left), int(r.top), int(r.right), int(r.bottom)]
            entry["confidence"] = r.confidence
        elif isinstance(r, SegmentationMaskRecord):
            objects.setdefault(r.object_id, {}).setdefault(idx, {})["mask"] = r  # later revisions win
    return objects


def frame_ranges(indices):
    """[0, 1, 2, 5, 7, 8] -> "0-2, 5, 7-8" """
    out, start, prev = [], None, None
    for i in sorted(indices):
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = i
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else str(start))
    return ", ".join(out)


def best_frame(frames):
    return max(frames, key=lambda i: (frames[i].get("confidence") or 0.0, -i))


def summarize(objects):
    """What the model sees per object: no masks, just where and how confidently."""
    summary = []
    for object_id, frames in objects.items():
        confs = [e["confidence"] for e in frames.values() if e.get("confidence") is not None]
        best = best_frame(frames)
        summary.append({
            "object_id": object_id,
            "frames": frame_ranges(frames),
            "frame_count": len(frames),
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
            "best_frame": best,
            "box_at_best_frame": frames[best].get("box"),
        })
    return summary


def mask_to_frame(mask_record, frame_shape):
    """Full-frame binary mask from a box-local SAM mask."""
    h, w = frame_shape[:2]
    raster = np.frombuffer(decode_mask_to_raster(mask_record.mask), dtype=np.uint8)
    raster = raster.reshape(mask_record.mask.height, mask_record.mask.width)
    b = mask_record.bounds
    x1, y1, x2, y2 = int(b.left), int(b.top), int(b.right), int(b.bottom)
    full = np.zeros((h, w), np.uint8)
    local = cv2.resize(raster, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    full[y1:min(y2, h), x1:min(x2, w)] = local[: min(y2, h) - y1, : min(x2, w) - x1]
    return full


def overlay_wire(overlay):
    """Rebuild SAM wire lines from a saved overlay, one track per finding (track id = finding
    index + 1), so the frontend can feed them to @meta-sam/parser and get a real result object."""
    w, h = overlay["frame_width"], overlay["frame_height"]
    by_frame = {}
    for obj in overlay["objects"]:
        for f in obj["frames"]:
            if f["box"] is None or f["mask"] is None:
                continue
            x1, y1, x2, y2 = f["box"]
            c = "" if f["confidence"] is None else f";c={f['confidence']}"
            m = f["mask"]
            by_frame.setdefault(f["frame_index"], []).append(
                f"{obj['finding'] + 1}<|box;x1={x1};y1={y1};x2={x2 - 1};y2={y2 - 1};w={w};h={h}{c}|>"
                f"<|mask;x=0;y=0{c};data={m['height']},{m['width']},{m['payload']}|>"
            )
    return "".join(f"<{i}f>{','.join(records)}\n" for i, records in sorted(by_frame.items()))


def overlay_frames(frames):
    """JSON-safe per-frame data for a later video overlay (masks stay encoded)."""
    out = []
    for idx in sorted(frames):
        e = frames[idx]
        m = e.get("mask")
        out.append({
            "frame_index": idx,
            "box": e.get("box"),
            "confidence": e.get("confidence"),
            "mask": None if m is None else {
                "encoding": m.mask.encoding, "payload": m.mask.payload,
                "width": m.mask.width, "height": m.mask.height,
            },
        })
    return out
