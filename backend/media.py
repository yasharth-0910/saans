"""Frame sampling, privacy blur and evidence-frame rendering."""

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov"}
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "1"))

COLORS = [(0, 200, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0)]  # BGR


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTS


def sample_frames(video_path, out_dir, fps=SAMPLE_FPS):
    """Extract frames at `fps` with ffmpeg into out_dir. Returns [(source_frame_index, path)]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vf", f"fps={fps}",
         "-q:v", "2", str(out_dir / "sample_%05d.jpg")],
        check=True,
    )
    paths = sorted(out_dir.glob("sample_*.jpg"))
    # ponytail: index from timestamp assumes constant frame rate; read pts via ffprobe if VFR clips drift.
    return [(round(i / fps * src_fps), p) for i, p in enumerate(paths)]


def video_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def trim_video(path, max_seconds):
    """Return path unchanged if it is short enough, else a re-encoded H.264 copy of the first max_seconds."""
    path = Path(path)
    if video_duration(path) <= max_seconds:
        return path
    out = path.with_name(f"{path.stem}_trimmed.mp4")
    # Re-encode instead of stream copy: copy cuts on keyframes and can garble the first frames.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-t", str(max_seconds),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-movflags", "+faststart", str(out)],
        check=True,
    )
    return out


def read_frame(media_path, frame_index=0):
    """Read one frame as BGR. Images ignore frame_index."""
    if not is_video(media_path):
        return cv2.imread(str(media_path))  # applies EXIF rotation
    cap = cv2.VideoCapture(str(media_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def blur_regions(frame, regions):
    """Strong Gaussian blur inside each region. A region is a box [x1, y1, x2, y2]
    (pixels, half-open) or a binary HxW mask the size of the frame."""
    if not regions:
        return frame
    h, w = frame.shape[:2]
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=max(h, w) / 25)
    out = frame.copy()
    for region in regions:
        if isinstance(region, np.ndarray) and region.ndim == 2:
            m = region.astype(bool)
            out[m] = blurred[m]
        else:
            x1, y1, x2, y2 = (int(v) for v in region)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            out[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]
    return out


def boxes_near(privacy_boxes, frame_index, window):
    """Privacy boxes from frames within `window` of frame_index, to cover detection gaps."""
    return [b for i in range(frame_index - window, frame_index + window + 1) for b in privacy_boxes.get(i, [])]


def blur_video(src, out, privacy_boxes, window):
    """Write an H.264 MP4 of src with privacy_boxes ({frame_index: [box]}) blurred on every frame.
    Frames are written one for one, so SAM frame indices still line up. Audio is dropped."""
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read the video")
    h, w = frame.shape[:2]  # decoded size, after any rotation (the size SAM reports)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    count = 0
    try:
        while ok:
            if frame.shape[:2] != (h, w):
                raise ValueError("Frame size changed mid-video")
            proc.stdin.write(blur_regions(frame, boxes_near(privacy_boxes, count, window)).tobytes())
            count += 1
            ok, frame = cap.read()
    finally:
        cap.release()
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the blurred video")
    return count


def draw_findings(frame, findings, masks=None):
    """Draw each finding's box, label and (if given) full-frame mask outline."""
    for i, f in enumerate(findings):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = (int(v) for v in f["box"])
        thickness = max(2, frame.shape[1] // 400)
        if masks and masks[i] is not None:
            contours, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), color, thickness)
        label = f'{f["violation_type"].replace("_", " ")} {f["confidence"]:.0%}'
        scale = max(0.5, frame.shape[1] / 1600)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ty = max(y1, th + 8)
        cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
        cv2.putText(frame, label, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
    return frame


def save_evidence(media_path, findings, redactions, out_dir, masks=None):
    """Render one blurred, marked-up JPEG per frame that has findings. masks, if given, holds a
    full-frame binary mask (or None) per finding. Returns file names."""
    masks = masks or [None] * len(findings)
    names = []
    for idx in sorted({f["frame_index"] for f in findings}):
        frame = read_frame(media_path, idx)
        if frame is None:
            continue
        # Blur first, so nothing unblurred is ever written to disk.
        frame = blur_regions(frame, redactions.get(idx, []))
        here = [i for i, f in enumerate(findings) if f["frame_index"] == idx]
        frame = draw_findings(frame, [findings[i] for i in here], [masks[i] for i in here])
        name = f"frame_{idx:05d}.jpg"
        cv2.imwrite(str(Path(out_dir) / name), frame)
        names.append(name)
    return names


if __name__ == "__main__":
    img = np.full((100, 100, 3), 255, np.uint8)
    img[::2, ::2] = 0  # checkerboard, so blur changes it
    out = blur_regions(img, [[10, 10, 50, 50]])
    assert not np.array_equal(out[10:50, 10:50], img[10:50, 10:50]), "box not blurred"
    assert np.array_equal(out[60:, 60:], img[60:, 60:]), "outside box changed"
    mask = np.zeros((100, 100), np.uint8)
    mask[70:90, 70:90] = 1
    out = blur_regions(img, [mask])
    assert not np.array_equal(out[70:90, 70:90], img[70:90, 70:90]), "mask not blurred"
    assert np.array_equal(out[:60, :60], img[:60, :60]), "outside mask changed"
    print("media self-check ok")
