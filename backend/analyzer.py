"""Analyzer interface. The rest of the app only calls get_analyzer().analyze(path)."""

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import openai

import sam
from db import load_rules
from media import boxes_near, is_video, read_frame, trim_video
from mock_agent import ScriptedClient

MUSE_MODEL = "muse-spark-1.3"
PRIVACY_PROMPTS = ("face", "license plate")
PRIVACY_WINDOW = 5  # also blur detections this many frames either side of an evidence frame
PRIVACY_PAD = 0.15  # grow privacy boxes by this fraction on each side


@dataclass
class Finding:
    violation_type: str  # garbage_burning, construction_dust, uncovered_debris, uncovered_vehicle
    rule_id: str  # id in data/grap_rules.json
    confidence: float  # 0..1
    frame_index: int  # source frame index; 0 for images
    box: list[int]  # [x1, y1, x2, y2] source pixels, half-open
    description: str
    sam_object: str | None = None  # "<segment_id>:<object_id>" the finding is backed by


@dataclass
class AnalysisResult:
    findings: list[Finding]
    summary: str
    # Faces and number plates to blur, keyed by frame index: boxes or binary masks.
    redactions: dict[int, list] = field(default_factory=dict)
    # Full-frame binary mask per finding (aligned with findings), or None.
    masks: list = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    overlay: dict | None = None  # SAM data for a later video overlay
    media_path: Path | None = None  # file the frame indices refer to, if not the upload
    # Padded face and plate boxes on every frame where SAM saw them: {frame_index: [box]}.
    privacy_boxes: dict[int, list] = field(default_factory=dict)


class Analyzer(Protocol):
    def analyze(self, media_path) -> AnalysisResult: ...


class MockAnalyzer:
    """Runs the real MetaAnalyzer against a scripted client: no API calls, real output shapes."""

    def __init__(self, scenario="burning_and_debris"):
        self.scenario = scenario

    def analyze(self, media_path):
        media_path = Path(media_path)
        frame_count = 1
        if is_video(media_path):
            media_path = trim_video(media_path, float(os.getenv("MAX_VIDEO_SECONDS", "20")))
            cap = cv2.VideoCapture(str(media_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cap.release()
        h, w = read_frame(media_path, 0).shape[:2]
        client = ScriptedClient(self.scenario, w, h, frame_count)
        return MetaAnalyzer(client=client).analyze(media_path)


INSTRUCTIONS = """You are Saans, an evidence assistant for air-pollution rule violations in Delhi NCR under GRAP (the Graded Response Action Plan).

You get one photo or short video. Your job:
1. Look at the media and decide which violations may be visible: open burning of garbage or biomass, construction or demolition without dust control, uncovered construction debris or material, and trucks carrying debris or construction material without a cover.
2. Call get_grap_rules to see the rules, the violation types each one covers and the GRAP stage each applies from. You are not told which stage is in force, so when several rules cover a violation, cite the one with the lowest stage.
3. Verify every suspected violation with segment(prompt). The prompt is one short noun phrase naming one visible thing, like "smoke", "burning garbage", "debris pile" or "dump truck". If a prompt finds nothing or only low-confidence objects, try a different phrase. In a video, prefer objects that hold across many frames over one-frame detections.
4. Report only what the segmentation evidence supports. Each finding must point to an object segment returned, written "<segment_id>:<object_id>", for example "s2:1".
5. Finish by calling submit_findings. If nothing is supported, submit an empty findings list and say so in the summary.

Describe objects and activities only. Never identify people, and never read out number plates. Give no health advice. Keep a neutral, factual tone and do not comment on authorities or politics."""

USER_PROMPT = "Check this media for visible GRAP violations and submit your verified findings."
LAST_TURN_PROMPT = "This is your last turn. Call submit_findings now with only the findings you have verified."
NUDGE_PROMPT = "Call submit_findings to finish, with an empty list if nothing is verified."


def _tools(rules):
    violation_types = sorted({t for r in rules for t in r["violation_types"]})
    return [
        {
            "type": "function",
            "name": "segment",
            "description": "Run SAM 3.1 segmentation and tracking on the media for one short noun phrase. "
                           "Returns a segment_id and, per object, the frames it appears in, its mean "
                           "confidence and its box at the best frame.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "One short noun phrase, e.g. \"smoke\"."}},
                "required": ["prompt"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "get_grap_rules",
            "description": "Return the GRAP rules with their ids and the violation types they cover.",
            "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            "strict": True,
        },
        {
            "type": "function",
            "name": "submit_findings",
            "description": "Submit the final verified findings and a short summary. Ends the analysis "
                           "unless some findings are rejected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "violation_type": {"type": "string", "enum": violation_types},
                                "rule_id": {"type": "string", "enum": [r["id"] for r in rules]},
                                "sam_object": {"type": "string", "description": "\"<segment_id>:<object_id>\""},
                                "confidence": {"type": "number", "description": "0 to 1"},
                                "description": {"type": "string", "description": "One or two factual sentences."},
                            },
                            "required": ["violation_type", "rule_id", "sam_object", "confidence", "description"],
                            "additionalProperties": False,
                        },
                    },
                    "summary": {"type": "string", "description": "Two or three factual sentences."},
                },
                "required": ["findings", "summary"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def _short(value, limit=1500):
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[:limit] + "…"


class MetaAnalyzer:
    """Muse Spark agent that verifies violations with SAM 3.1, then a privacy pass."""

    def __init__(self, client=None):
        self.client = client or openai.OpenAI(
            base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"], max_retries=2
        )
        self.max_turns = int(os.getenv("AGENT_MAX_TURNS", "8"))
        self.max_segments = int(os.getenv("AGENT_MAX_SEGMENT_CALLS", "8"))
        self.muse_timeout = float(os.getenv("MUSE_TIMEOUT_S", "120"))
        self.sam_timeout = float(os.getenv("SAM_TIMEOUT_S", "180"))
        self.max_video_s = float(os.getenv("MAX_VIDEO_SECONDS", "20"))
        self.rules = load_rules()
        self.tools = _tools(self.rules)

    def analyze(self, media_path):
        media_path = Path(media_path)
        video = is_video(media_path)
        if video:
            media_path = trim_video(media_path, self.max_video_s)
        with open(media_path, "rb") as f:
            uploaded = self.client.files.create(purpose="user_data", file=f)
        try:
            if video:
                sam_media = {"type": "input_video", "file_id": uploaded.id}
            else:
                # The SAM docs only show image_url (URL or data URL) for images, not file_id.
                mime = mimetypes.guess_type(media_path.name)[0] or "image/jpeg"
                data = base64.b64encode(media_path.read_bytes()).decode()
                sam_media = {"type": "input_image", "image_url": f"data:{mime};base64,{data}"}
            run = _AgentRun(self, uploaded.id, sam_media, video)
            run.loop()
            return run.result(read_frame(media_path, 0).shape, media_path)
        finally:
            self.client.files.delete(uploaded.id)


class _AgentRun:
    """State for one report: SAM results, trace, and the best verified submission so far."""

    def __init__(self, analyzer, file_id, sam_media, video):
        self.a = analyzer
        self.file_id = file_id
        self.sam_media = sam_media
        self.video = video
        self.segments = {}  # segment_id -> {"prompt": str, "objects": {...}}
        self.by_prompt = {}  # normalized prompt -> segment_id
        self.trace = []
        self.accepted = []  # validated finding dicts from the latest submission
        self.summary = ""
        self.done = False
        self.note = ""

    # --- agent loop ---------------------------------------------------------

    def loop(self):
        a = self.a
        input_items = [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_PROMPT},
                {"type": "input_file", "file_id": self.file_id},
            ],
        }]
        previous_id = None
        for turn in range(1, a.max_turns + 1):
            if turn == a.max_turns:
                input_items.append({"role": "user", "content": LAST_TURN_PROMPT})
            kwargs = {"previous_response_id": previous_id} if previous_id else {}
            try:
                response = a.client.with_options(timeout=a.muse_timeout).responses.create(
                    model=MUSE_MODEL,
                    instructions=INSTRUCTIONS,  # not carried over by previous_response_id
                    input=input_items,
                    tools=a.tools,
                    reasoning={"summary": "auto"},
                    **kwargs,
                )
            except openai.APIError as e:
                self.note = f"The analysis stopped early after a model error ({type(e).__name__})."
                self.trace.append({"type": "error", "turn": turn, "text": str(e)[:300]})
                return
            previous_id = response.id

            calls = []
            for item in response.output:
                if item.type == "reasoning":
                    text = " ".join(getattr(s, "text", "") for s in (item.summary or [])).strip()
                    if text:
                        self.trace.append({"type": "reasoning", "turn": turn, "text": text})
                elif item.type == "message":
                    text = "".join(getattr(c, "text", "") for c in item.content).strip()
                    if text:
                        self.trace.append({"type": "message", "turn": turn, "text": text})
                elif item.type == "function_call":
                    calls.append(item)

            if not calls:
                input_items = [{"role": "user", "content": NUDGE_PROMPT}]
                continue
            input_items = []
            for call in calls:
                output = self.run_tool(turn, call.name, call.arguments)
                input_items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
                if self.done:
                    return
        self.note = f"The analysis hit its limit of {a.max_turns} turns; only findings verified by then are included."
        self.trace.append({"type": "limit", "text": self.note})

    def run_tool(self, turn, name, arguments):
        self.trace.append({"type": "tool_call", "turn": turn, "name": name, "arguments": _short(arguments, 2000)})
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            result = {"error": "Arguments must be a JSON object."}
        elif name == "segment":
            result = self.tool_segment(args)
        elif name == "get_grap_rules":
            result = self.a.rules
        elif name == "submit_findings":
            result = self.tool_submit(args)
        else:
            result = {"error": f"Unknown tool {name}."}
        output = json.dumps(result)
        self.trace.append({"type": "tool_result", "turn": turn, "name": name, "result": _short(output)})
        return output

    def tool_segment(self, args):
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return {"error": "prompt must be a non-empty noun phrase."}
        key = " ".join(prompt.lower().split())
        if key in self.by_prompt:  # repeated prompt: answer from cache, don't spend a call
            sid = self.by_prompt[key]
            return {"segment_id": sid, "prompt": prompt, "note": "Already segmented; same result.",
                    "objects": sam.summarize(self.segments[sid]["objects"])}
        sam_calls = len(self.segments)
        if sam_calls >= self.a.max_segments:
            return {"error": f"Segment limit of {self.a.max_segments} calls reached. Call submit_findings now."}
        try:
            objects = sam.segment(self.a.client, self.sam_media, prompt, self.video, self.a.sam_timeout)
        except openai.APIError as e:
            return {"error": f"Segmentation failed ({type(e).__name__}). Try again or use another phrase."}
        sid = f"s{sam_calls + 1}"
        self.segments[sid] = {"prompt": prompt, "objects": objects}
        self.by_prompt[key] = sid
        result = {"segment_id": sid, "prompt": prompt, "objects": sam.summarize(objects)}
        if not objects:
            result["note"] = "No matches. Try a different noun phrase if you still suspect this."
        return result

    def tool_submit(self, args):
        findings, summary = args.get("findings"), args.get("summary")
        if not isinstance(findings, list) or not isinstance(summary, str) or not summary.strip():
            return {"error": "Expected findings (a list) and summary (a non-empty string)."}
        accepted, rejected = [], []
        for i, f in enumerate(findings):
            problem = self.validate(f)
            if problem:
                rejected.append({"index": i, "reason": problem})
            else:
                accepted.append(f)
        # Keep the verified part even if the model never resubmits.
        self.accepted, self.summary = accepted, summary.strip()
        if rejected:
            return {"status": "rejected", "accepted": len(accepted), "rejected": rejected,
                    "message": "Fix or drop the rejected findings and call submit_findings again with the full list."}
        self.done = True
        return {"status": "accepted", "accepted": len(accepted)}

    def validate(self, f):
        """Check one submitted finding against the findings schema and the SAM results."""
        rules = {r["id"]: r for r in self.a.rules}
        if not isinstance(f, dict):
            return "Finding must be an object."
        expected = {"violation_type", "rule_id", "sam_object", "confidence", "description"}
        if set(f) != expected:
            return f"Finding must have exactly the fields {sorted(expected)}."
        if f["rule_id"] not in rules:
            return f"Unknown rule_id {f['rule_id']!r}."
        if f["violation_type"] not in rules[f["rule_id"]]["violation_types"]:
            return f"Rule {f['rule_id']} does not cover violation_type {f['violation_type']!r}."
        conf = f["confidence"]
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            return "confidence must be a number from 0 to 1."
        if not isinstance(f["description"], str) or not f["description"].strip():
            return "description must be a non-empty string."
        m = re.fullmatch(r"(s\d+):(\S+)", str(f["sam_object"]))
        if not m or m[1] not in self.segments or m[2] not in self.segments[m[1]]["objects"]:
            return f"sam_object {f['sam_object']!r} was never returned by segment."
        return None

    # --- results ------------------------------------------------------------

    def result(self, frame_shape, media_path=None):
        findings, masks, overlay_objects = [], [], []
        for i, f in enumerate(self.accepted):
            sid, oid = f["sam_object"].split(":", 1)
            frames = self.segments[sid]["objects"][oid]
            best = sam.best_frame(frames)
            entry = frames[best]
            findings.append(Finding(
                f["violation_type"], f["rule_id"], float(f["confidence"]), best,
                entry["box"], f["description"].strip(), f["sam_object"],
            ))
            masks.append(sam.mask_to_frame(entry["mask"], frame_shape) if entry.get("mask") else None)
            overlay_objects.append({
                "finding": i, "segment_id": sid, "prompt": self.segments[sid]["prompt"],
                "object_id": oid, "frames": sam.overlay_frames(frames),
            })

        evidence_frames = {f.frame_index for f in findings}
        privacy_boxes = self.privacy_pass(evidence_frames, frame_shape)
        redactions = {idx: boxes_near(privacy_boxes, idx, PRIVACY_WINDOW) for idx in evidence_frames}
        summary = self.summary or "No violations could be verified."
        if self.note:
            summary = f"{summary} Note: {self.note}"
        self.trace.append({"type": "final", "summary": summary, "findings": [
            {"sam_object": f.sam_object, "violation_type": f.violation_type, "rule_id": f.rule_id,
             "confidence": f.confidence, "frame_index": f.frame_index} for f in findings]})
        overlay = {"frame_width": frame_shape[1], "frame_height": frame_shape[0], "objects": overlay_objects}
        return AnalysisResult(findings, summary, redactions, masks, self.trace, overlay, media_path, privacy_boxes)

    def privacy_pass(self, evidence_frames, frame_shape):
        """Find faces and plates with SAM, whatever the agent did. Returns padded boxes for every
        frame they appear on. Runs for every video (the blurred copy needs it) and for images
        with evidence. A failure propagates, so nothing unblurred gets saved."""
        boxes = {}
        if not (self.video or evidence_frames):
            return boxes
        h, w = frame_shape[:2]
        for prompt in PRIVACY_PROMPTS:
            objects = sam.segment(self.a.client, self.sam_media, prompt, self.video, self.a.sam_timeout)
            regions = 0
            for frames in objects.values():
                for idx, entry in frames.items():
                    if entry.get("box"):
                        x1, y1, x2, y2 = entry["box"]
                        px, py = (x2 - x1) * PRIVACY_PAD, (y2 - y1) * PRIVACY_PAD
                        boxes.setdefault(idx, []).append([max(0, int(x1 - px)), max(0, int(y1 - py)),
                                                          min(w, int(x2 + px)), min(h, int(y2 + py))])
                        regions += 1
            self.trace.append({"type": "privacy", "prompt": prompt, "objects": len(objects), "regions": regions})
        return boxes


def get_analyzer() -> Analyzer:
    if os.getenv("USE_MOCK", "1") != "0":
        return MockAnalyzer()
    return MetaAnalyzer()
