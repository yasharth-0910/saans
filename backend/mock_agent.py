"""Scripted stand-in for the Meta API. MockAnalyzer runs the real MetaAnalyzer code against this
client, so the mock trace, findings and overlay.json have exactly the real shapes."""

import json
from types import SimpleNamespace as NS

# One-bit 24x24 SAM mask payloads (a smoke plume and a heap). SAM masks are box-local, so the
# same small raster stretches over any box.
PLUME = "!!!!!6D5*J-+IJhGSkU$he:5_rGkve"
MOUND = "!!!!!4AD4)Vww5_#)iuR0e{dhf=f"
MASK_SIZE = 24

# Each item is one violation the scripted model suspects. All prompts but the last find nothing,
# so the model retries; the last prompt finds the object at `box` (fractions of the frame).
SCENARIOS = {
    "burning_and_debris": {
        "look": "I can see grey smoke rising near the roadside and a heap of broken bricks and rubble in "
                "the lower left. Possible open burning and uncovered C&D waste. I'll check the GRAP rules "
                "and verify each with segmentation.",
        "items": [
            {"violation_type": "garbage_burning", "rule_id": "S1-OPEN-BURNING",
             "prompts": ["burning garbage", "smoke"], "box": (0.35, 0.12, 0.65, 0.70), "drift": -0.06, "peak": 0.35,
             "mask": PLUME, "confidence": 0.86,
             "description": "A column of grey smoke rises from a smouldering pile at the roadside and stays visible through the clip."},
            {"violation_type": "uncovered_debris", "rule_id": "S1-CD-WASTE-DUMPING",
             "prompts": ["debris pile"], "box": (0.05, 0.60, 0.42, 0.95), "drift": 0.0, "peak": 0.75,
             "mask": MOUND, "confidence": 0.78,
             "description": "A heap of construction and demolition debris lies in the open with no cover."},
        ],
        "rejected": {"label": "uncovered truck", "violation_type": "uncovered_vehicle", "rule_id": "S1-CD-TRANSPORT",
                     "description": "A truck carrying debris without a cover may be parked behind the heap."},
        "summary": "Two violations verified: open burning at the roadside, tracked as smoke across the clip, "
                   "and an uncovered heap of construction debris. Both match GRAP Stage I measures.",
    },
    "burning": {
        "look": "There is smoke near a pile of waste on vacant land. Possible open burning. I'll check the "
                "rules and confirm the smoke with segmentation.",
        "items": [
            {"violation_type": "garbage_burning", "rule_id": "S1-OPEN-BURNING",
             "prompts": ["garbage fire", "smoke"], "box": (0.30, 0.10, 0.70, 0.75), "drift": -0.05,
             "mask": PLUME, "confidence": 0.88,
             "description": "Smoke rising from a smouldering garbage pile on vacant land beside the road."},
        ],
        "rejected": {"label": "dumped waste", "violation_type": "uncovered_debris", "rule_id": "S1-CD-WASTE-DUMPING",
                     "description": "Loose waste may also be dumped next to the fire."},
        "summary": "Open burning verified: smoke rising from a garbage pile, matching the GRAP Stage I ban on open burning.",
    },
    "truck": {
        "look": "A tipper truck is passing with its load open to the air. Possible uncovered transport of "
                "construction material. I'll check the rules and segment the truck.",
        "items": [
            {"violation_type": "uncovered_vehicle", "rule_id": "S1-CD-TRANSPORT",
             "prompts": ["uncovered truck", "dump truck"], "box": (0.20, 0.30, 0.80, 0.85), "drift": 0.0,
             "mask": MOUND, "confidence": 0.77,
             "description": "Tipper truck carrying sand and debris with the load open to the air."},
        ],
        "rejected": {"label": "site dust", "violation_type": "construction_dust", "rule_id": "S1-CD-DUST",
                     "description": "Dust may be coming from a site behind the truck."},
        "summary": "Uncovered transport verified: a tipper truck carrying sand and debris without a cover, matching GRAP Stage I.",
    },
}


def _lane(**kw):
    return {"item_id": "msg_mock", "output_index": 0, "content_index": 0, **kw}


def _sam_events(item, width, height, frame_count):
    """A tracked object in SAM wire format. In a video it drifts a little and its confidence wavers."""
    fx1, fy1, fx2, fy2 = item["box"]
    lines = []
    for i in range(frame_count):
        t = i / max(frame_count - 1, 1)
        dy = item["drift"] * t
        x1, y1 = int(fx1 * width), max(0, int((fy1 + dy) * height))
        x2, y2 = min(width - 1, int(fx2 * width)), min(height - 1, int((fy2 + dy) * height))
        c = round(item["confidence"] - 0.15 * abs(t - item.get("peak", 0.5)), 3)  # clearest at `peak`
        lines.append(f"<{i}f>1<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height};c={c}|>"
                     f"<|mask;x=0;y=0;c={c};data={MASK_SIZE},{MASK_SIZE},{item['mask']}|>\n")
    events = [{"type": "response.output_text.delta", **_lane(delta=line)} for line in lines]
    return events + [{"type": "response.output_text.done", **_lane(text="".join(lines))},
                     {"type": "response.completed"}]


NO_MATCH = [{"type": "response.output_text.done", **_lane(text="")}, {"type": "response.completed"}]


def _call(name, args, call_id):
    return NS(type="function_call", name=name, arguments=json.dumps(args), call_id=call_id, status="completed")


def _think(text):
    return NS(type="reasoning", summary=[NS(type="summary_text", text=text)])


def _finding(item, sam_object):
    return {"violation_type": item["violation_type"], "rule_id": item["rule_id"], "sam_object": sam_object,
            "confidence": item["confidence"], "description": item["description"]}


class ScriptedClient:
    """Answers files.* and responses.create like the Meta API would for one scenario."""

    def __init__(self, scenario, width, height, frame_count):
        s = SCENARIOS[scenario]
        items = s["items"]
        self.hits = {item["prompts"][-1]: _sam_events(item, width, height, frame_count) for item in items}
        self.files = NS(create=lambda **kw: NS(id="file-mock"), delete=lambda file_id: None)
        self.responses = NS(create=self._create)

        # Segment ids follow call order: first prompts in turn 1, retries in turn 2.
        first = [_call("segment", {"prompt": it["prompts"][0]}, f"call_s{i}") for i, it in enumerate(items)]
        retries = [it for it in items if len(it["prompts"]) > 1]
        hit_sid = {}
        for i, it in enumerate(items):
            if len(it["prompts"]) == 1:
                hit_sid[id(it)] = f"s{i + 1}"
        for j, it in enumerate(retries):
            hit_sid[id(it)] = f"s{len(items) + j + 1}"
        verified = [_finding(it, f"{hit_sid[id(it)]}:1") for it in items]
        bogus = _finding({**s["rejected"], "confidence": 0.55}, f"{hit_sid[id(items[0])]}:7")
        seen = "hold across frames" if frame_count > 1 else "were found"
        missed = ", ".join(f"'{it['prompts'][0]}'" for it in retries)

        self.script = [
            NS(id="resp_1", output=[_think(s["look"]), _call("get_grap_rules", {}, "call_rules"), *first]),
            NS(id="resp_2", output=[
                _think(f"{missed} found nothing. SAM needs the visible thing itself, so I'll try a plainer phrase."),
                *[_call("segment", {"prompt": it["prompts"][-1]}, f"call_r{j}") for j, it in enumerate(retries)],
            ]) if retries else None,
            NS(id="resp_3", output=[
                _think(f"The objects {seen} with good confidence. Submitting what I found."),
                _call("submit_findings", {"findings": verified + [bogus], "summary": s["summary"]}, "call_submit_1"),
            ]),
            NS(id="resp_4", output=[
                _think(f"The {s['rejected']['label']} finding was rejected: no segmentation "
                       "backs it. Dropping it and resubmitting only the verified findings."),
                _call("submit_findings", {"findings": verified, "summary": s["summary"]}, "call_submit_2"),
            ]),
        ]
        self.script = [r for r in self.script if r is not None]

    def with_options(self, **_):
        return self

    def _create(self, **kw):
        if kw["model"] == "sam-3.1":
            return iter(self.hits.get(kw["input"][0]["content"][0]["text"], NO_MATCH))
        return self.script.pop(0)
