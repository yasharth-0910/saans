"""Seed demo reports around Greater Noida, Noida and Ghaziabad. Safe to re-run.

Usage (from repo root): backend/.venv/bin/python backend/scripts/seed.py
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import db  # noqa: E402
from analyzer import MetaAnalyzer, _AgentRun  # noqa: E402
from hotspots import find_hotspots  # noqa: E402
from mock_agent import ScriptedClient  # noqa: E402

# Seeds that get a full agent trace, from the same scripted run MockAnalyzer uses.
TRACED = {"seed-1": "burning", "seed-6": "truck"}


def scripted_run(scenario):
    """Run the agent loop on a 1280x720 still with no media file. Returns (findings, summary, trace)."""
    client = ScriptedClient(scenario, 1280, 720, 1)
    run = _AgentRun(MetaAnalyzer(client=client), "file-mock", {"type": "input_image", "image_url": "data:"}, False)
    run.loop()
    result = run.result((720, 1280, 3))
    return [asdict(f) for f in result.findings], result.summary, result.trace

BURNING = ("garbage_burning", "S1-OPEN-BURNING",
           "Smoke from a smouldering garbage pile on vacant land beside the road.")
DEBRIS = ("uncovered_debris", "S1-CD-WASTE-DUMPING",
          "Construction debris dumped along the service lane with no cover.")
DUST = ("construction_dust", "S1-CD-DUST",
        "Active excavation raising visible dust; no screens or water sprinkling seen.")
TRUCK = ("uncovered_vehicle", "S1-CD-TRANSPORT",
         "Tipper truck carrying sand and debris with the load open to the air.")

# (lat, lng, place, finding, confidence). The Knowledge Park and Indirapuram points
# sit within 200 m of each other so they show up as hotspots.
SEEDS = [
    (28.4595, 77.4920, "Knowledge Park II, Greater Noida", BURNING, 0.88),
    (28.4607, 77.4929, "Knowledge Park II, Greater Noida", BURNING, 0.81),
    (28.4589, 77.4935, "Knowledge Park II, Greater Noida", DEBRIS, 0.72),
    (28.6270, 77.3650, "Sector 62, Noida", DUST, 0.79),
    (28.5700, 77.3260, "Sector 18, Noida", TRUCK, 0.84),
    (28.6415, 77.3710, "Indirapuram, Ghaziabad", TRUCK, 0.77),
    (28.6426, 77.3718, "Indirapuram, Ghaziabad", DUST, 0.69),
    (28.6960, 77.4250, "Raj Nagar Extension, Ghaziabad", BURNING, 0.90),
]


def main():
    now = datetime.now(timezone.utc)
    for i, (lat, lng, place, (vtype, rule_id, desc), conf) in enumerate(SEEDS, start=1):
        report_id = f"seed-{i}"
        summary = f"Demo report near {place}: {vtype.replace('_', ' ')} matching a GRAP Stage I measure."
        findings = [{
            "violation_type": vtype, "rule_id": rule_id, "confidence": conf,
            "frame_index": 0, "box": [320, 180, 960, 620], "description": desc,
        }]
        if report_id in TRACED:
            findings, summary, trace = scripted_run(TRACED[report_id])
            folder = db.UPLOADS_DIR / report_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "trace.json").write_text(json.dumps(trace, indent=1))
        db.save_report({
            "id": report_id,
            "created_at": (now - timedelta(days=i, hours=3 * i)).isoformat(),
            "lat": lat,
            "lng": lng,
            "media_type": "seed",
            "summary": summary,
            "findings": findings,
            "frames": [],
        })
    hotspots = find_hotspots(db.list_reports())
    print(f"Seeded {len(SEEDS)} reports into {db.DB_PATH} ({', '.join(TRACED)} with agent traces)")
    for h in hotspots:
        print(f"  hotspot: {h['count']} reports {h['report_ids']}")


if __name__ == "__main__":
    main()
