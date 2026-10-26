"""Ask muse-spark-1.3 to describe air-pollution violations in a local video.

Usage: python scripts/test_muse.py <video.mp4>
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]

PROMPT = (
    "You are reviewing a short clip from Delhi NCR. Describe any visible air-pollution "
    "violations: open garbage or biomass burning, construction or demolition without dust "
    "control, uncovered debris or sand piles, or uncovered trucks carrying debris. For each, "
    "say what you see, roughly when it appears in the clip, and how confident you are. "
    "If you see none, say so plainly. Describe only objects and activities, not people's "
    "identities, and give no health advice."
)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])

    load_dotenv(REPO_ROOT / ".env")
    client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])

    with open(path, "rb") as f:
        uploaded = client.files.create(purpose="user_data", file=f)
    try:
        # Streaming keeps long video requests from hitting idle timeouts.
        with client.responses.stream(
            model="muse-spark-1.3",
            input=[{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_file", "file_id": uploaded.id},
                ],
            }],
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
        print()
    finally:
        client.files.delete(uploaded.id)


if __name__ == "__main__":
    main()
