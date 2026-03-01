import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


FEEDBACK_DIR = Path("data/feedback")
FEEDBACK_FILE = FEEDBACK_DIR / "feedback_log.jsonl"


def log_feedback(event: Dict[str, Any]) -> None:
    """
    Append a single feedback event as one JSON line.

    Each event might contain:
      - query_text: str
      - answer: str
      - citations: list[dict]
      - rating: str ("up" / "down")
      - comment: optional str
      - metadata: optional dict (e.g. version, user id, etc.)

    We also automatically add:
      - timestamp: ISO8601 string
    """
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    event_with_ts = {
        **event,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event_with_ts) + "\n")