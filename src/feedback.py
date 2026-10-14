import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_FILE = ROOT / "data" / "evaluation" / "feedback.csv"

FIELDS = [
    "response_id",
    "session_id",
    "timestamp",
    "mode",
    "question",
    "answer",
    "rating",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "model",
    "source_ids"
]


class FeedbackManager:
    def __init__(self, file_path: Path = FEEDBACK_FILE):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_rows(self) -> list[dict]:
        #korábbi értékelések read
        if not self.file_path.exists():
            return []

        with self.file_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as f:
            return list(csv.DictReader(f))

    def _write_rows(self, rows: list[dict]):
        #csv cseréje tempfile-on át
        tmp_path = None

        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                dir=self.file_path.parent,
                suffix=".tmp",
                delete=False
            ) as f:
                tmp_path = Path(f.name)
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            os.replace(tmp_path, self.file_path)

        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    def save_feedback(
        self,
        response_id: str,
        session_id: str,
        mode: str,
        result: dict,
        rating: int,
        model: str
    ):
        #mentés vagy refresh

        if mode not in ("rag", "baseline"):
            raise ValueError("A mode csak 'rag' vagy 'baseline' lehet.")

        if type(rating) is not int or not 1 <= rating <= 5:
            raise ValueError("Az értékelés 1 és 5 közötti egész szám.")

        if not response_id or not session_id:
            raise ValueError("Hiányzó response_id vagy session_id.")

        usage = result.get("usage", {})

        source_ids = [
            source["id"]
            for source in result.get("sources", [])
        ]

        new_row = {
            "response_id": response_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "question": result["question"],
            "answer": result["answer"],
            "rating": rating,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "model": model,
            "source_ids": json.dumps(source_ids)
        }

        rows = self._read_rows()

        #ugyanaz a válasz csak egyszer szerepeljen
        for i, row in enumerate(rows):
            if row["response_id"] == response_id:
                rows[i] = new_row
                break
        else:
            rows.append(new_row)

        self._write_rows(rows)

    def get_feedback_stats(self) -> dict:
        #átlagos értékelés és tokenhasználat 

        rows = self._read_rows()
        stats = {}

        for mode in ("rag", "baseline"):
            mode_rows = [
                row for row in rows if row["mode"] == mode
            ]

            count = len(mode_rows)

            if count == 0:
                stats[mode] = {
                    "count": 0,
                    "avg_rating": None,
                    "avg_tokens": None
                }
                continue

            stats[mode] = {
                "count": count,
                "avg_rating": round(
                    sum(int(r["rating"]) for r in mode_rows) / count,
                    1
                ),
                "avg_tokens": round(
                    sum(int(r["total_tokens"]) for r in mode_rows)
                    / count,
                    1
                )
            }

        return stats
