
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
    "source_ids",
    # Új evaluation mezők
    "relevant_chunk_ids",
    "top_k",
    "groundedness",
    "citation_accuracy",
    "answer_relevance",
    "evaluation_run_id"
]

EVALUATION_FIELDS = {
    "relevant_chunk_ids",
    "groundedness",
    "citation_accuracy",
    "answer_relevance",
    "evaluation_run_id"
}

CHOICES = {"yes", "no", "unknown"}


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

                #hiányzó mezők feltöltése üres értékekkel
                for row in rows:
                    writer.writerow({
                        field: row.get(field, "")
                        for field in FIELDS
                    })

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
        rating: int | None = None,
        model: str = "",
        relevant_chunk_ids: list[str] | None = None,
        groundedness: str | None = None,
        citation_accuracy: str | None = None,
        answer_relevance: str | None = None,
        evaluation_run_id: str | None = None
    ):
        #mentés vagy refresh

        if mode not in ("rag", "baseline"):
            raise ValueError("A mode csak 'rag' vagy 'baseline' lehet.")

        if rating is not None:
            if type(rating) is not int or not 1 <= rating <= 5:
                raise ValueError(
                    "Az értékelés 1 és 5 közötti egész szám."
                )

        if not response_id or not session_id:
            raise ValueError("Hiányzó response_id vagy session_id.")

        for name, value in {
            "groundedness": groundedness,
            "citation_accuracy": citation_accuracy,
            "answer_relevance": answer_relevance
        }.items():
            if value is not None and value not in CHOICES:
                raise ValueError(
                    f"{name}: yes, no vagy unknown lehet."
                )

        if mode == "baseline":
            if relevant_chunk_ids is not None or citation_accuracy is not None:
                raise ValueError(
                    "Baseline módban nincs chunkrelevancia vagy hivatkozáspontosság."
                )

        source_ids = [
            source["id"]
            for source in result.get("sources", [])
        ]

        if relevant_chunk_ids is not None:
            if not isinstance(relevant_chunk_ids, list):
                raise ValueError("A relevant_chunk_ids lista kell legyen.")

            if any(
                not isinstance(item, str) or item not in source_ids
                for item in relevant_chunk_ids
            ):
                raise ValueError(
                    "Csak a visszakeresett chunkok ID-jai jelölhetők."
                )

            if len(relevant_chunk_ids) != len(set(relevant_chunk_ids)):
                raise ValueError("Duplikált chunkazonosító.")

        usage = result.get("usage", {})

        rows = self._read_rows()

        existing = next(
            (
                row for row in rows
                if row.get("response_id") == response_id
            ),
            None
        )

        #már korábban mentett evaluation mezőket megtartjuk
        new_row = {
            field: existing.get(field, "") if existing else ""
            for field in FIELDS
        }

        new_row.update({
            "response_id": response_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "question": result["question"],
            "answer": result["answer"],
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "model": model,
            "source_ids": json.dumps(source_ids, ensure_ascii=False),
            "top_k": len(source_ids)
        })

        # Csak a most megadott értékeket frissítjük
        if rating is not None:
            new_row["rating"] = rating

        if relevant_chunk_ids is not None:
            new_row["relevant_chunk_ids"] = json.dumps(
                relevant_chunk_ids, ensure_ascii=False
            )

        if groundedness is not None:
            new_row["groundedness"] = groundedness

        if citation_accuracy is not None:
            new_row["citation_accuracy"] = citation_accuracy

        if answer_relevance is not None:
            new_row["answer_relevance"] = answer_relevance

        if evaluation_run_id is not None:
            new_row["evaluation_run_id"] = evaluation_run_id

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
            #afőoldali KPI-k csak az 1-5 skálán értékelt válaszokból
            mode_rows = [
                row for row in rows
                if row.get("mode") == mode
                and row.get("rating") not in ("", None)
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

    def get_evaluation_stats(self) -> dict:
        #részletes evaluation dashboard aggregációk

        rows = self._read_rows()
        stats = {}

        for mode in ("rag", "baseline"):
            mode_rows = [
                row for row in rows
                if row.get("mode") == mode
            ]

            mode_stats = {"count": len(mode_rows)}

            for field in (
                "groundedness",
                "citation_accuracy",
                "answer_relevance"
            ):
                valid = [
                    row[field]
                    for row in mode_rows
                    if row.get(field) in ("yes", "no")
                ]

                mode_stats[f"{field}_count"] = len(valid)
                mode_stats[f"{field}_rate"] = (
                    round(valid.count("yes") / len(valid), 3)
                    if valid else None
                )

            stats[mode] = mode_stats

        return stats
