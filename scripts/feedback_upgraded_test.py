from pathlib import Path
from tempfile import TemporaryDirectory

from src.feedback import FeedbackManager


def main():
    with TemporaryDirectory() as tmp:
        feedback = FeedbackManager(
            file_path=Path(tmp) / "feedback_test.csv"
        )

        result = {
            "question": "Mekkora levonás jár a hajlított térdért?",
            "answer": "0,10 / 0,30 / 0,50 pont. [2]",
            "sources": [
                {"id": "chunk_001"},
                {"id": "chunk_002"},
                {"id": "chunk_003"}
            ],
            "usage": {
                "prompt_tokens": 1651,
                "completion_tokens": 264,
                "total_tokens": 1915
            }
        }

        common = {
            "response_id": "test_response_001",
            "session_id": "test_session",
            "mode": "rag",
            "result": result,
            "model": "openai/gpt-oss-120b"
        }

        #Helyességi rating
        feedback.save_feedback(**common, rating=5)

        #Chunk relevancia
        feedback.save_feedback(
            **common,
            relevant_chunk_ids=["chunk_002", "chunk_003"]
        )

        #Generálási metrikák
        feedback.save_feedback(
            **common,
            groundedness="yes",
            citation_accuracy="yes",
            answer_relevance="no"
        )

        #Újraértékelés
        feedback.save_feedback(**common, rating=4)

        rows = feedback._read_rows()

        assert len(rows) == 1
        assert rows[0]["rating"] == "4"
        assert rows[0]["relevant_chunk_ids"] == (
            '["chunk_002", "chunk_003"]'
        )
        assert rows[0]["groundedness"] == "yes"
        assert rows[0]["citation_accuracy"] == "yes"
        assert rows[0]["answer_relevance"] == "no"
        assert rows[0]["top_k"] == "3"

        print("Duplikáció elleni teszt: OK")
        print("Részleges értékelés mentése: OK")
        print("Korábbi értékelések megőrzése: OK")
        print("Dinamikus top-k tárolása: OK")

        print("\nFőoldali KPI:")
        print(feedback.get_feedback_stats()["rag"])

        print("\nEvaluation metrikák:")
        print(feedback.get_evaluation_stats()["rag"])


if __name__ == "__main__":
    main()
