from pathlib import Path
from uuid import uuid4

from src.feedback import FeedbackManager


def main():
    #külön teszt-CSV, nem írjuk felül az éles értékeléseket.
    test_file = (
        Path(__file__).resolve().parents[1]
        / "data" / "evaluation" / "feedback_test.csv"
    )

    feedback = FeedbackManager(file_path=test_file)

    session_id = str(uuid4())
    response_id = str(uuid4())

    rag_result = {
        "question": "Mekkora levonás jár a hajlított térdért?",
        "answer": "0,10 / 0,30 / 0,50 pont.",
        "sources": [
            {"id": "deduction_002"},
            {"id": "pdf_p032_001"}
        ],
        "usage": {
            "prompt_tokens": 1651,
            "completion_tokens": 264,
            "total_tokens": 1915
        }
    }

    #Első értékelés
    feedback.save_feedback(
        response_id=response_id,
        session_id=session_id,
        mode="rag",
        result=rag_result,
        rating=5,
        model="openai/gpt-oss-120b"
    )

    #Ugyanazt a választ újraértékeljük
    feedback.save_feedback(
        response_id=response_id,
        session_id=session_id,
        mode="rag",
        result=rag_result,
        rating=4,
        model="openai/gpt-oss-120b"
    )

    #Ellenőrizzük, hogy nem lett duplikáció
    rows = feedback._read_rows()

    assert len(rows) == 1
    assert rows[0]["rating"] == "4"

    print("Duplikáció elleni teszt: OK")

    #Statisztika
    stats = feedback.get_feedback_stats()

    print("\nRAG statisztika:")
    print(stats["rag"])

    print("\nBaseline statisztika:")
    print(stats["baseline"])

    print(f"\nTesztfájl: {test_file}")


if __name__ == "__main__":
    main()
