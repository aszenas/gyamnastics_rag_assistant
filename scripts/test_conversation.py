from src.conversation import ConversationManager
from src.rag_pipeline import RAGPipeline
from src.baseline import BaselinePipeline


def main():
    memory = ConversationManager(max_turns=3)

    rag = RAGPipeline(top_k=5)
    baseline = BaselinePipeline()

    tests = [
        ("rag", "Mekkora levonás jár a hajlított térdért?"),
        ("rag", "És ezt melyik zsűri vonja le?"),
        ("baseline", "Mekkora levonás jár a hajlított térdért?"),
        ("baseline", "És ezt melyik zsűri vonja le?")
    ]

    for mode, question in tests:
        print("\n" + "=" * 70)
        print(f"MÓD: {mode.upper()}")
        print(f"KÉRDÉS: {question}")

        #csak a kiválasztott mód historyja
        history = memory.get_context(mode)

        pipeline = rag if mode == "rag" else baseline

        result = pipeline.ask(
            question,
            history=history
        )

        print("\nVÁLASZ:")
        print(result["answer"])

        print("\nTOKENHASZNÁLAT:")
        print(result["usage"]["total_tokens"])

        #csak sikeres válasz után mentünk
        memory.add_turn(
            mode,
            question,
            result["answer"]
        )

    print("\n" + "=" * 70)
    print("MEMÓRIA ELLENŐRZÉSE")
    print(f"RAG üzenetek: {len(memory.get_history('rag'))}")
    print(f"Baseline üzenetek: {len(memory.get_history('baseline'))}")


if __name__ == "__main__":
    main()
