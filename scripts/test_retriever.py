from src.retriever import Retriever


def main():
    retriever = Retriever(top_k=5)

    questions = [
        "Mennyi idő van visszamászni a gerendára leesés után?",
        "Mekkora levonás jár a hajlított térdért?",
        "Milyen követelmények vannak a spárgafelugrásnál?",
        "Milyen nehézségi értékű a kézenátfordulás?"
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"KÉRDÉS: {question}")
        print("=" * 80)

        results = retriever.retrieve(question)

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print(f"\n{rank}. TALÁLAT")
            print(f"Score: {result['score']:.4f}")
            print(f"Forrás: {metadata.get('source', 'N/A')}")
            print(f"Oldal: {metadata.get('original_page', 'N/A')}")
            print(f"Típus: {metadata.get('source_type', 'N/A')}")
            print(f"Szöveg:\n{result['text']}")
            print("-" * 80)


if __name__ == "__main__":
    main()
