from src.rag_pipeline import RAGPipeline

def main():
    rag = RAGPipeline(top_k=5)

    question = "Mekkora levonás jár a hajlított térdért?"

    print(f"\nKÉRDÉS: {question}")

    result = rag.ask(question)

    print("\n" + "=" * 70)
    print("LLM VÁLASZ:")
    print(result["answer"])

    print("\nFORRÁSOK:")
    for i, source in enumerate(result["sources"], start=1):
        metadata = source["metadata"]
        print(
            f"[{i}] "
            f"Oldal: {metadata.get('original_page', 'N/A')} | "
            f"Score: {source['score']:.4f}"
        )

    print("\nTOKENHASZNÁLAT:")
    usage = result["usage"]
    print(f"Input: {usage['prompt_tokens']}")
    print(f"Output: {usage['completion_tokens']}")
    print(f"Összesen: {usage['total_tokens']}")


if __name__ == "__main__":
    main()
