from src.baseline import BaselinePipeline


def main():
    baseline = BaselinePipeline()

    question = "Mekkora levonás jár a hajlított térdért?"

    result = baseline.ask(question)

    print("=" * 70)
    print(f"KÉRDÉS: {question}")

    print("\nBASELINE LLM VÁLASZ:")
    print(result["answer"])

    print("\nTOKENHASZNÁLAT:")
    print(f"Input: {result['usage']['prompt_tokens']}")
    print(f"Output: {result['usage']['completion_tokens']}")
    print(f"Összesen: {result['usage']['total_tokens']}")


if __name__ == "__main__":
    main()
