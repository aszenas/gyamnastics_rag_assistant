from src.retriever import Retriever
from src.generation import AnswerGenerator


class RAGPipeline:
    def __init__(self, top_k: int = 5):
        self.retriever = Retriever(top_k=top_k)
        self.generator = AnswerGenerator()

    def ask(
        self,
        question: str,
        history: list[dict] | None = None
    ) -> dict:

        if not question.strip():
            raise ValueError("A kérdés nem lehet üres.")

        history = history or []

        #legutóbbi felhasználói kérdés megkeresése
        previous_question = next(
            (
                msg["content"]
                for msg in reversed(history)
                if msg["role"] == "user"
            ),
            None
        )

        #követőkérdés esetén a keresés kontextust kap
        search_query = question

        if previous_question:
            search_query = (
                f"Előző kérdés: {previous_question}\n"
                f"Jelenlegi kérdés: {question}"
            )

        #pinecone keresés
        documents = self.retriever.retrieve(search_query)

        #LLM válasz előzményekkel
        generation = self.generator.generate(
            question,
            documents,
            history=history
        )

        return {
            "question": question,
            "answer": generation["answer"],
            "sources": documents,
            "usage": generation["usage"]
        }
