from src.retriever import Retriever
from src.generation import AnswerGenerator


class RAGPipeline:
    def __init__(self, top_k: int = 5):
        self.retriever = Retriever(top_k=top_k)
        self.generator = AnswerGenerator()

    def ask(self, question: str) -> dict:
        if not question.strip():
            raise ValueError("A kérdés nem lehet üres.")

        #keresés
        documents = self.retriever.retrieve(question)

        #LLM válasz generálás
        generation = self.generator.generate(question, documents)

        return {
            "question": question,
            "answer": generation["answer"],
            "sources": documents,
            "usage": generation["usage"]
        }
