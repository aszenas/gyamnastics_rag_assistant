from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


class EmbeddingModel:
    def __init__(self):
        print(f"Embedding modell betöltése: {MODEL_NAME}")
        self.model = SentenceTransformer(
            MODEL_NAME
        )

    def embed_documents(self, texts: list[str]):
        formatted = [f"passage: {text}" for text in texts]

        return self.model.encode(
            formatted,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def embed_query(self, query: str):
        return self.model.encode(
            f"query: {query}",
            normalize_embeddings=True
        )
