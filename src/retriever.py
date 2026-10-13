from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore

class Retriever:
    def __init__(self, top_k: int = 5):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        #top 5 leghasonlóbb dokumentumrészlet search

        if not query.strip():
            raise ValueError("A keresőkérdés nem lehet üres.")

        #kérdésből embedding
        query_embedding = self.embedding_model.embed_query(query)

        #semantic search
        response = self.vector_store.index.query(
            vector=query_embedding.tolist(),
            top_k=self.top_k,
            namespace=self.vector_store.namespace,
            include_metadata=True
        )

        #get results
        results = []

        for match in response.matches:
            metadata = dict(match.metadata or {})

            results.append({
                "id": match.id,
                "score": float(match.score),
                "text": metadata.pop("text", ""),
                "metadata": metadata
            })

        return results
