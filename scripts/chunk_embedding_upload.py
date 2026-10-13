import json
from pathlib import Path
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "rag_chunks.jsonl"


def main():

    #input read
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"beolvasott chunkok: {len(chunks)}")

    #embedding model load az embeddings.py-bol
    embedding_model = EmbeddingModel()

    #embeddingek elkészítése
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.embed_documents(texts)

    print(f"Embeddingek alakja: {embeddings.shape}")

    #pinecone kapcsolat
    vector_store = VectorStore()

    # feltöltés
    vector_store.upsert_chunks(chunks, embeddings)

    # stats
    stats = vector_store.get_stats()
    print("\nIndex statisztika:")
    print(stats)

    print("\nFeltöltési folyamat befejeződött!")


if __name__ == "__main__":
    main()
