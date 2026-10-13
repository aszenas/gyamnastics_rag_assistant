import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


class VectorStore:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")

        self.namespace = os.getenv(
            "PINECONE_NAMESPACE", "__default__"
        )

        self.pc = Pinecone(api_key=api_key)

        #index konfiguráció check
        info = self.pc.describe_index(name=index_name)

        if info.dimension != 384:
            raise ValueError(
                f"Hibás indexdimenzió: {info.dimension}, "
                "elvárt: 384"
            )

        if info.metric != "cosine":
            raise ValueError(
                f"Hibás metric: {info.metric}, elvárt: cosine"
            )

        self.index = self.pc.Index(name=index_name)

        print(f"Pinecone index: {index_name}")
        print(f"Namespace: {self.namespace}")

    def upsert_chunks(self, chunks, embeddings, batch_size=50):

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_embeddings = embeddings[start:start + batch_size]

            vectors = []

            for chunk, embedding in zip(
                batch_chunks, batch_embeddings
            ):
                metadata = {
                    key: value
                    for key, value in chunk["metadata"].items()
                    if value is not None
                }

                #a szöveget is tárolom
                metadata["text"] = chunk["text"]

                vectors.append({
                    "id": chunk["id"],
                    "values": embedding.tolist(),
                    "metadata": metadata
                })

            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )

            print(
                f"Feltöltve: "
                f"{min(start + batch_size, len(chunks))}"
                f"/{len(chunks)}"
            )

    def get_stats(self):
        return self.index.describe_index_stats()
