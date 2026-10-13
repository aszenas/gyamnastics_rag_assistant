
from src.embeddings import EmbeddingModel
import numpy as np

embedding_model = EmbeddingModel()

documents = [
    "Gerendán a tornásznak 10 másodperce van a leesés után visszamászni.",
    "Talajon a gyakorlat maximális időtartama 90 másodperc.",
    "A hajlított térdért az E-zsűri pontlevonást alkalmaz."
]

query = "Mennyi időn belül kell visszamászni a gerendára leesés után?"

#embeddingek generálása
doc_embeddings = embedding_model.embed_documents(documents)
query_embedding = embedding_model.embed_query(query)

#hasonlóság kiszámítása
similarities = doc_embeddings @ query_embedding

print(f"dokumentum embeddingek alakja: {doc_embeddings.shape}")
print(f"kérdés embedding alakja: {query_embedding.shape}")

print("\nszemantikus hasonlóságok:")
for idx in np.argsort(similarities)[::-1]:
    print(f"{similarities[idx]:.4f} | {documents[idx]}")
