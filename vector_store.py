import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model = SentenceTransformer(model_name)
        # Initialize a flat L2 distance index for precise nearest-neighbor search
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add_chunks(self, chunks: list[str]):
        """Encodes text chunks and adds them to the FAISS index."""
        if not chunks:
            return

        embeddings = self.model.encode(chunks)
        # FAISS strictly requires float32 data types
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """Converts the query to a vector and retrieves the closest matches."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.index.ntotal == 0:
            return []

        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append({
                    "score": float(dist),
                    "text": self.chunks[idx]
                })
        return results
