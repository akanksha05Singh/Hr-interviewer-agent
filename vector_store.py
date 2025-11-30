import numpy as np
import faiss
import pickle
import os
from typing import List, Dict
from llm_client import client

# OpenAI Embedding Model
EMB_MODEL = "text-embedding-3-small"
EMB_DIM = 1536

class SimpleVectorStore:
    def __init__(self, index_path=None):
        self.dim = EMB_DIM
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadatas = []  # parallel list for metadata
        self.index_path = index_path

    def _get_embedding(self, text: str):
        try:
            text = text.replace("\n", " ")
            return client.embeddings.create(input=[text], model=EMB_MODEL).data[0].embedding
        except Exception as e:
            print(f"Embedding Error: {e}")
            # Return zero vector of correct dimension on failure
            return [0.0] * self.dim

    def _get_embeddings_batch(self, texts: List[str]):
        try:
            # OpenAI batch embedding
            # Clean newlines
            texts = [t.replace("\n", " ") for t in texts]
            resp = client.embeddings.create(input=texts, model=EMB_MODEL)
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"Batch Embedding Error: {e}")
            return [[0.0] * self.dim for _ in texts]

    def add_documents(self, docs: List[Dict]):
        # docs: list of {"id":..., "text":..., "meta": {...}}
        if not docs:
            return
        texts = [d["text"] for d in docs]
        
        try:
            vecs = self._get_embeddings_batch(texts)
            vecs_np = np.array(vecs, dtype=np.float32)
            self.index.add(vecs_np)
            self.metadatas.extend([{"id": d["id"], "meta": d.get("meta", {}), "text": d["text"]} for d in docs])
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")

    def save(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path + ".index")
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self, path):
        if not os.path.exists(path + ".index"):
            raise FileNotFoundError(f"Index not found at {path}.index")
        self.index = faiss.read_index(path + ".index")
        with open(path + ".meta.pkl", "rb") as f:
            self.metadatas = pickle.load(f)

    def query(self, q: str, k=5):
        if self.index.ntotal == 0:
            return []
        try:
            v = self._get_embedding(q)
            v_np = np.array([v], dtype=np.float32)
            D, I = self.index.search(v_np, k)
            res = []
            for idx in I[0]:
                if idx != -1 and idx < len(self.metadatas):
                    res.append(self.metadatas[idx])
            return res
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []
