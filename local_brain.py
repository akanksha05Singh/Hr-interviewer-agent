import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# Cache the model so it only loads once
@st.cache_resource
def load_local_model():
    # all-MiniLM-L6-v2 is a small, fast model perfect for local CPU inference
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_local_embedding(text: str):
    """Generate embedding using local CPU model."""
    model = load_local_model()
    return model.encode(text)

def calculate_local_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity using local embeddings."""
    if not text1 or not text2:
        return 0.0
        
    try:
        v1 = get_local_embedding(text1)
        v2 = get_local_embedding(text2)
        
        # Compute cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        sim = float(dot_product / (norm_v1 * norm_v2))
        # Normalize to 0-1 range (MiniLM is usually cosine similarity, so -1 to 1)
        return max(0.0, sim)
    except Exception as e:
        print(f"Local similarity error: {e}")
        return 0.0
