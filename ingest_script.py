import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
with open('timscdr_data.json', 'r') as f:
    data = json.load(f)

meeting_id = data['id']
text = data['text']

# Chunk text
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - overlap if end < text_len else end
    return chunks

chunks = chunk_text(text)

# Embed
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embeddings = embedding_model.encode(chunks)
embedding_vectors = np.array(embeddings).astype("float32")

# Create index
dimension = embedding_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_vectors)

# Save
os.makedirs("faiss_indexes", exist_ok=True)
index_path = f"faiss_indexes/{meeting_id}.index"
faiss.write_index(index, index_path)

metadata_path = f"faiss_indexes/{meeting_id}_chunks.json"
with open(metadata_path, "w") as f:
    json.dump(chunks, f)

print(f"Ingested {len(chunks)} chunks for meeting {meeting_id}")