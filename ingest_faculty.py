import csv
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load CSV data
csv_file = 'faculty_details.csv'
chunks = []

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Combine relevant fields into a single text chunk
        text = f"Name: {row['Name']}, Designation: {row['Designation']}, Details: {row['Details']}"
        chunks.append(text)

# Embed
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embeddings = embedding_model.encode(chunks)
embedding_vectors = np.array(embeddings).astype("float32")

# Create index
dimension = embedding_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_vectors)

# Save
meeting_id = "faculty_details"
os.makedirs("faiss_indexes", exist_ok=True)
index_path = f"faiss_indexes/{meeting_id}.index"
faiss.write_index(index, index_path)

metadata_path = f"faiss_indexes/{meeting_id}_chunks.json"
with open(metadata_path, "w") as f:
    json.dump(chunks, f)

print(f"Ingested {len(chunks)} chunks from CSV for meeting {meeting_id}")