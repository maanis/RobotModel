from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import requests
import whisper
import tempfile
import traceback
import time
import psutil
import torch
from functools import wraps
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from yt_dlp.utils import DownloadError
from yt_dlp import YoutubeDL
from pytube import YouTube

app = FastAPI()

# Set environment variables for better GPU utilization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["OMP_NUM_THREADS"] = "1"

# Check and setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            print(f"GPU memory before: {start_gpu_memory:.2f} MB")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            print(f"GPU memory after: {end_gpu_memory:.2f} MB")
            print(f"GPU memory used: {end_gpu_memory - start_gpu_memory:.2f} MB")
        
        print(f"Function {func.__name__} took {end_time - start_time:.2f}s")
        print(f"Memory used: {end_memory - start_memory:.2f} MB")
        
        return result
    return wrapper

# Load model with optimizations

# Initialize models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


# Optimized transcription function

@app.on_event("startup")
def load_index():
    global index, chunks_dict
    index = None
    chunks_dict = {}
    
    # Print GPU info on startup
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

class Query(BaseModel):
    question: str

class EmbedRequest(BaseModel):
    chunks: list[str]
    meeting_id: str

class IngestRequest(BaseModel):
    id: str
    text: str

def merge_transcript_chunks(transcript, max_duration=10):
    merged = []
    current_chunk = {
        "start": None,
        "end": None,
        "text": "",
    }

    for item in transcript:
        if current_chunk["start"] is None:
            current_chunk["start"] = item["start"]
            current_chunk["end"] = item["end"]
            current_chunk["text"] = item["text"]
        else:
            # Check if adding this item exceeds max duration
            if item["end"] - current_chunk["start"] <= max_duration:
                current_chunk["end"] = item["end"]
                current_chunk["text"] += " " + item["text"]
            else:
                merged.append(current_chunk)
                # start new chunk
                current_chunk = {
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                }

    if current_chunk["text"]:
        merged.append(current_chunk)

    return merged

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start = end - overlap if end < text_len else end
    return chunks

@app.post("/get-answer")
def get_answer(query: Query):
    meeting_id = "timscdr_full_profile"  # Default meeting ID
    question = query.question

    try:
        global index, chunks_dict
        if meeting_id not in chunks_dict:
            idx_path = f"faiss_indexes/{meeting_id}.index"
            chunks_path = f"faiss_indexes/{meeting_id}_chunks.json"
            if not os.path.exists(idx_path) or not os.path.exists(chunks_path):
                raise HTTPException(status_code=404, detail="Meeting data not found")
            index = faiss.read_index(idx_path)
            with open(chunks_path, "r") as f:
                chunks_dict[meeting_id] = json.load(f)
        
        chunks = chunks_dict[meeting_id]
        
        question_embedding = model.encode([question]).astype("float32")
        D, I = index.search(question_embedding, k=3)
        relevant_chunks = [chunks[i] for i in I[0]]
        context = "\n".join(relevant_chunks)
        
        prompt = f"""
You are a helpful assistant for TIMSCDR (Thakur Institute of Management Studies, Career Development & Research, Kandivali East, Mumbai).
You must answer ONLY based on the context given.
If the answer is not in the context, reply exactly:
"I'm sorry, I can only answer questions about TIMSCDR."

Question: {question}

Context:
{context}


"""

        GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyBJrJArjXtRDjf5oVx3tG86EAZHIf1MhqE"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}]
        }
        res = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=payload)
        if res.status_code == 200:
            data = res.json()
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"answer": answer}
        else:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {res.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        meeting_id = req.id
        text = req.text

        chunks = chunk_text(text)
        
        embeddings = embedding_model.encode(chunks)
        embedding_vectors = np.array(embeddings).astype("float32")

        dimension = embedding_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_vectors)

        os.makedirs("faiss_indexes", exist_ok=True)
        index_path = f"faiss_indexes/{meeting_id}.index"
        faiss.write_index(index, index_path)

        metadata_path = f"faiss_indexes/{meeting_id}_chunks.json"
        with open(metadata_path, "w") as f:
            json.dump(chunks, f)

        return {
            "meeting_id": meeting_id,
            "num_vectors": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/")
def embed_chunks(req: EmbedRequest):
    try:
        chunks = req.chunks
        meeting_id = req.meeting_id

        embeddings = embedding_model.encode(chunks)
        embedding_vectors = np.array(embeddings).astype("float32")

        dimension = embedding_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_vectors)

        os.makedirs("faiss_indexes", exist_ok=True)
        index_path = f"faiss_indexes/{meeting_id}.index"
        faiss.write_index(index, index_path)

        metadata_path = f"faiss_indexes/{meeting_id}_chunks.json"
        with open(metadata_path, "w") as f:
            json.dump(chunks, f)

        return {
            "meeting_id": meeting_id,
            "num_vectors": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)