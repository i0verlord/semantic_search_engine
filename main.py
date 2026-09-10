import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ingest import extract_text, chunk_text
from vector_store import VectorStore

app = FastAPI(title="Semantic Search Engine")
store = VectorStore()

class SearchQuery(BaseModel):
    query: str
    top_k: int = 4

@app.get("/")
async def serve_frontend():
    """Serves the static frontend application."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found in directory.")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accepts a PDF, writes to a temp file, chunks it, and indexes the vectors."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Save uploaded file temporarily for PyMuPDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        text = extract_text(tmp_path)
        os.remove(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        chunks = chunk_text(text)
        store.add_chunks(chunks)

        return {"message": f"Successfully processed '{file.filename}' into {len(chunks)} searchable chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_documents(payload: SearchQuery):
    """Executes a semantic search against the FAISS index."""
    results = store.search(payload.query, top_k=payload.top_k)
    return {"results": results}
