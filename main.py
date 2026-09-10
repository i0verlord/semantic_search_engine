import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from ingest import extract_text, chunk_text
from vector_store import VectorStore

app = FastAPI(title="Semantic Search Engine")
store = VectorStore()

class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=4, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be blank")
        return value

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
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = None
    try:
        # Save uploaded file temporarily for PyMuPDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        text = extract_text(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        chunks = chunk_text(text)
        store.add_chunks(chunks)

        return {"message": f"Successfully processed '{file.filename}' into {len(chunks)} searchable chunks."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process PDF: {e}") from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/search")
async def search_documents(payload: SearchQuery):
    """Executes a semantic search against the FAISS index."""
    results = store.search(payload.query, top_k=payload.top_k)
    return {"results": results}
