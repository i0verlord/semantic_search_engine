import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    """Extracts raw text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    """Splits text into smaller overlapping chunks (by word count)."""
    words = text.split()
    chunks = []
    # Step through the text creating windows of size 'chunk_size'
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
