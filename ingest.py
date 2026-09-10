import fitz  # PyMuPDF


def extract_text(pdf_path: str) -> str:
    """Extracts raw text from a PDF file."""
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc) + "\n"


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    """Splits text into smaller overlapping chunks (by word count)."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and greater than overlap")

    words = text.split()
    chunks = []
    # Step through the text creating windows of size 'chunk_size'
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks
