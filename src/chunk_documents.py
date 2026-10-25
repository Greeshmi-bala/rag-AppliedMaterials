import json
import re
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_FILE = PROJECT_ROOT / "data" / "company_documents.jsonl"
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


def split_text(text, chunk_size=800, chunk_overlap=120):
    """
    Split text into overlapping chunks without external libraries.

    Args:
        text (str): Input text to split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters.

    Returns:
        listList of text chunks.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending at a word boundary
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def main():
    """
    Load company documents, split them into chunks,
    and save the chunks as JSONL.
    """

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Company documents not found: {DOCUMENTS_FILE}"
        )

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    documents = []

    with DOCUMENTS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    if not documents:
        raise ValueError("The company document dataset is empty.")

    chunks = []

    for document in documents:
        text = document.get("text", "").strip()

        if not text:
            continue

        passages = split_text(text)

        for index, passage in enumerate(passages, start=1):
            chunk = {
                "chunk_id": f"{document['document_id']}_CHUNK_{index:03d}",
                "document_id": document["document_id"],
                "title": document.get("title", ""),
                "category": document.get("category", ""),
                "source_url": document.get("source_url", ""),
                "text": passage,
            }

            chunks.append(chunk)

    with CHUNKS_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print("=" * 50)
    print("DOCUMENT CHUNKING COMPLETED")
    print("=" * 50)
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Output file: {CHUNKS_FILE}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0]["text"])


if __name__ == "__main__":
    main()