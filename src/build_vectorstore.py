import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input: the chunks created in the previous step
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"

# Output: the saved FAISS database and metadata
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
INDEX_FILE = VECTORSTORE_DIR / "company_index.faiss"
METADATA_FILE = VECTORSTORE_DIR / "metadata.jsonl"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("=" * 55)
    print("BUILDING THE FAISS VECTOR DATABASE")
    print("=" * 55)

    # Check whether the chunks file exists
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}\n"
            "Run: python src/chunk_documents.py"
        )

    # Load all document chunks
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("No chunks were found in the input file.")

    # Extract the text that will be converted into embeddings
    texts = [chunk["text"] for chunk in chunks]

    print(f"Chunks loaded: {len(texts)}")
    print(f"Loading embedding model: {MODEL_NAME}")

    # Load the model (downloads on first run)
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # FAISS works with float32 vectors
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Create an index using cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to the index
    index.add(embeddings)

    # Create output directory
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # Save the FAISS index
    faiss.write_index(index, str(INDEX_FILE))

    # Save metadata in the same order as embeddings
    with METADATA_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 55)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 55)
    print(f"Number of vectors: {index.ntotal}")
    print(f"Embedding dimensions: {dimension}")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()