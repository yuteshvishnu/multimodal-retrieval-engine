import os
from pathlib import Path
import numpy as np
import json

from backend.encoders.text_encoder import TextEncoder  # we’ll use this soon

RAW_DOCS_DIR = Path("data/raw_docs")
PROCESSED_DIR = Path("data/processed")  # later we’ll save index/metadata here


def chunk_text_sliding(
    text: str,
    window_size: int = 10,
    overlap: int = 5,
) -> list[str]:
    """
    Split text into overlapping word windows.

    Example:
        window_size = 120, overlap = 40
        -> chunks of ~120 words, where consecutive chunks share ~40 words.

    This approximates token-based chunking without needing a tokenizer.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    n = len(words)
    print(f"n={n}")

    while start < n:
        end = min(start + window_size, n)
        chunk_words = words[start:end]
        print(f"Chunking text: {chunk_words}")
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end == n:
            break

        # move the window with overlap
        start = end - overlap

    return chunks


def main():
    print("Building index from raw docs...")
    print(f"Reading from: {RAW_DOCS_DIR.resolve()}")

    if not RAW_DOCS_DIR.exists():
        print("ERROR: raw docs directory does not exist.")
        return

    txt_files = sorted(RAW_DOCS_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} text files")

    if not txt_files:
        print("No .txt files found in data/raw_docs. Add some docs first.")
        return

    # docs = []
    # for path in txt_files:
    #     text = path.read_text(encoding="utf-8")
    #     docs.append({"id": path.stem, "path": str(path), "text": text})
    #     preview = text[:80].replace("\n", " ")
    #     print(f"- Loaded {path.name}, first 80 chars: {preview!r}")

    # print(f"Loaded {len(docs)} documents into memory.")

    docs = []
    chunk_count = 0

    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        # Simple chunking: split on double newlines (paragraphs)
        raw_chunks = chunk_text_sliding(
            text,
            window_size=10,
            overlap=5,
        )

        print(f"Loaded {len(raw_chunks)} chunks from {path.name}")

        for idx, chunk in enumerate(raw_chunks):
            doc_id = f"{path.stem}_chunk{idx}"
            print(f"path,{path.stem}")
            docs.append(
                {
                    "id": doc_id,
                    "path": str(path),
                    "text": chunk,
                    "source": str(path.stem)
                }
            )
            chunk_count += 1

        preview = text[:80].replace("\n", " ")
        print(
            f"- Loaded {path.name} with {len(raw_chunks)} chunks, "
            f"first 80 chars of file: {preview!r}"
        )

    print(f"Total chunks loaded into memory: {chunk_count}")

    print("Initializing TextEncoder...")
    encoder = TextEncoder()

    embeddings = []
    metadata = []

    print("Encoding documents...")

    for doc in docs:
        vec = encoder.encode(doc["text"])
        embeddings.append(vec)
        metadata.append({
            "id": doc["id"],
            "path": doc["path"],
            "text": doc["text"],
            "source": doc["source"]
        })

        print(f"- Embedded {doc['id']} (dim={vec.shape[0]})")

    embeddings = np.vstack(embeddings)  # shape: (num_docs, dim)

    print(f"Final embeddings shape: {embeddings.shape}")

    # --- Simple similarity test (no index yet) ---
    test_query = "What does the heart do in the body?"
    print(f"\nRunning simple similarity test with query: {test_query!r}")
    q_vec = encoder.encode(test_query)

    # cosine similarity between query and each doc
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    sims = []
    for doc, doc_vec in zip(metadata, embeddings):
        sim = cosine_sim(q_vec, doc_vec)
        sims.append((doc["id"], sim))

    # sort by similarity descending
    sims.sort(key=lambda x: x[1], reverse=True)

    print("Similarity scores (highest is most relevant):")
    for doc_id, score in sims:
        print(f"- {doc_id}: {score:.3f}")

    print("\nEmbedding + similarity test complete. (We still haven't saved an index yet.)")

        # --- Save embeddings + metadata to disk ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    embeddings_path = PROCESSED_DIR / "embeddings.npy"
    metadata_path = PROCESSED_DIR / "metadata.json"

    np.save(embeddings_path, embeddings)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nSaved embeddings to: {embeddings_path}")
    print(f"Saved metadata to:   {metadata_path}")
    print("Index data is now persisted (still no Faiss index, just arrays + metadata).")


if __name__ == "__main__":
    main()