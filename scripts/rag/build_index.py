"""
Rebuilds Scribe's search index from every document in data/rag/raw_docs/
(.txt, .md, .pdf, .docx): chunk -> embed -> save to data/rag/processed/.

Documents uploaded in the app are added to the index right away; run this only to
rebuild everything, e.g. after editing or deleting files by hand.

    python -m scripts.rag.build_index
"""
import json
from pathlib import Path

import numpy as np

from backend.rag.documents import RAW_DOCS_DIR, SUPPORTED, document_chunks, extract_text
from backend.rag.encoders.text_encoder import get_text_encoder

PROCESSED_DIR = Path("data/rag/processed")


def main():
    print(f"Building index from {RAW_DOCS_DIR.resolve()}")
    files = sorted(p for p in RAW_DOCS_DIR.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No documents found. Add {', '.join(SUPPORTED)} files to {RAW_DOCS_DIR}.")
        return

    metadata = []
    for path in files:
        chunks = document_chunks(path, extract_text(path.name, path.read_bytes()))
        print(f"- {path.name}: {len(chunks)} chunks")
        metadata += chunks

    encoder = get_text_encoder()
    embeddings = np.vstack([encoder.encode(chunk["text"]) for chunk in metadata])
    print(f"Embedded {len(metadata)} chunks, shape {embeddings.shape}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "embeddings.npy", embeddings)
    (PROCESSED_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {PROCESSED_DIR}/embeddings.npy and metadata.json")


if __name__ == "__main__":
    main()
