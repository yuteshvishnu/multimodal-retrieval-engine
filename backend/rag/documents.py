"""
Scribe Version-2: the user's sources. Lists the documents Scribe can answer from, and adds
new ones (.txt, .md, .pdf, .docx) to the search index without a restart.

Each document is split into overlapping word windows (chunks), each chunk is embedded,
and the vectors are appended to data/rag/processed/ (embeddings.npy + metadata.json),
the same files scripts/rag/build_index.py writes.
"""
import json
import math
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import numpy as np

RAW_DOCS_DIR = Path("data/rag/raw_docs")
SUPPORTED = {".txt": "Text", ".md": "Markdown", ".pdf": "PDF", ".docx": "Word"}
CHUNK_WORDS = 40
CHUNK_OVERLAP = 10
PAGE_SIZE = 10
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# text that tries to give the model instructions (prompt injection); flagged on upload
INSTRUCTION_LIKE = re.compile(
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)|"
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|earlier)|forget\s+(all\s+|your\s+)?(previous\s+)?instructions|"
    r"\byou\s+are\s+now\b|\bsystem\s+prompt\b|\bnew\s+instructions\b|\bact\s+as\s+(a|an|the)\b",
    re.IGNORECASE,
)


def instruction_like(text: str) -> List[str]:
    """Phrases in a document that look like instructions to the model."""
    return sorted({m.group(0).lower() for m in INSTRUCTION_LIKE.finditer(text)})


def chunk_text_sliding(text: str, window_size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word windows, so ideas aren't cut at chunk boundaries."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + window_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def extract_text(name: str, content: bytes) -> str:
    """The plain text of a supported file."""
    suffix = Path(name).suffix.lower()
    if suffix in (".txt", ".md"):
        return content.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        from docx import Document

        return "\n".join(p.text for p in Document(BytesIO(content)).paragraphs)
    raise ValueError(f"Unsupported file type: {suffix or 'none'}")


def document_chunks(path: Path, text: str) -> List[Dict]:
    """Metadata entries for one document, in the format the vector index reads."""
    collection = "default" if path.parent.name == RAW_DOCS_DIR.name else path.parent.name
    return [
        {"id": f"{path.stem}_chunk{i}", "path": str(path), "text": chunk, "source": path.stem, "collection": collection}
        for i, chunk in enumerate(chunk_text_sliding(text))
    ]


def page_of(items: list, page: int, page_size: int = PAGE_SIZE) -> dict:
    pages = max(1, math.ceil(len(items) / page_size))
    page = min(max(1, page), pages)
    start = (page - 1) * page_size
    return {"items": items[start:start + page_size], "page": page, "pages": pages, "total": len(items)}


class DocumentStore:
    def __init__(self, index, encoder):
        self.index = index
        self.encoder = encoder
        self._lock = threading.Lock()

    def _metadata(self) -> List[Dict]:
        return self.index.metadata or []

    def sources(self) -> List[Dict]:
        """One card per document, in the order they were added."""
        cards: Dict[str, Dict] = {}
        for chunk in self._metadata():
            source = chunk.get("source", "text_corpus")
            if source not in cards:
                path = Path(chunk.get("path", ""))
                cards[source] = {
                    "name": source,
                    "file": path.name,
                    "type": SUPPORTED.get(path.suffix.lower(), path.suffix.lstrip(".").upper() or "Text"),
                    "size": path.stat().st_size if path.exists() else None,
                    "chunks": 0,
                    "preview": chunk["text"][:240],
                }
            cards[source]["chunks"] += 1
        return list(cards.values())

    def list(self, page: int, search: str = "") -> dict:
        words = search.lower().split()
        cards = [c for c in self.sources() if all(w in f"{c['name']} {c['file']} {c['preview']}".lower() for w in words)]
        return page_of(cards, page)

    def chunks(self, source: str, page: int) -> dict:
        items = [{"id": c["id"], "text": c["text"]} for c in self._metadata() if c.get("source") == source]
        if not items:
            raise KeyError(source)
        return page_of(items, page)

    def add(self, filename: str, content: bytes) -> Dict:
        """
        Saves the file to data/rag/raw_docs/, chunks and embeds it, and appends it to the
        index. Raises ValueError with a message for the user if it can't be added.
        """
        suffix = Path(filename or "").suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError("Supported files: " + ", ".join(SUPPORTED) + ".")
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("The file is larger than 50 MB.")

        # a clean name: it's the source name used to filter questions (no commas or spaces)
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_") or "document"
        try:
            text = extract_text(filename, content)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Could not read the file: {e}")
        if not text.strip():
            raise ValueError("No text found in the file (a scanned PDF has images, not text).")

        with self._lock:
            if any(c["name"].lower() == stem.lower() for c in self.sources()):
                raise ValueError(f"A document called '{stem}' already exists.")

            RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
            path = RAW_DOCS_DIR / f"{stem}{suffix}"
            path.write_bytes(content)

            chunks = document_chunks(path, text)
            vectors = np.vstack([self.encoder.encode(c["text"]) for c in chunks])
            self._append(chunks, vectors)

        card = next(c for c in self.sources() if c["name"] == stem)
        found = instruction_like(text)
        if found:
            card["warning"] = (
                "This document contains text that looks like instructions to the AI ("
                + ", ".join(f"“{f}”" for f in found[:3])
                + "). Scribe treats documents as data and ignores such text, but check the document is trusted."
            )
        return card

    def delete(self, source: str) -> None:
        """
        Removes a document: its passages and vectors from the index, and its file from
        data/rag/raw_docs/. Raises KeyError if there's no such document.
        """
        with self._lock:
            metadata = self._metadata()
            keep = [i for i, c in enumerate(metadata) if c.get("source") != source]
            if len(keep) == len(metadata):
                raise KeyError(source)
            paths = {Path(c["path"]) for c in metadata if c.get("source") == source and c.get("path")}

            vectors = self.index.embeddings[keep] if keep else np.zeros((0, self.index.dim or 768), dtype="float32")
            self._save([metadata[i] for i in keep], vectors)
            for path in paths:
                # only files inside the documents folder
                if path.resolve().is_relative_to(RAW_DOCS_DIR.resolve()) and path.exists():
                    path.unlink()

    def _append(self, chunks: List[Dict], vectors: np.ndarray) -> None:
        """Adds chunks to the saved index files, then reloads the index."""
        metadata = self._metadata() + chunks
        if self.index.embeddings is not None and len(self.index.embeddings):
            vectors = np.vstack([self.index.embeddings, vectors])
        self._save(metadata, vectors)

    def _save(self, metadata: List[Dict], vectors: np.ndarray) -> None:
        """Writes the index files, then reloads the index."""
        embeddings_path, metadata_path = self.index.embeddings_path, self.index.metadata_path
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, vectors)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self.index._load_index()
