# Multimodal Retrieval and Reasoning Engine

# Version-1 : Basic Text Semantic Search + Bullet Summary
we build a basic architecture

Frontend: takes text as input (like a search bar), and gives relevant output
Backend:
    1. reads the data sources we provide, breaks them into chunks and create their respective vector embeddings and stores it 
    2. takes the input, compares the query string with the chunks we have and finds the nearest one, (i.e) in terms of difference between norms
    3. find the cosine similarity, sort by similarity, normalize scores between 0 to 1
    4.  list out the ones above threshold
    5. return in points


# Version-2 : LLM-Powered Answering (True RAG)

# Version-3 : Smarter Chunking (Token-based + Overlap)

# Version-4 : Two-Stage Retrieval (Re-Ranking)

# Version-5 : Metadata-Aware Search (Filters & Scopes)

# Version-6 : Evaluation & Metrics (Text-Only Bench)

# Version-7 : Relevance Feedback Loop (Learning from Thumbs-Up/Down)

# Version-8 : Multi-Collection / Namespace Support

# Version-9 : Polished UX: Rich Text UI + Citation Viewer

# Version-10 : Multimodal Extension (Text + Image Retrieval)

