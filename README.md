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
Now that we created a small pipeline, where we give text as input, and relevant top k chunks from the vector database, we are just returning the top k chunks, now lets use a actual llm to give a proper reasoning

we have used : ##microsoft/Phi-3-mini-4k-instruct##, and it is taking so much long time
it provides a decent reasoning by taking lot of time
for now, we are just showing relevant chunks (top k)

# Version-3 : Smarter Chunking (Token-based + Overlap)
We were using a line in a paragraph as a chunk to store in our vector embeddings store, now we broke them into words and set a limit of 80 characters, with 40 characters overlap, to get relevant chunks, (concept used: sliding window)

understanding importance of chunk length and overlap
chunk length: too big, captures more context but leads to fewer chunks, making it heavy embedding, thereby containing multiple topics and too small, its more precise but may miss the context
overlap: safe boundaries but creates redundancy

chunk length: 80, overlap length: 40 worked the best


# Version-4 : Two-Stage Retrieval (Re-Ranking)
Stage-1: we were directly embedding the query as vector and getting sim score and returning top k chunks using similarity score
Stage-2: Now to make it more precise, we select from those, we device a better matching algorithm with the query and top k chunks we have

1. (what we are using now) checking how many words are common between query and chunk : 
    Store each word of query in one map
    string stream each word in chunk and check if its in the map
    TC: O(n), but not very efficient

2. checking sim score between each word of query and chunk : 
    for each chunk (lets say n), we need to do sim score (lets say it takes O(1) time), each chunk has k words, and query has m words 
    then TC: O(nxkxm), expensive

3. Incorporating the modern concept of cross-encoders
These are basically transformer based architecture which uses self attention and calculates scores for query and chunk to get a relevance score
So we pass the query and each chunk/snippet to this transformer architecture and get relevant score and sort them based on that
we get pre-trained models, we can plugin to get the scores, such as ##cross-encoder/ms-marco-MiniLM-L-12-v2##


# Version-5 : Metadata-Aware Search (Filters & Scopes)
current chunk meta 
    {"id": "doc1_chunk0", "path": "...", "text": "..."}

This is not enough to 
    1. filter by document type
    2. group by source
    3. only search in this subset

Improvement
    1. we shall start by adding a source field in the metadata : to understand from where we got this chunk
    2. lets add source filter, to basically generate output based on only selected sources: what we are doing here, is setting the input space from which we need our answers
    3. we provided an option to user to select the sources from which they want the output, along with provided source information from the selected chunks

# Version-6 : Evaluation & Metrics (Text-Only Bench)
We were able to create a basic work flow we can see certain results based on the input space and query asked, but we dont have specific metrics to evaluate 
    1. verifying if correct source is used
    2. verifying if correct snippet is part of output


# Version-7 : Relevance Feedback Loop (Learning from Thumbs-Up/Down)
If we don’t have a real model that learns, what is the point of collecting user feedback?
    Currently this are the steps we follow to get relevant context chunks
        1. top 10 chunks
        2. rerank and get top 5
        3. filter score by threshold
    
    But we need to know
        1. is threshold too high or too low ?
        2. do users want fever chunks
        3. are we retriveing irrevalent items frequently, how to get better?

    Improvements to
        1. tuning threshold
        2. tune re-rank behavior, choose the right logic
        3. understand chunk issues and make it better
        4. provide live performance metrics
        5. real training data for future LLM Integration
        6. making it production ready




# Version-8 : Multi-Collection / Namespace Support

# Version-9 : Polished UX: Rich Text UI + Citation Viewer

# Version-10 : Multimodal Extension (Text + Image Retrieval)

