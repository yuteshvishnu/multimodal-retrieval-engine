# Version-1: Multimodal Retrieval and Reasoning Engine

# Version-1.1 : Basic Text Semantic Search + Bullet Summary
we build a basic architecture

Frontend: takes text as input (like a search bar), and gives relevant output
Backend:
    1. reads the data sources we provide, breaks them into chunks and create their respective vector embeddings and stores it 
    2. takes the input, compares the query string with the chunks we have and finds the nearest one, (i.e) in terms of difference between norms
    3. find the cosine similarity, sort by similarity, normalize scores between 0 to 1
    4.  list out the ones above threshold
    5. return in points


# Version-1.2 : LLM-Powered Answering (True RAG)
Now that we created a small pipeline, where we give text as input, and relevant top k chunks from the vector database, we are just returning the top k chunks, now lets use a actual llm to give a proper reasoning

we have used : ##microsoft/Phi-3-mini-4k-instruct##, and it is taking so much long time
it provides a decent reasoning by taking lot of time
for now, we are just showing relevant chunks (top k)

# Version-1.3 : Smarter Chunking (Token-based + Overlap)
We were using a line in a paragraph as a chunk to store in our vector embeddings store, now we broke them into words and set a limit of 80 characters, with 40 characters overlap, to get relevant chunks, (concept used: sliding window)

understanding importance of chunk length and overlap
chunk length: too big, captures more context but leads to fewer chunks, making it heavy embedding, thereby containing multiple topics and too small, its more precise but may miss the context
overlap: safe boundaries but creates redundancy

chunk length: 80, overlap length: 40 worked the best


# Version-1.4 : Two-Stage Retrieval (Re-Ranking)
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


# Version-1.5 : Metadata-Aware Search (Filters & Scopes)
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

# Version-1.6 : Evaluation & Metrics (Text-Only Bench)
We were able to create a basic work flow we can see certain results based on the input space and query asked, but we dont have specific metrics to evaluate 
    1. verifying if correct source is used
    2. verifying if correct snippet is part of output


# Version-1.7 : Relevance Feedback Loop (Learning from Thumbs-Up/Down)
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




# Version-1.8 : Multi-Collection / Namespace Support
Till now we have this single point of data dump, where we have all sources from which we generate the chunks, but as we build, we require bigger buckets which consists of specific files

1. we added an option to incldue a specific collection first, and then the source to search inside of collection, giving more control to the user to specific select the sources or context from which we need the model to answer


# Version-1.9 : Polished UX: Rich Text UI + Citation Viewer
Providing a better UI experience with better layout

# Version-1.10 : Multimodal Extension (Text + Image Retrieval)
Now that we completed the flow for the text, we shall do for the image, 

It does not change the value, only the type and how it is stored.

Raw Image Bytes
        |
        v
   SHA-256 Hash (32 bytes)
        |
        v
 Convert to float32 array (length=32)
        |
        v
   Repeat until length=768
        |
        v
   Normalize (unit vector)
        |
        v
  Final Embedding (768-dim)


  This concludes the basic skeleton implementation of a working multi-modal RAG based approach for Text + Image inputs

# Summary of Version-1

## Complete Life-cycle
In the First version of our RAG modal, we were able to create a personal CHATGPT Based model, 
    1. A UI which provides answers to user queries based on pre-stored data
    2. Allows users to choose the sources from which they would like the model to give the output
    3. Output includes relevant answer + its relevant reasoning + related citations from the sources

## Concepts used
 1. Chunk based (limit + overlap configured, sliding window pattern) tokens stored in vector array database (dim = 768)
 2. query encoding and similarity finding using loat(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)), to get top 20 tokens
 3. then concept of re-ranking: term-based heuristic on the candidates to get final top 5 candidates
 4. reason through online groq model

 ## Models used
 sentence transformer: all-mpnet-base-v2
 reasoning: llama-3.1-8b-instant

## Evaluation Metrics
Used RAGAS score to below metric scores and final overal score, tests used : 3


Faithfulness         : 0.5333
Relevancy            : 0.8713
Context Precision    : 0.4444
------------------------------------------
OVERALL RAGAS SCORE  : 0.5689

INSIGHT:
- Issue Detected: Low Faithfulness suggests hallucinations. Check context chunks.

## Conclusion
It is clearly evident that our RAG modal still requires better statistics in terms of performance, and that would be the goal of version-2 of this project

# Version-2: RAG Performance Improvements

// CURRENTLY IN PROGRESS



