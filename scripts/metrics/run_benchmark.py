import os
import json
from backend.core.pipeline import MultimodalPipeline

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(EVAL_DIR, exist_ok=True)

# 2. Define questions
TEST_QUESTIONS = [
    "How many chambers does the human heart have, and what are the specific names and functions of each?",
    "What is the role of the Sinoatrial (SA) node in the heart’s electrical system, and how does it act as a natural pacemaker?",
    "What is the clinical relationship between coronary artery plaque buildup and the occurrence of a myocardial infarction?"
]

def run_and_save():
    pipeline = MultimodalPipeline()
    captured_data = []

    print("--- Phase 1: Running RAG Pipeline ---")
    for q in TEST_QUESTIONS:
        print(f"Querying: {q}")
        response = pipeline.run(
            query_text=q,
            sources=["heart_notes"],       # Required as per your pipeline
            collections=["medical_notes"]   # Required as per your pipeline
        )

        print(f"response: {response}")

        # Structure the data for Ragas
        chunks = response.get("citations", [])
        contexts = [c.get("snippet", "") for c in chunks]

        captured_data.append({
            "question": q,
            "answer": response["answer"],
            "contexts": contexts
        })

    # Save to JSON so you don't have to run the pipeline again
    output_file = os.path.join(EVAL_DIR, "raw_responses.json")
    with open(output_file, "w") as f:
        json.dump(captured_data, f, indent=4)
    
    print(f"\n[Done] Responses saved to {output_file}")

if __name__ == "__main__":
    run_and_save()