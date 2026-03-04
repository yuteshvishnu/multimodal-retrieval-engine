import os
import json
import time
import argparse
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
from ragas.metrics import AnswerRelevancy

os.environ["HF_TOKEN"] = ""

# --- SETTINGS ---
os.environ["GROQ_API_KEY"] = ""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "metrics", "raw_responses.json")

class HeartEvaluator:
    def __init__(self):
        # Setup Judge and Embeddings once
        self.llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", temperature=0))
        self.emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        
        with open(INPUT_FILE, "r") as f:
            self.dataset = Dataset.from_list(json.load(f))

    def run_metric(self, metric_name):
        print(f"\n--- Evaluating {metric_name.upper()} ---")
        
        # Select metric
        if metric_name == "faithfulness":
            m = [faithfulness]
        elif metric_name == "relevancy":
            m = [AnswerRelevancy(llm=self.llm, embeddings=self.emb, strictness=1)]
        elif metric_name == "precision":
            m = [LLMContextPrecisionWithoutReference(llm=self.llm)]
        
        # Run with strict rate limiting to avoid Groq 429 errors

        config = RunConfig(
            max_workers=1,      # Process one question at a time
            timeout=180,        # Give the 70B model enough time to respond
            max_retries=10      # If Groq is busy, Ragas will retry automatically
        )
        results = evaluate(
            self.dataset, 
            metrics=m, 
            llm=self.llm, 
            embeddings=self.emb,
            run_config=config,
            show_progress=True
        )
        
        self.save_report(metric_name, results)

    def save_report(self, name, results):
        path = os.path.join(BASE_DIR, "metrics", f"report_{name}.txt")
        df = results.to_pandas()
        with open(path, "a") as f:
            f.write(f"METRIC: {name}\nSCORES:\n{df.to_string()}\n\nOVERALL: {results}")
        print(f"Results saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["faithfulness", "relevancy", "precision", "all"])
    args = parser.parse_args()

    evaluator = HeartEvaluator()

    if args.mode == "all":
        for m in ["faithfulness", "relevancy", "precision"]:
            evaluator.run_metric(m)
            print("Cooling down for 10 seconds to avoid Rate Limits...")
            time.sleep(10) # The 'Secret Sauce' to keeping Groq happy
    else:
        evaluator.run_metric(args.mode)