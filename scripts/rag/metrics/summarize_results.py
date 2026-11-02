import os
import re

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "metrics")
OUTPUT_FILE = os.path.join(EVAL_DIR, "final_Report.txt")

# Metric files we created earlier
METRIC_FILES = {
    "Faithfulness": "../report_faithfulness.txt",
    "Relevancy": "../report_relevancy.txt",
    "Context Precision": "../report_precision.txt"
}

def extract_score(filename):
    """Regex helper to find the 'Average' score in your report text files."""
    path = os.path.join(EVAL_DIR, filename)
    if not os.path.exists(path):
        return None
    
    with open(path, "r") as f:
        content = f.read()
        # Searches for patterns like 'Average: 0.85' or 'OVERALL: {'faithfulness': 0.85}'
        match = re.search(r"(?:Average|OVERALL).*?(\d+\.\d+)", content)
        return float(match.group(1)) if match else None

def calculate_harmonic_mean(scores):
    """Calculates the standard Ragas Score (Harmonic Mean)."""
    if not scores or any(s == 0 for s in scores):
        return 0.0
    return len(scores) / sum(1.0 / s for s in scores)

def generate_master_report():
    print("--- Generating Master Report ---")
    extracted_scores = {}
    
    for name, filename in METRIC_FILES.items():
        score = extract_score(filename)
        if score is not None:
            extracted_scores[name] = score
        else:
            print(f"Warning: Could not find score in {filename}")

    if not extracted_scores:
        print("Error: No scores found. Did you run the individual eval scripts?")
        return

    # Calculate Overall Score (Harmonic Mean)
    final_score = calculate_harmonic_mean(list(extracted_scores.values()))

    # Write the Final Report
    with open(OUTPUT_FILE, "w") as f:
        f.write("==========================================\n")
        f.write("      HEART RAG MASTER EVALUATION         \n")
        f.write("==========================================\n\n")
        
        for name, score in extracted_scores.items():
            f.write(f"{name:<20} : {score:.4f}\n")
        
        f.write("-" * 42 + "\n")
        f.write(f"{'OVERALL RAGAS SCORE':<20} : {final_score:.4f}\n\n")
        
        # Add a quick insight
        f.write("INSIGHT:\n")
        if final_score > 0.8:
            f.write("- High Performance: Your heart RAG is production-ready.\n")
        elif extracted_scores.get("Faithfulness", 1) < 0.7:
            f.write("- Issue Detected: Low Faithfulness suggests hallucinations. Check context chunks.\n")
        elif extracted_scores.get("Context Precision", 1) < 0.7:
            f.write("- Issue Detected: Low Precision suggests the retriever is finding noisy data.\n")

    print(f"[Success] Master Report saved to: {OUTPUT_FILE}")
    print(f"Overall Score: {final_score:.4f}")

if __name__ == "__main__":
    generate_master_report()