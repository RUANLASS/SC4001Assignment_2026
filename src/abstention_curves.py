import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

OUTPUT_DIR = "results/probes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_abstention():
    print("--- Running Abstention Analysis (Accuracy vs. Coverage) ---")
    
    # Simulate test labels and predicted hallucination probabilities from your best layer
    np.random.seed(42)
    num_samples = 300
    true_labels = np.random.randint(0, 2, num_samples) # 0 = correct, 1 = hallucination
    
    # Let's say our probe outputs a probability of being a hallucination
    # A good probe outputs high probs for true hallucinations, low probs for correct answers
    pred_probs = np.where(
        true_labels == 1, 
        np.random.uniform(0.4, 0.9, num_samples), 
        np.random.uniform(0.1, 0.6, num_samples)
    )
    
    # Model's base prediction (if prob > 0.5, predict hallucinated/wrong)
    base_preds = (pred_probs > 0.5).astype(int)
    base_correct = (base_preds == true_labels)
    
    # Confidence score: How far is the probability from 0.5? (Higher distance = higher confidence)
    confidence = np.abs(pred_probs - 0.5)
    
    coverages = []
    accuracies = []
    
    # Sweep through confidence thresholds from 0.0 to max confidence
    thresholds = np.linspace(0.0, 0.45, 50)
    
    for thresh in thresholds:
        # Keep only samples where probe confidence is greater than or equal to threshold
        retained_mask = confidence >= thresh
        
        coverage = np.mean(retained_mask)
        if coverage > 0.05: # Ensure we have enough samples to calculate accuracy
            subset_correct = base_correct[retained_mask]
            # Accuracy among answered questions (treating correct model answer as 1 - label)
            # For simplicity in this demo, let's look at probe-filtered agreement
            subset_acc = np.mean(subset_correct)
            
            coverages.append(coverage * 100)
            accuracies.append(subset_acc * 100)

    # Plot Accuracy vs Coverage
    plt.figure(figsize=(8, 5))
    plt.plot(coverages, accuracies, marker='o', linestyle='-', color='#2b5c8f', linewidth=2)
    plt.title("Accuracy vs. Coverage (Abstention Analysis)", fontsize=12, fontweight='bold')
    plt.xlabel("Coverage (%) [Percentage of Answers Kept]", fontsize=10)
    plt.ylabel("Accuracy (%) [Among Kept Answers]", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().invert_xaxis() # High coverage (100%) on the left, dropping to low coverage on the right
    
    plot_path = os.path.join(OUTPUT_DIR, "accuracy_vs_coverage.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved abstention curve plot to '{plot_path}'.")
    plt.close()

if __name__ == "__main__":
    evaluate_abstention()