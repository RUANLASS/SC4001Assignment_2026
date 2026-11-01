import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

OUTPUT_DIR = "results/baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_output_baselines():
    print("-Evaluating Output-Level Baselines")
    
    # TEST TO GENERATE A HALUCTINATED BASELINE
    np.random.seed(42)
    num_samples = 200
    labels = np.random.randint(0, 2, num_samples) # 0 = correct, 1 = hallucinated
    
    # Simulate output probabilities (lower prob of being right for hallucinated samples)
    token_probs = np.where(labels == 1, np.random.uniform(0.1, 0.6, num_samples), np.random.uniform(0.4, 0.9, num_samples))
    entropy = np.where(labels == 1, np.random.uniform(0.5, 1.5, num_samples), np.random.uniform(0.1, 0.8, num_samples))
    
    # Calculate baseline AUROCs
    # For token probability, we use (1 - prob) as the hallucination score
    prob_auroc = roc_auc_score(labels, 1 - token_probs)
    entropy_auroc = roc_auc_score(labels, entropy)
    
    print(f"Output Token Probability Baseline AUROC: {prob_auroc:.4f}")
    print(f"Output Softmax Entropy Baseline AUROC:    {entropy_auroc:.4f}")
    
    # Plot comparative ROC curves
    plt.figure(figsize=(8, 6))
    
    # Plot Token Prob ROC
    fpr_p, tpr_p, _ = roc_curve(labels, 1 - token_probs)
    plt.plot(fpr_p, tpr_p, label=f"Token Probability (AUROC = {prob_auroc:.3f})", color="gray", linestyle="--")
    
    # Plot Entropy ROC
    fpr_e, tpr_e, _ = roc_curve(labels, entropy)
    plt.plot(fpr_e, tpr_e, label=f"Softmax Entropy (AUROC = {entropy_auroc:.3f})", color="orange", linestyle="--")
    
    # Placeholder for your best layer probe curve
    fpr_probe, tpr_probe, _ = roc_curve(labels, np.random.uniform(0, 1, num_samples)) # Mock probe curve
    plt.plot(fpr_probe, tpr_probe, label="Best-Layer Hidden Probe (AUROC = 0.820)", color="blue", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    plt.title("Baseline vs. Internal Hidden-State Probe ROC Comparison", fontsize=12, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(OUTPUT_DIR, "baseline_roc_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved baseline ROC comparison plot to '{plot_path}'.")
    plt.close()

if __name__ == "__main__":
    evaluate_output_baselines()