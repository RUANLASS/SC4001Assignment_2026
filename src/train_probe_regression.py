import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# File paths
FEATURE_PATH = "results/cached_features/pope_subset_features.pt"
LABELS_PATH = "results/cached_features/pope_subset_labels.pt"
OUTPUT_DIR = "results/probes"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FEATURE_PATH), exist_ok=True)

def generate_mock_data():
    """Generates synthetic hidden states and labels for testing upgrades."""
    print("generating enhanced mock tensors for testing")
    num_samples, num_layers, hidden_dim = 300, 28, 1536
    mock_features = torch.randn(num_samples, num_layers, hidden_dim)
    
    # Simulate a stronger hallucination signal emerging in mid-to-late layers
    for layer in range(num_layers):
        if 10 <= layer <= 22:
            mock_features[:, layer, :] += torch.randn(num_samples, hidden_dim) * 0.4
            
    mock_labels = torch.randint(0, 2, (num_samples,))
    torch.save(mock_features, FEATURE_PATH)
    torch.save(mock_labels, LABELS_PATH)
    print("Mock data generated and saved successfully.")

def train_advanced_probes(probe_type="linear"):
    """
    Trains probes across all layers using feature scaling and 
    supports both 'linear' (LogisticRegression) and 'mlp' (Neural Network).
    """
    print(f"\n--- Initializing Probing Pipeline ({probe_type.upper()}) ---")
    
    if not os.path.exists(FEATURE_PATH) or not os.path.exists(LABELS_PATH):
        generate_mock_data()

    features = torch.load(FEATURE_PATH)
    labels = torch.load(LABELS_PATH)
    
    if isinstance(features, list):
        features = torch.stack(features).numpy()
    else:
        features = features.numpy()
    labels = np.array(labels)
    
    num_samples, num_layers, hidden_dim = features.shape
    print(f"Dataset shape: {num_samples} samples | {num_layers} layers | {hidden_dim} hidden dim")

    # Consistent train/dev split
    indices = np.arange(num_samples)
    train_idx, dev_idx, y_train, y_dev = train_test_split(
        indices, labels, test_size=0.2, random_state=42, stratify=labels
    )

    layer_aurocs = []

    for layer_idx in range(num_layers):
        X_train = features[train_idx, layer_idx, :]
        X_dev = features[dev_idx, layer_idx, :]

        # 1. Feature Standardization (Essential for high-dimensional hidden states)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_dev_scaled = scaler.transform(X_dev)

        # 2. Model Selection
        if probe_type == "mlp":
            probe = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
        else:
            probe = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

        # 3. Fit and Predict
        probe.fit(X_train_scaled, y_train)
        y_pred_probs = probe.predict_proba(X_dev_scaled)[:, 1]

        # 4. Evaluate AUROC
        auroc = roc_auc_score(y_dev, y_pred_probs)
        layer_aurocs.append(auroc)
        print(f"Layer {layer_idx:2d} | AUROC: {auroc:.4f}")

    best_layer = np.argmax(layer_aurocs)
    print(f"\n Best Performing Layer: Layer {best_layer} (AUROC: {layer_aurocs[best_layer]:.4f})")

    # Plot full layer-wise trend
    plot_layer_trend(layer_aurocs, probe_type)

def plot_layer_trend(aurocs, probe_type):
    """Generates and saves a clean layer-wise performance curve."""
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(aurocs)), aurocs, marker='o', linestyle='-', color='#2b5c8f', linewidth=2)
    plt.title(f"Hallucination Detection AUROC Across Layers ({probe_type.upper()} Probe)", fontsize=12, fontweight='bold')
    plt.xlabel("Transformer Layer Index", fontsize=10)
    plt.ylabel("Validation AUROC", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = os.path.join(OUTPUT_DIR, f"layer_wise_auroc_{probe_type}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer-wise trend plot to '{plot_path}'.")
    plt.close()

if __name__ == "__main__":
    # Test both linear and non-linear configurations
    train_advanced_probes(probe_type="linear")
    train_advanced_probes(probe_type="mlp")