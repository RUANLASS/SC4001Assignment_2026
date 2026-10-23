import os
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from download_datasets import load_or_download_pope

# Configuration
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR = "results/cached_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features():
    print("--- Starting Feature Extraction ---")
    
    # 1. Load Dataset
    pope_dataset = load_or_download_pope()
    
    split_name = "test" # Adjust depending on the dataset structure loaded
    dataset_split = pope_dataset[split_name]
    
    # 2. Load Model and Processor
    print(f"Loading VLM: {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    print(f"Model loaded successfully on device: {device}")
    
    # 3. Processing loop (Example over a small subset first, e.g., 100 samples)
    subset_size = 100 
    print(f"Extracting features for the first {subset_size} samples...")
    
    all_hidden_states = []
    all_labels = []
    
    for i in tqdm(range(min(subset_size, len(dataset_split)))):
        sample = dataset_split[i]
        image = sample.get("image")
        question = sample.get("question")
        ground_truth = sample.get("answer").strip().lower() # "yes" or "no"
        
        # Format input for Qwen2-VL or similar chat models
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question} Answer yes or no."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = processor.image_processor(image, return_tensors="pt") if image else (None, None)
        
        inputs = processor(
            text=[text],
            images=image,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Forward pass with hidden states requested
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # outputs.hidden_states is a tuple: (num_layers + 1,) tensors of shape (batch, seq_len, hidden_dim)
        # We want to extract the hidden state of the final token of the prompt/generation
        layer_states = [layer_state[0, -1, :].cpu() for layer_state in outputs.hidden_states]
        
        # Determine model prediction vs Ground Truth to assign hallucination label (1 if hallucinated, 0 if correct)
        # For POPE: Ground truth "no" means object is absent. If model says "yes", it's a hallucination.
        # (You will parse the model's generated token text here to see what it answered)
        
        # For now, let's append the tensor stack
        all_hidden_states.append(torch.stack(layer_states)) # Shape: (num_layers, hidden_dim)

    # 4. Save cached features to disk
    save_path = os.path.join(OUTPUT_DIR, "pope_subset_features.pt")
    torch.save(all_hidden_states, save_path)
    print(f"Saved extracted hidden states to {save_path}!")

if __name__ == "__main__":
    extract_features()