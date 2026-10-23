import os
from datasets import load_dataset, load_from_disk

# Define your local data path
LOCAL_DATA_PATH = "data/pope"

def load_or_download_pope():
    """
    Checks if the local data directory is empty/missing.
    If so, downloads the POPE dataset from Hugging Face and saves it locally.
    Otherwise, loads it directly from the local disk.
    """
    # Check if directory exists and contains files
    if os.path.exists(LOCAL_DATA_PATH) and len(os.listdir(LOCAL_DATA_PATH)) > 0:
        print(f"Found local POPE dataset at '{LOCAL_DATA_PATH}'. Loading from disk...")
        dataset = load_from_disk(LOCAL_DATA_PATH)
    else:
        print(f"Local directory '{LOCAL_DATA_PATH}' is empty or missing.")
        print("Downloading POPE dataset from Hugging Face...")
        
        # Download standard POPE dataset (using a common community repository like lmms-lab-encoder/POPE)
        dataset = load_dataset("lmms-lab-encoder/POPE")
        
        # Save locally so it's ready for future runs without re-downloading
        os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
        dataset.save_to_disk(LOCAL_DATA_PATH)
        print(f"Successfully downloaded and saved dataset to '{LOCAL_DATA_PATH}'.")
        
    return dataset

if __name__ == "__main__":
    # Test the loader
    pope_data = load_or_download_pope()
    print("\nDataset structure loaded:")
    print(pope_data)
    
    # Inspect a single split (e.g., random split)
    if "test" in pope_data:
        sample = pope_data["test"][0]
        print("\nSample question from test split:")
        print(f"Question: {sample.get('question')}")
        print(f"Answer: {sample.get('answer')}")