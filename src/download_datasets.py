import os
from datasets import load_dataset, load_from_disk

LOCAL_DATA_PATH = "data/pope"

def load_or_download_pope():
    """Load POPE from disk, downloading it from Hugging Face first if missing."""
    if os.path.exists(LOCAL_DATA_PATH) and len(os.listdir(LOCAL_DATA_PATH)) > 0:
        print(f"Found local POPE dataset at '{LOCAL_DATA_PATH}'. Loading from disk...")
        dataset = load_from_disk(LOCAL_DATA_PATH)
    else:
        print(f"Local directory '{LOCAL_DATA_PATH}' is empty or missing.")
        print("Downloading POPE dataset from Hugging Face...")

        dataset = load_dataset("lmms-lab-encoder/POPE")

        os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
        dataset.save_to_disk(LOCAL_DATA_PATH)
        print(f"Successfully downloaded and saved dataset to '{LOCAL_DATA_PATH}'.")
        
    return dataset

if __name__ == "__main__":
    pope_data = load_or_download_pope()
    print("\nDataset structure loaded:")
    print(pope_data)

    if "test" in pope_data:
        sample = pope_data["test"][0]
        print("\nSample question from test split:")
        print(f"Question: {sample.get('question')}")
        print(f"Answer: {sample.get('answer')}")