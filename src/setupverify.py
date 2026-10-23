import torch
from transformers import AutoProcessor

def test_environment():
    print("--- Environment Smoke Test ---")
    
    # 1. Check PyTorch & CUDA
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (Note: Feature extraction will be slower).")

    # 2. Test importing a lightweight model architecture (e.g., Qwen2-VL-2B)
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"\nChecking ability to reference model: {model_id}")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Success: Processor loaded successfully.")
    except Exception as e:
        print(f"Note: Could not load processor yet (check internet/auth if needed): {e}")

    print("\nEnvironment groundwork is ready!")

if __name__ == "__main__":
    test_environment()