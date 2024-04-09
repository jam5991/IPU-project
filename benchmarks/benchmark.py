import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def benchmark_model(model_path, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # Prepare the model for inference
    model.eval()

    texts = ["Your text to analyze"] * 100  # Example texts repeated to simulate workload
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)

    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(**inputs)
    elapsed_time = time.time() - start_time
    print(f"Average inference time on {device}: {elapsed_time / 100:.3f} seconds")

if __name__ == "__main__":
    benchmark_model("./model", device="cpu")  # Repeat for "cuda" and IPU-specific settings
