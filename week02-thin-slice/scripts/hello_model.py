"""Step 1: talk to a model directly, no server. Prints the answer word by word."""

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = os.getenv("DEVICE") or pick_device()
    print(f"Model: {MODEL_ID}  |  device: {device}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    messages = [{"role": "user", "content": "Explain what a GPU is in two sentences."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("--- what the model actually sees (chat template applied) ---")
    print(prompt)
    print("--- answer ---")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    t0 = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, streamer=streamer)
    elapsed = time.perf_counter() - t0

    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    print(f"\n{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tokens/s")


if __name__ == "__main__":
    main()
