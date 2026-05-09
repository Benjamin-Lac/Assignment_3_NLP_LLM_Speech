import os
import pickle

import numpy as np
import torch
import faiss
import pyttsx3
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# 1. Load the saved email RAG data
# ----------------------------
index = faiss.read_index("rag_data/email_index.faiss")
with open("rag_data/email_chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Loaded email RAG index with {index.ntotal} vectors")


# ----------------------------
# 2. Load LLM
# ----------------------------
# TinyLlama can be memory-heavy on CPU. Use a smaller default on CPU for reliability.
default_model_name = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if torch.cuda.is_available()
    else "Qwen/Qwen2.5-0.5B-Instruct"
)
model_name = os.getenv("RAG_MODEL_NAME", default_model_name)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading model (first run may take several minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Model loaded and moved to {device}.")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# ----------------------------
# 3. Build the RAG answer helper
# ----------------------------
def ask(question, k=4):
    query_embedding = embed_model.encode([question], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(query_embedding, k)
    context = "\n\n".join([chunks[i] for i in indices[0] if i >= 0])

    prompt = (
        "<|system|>\nYou answer questions about the email dataset.\n</s>\n"
        f"<|user|>\nContext:\n{context}\n\nQuestion: {question}\n</s>\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in answer:
        answer = answer.split("<|assistant|>")[-1].strip()
    return answer


# ----------------------------
# 4. Ask a typed question and speak the answer
# ----------------------------
question = input("Type a question about the email data: ").strip()

if question:
    answer = ask(question)
    print("\nAnswer:")
    print(answer)

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(answer)
    engine.runAndWait()
else:
    print("No question entered.")
