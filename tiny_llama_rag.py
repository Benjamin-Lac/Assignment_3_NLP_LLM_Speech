import os
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss

# ----------------------------
# 1. Load emails from CSV
# ----------------------------
df = pd.read_csv('emails.csv')
print(f"Loaded {len(df)} emails ({df['spam'].sum()} spam, {(df['spam']==0).sum()} ham)")

# Each email becomes one chunk; prepend label for richer context
def build_chunks(df, max_chars=500):
    chunks = []
    for _, row in df.iterrows():
        label = "SPAM" if row['spam'] == 1 else "HAM"
        chunks.append(f"[{label}] {str(row['text'])[:max_chars]}")
    return chunks

chunks = build_chunks(df)
print(f"Number of chunks: {len(chunks)}")

# ----------------------------
# 2. Embed chunks using SentenceTransformer
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# ----------------------------
# 3. Build FAISS index
# ----------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype(np.float32))
print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

# ----------------------------
# 4. Retrieval function
# ----------------------------
def retrieve(query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

# ----------------------------
# 5. Load TinyLlama
# ----------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ----------------------------
# 6. RAG query example
# ----------------------------
question = "What words or phrases are commonly found in spam emails?"
context  = "\n\n".join(retrieve(question))

prompt = (
    "<|system|>\nYou are a helpful assistant that answers questions "
    "about email data.\n</s>\n"
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
        pad_token_id=tokenizer.eos_token_id
    )

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
if "<|assistant|>" in answer:
    answer = answer.split("<|assistant|>")[-1].strip()
print("Answer:", answer)

# ----------------------------
# 7. Save RAG index and chunks for tiny_llama_test.py
# ----------------------------
os.makedirs("rag_data", exist_ok=True)
faiss.write_index(index, "rag_data/email_index.faiss")
with open("rag_data/email_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("RAG data saved to rag_data/")
