# tinyllama_rag.py
# Applies Retrieval-Augmented Generation (RAG) using TinyLlama on emails.csv.
# Builds a FAISS vector index from email documents, then saves the index and
# text chunks so tinyllama_test.py can load them for evaluation.
#
# Dependencies:
#   pip install transformers torch faiss-cpu sentence-transformers pandas pickle5

import os
import pickle
import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. Load and chunk email data
# ============================================================
print("Loading emails.csv ...")
df = pd.read_csv("emails.csv")
print(f"  Loaded {len(df)} emails  (spam={df['spam'].sum()}, ham={(df['spam']==0).sum()})")

# Build plain-text chunks: prefix each email with its label for context
chunks = []
for _, row in df.iterrows():
    label_str = "SPAM" if row['spam'] == 1 else "HAM"
    # Truncate very long emails to keep chunk size reasonable
    body = str(row['text'])[:800]
    chunks.append(f"[{label_str}] {body}")

print(f"  Total chunks: {len(chunks)}")

# ============================================================
# 2. Embed chunks with SentenceTransformer
# ============================================================
print("\nLoading embedding model (all-MiniLM-L6-v2) ...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding chunks (this may take a minute) ...")
embeddings = embed_model.encode(
    chunks,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
print(f"  Embeddings shape: {embeddings.shape}")

# ============================================================
# 3. Build FAISS index
# ============================================================
print("\nBuilding FAISS index ...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)       # exact L2 nearest-neighbour
index.add(embeddings.astype(np.float32))
print(f"  FAISS index built  ({index.ntotal} vectors, dim={dim})")

# ============================================================
# 4. Save RAG artefacts for later use by tinyllama_test.py
# ============================================================
os.makedirs("rag_model", exist_ok=True)

faiss.write_index(index, "rag_model/email_faiss.index")
with open("rag_model/email_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("\nRAG artefacts saved to rag_model/")
print("  rag_model/email_faiss.index")
print("  rag_model/email_chunks.pkl")

# ============================================================
# 5. Load TinyLlama generation model
# ============================================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"\nLoading TinyLlama: {MODEL_NAME} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.to(device)
model.eval()
print(f"  Model loaded on {device}")

# ============================================================
# 6. RAG inference helper
# ============================================================
def retrieve(query: str, k: int = 5) -> list[str]:
    """Return top-k email chunks most relevant to the query."""
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]


def rag_answer(query: str, k: int = 3, max_new_tokens: int = 150) -> str:
    """Retrieve relevant context, then generate an answer with TinyLlama."""
    context_chunks = retrieve(query, k)
    context = "\n".join(c[:200] for c in context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Read the email excerpts and answer "
                "the question in 2-4 clear sentences. Do not copy text from the emails."
            ),
        },
        {
            "role": "user",
            "content": f"Email excerpts:\n{context}\n\nQuestion: {query}",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# 7. Demo query to verify the RAG pipeline works
# ============================================================
print("\n" + "=" * 60)
print("DEMO RAG QUERY")
print("=" * 60)
demo_q = "What are common topics found in spam emails?"
print(f"Question: {demo_q}")
answer = rag_answer(demo_q)
print(f"Answer:   {answer}")
print("\nRAG setup complete. Run tinyllama_test.py to evaluate the model.")
