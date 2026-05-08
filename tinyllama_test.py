# tinyllama_test.py
# Loads the RAG artefacts saved by tinyllama_rag.py, then evaluates the
# RAG-tuned TinyLlama model by asking three email-dataset-specific questions.
#
# Run tinyllama_rag.py first to generate rag_model/email_faiss.index
# and rag_model/email_chunks.pkl.

import pickle
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. Load saved RAG artefacts
# ============================================================
print("Loading RAG artefacts ...")
index = faiss.read_index("rag_model/email_faiss.index")
with open("rag_model/email_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"  FAISS index loaded  ({index.ntotal} vectors)")
print(f"  Email chunks loaded ({len(chunks)} chunks)")

# ============================================================
# 2. Load embedding model (must match the one used in rag.py)
# ============================================================
print("\nLoading embedding model ...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# 3. Load TinyLlama
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
# 4. RAG inference helpers
# ============================================================
def retrieve(query: str, k: int = 3) -> list[str]:
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]


def retrieve_mixed(k_spam: int = 2, k_ham: int = 2) -> list[str]:
    """Return k_spam SPAM chunks and k_ham HAM chunks from the index."""
    spam, ham = [], []
    # Scan a broad neighbourhood to find both classes
    q_emb = embed_model.encode(["email message"], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(q_emb, 200)
    for i in indices[0]:
        if chunks[i].startswith("[SPAM]") and len(spam) < k_spam:
            spam.append(chunks[i])
        elif chunks[i].startswith("[HAM]") and len(ham) < k_ham:
            ham.append(chunks[i])
        if len(spam) >= k_spam and len(ham) >= k_ham:
            break
    return spam + ham


def rag_answer(query: str, context_chunks: list[str], max_new_tokens: int = 150) -> str:
    # Truncate each chunk to 200 chars so the context stays short
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

    # apply_chat_template handles the </s> tokens correctly for TinyLlama
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
# 5. Three course-specific questions about the email dataset
# ============================================================
questions = [
    "What kinds of products or services are commonly advertised in spam emails?",
    "How do spam emails typically try to persuade the reader to take action?",
    "What language or phrases appear frequently in ham (non-spam) emails compared to spam emails?",
]

print("\n" + "=" * 60)
print("RAG-TUNED TINYLLAMA — EVALUATION")
print("=" * 60)

for i, question in enumerate(questions, 1):
    print(f"\n--- Question {i} ---")
    print(f"Q: {question}")

    # Q3 needs both HAM and SPAM chunks for a fair comparison
    if i == 3:
        retrieved = retrieve_mixed(k_spam=2, k_ham=2)
    else:
        retrieved = retrieve(question, k=3)

    print("\nTop retrieved email excerpts:")
    for j, chunk in enumerate(retrieved, 1):
        print(f"  [{j}] {chunk[:150]}...")

    print("\nGenerating answer ...")
    answer = rag_answer(question, retrieved)
    print(f"\nA: {answer}")
    print("-" * 60)
