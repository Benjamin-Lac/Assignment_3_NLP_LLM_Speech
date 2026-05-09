import os
import pickle
from typing import Any

import numpy as np
import torch
import faiss  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# 1. Load the true receipt text
# ----------------------------
true_receipt_path = "true_receipt_contents.txt"
extracted_receipt_path = "extracted_receipt_contents.txt"

with open(true_receipt_path, "r", encoding="utf-8") as file:
    true_receipt_text = file.read()


# Keep only the useful receipt lines and skip comments / blank lines.
def build_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        chunks.append(line)
    return chunks


chunks: list[str] = build_chunks(true_receipt_text)
print(f"Loaded {len(chunks)} receipt chunks from {true_receipt_path}")


# ----------------------------
# 2. Embed the receipt chunks
# ----------------------------
embed_model: Any = SentenceTransformer("all-MiniLM-L6-v2")
embeddings: NDArray[np.float32] = embed_model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=True,
)


# ----------------------------
# 3. Build and save the FAISS index
# ----------------------------
dim = embeddings.shape[1]
index: Any = faiss.IndexFlatL2(dim)
index.add(embeddings.astype(np.float32))

os.makedirs("rag_data", exist_ok=True)
getattr(faiss, "write_index")(index, "rag_data/receipt_index.faiss")
with open("rag_data/receipt_chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)

print(f"FAISS index saved with {index.ntotal} receipt vectors")


# ----------------------------
# 4. Load TinyLlama
# ----------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = getattr(AutoTokenizer, "from_pretrained")(model_name)
load_model: Any = getattr(AutoModelForCausalLM, "from_pretrained")
model = load_model(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# ----------------------------
# 5. Retrieval helper
# ----------------------------
def retrieve(query: str, k: int = 4) -> list[str]:
    query_embedding: NDArray[np.float32] = embed_model.encode(
        [query],
        convert_to_numpy=True,
    ).astype(np.float32)
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]


# ----------------------------
# 6. Ask the model about the OCR text
# ----------------------------
with open(extracted_receipt_path, "r", encoding="utf-8") as file:
    extracted_receipt_text = file.read()

question = (
    "Use the OCR text below to identify all purchased items from the receipts. "
    "Ignore store names, totals, taxes, and change lines. Return only the item names.\n\n"
    f"OCR text:\n{extracted_receipt_text}"
)

context = "\n\n".join(retrieve(question))
prompt = (
    "<|system|>\nYou are a helpful assistant that extracts receipt items.\n</s>\n"
    f"<|user|>\nContext:\n{context}\n\nQuestion:\n{question}\n</s>\n"
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

print("\n=== OCR RAG Answer ===\n")
print(answer)
