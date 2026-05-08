import os
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----------------------------
# 1. Extract text from PDF
# ----------------------------
text = fitz.open("./data/CPSC_254_SYL.pdf")[0].get_text()

# Split text into chunks (for retrieval)
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = chunk_text(text)
print(f"Number of chunks: {len(chunks)}")

# ----------------------------
# 2. Embed chunks using SentenceTransformer
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks, convert_to_numpy=True)

# ----------------------------
# 3. Build FAISS index
# ----------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# ----------------------------
# 4. Prepare retriever
# ----------------------------
class SimpleFAISSRetriever(RagRetriever):
    def __init__(self, index, chunks, tokenizer):
        self.index = index
        self.chunks = chunks
        self.tokenizer = tokenizer
    
    def __call__(self, question, **kwargs):
        q_emb = embed_model.encode([question], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k=5)
        return {"input_ids": [self.tokenizer(chunk, return_tensors="pt").input_ids.squeeze() for chunk in np.array(self.chunks)[I[0]]]}

# ----------------------------
# 5. Load RAG model
# ----------------------------
token = os.environ.get("HF_TOKEN")
model_name = "facebook/rag-sequence-nq"
tokenizer = RagTokenizer.from_pretrained(model_name, use_auth_token=token)
rag_model = RagSequenceForGeneration.from_pretrained(model_name, use_auth_token=token)

# Create retriever
retriever = SimpleFAISSRetriever(index, chunks, tokenizer)
rag_model.set_retriever(retriever)

# ----------------------------
# 6. Query Example
# ----------------------------
question = "What topics are covered in CPSC 254?"
inputs = tokenizer(question, return_tensors="pt")
generated = rag_model.generate(**inputs, max_length=200)
answer = tokenizer.decode(generated[0], skip_special_tokens=True)
print("Answer:", answer)
