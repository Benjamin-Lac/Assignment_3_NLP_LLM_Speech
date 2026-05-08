import pickle
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# 1. Load saved RAG index
# ----------------------------
index = faiss.read_index("rag_data/email_index.faiss")
with open("rag_data/email_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"RAG index loaded: {index.ntotal} email vectors")

# ----------------------------
# 2. Load TinyLlama model
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
# 3. RAG query function
# ----------------------------
def ask(question, k=4):
    # Retrieve relevant email chunks
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype(np.float32)
    _, I  = index.search(q_emb, k)
    context = "\n\n".join([chunks[i] for i in I[0]])

    prompt = (
        "<|system|>\nYou are a helpful assistant that answers questions "
        "about a spam email dataset. Use the provided examples.\n</s>\n"
        f"<|user|>\nEmail examples:\n{context}\n\nQuestion: {question}\n</s>\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in result:
        result = result.split("<|assistant|>")[-1].strip()
    return result

# ----------------------------
# 4. Three course-specific questions about emails.csv
# ----------------------------
questions = [
    "What words or phrases are most commonly used in spam emails to persuade recipients to click a link?",
    "How do legitimate (ham) emails differ from spam emails in terms of language and content?",
    "What are the top three characteristics that reliably identify a spam email?",
]

for i, q in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"Question {i}: {q}")
    print(f"{'='*60}")
    print("Answer:", ask(q))
