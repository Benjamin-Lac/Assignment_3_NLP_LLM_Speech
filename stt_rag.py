import os
import pickle
import importlib
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from sentence_transformers import SentenceTransformer


class FaissIndex(Protocol):
    ntotal: int

    def search(
        self, x: NDArray[np.float32], k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...


class FaissModule(Protocol):
    def read_index(self, path: str) -> FaissIndex: ...


class RecognizerLike(Protocol):
    def adjust_for_ambient_noise(self, source: Any, duration: float = 1) -> None: ...

    def listen(self, source: Any) -> Any: ...

    def recognize_google(self, audio_data: Any) -> str: ...


class MicrophoneLike(Protocol):
    def __enter__(self) -> Any: ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool | None: ...


class SpeechRecognitionModule(Protocol):
    Recognizer: type[RecognizerLike]
    Microphone: type[MicrophoneLike]
    UnknownValueError: type[Exception]
    RequestError: type[Exception]


class BatchEncodingLike(Protocol):
    def to(self, device: str) -> dict[str, Tensor]: ...


class TokenizerLike(Protocol):
    pad_token_id: int | None
    eos_token_id: int
    eos_token: str
    pad_token: str | None

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> BatchEncodingLike: ...

    def decode(self, token_ids: Tensor, *, skip_special_tokens: bool) -> str: ...


class CausalLMModelLike(Protocol):
    def to(self, device: str) -> Any: ...

    def eval(self) -> Any: ...

    def generate(self, **kwargs: Any) -> Tensor: ...


class EmbedderLike(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
    ) -> NDArray[np.float32]: ...


# ----------------------------
# 1. Load the saved email RAG data
# ----------------------------
INDEX_PATH = "rag_data/email_index.faiss"
CHUNKS_PATH = "rag_data/email_chunks.pkl"

faiss = cast(FaissModule, importlib.import_module("faiss"))
sr = cast(SpeechRecognitionModule, importlib.import_module("speech_recognition"))
transformers_mod: Any = importlib.import_module("transformers")

if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
    raise FileNotFoundError(
        "Missing RAG files. Build them first with tiny_llama_rag.py "
        "(expected rag_data/email_index.faiss and rag_data/email_chunks.pkl)."
    )

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as file:
    chunks = cast(list[str], pickle.load(file))

embed_model = cast(EmbedderLike, SentenceTransformer("all-MiniLM-L6-v2"))
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
tokenizer_factory: Any = transformers_mod.AutoTokenizer
model_factory: Any = transformers_mod.AutoModelForCausalLM

print("Loading TinyLlama tokenizer...")
tokenizer = cast(TokenizerLike, tokenizer_factory.from_pretrained(model_name))
print("Loading TinyLlama model (first run may take several minutes)...")
model = cast(
    CausalLMModelLike,
    model_factory.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    ),
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"Model loaded and moved to {device}.")

# Avoid generate() errors when a pad token is not configured.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# ----------------------------
# 3. Build the RAG prompt
# ----------------------------
def ask(question: str, k: int = 4) -> str:
    query_embedding = embed_model.encode([question], convert_to_numpy=True).astype(np.float32)
    _, indices = index.search(query_embedding, k)
    context = "\n\n".join([chunks[int(i)] for i in indices[0] if int(i) >= 0])

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
# 4. Record speech and query the model
# ----------------------------
recognizer = sr.Recognizer()

question: str | None = None

try:
    with sr.Microphone() as source:
        print("Speak your question about the email data...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio_data = recognizer.listen(source)

    question = recognizer.recognize_google(audio_data)
    print("You said:", question)
except OSError:
    print("Microphone backend not available. Install with: pip install pyaudio")
except AttributeError:
    print(
        "SpeechRecognition backend issue. Reinstall with: "
        "pip install --upgrade SpeechRecognition"
    )
except sr.UnknownValueError:
    print("Sorry, I could not understand the audio.")
except sr.RequestError as error:
    print(f"Could not request speech recognition results: {error}")

if not question:
    question = input("Type your question instead: ").strip()

if question:
    print("\nAnswer:")
    print(ask(question))
else:
    print("No question provided.")
