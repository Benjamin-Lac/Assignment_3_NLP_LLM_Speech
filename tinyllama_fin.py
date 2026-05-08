import os
import fitz  # PyMuPDF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Extract text from PDF
text = fitz.open("./data/CPSC_254_SYL.pdf")[0].get_text()
# Wrap into Dataset
dataset = Dataset.from_dict({"text": [text]})
# Hugging Face token
token = os.environ.get("HF_TOKEN")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=token)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=token)

# Tokenize
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = dataset.map(tokenize_function)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=2,
)

# Train
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
trainer.train()

# Save
model.save_pretrained("./fine_tuned_tinyllama")
tokenizer.save_pretrained("./fine_tuned_tinyllama")
