from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Few-shot prompt for math reasoning using a consistent question-answer format
prompt = """Q: If you have 3 apples and you buy 2 more, how many apples do you have?
A: 3 + 2 = 5. You have 5 apples.

Q: Sarah had 10 candies. She gave 4 to her friend. How many candies does she have left?
A: 10 - 4 = 6. She has 6 candies.

Q: Tom has 8 pencils. He buys 7 more. How many pencils does he have?
A:"""

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and display output
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== Model Response ===\n")
print(result)
