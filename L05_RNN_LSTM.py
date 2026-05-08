import torch
import torch.nn as nn
import numpy as np

# ----------------------------
# 1. Load and encode text
# ----------------------------
with open("lyrics.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

chars = sorted(list(set(text)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
vocab_size = len(chars)

data = [char_to_idx[ch] for ch in text]

seq_length = 40
step = 3
X = []
Y = []

for i in range(0, len(data) - seq_length, step):
    X.append(data[i:i+seq_length])
    Y.append(data[i+seq_length])

X = torch.tensor(X)
Y = torch.tensor(Y)

# ----------------------------
# 2. LSTM Model
# ----------------------------
class LyricsLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, 128),
                torch.zeros(1, batch_size, 128))

model = LyricsLSTM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# ----------------------------
# 3. Training Loop
# ----------------------------
epochs = 40
batch_size = 256

for epoch in range(epochs):
    hidden = model.init_hidden(X.size(0))
    optimizer.zero_grad()

    output, hidden = model(X, hidden)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ----------------------------
# 4. Generate Lyrics
# ----------------------------
def generate(model, seed="love ", length=300):
    model.eval()
    seed = seed.lower()
    chars = [char_to_idx[ch] for ch in seed if ch in char_to_idx]
    hidden = model.init_hidden(1)
    
    for _ in range(length):
        x = torch.tensor(chars[-seq_length:]).unsqueeze(0)
        output, hidden = model(x, hidden)
        
        prob = torch.softmax(output, dim=1).detach().numpy().ravel()
        next_idx = np.random.choice(range(vocab_size), p=prob)
        chars.append(next_idx)
    
    return ''.join(idx_to_char[i] for i in chars)

print("\nGenerated Lyrics:")
print(generate(model, "love "))
