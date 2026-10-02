import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)

# ---------- 1. 读数据 ----------
words = open("names.txt", "r").read().splitlines()
random.shuffle(words)

# ---------- 2. 字符表 ----------
chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
V = len(stoi)
print("vocab size:", V)

# ---------- 3. 构建数据集：用前 3 个字符预测下一个 ----------
block_size = 3   # 上下文长度

def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X, device=device), torch.tensor(Y, device=device)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
print("训练样本:", len(Xtr))

# ---------- 4. 模型 ----------
n_embd = 10     # 每个字符的嵌入维度
n_hidden = 200  # 隐藏层大小

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Embedding(V, n_embd)
        self.fc1 = nn.Linear(block_size * n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, V)

    def forward(self, x):
        emb = self.C(x)                       # (B, block_size, n_embd)
        emb = emb.view(emb.shape[0], -1)      # (B, block_size * n_embd)
        h = torch.tanh(self.fc1(emb))
        logits = self.fc2(h)
        return logits

model = MLP().to(device)
print("参数量:", sum(p.numel() for p in model.parameters()))

# ---------- 5. 训练 ----------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
batch_size = 256

for step in range(5000):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

# ---------- 6. 采样 ----------
print("\n生成的名字：")
with torch.no_grad():
    for _ in range(15):
        context = [0] * block_size
        out = []
        while True:
            x = torch.tensor([context], device=device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            if ix == 0:
                break
            out.append(itos[ix])
            context = context[1:] + [ix]
        print("".join(out))