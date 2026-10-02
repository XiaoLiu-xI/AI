import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# 1. 数据：一小批英文名字
words = """emma olivia ava isabella sophia charlotte mia amelia harper evelyn
abigail emily elizabeth mila ella avery sofia camila aria scarlett
victoria madison luna grace chloe penelope layla riley zoey nora
lily eleanor hannah lillian addison aubrey ellie stella natalie zoe
leah hazel violet aurora savannah audrey brooklyn bella claire skylar""".split()

# 2. 构建字符表
chars = sorted(set("".join(words)))
stoi = {c: i for i, c in enumerate(chars)}
stoi["."] = len(chars)     # "." 表示名字的开始和结束
itos = {i: c for c, i in stoi.items()}
V = len(stoi)
print("vocab size:", V)

# 3. 构建 bigram 训练数据
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        xs.append(stoi[a])
        ys.append(stoi[b])

xs = torch.tensor(xs, device=device)
ys = torch.tensor(ys, device=device)
print("训练样本数:", len(xs))

# 4. 模型：一个查找表 (V, V)
# 每一行代表"当前字符"的 logits，下一字符的概率由 softmax 得到
W = torch.randn((V, V), device=device, requires_grad=True)

# 5. 训练
optimizer = torch.optim.Adam([W], lr=0.1)
for i in range(200):
    logits = W[xs]                       # (N, V)
    loss = F.cross_entropy(logits, ys)   # 交叉熵
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 20 == 0:
        print(f"step {i:3d}  loss {loss.item():.4f}")

# 6. 采样：从 "." 开始，逐字符生成，直到再次遇到 "."
print("\n生成的名字：")
for _ in range(15):
    out = []
    idx = stoi["."]
    while True:
        logits = W[idx]
        probs = F.softmax(logits, dim=0)
        idx = torch.multinomial(probs, num_samples=1).item()
        if idx == stoi["."]:
            break
        out.append(itos[idx])
    print("".join(out))