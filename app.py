import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template_string

# ============ 模型定义（必须和训练时一致）============
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("minigpt.pt", map_location=device)
stoi = ckpt["stoi"]
itos = ckpt["itos"]
cfg = ckpt["config"]
V = len(stoi)

block_size = cfg["block_size"]
n_embd = cfg["n_embd"]
n_head = cfg["n_head"]
n_layer = cfg["n_layer"]
dropout = 0.0    # 推理时关闭 dropout

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return self.dropout(wei) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(V, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = self.blocks(tok + pos)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = MiniGPT().to(device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"模型加载完成，device={device}，参数量={sum(p.numel() for p in model.parameters())}")

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join(itos[i] for i in l)

# ============ Flask ============
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>MiniGPT Shakespeare</title>
<style>
body { font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #1a1a1a; color: #eee; }
h1 { color: #d4af37; }
textarea { width: 100%; height: 100px; padding: 10px; font-size: 16px; font-family: Georgia, serif; background: #2a2a2a; color: #eee; border: 1px solid #444; }
button { padding: 10px 30px; font-size: 16px; background: #d4af37; border: none; cursor: pointer; margin-top: 10px; }
button:hover { background: #b8941f; }
#output { white-space: pre-wrap; background: #2a2a2a; padding: 20px; margin-top: 20px; border-left: 4px solid #d4af37; min-height: 100px; }
.row { display: flex; gap: 20px; align-items: center; margin-top: 10px; }
label { color: #aaa; }
input[type=range] { flex: 1; }
</style>
</head>
<body>
<h1>MiniGPT Shakespeare</h1>
<p style="color:#aaa;">输入一个开头，比如 <code>ROMEO:</code> 或 <code>JULIET:</code>，让模型续写。</p>
<textarea id="prompt">ROMEO:</textarea>
<div class="row">
  <label>temperature</label>
  <input type="range" id="temp" min="0.3" max="1.5" step="0.1" value="0.8">
  <span id="tempv">0.8</span>
</div>
<div class="row">
  <label>top-k</label>
  <input type="range" id="topk" min="0" max="65" step="1" value="20">
  <span id="topkv">20</span>
</div>
<div class="row">
  <label>生成长度</label>
  <input type="range" id="len" min="100" max="1000" step="50" value="400">
  <span id="lenv">400</span>
</div>
<button onclick="generate()">生成</button>
<div id="output"></div>

<script>
document.querySelectorAll('input[type=range]').forEach(el => {
  el.addEventListener('input', () => {
    document.getElementById(el.id + 'v').textContent = el.value;
  });
});

async function generate() {
  const out = document.getElementById('output');
  out.textContent = '生成中...';
  const res = await fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      prompt: document.getElementById('prompt').value,
      temperature: parseFloat(document.getElementById('temp').value),
      top_k: parseInt(document.getElementById('topk').value) || null,
      max_tokens: parseInt(document.getElementById('len').value),
    })
  });
  const data = await res.json();
  out.textContent = data.text;
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.8))
    top_k = data.get("top_k", 20)
    max_tokens = int(data.get("max_tokens", 400))

    if not prompt:
        return jsonify({"text": "请输入开头"})

    idx = encode(prompt)
    if not idx:
        return jsonify({"text": "输入包含未知字符，请用英文和常见标点"})

    x = torch.tensor([idx], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    text = decode(out[0].tolist())
    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)