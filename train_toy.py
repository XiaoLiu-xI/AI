import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

torch.manual_seed(0)

# 玩具数据：y = 3x + 2 + 噪声
X = torch.randn(1000, 1)
y = 3 * X + 2 + 0.1 * torch.randn(1000, 1)

X, y = X.to(device), y.to(device)

model = nn.Linear(1, 1).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

print("训练后 weight:", model.weight.item())
print("训练后 bias:", model.bias.item())