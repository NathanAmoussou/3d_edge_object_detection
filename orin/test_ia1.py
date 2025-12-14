cat << 'EOF' > test_ia.py
import torch
import torch.nn as nn
import torch.optim as optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("-" * 30)
if device.type == 'cuda':
    print(f"✅ SUCCÈS : GPU détecté -> {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ ATTENTION : GPU non détecté. Le code tourne sur le CPU.")
print("-" * 30)

class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = TinyModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]], device=device)

print("🧠 Début de l'entraînement...")
start_time = time.time()

for epoch in range(100):
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

end_time = time.time()
print("-" * 30)
print(f"⏱️ Entraînement terminé en {end_time - start_time:.4f} secondes.")

test_val = torch.tensor([[10.0]], device=device)
prediction = model(test_val).item()
print(f"❓ Test : Si je donne 10 à l'IA, elle devine : {prediction:.2f}")
print("-" * 30)
EOF