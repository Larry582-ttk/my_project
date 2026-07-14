import os
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error as mae, r2_score
import matplotlib.pyplot as plt

from model import LSTMModel
from loss_fn import Loss_fn
from utils.data_utils import load_and_prepare_data

# ========== 创建输出目录 ==========
os.makedirs('outputs', exist_ok=True)

# ========== 超参数 ==========
input_size = 1
hidden_size = 64
num_layers = 2
num_epochs = 100
learning_rate = 0.001
npz_path = r'dataset/pems04.npz'

# ========== 设备 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))

# ========== 数据加载 ==========
train_loader, x_test_t, y_test_t, scaler_y = load_and_prepare_data(npz_path)

# ========== 模型、损失函数、优化器 ==========
model = LSTMModel(input_size, hidden_size, num_layers).to(device)
criterion = Loss_fn()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ========== 训练循环 ==========
train_log = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    batch_count = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count
    train_log.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print("Epoch [{}/{}], Loss: {:.6f}".format(epoch + 1, num_epochs, avg_loss))

# ========== 评估 ==========
model.eval()
with torch.no_grad():
    y_pred_t = model(x_test_t.to(device))

y_pred = scaler_y.inverse_transform(y_pred_t.cpu().numpy())
y_true = scaler_y.inverse_transform(y_test_t.numpy())

r2 = r2_score(y_true, y_pred)
mae_score = mae(y_true, y_pred)
print("\nR² score: {:.4f}".format(r2))
print("MAE: {:.4f}".format(mae_score))

# ========== 画图 ==========
plt.figure(figsize=(12, 5))

# 子图1：前200个测试样本对比
plt.subplot(1, 2, 1)
plt.plot(y_true[:200], color='blue', alpha=0.5, label='True value')
plt.plot(y_pred[:200], color='red', alpha=0.5, label='Predicted value')
plt.xlabel('Sample index')
plt.ylabel('Traffic flow')
plt.title('Comparison of first 200 test samples')
plt.legend()
plt.grid(True)

# 子图2：散点图
plt.subplot(1, 2, 2)
plt.scatter(y_true, y_pred, alpha=0.3, s=1)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=1)
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.title('R² = {:.4f}'.format(r2))
plt.grid(True)

plt.tight_layout()
plt.savefig(r'outputs/prediction_results.png', dpi=300, bbox_inches='tight')
print(" save to outputs/prediction_results.png")
plt.show()
