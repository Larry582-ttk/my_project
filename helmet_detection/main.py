import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils.load_yolo_dataset import YOLOdataset
from model import Model
from loss_fn import Loss_fn

if __name__ == '__main__':
    # ========== 创建输出目录 ==========
    os.makedirs('outputs', exist_ok=True)

    # ========== 超参数 ==========
    batch_size = 10
    epochs = 10
    learning_rate = 1e-2

    # ========== 设备 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: {}'.format(device))

    # ========== 数据加载 ==========
    train_set = YOLOdataset(
        r'dataset/HelmetDataset-YOLO-Train/images',
        r'dataset/HelmetDataset-YOLO-Train/labels',
        transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((224, 224))]),
        None
    )

    test_set = YOLOdataset(
        r'dataset/HelmetDataset-YOLO-Val/HelmetDataset-YOLO-Val/images',
        r'dataset/HelmetDataset-YOLO-Val/HelmetDataset-YOLO-Val/labels',
        transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((224, 224))]),
        None
    )

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ========== 模型、损失函数、优化器 ==========
    model = Model().to(device)
    loss_fn = Loss_fn()
    optimiter = torch.optim.SGD(model.parameters(), lr=learning_rate)

    total_train_steps = 0
    total_test_steps = 0

    # ========== 训练循环 ==========
    for epoch in range(epochs):
        print('\n========== Epoch {} =========='.format(epoch + 1))

        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0

        for batch_idx, (imgs, targets) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimiter.zero_grad()
            loss.backward()
            optimiter.step()

            total_train_loss += loss.item()
            total_train_steps += 1

            if total_train_steps % 200 == 0:
                print('[Train] Step: {}, Loss: {:.4f}'.format(total_train_steps, loss.item()))

        avg_train_loss = total_train_loss / len(train_dataloader)
        print('[Train] Epoch {} avg Loss: {:.4f}'.format(epoch + 1, avg_train_loss))

        # --- 验证阶段 ---
        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(test_dataloader):
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, targets)

                total_test_loss += loss.item()
                total_test_steps += 1

        avg_test_loss = total_test_loss / len(test_dataloader)
        print('[Val]   Epoch {} avg Loss: {:.4f}'.format(epoch + 1, avg_test_loss))

    # ========== 保存模型 ==========
    torch.save(model.state_dict(), 'outputs/model_weights.pth')
    print('\noutputs/model_weights.pth saved')

    # 导出 ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, 'outputs/model.onnx')
    print('outputs/model.onnx saved')
