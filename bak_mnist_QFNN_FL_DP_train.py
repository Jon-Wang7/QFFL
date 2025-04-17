import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from bak_min_QFNN_DP import QFNN
from pyvacy import optim as pyvacy_optim
from pyvacy.analysis import moments_accountant
import matplotlib.pyplot as plt
import time
import os
import random
from tqdm import tqdm

# 在文件开头，其他参数声明的地方添加
noise_multiplier = 0.1  # 默认值
l2_norm_clip = 1.0     # 默认值

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 将数据集分成10个客户端
num_clients = 10
client_datasets = []
for i in range(num_clients):
    indices = list(range(i * 6000, (i + 1) * 6000))
    client_datasets.append(Subset(train_dataset, indices))

# 差分隐私参数
batch_size = 256

# 97.08%
# noise_multiplier = 0.01 
# l2_norm_clip = 10.0

# 90.30%
# noise_multiplier = 0.05
# l2_norm_clip = 10.0

# 75.13%
# noise_multiplier = 0.1
# l2_norm_clip = 10.0

# 20.69%
# noise_multiplier = 0.2
# l2_norm_clip = 10.0

# 10.26%
# noise_multiplier = 0.5
# l2_norm_clip = 10.0

# 98.68%
# noise_multiplier = 0.01
# l2_norm_clip = 1

# 96.55%
# noise_multiplier = 0.05
# l2_norm_clip = 1

# 93.43%
# noise_multiplier = 0.1
# l2_norm_clip = 1

# l2_norm_clip = 10.0
lr = 0.001
epochs = 20

print("训练参数:")
print(f"epochs: {epochs}")
print(f"batch_size: {batch_size}")
print("差分隐私参数:")
print(f"noise_multiplier: {noise_multiplier}")
print(f"l2_norm_clip: {l2_norm_clip}")
print(f"learning_rate: {lr}")


# 训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return total_loss / len(train_loader), 100. * correct / total


# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


# 联邦学习训练
def federated_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化全局模型
    global_model = QFNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # 测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    start_time = time.time()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # 客户端训练
        client_models = []
        for client_id in range(num_clients):
            print(f"\nTraining client {client_id + 1}/{num_clients}")
            client_model = QFNN().to(device)
            client_model.load_state_dict(global_model.state_dict())

            # 使用PyVacy的DP优化器
            optimizer = pyvacy_optim.DPAdam(
                params=client_model.parameters(),
                batch_size=batch_size,
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                lr=lr
            )

            client_loader = DataLoader(
                client_datasets[client_id],
                batch_size=batch_size,
                shuffle=True
            )

            loss, acc = train(client_model, client_loader, optimizer, criterion, device)
            print(f"Client {client_id + 1} - Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
            client_models.append(client_model)

        # 模型聚合（FedAvg）
        with torch.no_grad():
            for param in global_model.parameters():
                param.data.zero_()

            for client_model in client_models:
                for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                    global_param.data += client_param.data / num_clients

        # 测试全局模型
        test_loss, test_acc = test(global_model, test_loader, criterion, device)
        print(f"\nGlobal Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        # 记录训练过程
        train_losses.append(loss)
        train_accuracies.append(acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds")

    # 保存模型
    model_path = os.path.join(dp_results_dir, f'noise_{noise_multiplier}_clip_{l2_norm_clip}', 'model.pth')
    torch.save(global_model.state_dict(), model_path)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    return test_acc


if __name__ == "__main__":
    # 创建主文件夹
    dp_results_dir = 'result/model/dp_experiments'
    os.makedirs(dp_results_dir, exist_ok=True)

    # 参数组合 0.01, 0.05,
    noise_multipliers = [0.1, 0.2, 0.5]
    l2_norm_clips = [1.0, 10.0]

    # 创建子文件夹并运行实验
    for noise in noise_multipliers:
        for clip in l2_norm_clips:
            print(f"\n=== 开始实验：noise={noise}, clip={clip} ===")
            
            # 更新全局参数（不需要再声明global）
            noise_multiplier = noise
            l2_norm_clip = clip
            
            # 创建实验文件夹
            folder_name = f'noise_{noise}_clip_{clip}'
            folder_path = os.path.join(dp_results_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            try:
                # 运行联邦训练
                test_acc = federated_train()
                
                # 保存实验信息
                info_path = os.path.join(folder_path, 'experiment_info.txt')
                with open(info_path, 'w') as f:
                    f.write(f'Noise Multiplier: {noise}\n')
                    f.write(f'L2 Norm Clip: {clip}\n')
                    f.write(f'Final Test Accuracy: {test_acc:.4f}\n')
                
                print(f"=== 实验完成：noise={noise}, clip={clip}, acc={test_acc:.4f} ===\n")
                
            except Exception as e:
                print(f"实验失败：noise={noise}, clip={clip}")
                print(f"错误信息：{str(e)}\n")
                continue