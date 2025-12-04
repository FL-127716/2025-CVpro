import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np


# ======================== 1. 定义ResNet18模型 ========================
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y


class ResNet18(nn.Module):
    def __init__(self, Residual):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(Residual(64, 64, use_1conv=False, strides=1),
                                Residual(64, 64, use_1conv=False, strides=1))

        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2),
                                Residual(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2),
                                Residual(512, 512, use_1conv=False, strides=1))

        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, 10))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


# ======================== 2. 定义GoogLeNet模型 ========================
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()
        # 路线1，单1×1卷积层
        self.p1_1 = nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
        # 路线2，1×1卷积层, 3×3的卷积
        self.p2_1 = nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
        # 路线3，1×1卷积层, 5×5的卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        # 路线4，3×3的最大池化, 1×1的卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)

    def forward(self, x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, Inception):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 10))

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x


# ======================== 3. 通用训练和测试函数 ========================
def train_model(model, train_loader, criterion, optimizer, device, epoch, model_name):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            print(f'{model_name} - Epoch {epoch}, Batch {batch_idx}: Loss: {loss.item():.4f}, '
                  f'Train Acc: {100. * correct / total:.2f}%')

    train_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return avg_loss, train_acc, train_time


def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return avg_loss, test_acc


# ======================== 4. 性能对比主函数 ========================
def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    # 训练参数
    epochs = 3  # 可根据需要增加，建议3-5轮
    results = {
        'ResNet18': {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'train_time': []},
        'GoogLeNet': {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'train_time': []}
    }

    # ======================== 训练ResNet18 ========================
    print("\n" + "=" * 50)
    print("Training ResNet18...")
    print("=" * 50)
    resnet_model = ResNet18(Residual).to(device)
    resnet_criterion = nn.CrossEntropyLoss()
    resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)

    # 打印模型信息
    print("\nResNet18 Model Summary:")
    summary(resnet_model, (1, 224, 224))

    for epoch in range(1, epochs + 1):
        print(f"\nResNet18 Epoch {epoch}/{epochs}")
        train_loss, train_acc, train_time = train_model(
            resnet_model, train_loader, resnet_criterion, resnet_optimizer, device, epoch, "ResNet18"
        )
        test_loss, test_acc = test_model(resnet_model, test_loader, resnet_criterion, device)

        results['ResNet18']['train_loss'].append(train_loss)
        results['ResNet18']['test_loss'].append(test_loss)
        results['ResNet18']['train_acc'].append(train_acc)
        results['ResNet18']['test_acc'].append(test_acc)
        results['ResNet18']['train_time'].append(train_time)

        print(f'ResNet18 - Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Training time for this epoch: {train_time:.2f} seconds')

    # ======================== 训练GoogLeNet ========================
    print("\n" + "=" * 50)
    print("Training GoogLeNet...")
    print("=" * 50)
    googlenet_model = GoogLeNet(Inception).to(device)
    googlenet_criterion = nn.CrossEntropyLoss()
    googlenet_optimizer = optim.Adam(googlenet_model.parameters(), lr=0.0001)

    # 打印模型信息
    print("\nGoogLeNet Model Summary:")
    summary(googlenet_model, (1, 224, 224))

    for epoch in range(1, epochs + 1):
        print(f"\nGoogLeNet Epoch {epoch}/{epochs}")
        train_loss, train_acc, train_time = train_model(
            googlenet_model, train_loader, googlenet_criterion, googlenet_optimizer, device, epoch, "GoogLeNet"
        )
        test_loss, test_acc = test_model(googlenet_model, test_loader, googlenet_criterion, device)

        results['GoogLeNet']['train_loss'].append(train_loss)
        results['GoogLeNet']['test_loss'].append(test_loss)
        results['GoogLeNet']['train_acc'].append(train_acc)
        results['GoogLeNet']['test_acc'].append(test_acc)
        results['GoogLeNet']['train_time'].append(train_time)

        print(f'GoogLeNet - Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'Training time for this epoch: {train_time:.2f} seconds')

    # ======================== 性能对比可视化 ========================
    print("\n" + "=" * 50)
    print("Generating performance comparison plots...")
    print("=" * 50)

    # 设置图表样式
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (15, 10)

    # 创建子图
    fig, axes = plt.subplots(2, 2)

    # 1. 训练损失对比
    axes[0, 0].plot(range(1, epochs + 1), results['ResNet18']['train_loss'],
                    'b-o', label='ResNet18', linewidth=2, markersize=6)
    axes[0, 0].plot(range(1, epochs + 1), results['GoogLeNet']['train_loss'],
                    'r-s', label='GoogLeNet', linewidth=2, markersize=6)
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 测试损失对比
    axes[0, 1].plot(range(1, epochs + 1), results['ResNet18']['test_loss'],
                    'b-o', label='ResNet18', linewidth=2, markersize=6)
    axes[0, 1].plot(range(1, epochs + 1), results['GoogLeNet']['test_loss'],
                    'r-s', label='GoogLeNet', linewidth=2, markersize=6)
    axes[0, 1].set_title('Test Loss Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 训练准确率对比
    axes[1, 0].plot(range(1, epochs + 1), results['ResNet18']['train_acc'],
                    'b-o', label='ResNet18', linewidth=2, markersize=6)
    axes[1, 0].plot(range(1, epochs + 1), results['GoogLeNet']['train_acc'],
                    'r-s', label='GoogLeNet', linewidth=2, markersize=6)
    axes[1, 0].set_title('Training Accuracy Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 测试准确率对比
    axes[1, 1].plot(range(1, epochs + 1), results['ResNet18']['test_acc'],
                    'b-o', label='ResNet18', linewidth=2, markersize=6)
    axes[1, 1].plot(range(1, epochs + 1), results['GoogLeNet']['test_acc'],
                    'r-s', label='GoogLeNet', linewidth=2, markersize=6)
    axes[1, 1].set_title('Test Accuracy Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ======================== 性能总结表格 ========================
    print("\n" + "=" * 80)
    print("Model Performance Summary")
    print("=" * 80)

    # 计算平均训练时间
    resnet_avg_time = np.mean(results['ResNet18']['train_time'])
    googlenet_avg_time = np.mean(results['GoogLeNet']['train_time'])

    # 最终测试准确率
    resnet_final_acc = results['ResNet18']['test_acc'][-1]
    googlenet_final_acc = results['GoogLeNet']['test_acc'][-1]

    # 最终测试损失
    resnet_final_loss = results['ResNet18']['test_loss'][-1]
    googlenet_final_loss = results['GoogLeNet']['test_loss'][-1]

    print(f"{'Metric':<20} {'ResNet18':<15} {'GoogLeNet':<15} {'Better Model':<15}")
    print("-" * 80)
    print(f"{'Final Test Accuracy (%)':<20} {resnet_final_acc:<15.2f} {googlenet_final_acc:<15.2f} "
          f"{'ResNet18' if resnet_final_acc > googlenet_final_acc else 'GoogLeNet'}")
    print(f"{'Final Test Loss':<20} {resnet_final_loss:<15.4f} {googlenet_final_loss:<15.4f} "
          f"{'ResNet18' if resnet_final_loss < googlenet_final_loss else 'GoogLeNet'}")
    print(f"{'Avg Training Time (s/epoch)':<20} {resnet_avg_time:<15.2f} {googlenet_avg_time:<15.2f} "
          f"{'ResNet18' if resnet_avg_time < googlenet_avg_time else 'GoogLeNet'}")

    # 保存结果
    torch.save({
        'ResNet18': {
            'model_state_dict': resnet_model.state_dict(),
            'final_test_acc': resnet_final_acc,
            'avg_train_time': resnet_avg_time
        },
        'GoogLeNet': {
            'model_state_dict': googlenet_model.state_dict(),
            'final_test_acc': googlenet_final_acc,
            'avg_train_time': googlenet_avg_time
        },
        'results': results
    }, 'model_comparison_results.pth')

    print("\nAll results saved to 'model_comparison_results.pth'")
    print("Performance plot saved to 'model_performance_comparison.png'")


if __name__ == "__main__":
    main()