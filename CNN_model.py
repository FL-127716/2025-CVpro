import torch
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score
import torch.nn as nn
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        # 1x1 -> 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),  # 5x5卷积，padding=2保持尺寸
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        # 3x3池化 -> 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        )

        self.inception1 = Inception(32, 16, 16, 32, 8, 16, 8)  # 输出: 16+32+16+8=72
        self.inception2 = Inception(72, 32, 32, 48, 12, 24, 12)  # 输出: 32+48+24+12=116

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(116, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 输入通道改为1（灰度）
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)

        )
        self.layer1 = self._make_layer(32, 32, 2, stride=1)  # 28x28 -> 28x28
        self.layer2 = self._make_layer(32, 64, 2, stride=2)  # 28x28 -> 14x14
        self.layer3 = self._make_layer(64, 128, 2, stride=2)  # 14x14 -> 7x7
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model3=GoogLeNet(num_classes=10)
model4=ResNet()

models=[model3,model4]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 加载数据集
train_dataset = datasets.MNIST(root='D:\deeplearn\mnist', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='D:\deeplearn\mnist', train=False, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

names=['GoogLeNet','ResNet']
for model,name in zip(models,names):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses=[]
    train_accuracies=[]
    test_accuracies=[]
    test_recalls = []
    test_f1_scores = []
    ecpohs=10

    model.train()
    print(f'开始{name}训练')
    for epoch in range(ecpohs):
        total_loss = 0
        correct = 0
        total = 0

        for image,lables in train_loader:
            image, lables = image.to(device), lables.to(device)
            output=model(image)
            loss=criterion(output,lables)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += lables.size(0)
            correct += (predicted == lables).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []  # 收集所有预测结果
        all_labels = []  # 收集所有真实标签

        with torch.no_grad():
            for image, labels in test_loader:
                image, labels = image.to(device), labels.to(device)
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                # 收集预测结果和真实标签用于计算召回率和F1分数
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)

        # 计算召回率 (Recall)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        test_recalls.append(recall)

        # 计算F1分数 (F1-Score)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0) * 100
        test_f1_scores.append(f1)

        model.train()  # 切换回训练模式

        print(
            f'轮次 = {epoch + 1}/{ecpohs} ({(epoch + 1) / ecpohs * 100:.1f}%), '
            f'损失 = {train_loss:.4f}, 训练准确率 = {train_accuracy:.2f}%, '
            f'测试准确率 = {test_accuracy:.2f}%, '
            f'召回率 = {recall:.2f}%, F1分数 = {f1:.2f}%'
        )

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    epochs_range = range(1, ecpohs + 1)
    plt.plot(epochs_range, train_accuracies, 'b-', label='训练准确率', marker='o', linewidth=2)
    plt.plot(epochs_range, test_accuracies, 'r-', label='测试准确率', marker='s', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.title(f'{name}训练和测试准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    # 损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_losses, 'g-', label='训练损失', marker='o', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title(f'{name}训练损失')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, test_accuracies, 'b-', label='测试准确率', marker='o', linewidth=2)
    plt.plot(epochs_range, test_recalls, 'orange', label='测试召回率', marker='s', linewidth=2)
    plt.plot(epochs_range, test_f1_scores, 'g', label='测试F1分数', marker='^', linewidth=2)
    plt.xlabel('训练轮次')
    plt.ylabel('百分比 (%)')
    plt.title('测试集性能指标对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()
