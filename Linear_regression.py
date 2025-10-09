import torch
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('train.csv', encoding='gbk').dropna()

x_data = torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1)
y_data = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1)

x_mean, x_std = x_data.mean(), x_data.std()
y_mean, y_std = y_data.mean(), y_data.std()
x_data = (x_data - x_mean) / x_std
y_data = (y_data - y_mean) / y_std

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=1)
        torch.nn.init.normal_(self.linear.bias, mean=0, std=1)

    def forward(self, x):
        return self.linear(x)

criterion = torch.nn.MSELoss()

optimizer_list = [
    {'name': 'Adagrad', 'optimizer': torch.optim.Adagrad, 'lr': 0.1},
    {'name': 'Adam', 'optimizer': torch.optim.Adam, 'lr': 0.01},
    {'name': 'SGD', 'optimizer': torch.optim.SGD, 'lr': 0.001}
]

for opt_info in optimizer_list:
    model = LinearModel()
    optimizer = opt_info['optimizer'](model.parameters(), lr=opt_info['lr'])
    opt_name = opt_info['name']

    w_list, b_list, loss_list, epoch_list = [], [], [], []
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        w_list.append(model.linear.weight.item())
        b_list.append(model.linear.bias.item())
        loss_list.append(loss.item())
        epoch_list.append(epoch)

    best_loss_idx = loss_list.index(min(loss_list))
    w = w_list[best_loss_idx] * (y_std / x_std)
    b = y_mean - w * x_mean

    print(f'{opt_name}优化器结果：')
    print(f'最佳权重 w = {w:.3f}')
    print(f'最佳偏置 b = {b:.3f}')
    print(f'最小损失 loss = {min(loss_list):.3f}')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{opt_name}优化器 - 线性回归训练过程分析')

    with torch.no_grad():
        y_pred = model(x_data).numpy()
    axes[0, 0].scatter(x_data.numpy(), y_data.numpy(), alpha=0.5, label='原始数据')
    axes[0, 0].plot(x_data.numpy(), y_pred, 'r-', label=f'拟合直线: y={w:.3f}x + {b:.3f}')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('数据拟合结果')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(w_list, loss_list, alpha=0.5)
    axes[0, 1].set_xlabel('权重 w')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].set_title('权重 w 与损失的关系')

    axes[1, 0].plot(epoch_list, loss_list, alpha=0.5)
    axes[1, 0].set_xlabel('迭代次数')
    axes[1, 0].set_ylabel('损失')
    axes[1, 0].set_title('迭代次数与损失的关系')

    axes[1, 1].plot(epoch_list, w_list, alpha=0.5)
    axes[1, 1].set_xlabel('迭代次数')
    axes[1, 1].set_ylabel('权重 w')
    axes[1, 1].set_title('迭代次数与权重 w 的关系')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    lrs = [0.001, 0.01, 0.1]
    fig, ax = plt.subplots(figsize=(10, 6))
    for lr in lrs:
        model_lr = LinearModel()
        optimizer_lr = opt_info['optimizer'](model_lr.parameters(), lr=lr)
        loss_lr_list = []
        for epoch in range(300):
            y_pred_lr = model_lr(x_data)
            loss_lr = criterion(y_pred_lr, y_data)
            optimizer_lr.zero_grad()
            loss_lr.backward()
            optimizer_lr.step()
            loss_lr_list.append(loss_lr.item())
        ax.plot(range(300), loss_lr_list, label=f'学习率 {lr}')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('损失')
    ax.set_title(f'{opt_name}优化器不同学习率下损失变化')
    ax.legend()
    plt.show()

    epochs = [200, 500, 1000]
    fig, ax = plt.subplots(figsize=(10, 6))
    for epoch_num in epochs:
        model_epoch = LinearModel()
        optimizer_epoch = opt_info['optimizer'](model_epoch.parameters(), lr=opt_info['lr'])
        loss_epoch_list = []
        for epoch in range(epoch_num):
            y_pred_epoch = model_epoch(x_data)
            loss_epoch = criterion(y_pred_epoch, y_data)
            optimizer_epoch.zero_grad()
            loss_epoch.backward()
            optimizer_epoch.step()
            loss_epoch_list.append(loss_epoch.item())
        ax.plot(range(epoch_num), loss_epoch_list, label=f'迭代次数 {epoch_num}')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('损失')
    ax.set_title(f'{opt_name}优化器不同迭代次数下损失变化')
    ax.legend()
    plt.show()