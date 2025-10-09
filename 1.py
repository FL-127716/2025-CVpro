import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minpips'] = False

'''
第一部分：简单线性模型 y = w * x 的损失可视化
'''
# 定义数据集
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])


# 前向传播函数
def forward_simple(x):
    return x * w


# 损失计算函数
def loss_simple(x, y):
    y_pred = forward_simple(x)
    return (y_pred - y) ** 2


# 记录不同w值对应的MSE
w_list = []
mse_list = []

# 遍历不同的w值计算损失
for w in np.arange(0.0, 4.1, 0.1):
    print(f'w = {w:.1f}')
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward_simple(x_val)
        loss_val = loss_simple(x_val, y_val)
        loss_sum += loss_val
        print(f'\t{x_val} {y_val} {y_pred_val:.1f} {loss_val:.2f}')
    mse = loss_sum / 3
    print(f'MSE = {mse:.2f}\n')
    w_list.append(w)
    mse_list.append(mse)

# 绘制w与MSE的关系图
plt.plot(w_list, mse_list)
plt.xlabel('w')
plt.ylabel('均方误差 (MSE)')
plt.title('y = w*x 模型中w与MSE的关系')
plt.tight_layout()
plt.show()

'''
第二部分：带偏置的线性模型 y = w*x + b 的损失可视化
'''


# 带偏置的前向传播函数
def forward_bias(x, w, b):
    return x * w + b


# 带偏置的损失计算函数
def loss_bias(x, y, w, b):
    y_pred = forward_bias(x, w, b)
    return np.mean((y_pred - y) ** 2)


# 记录不同w、b值对应的MSE
w_list_2 = []
b_list_2 = []
mse_list_2 = []

# 遍历不同的w和b值计算损失
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        # 计算当前w和b对应的MSE
        mse = loss_bias(x_data, y_data, w, b)
        w_list_2.append(w)
        b_list_2.append(b)
        mse_list_2.append(mse)

# 绘制3D损失曲面
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter3D(w_list_2, b_list_2, mse_list_2, c=mse_list_2, cmap='viridis', s=10)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('均方误差 (MSE)')
ax.set_title('y = w*x + b 模型的损失曲面')
fig.colorbar(scatter, label='MSE值')
plt.tight_layout()
plt.show()

'''
第三部分：使用梯度下降法拟合训练数据
'''
# 读取训练数据
df = pd.read_csv(r'train.csv')
df = df.dropna()  # 去除缺失值
x_train = df['x'].values
y_train = df['y'].values


# 定义模型函数
def model(x, w, b):
    return w * x + b


# 定义损失函数
def compute_loss(x, y, w, b):
    y_pred = model(x, w, b)
    return np.mean((y_pred - y) ** 2)


# 初始化参数
w = 0.0
b = 0.0
learning_rate = 0.0001  # 学习率
num_samples = len(x_train)
iterations = 0  # 迭代计数器

# 记录训练过程
w_history = []
b_history = []
loss_history = []

# 梯度下降迭代
while iterations < num_samples:
    # 计算预测值
    y_pred = model(x_train, w, b)

    # 计算梯度并更新参数
    w_gradient = (-2 / num_samples) * np.sum(x_train * (y_train - y_pred))
    b_gradient = (-2 / num_samples) * np.sum(y_train - y_pred)
    w -= learning_rate * w_gradient
    b -= learning_rate * b_gradient

    # 计算当前损失
    current_loss = compute_loss(x_train, y_train, w, b)

    # 记录历史数据
    w_history.append(float(w))
    b_history.append(float(b))
    loss_history.append(float(current_loss))

    iterations += 1

# 找到最小损失对应的参数
min_loss_index = loss_history.index(min(loss_history))
best_w = w_history[min_loss_index]
best_b = b_history[min_loss_index]
min_mse = min(loss_history)

# 输出结果
print(f'误差最小值的索引为：{min_loss_index}')
print(f'最小MSE值为：{min_mse:.6f}')
print('误差最小的参数为：')
print(f'w = {best_w:.6f}')
print(f'b = {best_b:.6f}')
print(f'线性模型为：y = {best_w:.3f} * x + {best_b:.3f}')

# 绘制结果可视化
final_pred = model(x_train, w, b)
plt.figure(figsize=(15, 5))

# 原始数据与拟合直线
plt.subplot(1, 3, 1)
plt.scatter(x_train, y_train, alpha=0.5, label='原始数据', color='blue')
plt.plot(x_train, final_pred, 'r-', linewidth=2, label=f'拟合直线: y = {w:.3f}x + {b:.3f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('原始数据与拟合结果')
plt.legend()
plt.grid(True, alpha=0.3)

# w与损失的关系
plt.subplot(1, 3, 2)
plt.plot(w_history, loss_history, 'g-', linewidth=1, alpha=0.7)
plt.scatter(w_history[min_loss_index], loss_history[min_loss_index],
            color='red', s=100, label=f'最小MSE点 (w={best_w:.3f})')
plt.xlabel('权重 w')
plt.ylabel('均方误差 MSE')
plt.title('w与损失函数的关系')
plt.legend()
plt.grid(True, alpha=0.3)

# b与损失的关系
plt.subplot(1, 3, 3)
plt.plot(b_history, loss_history, 'g-', linewidth=1, alpha=0.7)
plt.scatter(b_history[min_loss_index], loss_history[min_loss_index],
            color='red', s=100, label=f'最小MSE点 (b={best_b:.3f})')
plt.xlabel('偏置 b')
plt.ylabel('均方误差 MSE')
plt.title('b与损失函数的关系')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()