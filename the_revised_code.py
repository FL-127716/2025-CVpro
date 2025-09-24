import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

try:
    data = pd.read_csv('train (1).csv', encoding='gbk')

    required_columns = ['x', 'y']
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"CSV文件缺少必要的列: {missing}")

    x_data = data['x'].values
    y_data = data['y'].values

    mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]

except FileNotFoundError:
    print("错误：找不到指定的CSV文件，请检查文件路径")
    exit()
except Exception as e:
    print(f"数据加载错误：{str(e)}")
    exit()

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)
W, B = np.meshgrid(w_range, b_range)

mse_values = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        w = W[i, j]
        b = B[i, j]
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val)
        mse_values[i, j] = l_sum / len(x_data)

fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(111, projection='3d')

surf = ax1.plot_surface(W, B, mse_values, cmap='viridis', alpha=0.8)

ax1.set_xlabel('w (权重)')
ax1.set_ylabel('b (偏置)')
ax1.set_zlabel('MSE (均方误差)')
ax1.set_title('MSE与权重w和偏置b的关系')

fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
