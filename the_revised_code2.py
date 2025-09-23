import numpy as np
import matplotlib.pyplot as plt
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


# 定义w和b的取值范围
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(-2.0, 2.1, 0.1)
W, B = np.meshgrid(w_range, b_range)

# 计算每个(w, b)组合的MSE
mse_values = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        w = W[i, j]
        b = B[i, j]
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            l_sum += loss(x_val, y_val)
        mse_values[i, j] = l_sum / len(x_data)

# 找到每个w对应的最佳b（使MSE最小的b）
best_b_for_w = np.argmin(mse_values, axis=0)
mse_for_w = [mse_values[best_b_for_w[i], i] for i in range(len(w_range))]

# 找到每个b对应的最佳w（使MSE最小的w）
best_w_for_b = np.argmin(mse_values, axis=1)
mse_for_b = [mse_values[i, best_w_for_b[i]] for i in range(len(b_range))]

# 创建2D图形
plt.figure(figsize=(12, 5))

# 第一个子图：MSE与w的关系
plt.subplot(1, 2, 1)
plt.plot(w_range, mse_for_w, 'b-', linewidth=2)
plt.xlabel('w (权重)')
plt.ylabel('MSE (均方误差)')
plt.title('MSE与权重w的关系')
plt.grid(True, alpha=0.3)

# 第二个子图：MSE与b的关系
plt.subplot(1, 2, 2)
plt.plot(b_range, mse_for_b, 'r-', linewidth=2)
plt.xlabel('b (偏置)')
plt.ylabel('MSE (均方误差)')
plt.title('MSE与偏置b的关系')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
