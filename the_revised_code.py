import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) **2

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
        mse_values[i, j] = l_sum / 3

fig = plt.figure(figsize=(8, 6))

ax1 = fig.add_subplot(111, projection='3d')
surf = ax1.plot_surface(W, B, mse_values, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w')
ax1.set_ylabel('b')
ax1.set_zlabel('MSE')
ax1.set_title('MSE vs w and b')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()
