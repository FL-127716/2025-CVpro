import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
'''

y=w*x

'''
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
x_data=np.array([1.0,2.0,3.0])
y_data=np.array([2.0,4.0,6.0])
def forword(x):
    return x * w

def loss(x,y):
    y_pred=forword(x)
    return (y_pred-y)**2

w_list=[]
mse_list=[]

for w in np.arange(0.0,4.1,0.1):
    print('w=',w)
    l_sum=0
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val=forword(x_val)
        loss_val=loss(x_val,y_val)
        l_sum+=loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)
    print('MSE',l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list,mse_list)
plt.tight_layout()
plt.show()
'''

y=w*x+b

'''
def forward(x, w, b):
    return x * w + b

def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return np.mean((y_pred - y) ** 2)

w_list = []
b_list = []
mse_list = []



for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):

        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val, w, b)
            mse = loss_val

        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
# 绘制散点图
scatter = ax.scatter3D(w_list, b_list, mse_list, c=mse_list, cmap='viridis', s=10)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE Loss')
ax.set_title('MSE Loss Landscape')
plt.show()

'''
读取train,用y=w*x+b
'''
#
df = pd.read_csv(r'train.csv')
df = df.dropna()
x_data = df['x'].values
y_data = df['y'].values
def f(x, w, b):
    return w * x + b
def loss(x, y, w, b):
    y_pred = f(x, w, b)
    return np.mean((y_pred - y) ** 2)
w_list = []
b_list = []
mse_list = []
w = 0
b = 0
a = 0.0001#学习率
n = len(x_data)
iteration = 0#循环初始值，同时也是记录循环次数

while iteration < n:

    y_pred = f(x_data, w, b)

    w = w - a * (-2 / n) * np.sum(x_data * (y_data - y_pred))
    b = b - a *(-2 / n) * np.sum(y_data - y_pred)


    mse = loss(x_data, y_data, w, b)

    w_list.append(float(w))
    b_list.append(float(b))
    mse_list.append(float(mse))
    iteration += 1
# print(iteration)

y_pred = f(x_data, w, b)
# print(w_list)

min_lable=mse_list.index(min(mse_list))
w_=w_list[min_lable]
b_=b_list[min_lable]


print(f'误差最小值的索引为：\n{min_lable}')
# print(w_list[min_lable])
# print(b_list[min_lable])
print(f'mse={min(mse_list)}')
print('误差最小的w,b为:')
print(f'w={w_}\nb={b_}\n线性模型为：y={w_:.3f} * x + {b_:.3f}')
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(x_data, y_data, alpha=0.5, label='原始数据', color='blue')
plt.plot(x_data, y_pred, 'r', linewidth=2, label=f'拟合直线: y = {w:.3f}x + {b:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(w_list, mse_list, 'g', linewidth=1, alpha=0.7)
plt.scatter(w_list[min_lable], mse_list[min_lable], color='red', s=100,
           label=f'最小MSE点 (w={w_:.3f}, MSE={min(mse_list):.3f})')
plt.xlabel('权重 w')
plt.ylabel('均方误差 MSE')
plt.title('权重 w 与损失函数 MSE 的关系')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(b_list, mse_list, 'g', linewidth=1, alpha=0.7)
plt.scatter(b_list[min_lable], mse_list[min_lable], color='red', s=100,
           label=f'最小MSE点 (w={b_:.3f}, MSE={min(mse_list):.3f})')
plt.xlabel('偏置 b')
plt.ylabel('均方误差 MSE')
plt.title('偏置 b 与损失函数 MSE 的关系')
plt.legend()

plt.tight_layout()
plt.show()