import torch
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('train.csv', encoding='gbk')
df = df.dropna()

x_data=torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1)
y_data=torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1)
# print(x_data[5])
# print(y_data[5])
x_mean,x_sta=x_data.mean(),x_data.std()
y_mean,y_sta=y_data.mean(),y_data.std()

x_data=(x_data-x_mean)/x_sta
y_data=(y_data-y_mean)/y_sta
# print(x_data[5])
# print(y_data[5])

class  LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=1)
        torch.nn.init.normal_(self.linear.bias, mean=0, std=1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
# w_list=[]
# b_list=[]
# mse_list=[]
# epoch_list=[]

# model=LinearModel()
criterion = torch.nn.MSELoss(size_average=False)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
# optimizer=torch.optim.Adagrad(model.parameters(),lr=0.1)
# optimizer=torch.optim.ASGD(model.parameters(),lr=0.0001)
optimizer_list=[
{'name': 'SGD', 'optimizer': torch.optim.SGD, 'lr': 0.0001},
    {'name': 'Adagrad', 'optimizer': torch.optim.Adagrad, 'lr': 0.1},
    {'name': 'ASGD', 'optimizer': torch.optim.ASGD, 'lr': 0.0001},
]

for i in optimizer_list:
    model = LinearModel()
    w_list = []
    b_list = []
    mse_list = []
    epoch_list = []
    optimizer=i['optimizer'](model.parameters(),lr=i['lr'])
    opt_name=i['name']
    for epoch in range(1000):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        # print(epoch, loss.item())
        # mse_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        w_list.append(model.linear.weight.item())
        b_list.append(model.linear.bias.item())
        mse_list.append(loss.item())
        epoch_list.append(epoch)
    # print('w = ', model.linear.weight.item())
    # print('b = ', model.linear.bias.item())
    # print(w_list)
    # plt.plot(w_list,mse_list)
    # plt.tight_layout()
    # plt.show()
    # print(mse_list)
    min_loss=mse_list.index(min(mse_list))
    w=w_list[min_loss]
    b=b_list[min_loss]
    w=w*(y_sta/x_sta)
    b=y_mean-w*x_mean

    print(f'{opt_name}优化器的结果：')
    print('w = ',w)
    print('b = ',b)

    with torch.no_grad():
        y_pred_tensor = model(x_data)
        y_pred = y_pred_tensor.numpy()
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    plt.suptitle(f'{opt_name}优化器 - 线性回归训练过程分析')

    axes[0,0].scatter(x_data.numpy(), y_data.numpy(), alpha=0.5, label='原始数据', color='blue')
    axes[0,0].plot(x_data.numpy(),y_pred,'r', linewidth=2,label=f'拟合直线:y={w:.3f} * x + ({b:.3f})')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].set_title('线性回归拟合结果')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(w_list,mse_list,alpha=0.5)
    axes[0,1].set_xlabel('w')
    axes[0,1].set_ylabel('mse')
    axes[0,1].set_title('w vs mse')

    axes[1,0].plot(epoch_list,mse_list,alpha=0.5)
    axes[1,0].set_xlabel('epoch')
    axes[1,0].set_ylabel('mse')
    axes[1,0].set_title('epoch vs mse')

    axes[1,1].plot(epoch_list,w_list,alpha=0.5)
    axes[1,1].set_xlabel('epoch')
    axes[1,1].set_ylabel('w')
    axes[1,1].set_title('epoch vs w')

    plt.tight_layout()
    plt.show()
