import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载数据
data_path = r'c:\Users\21204\Desktop\archive\mal_anime.csv'
df = pd.read_csv(data_path)

# 数据预处理
def preprocess_data(df):
    # 只保留有评分的数据
    df = df.dropna(subset=['Score'])
    
    # 合并文本特征，增加结构信息
    def create_enhanced_text(row):
        parts = []
        if pd.notna(row['title']):
            parts.append(f"标题: {row['title']}")
        if pd.notna(row['Genres']):
            parts.append(f"类型: {row['Genres']}")
        if pd.notna(row['Themes']):
            parts.append(f"主题: {row['Themes']}")
        if pd.notna(row['description']):
            parts.append(f"简介: {row['description']}")
        return ' '.join(parts)
    
    df['text_features'] = df.apply(create_enhanced_text, axis=1)
    
    # 清理文本
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df['clean_text'] = df['text_features'].apply(clean_text)
    
    # 过滤掉太短的文本
    df = df[df['clean_text'].str.len() > 10]
    
    return df

# 应用预处理
df_clean = preprocess_data(df)
print(f"预处理后数据集大小: {len(df_clean)}")

# 划分训练集和测试集
train_df, test_df = train_test_split(
    df_clean, 
    test_size=0.2, 
    random_state=42
)

print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")

# 自定义数据集类
class AnimeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.clean_text.tolist()
        self.targets = dataframe.Score.tolist()
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = str(self.text[index])
        
        # 使用新的tokenizer API
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移除批次维度
        ids = inputs['input_ids'].squeeze(0)
        mask = inputs['attention_mask'].squeeze(0)
        
        # 新版本默认可能不返回token_type_ids，如果需要可以手动添加
        token_type_ids = inputs.get('token_type_ids', torch.zeros_like(ids)).squeeze(0)
        
        target = torch.tensor(self.targets[index], dtype=torch.float)
        
        return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets': target
        }

# BERT回归模型
class BERTRegressor(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout_rate=0.3, bert_model=None):
        super(BERTRegressor, self).__init__()
        # 支持传入预加载的BERT模型
        if bert_model is not None:
            self.bert = bert_model
        else:
            self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        # 增加一个隐藏层，增强表达能力
        self.hidden = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.regressor = nn.Linear(256, 1)
        
        # 限制输出范围到合理区间（如1-10）
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS]标记
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # 通过隐藏层
        hidden_output = self.relu(self.hidden(pooled_output))
        
        output = self.regressor(hidden_output)
        
        # 将输出缩放到1-10分范围
        output = 1.0 + 9.0 * self.sigmoid(output)
        
        return output.squeeze()

# 组合损失函数
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, predictions, targets):
        return self.alpha * self.mse(predictions, targets) + (1 - self.alpha) * self.mae(predictions, targets)

# 训练函数
def train_epoch(model, data_loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="训练中"):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# 评估函数
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(ids, mask, token_type_ids)
            loss = nn.MSELoss()(outputs, targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return total_loss / len(data_loader), mse, mae, r2, predictions, actuals

# 模型保存函数
def save_model(model, tokenizer, output_dir="./bert_anime_model"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # 保存模型配置
    model.bert.config.save_pretrained(output_dir)
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    
    # 保存额外信息
    model_info = {
        'dropout_rate': model.dropout.p,
        'hidden_size': 256,
        'output_range': (1.0, 10.0)
    }
    
    import json
    with open(os.path.join(output_dir, "model_info.json"), 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"模型已完整保存到 {output_dir}")

# 模型加载函数
def load_model(model_dir="./bert_anime_model", device='cpu'):
    import os
    import json
    
    # 加载模型信息
    with open(os.path.join(model_dir, "model_info.json"), 'r', encoding='utf-8') as f:
        model_info = json.load(f)
    
    # 初始化模型架构
    model = BERTRegressor(dropout_rate=model_info['dropout_rate'])
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    
    print(f"模型已从 {model_dir} 加载")
    
    return model, tokenizer

# 防止网络问题的稳健加载方式 - Tokenizer
def safe_load_tokenizer(model_name='bert-base-uncased', local_path=None):
    import os
    from requests.exceptions import ConnectionError, Timeout
    
    if local_path is None:
        local_path = f"./{model_name}-local"
    
    # 首先尝试从本地加载
    if os.path.exists(local_path) and os.listdir(local_path):
        print(f"从本地加载tokenizer: {local_path}")
        return BertTokenizer.from_pretrained(local_path)
    
    # 本地不存在，尝试在线下载
    try:
        print(f"从HuggingFace下载tokenizer: {model_name}")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 下载后保存到本地
        os.makedirs(local_path, exist_ok=True)
        tokenizer.save_pretrained(local_path)
        print(f"Tokenizer已保存到本地: {local_path}")
        
        return tokenizer
        
    except (ConnectionError, Timeout) as e:
        print(f"网络连接失败: {e}")
        print("请尝试以下解决方案:")
        print(f"1. 手动下载模型到 {local_path} 目录")
        print("2. 设置环境变量: $env:HF_ENDPOINT='https://hf-mirror.com'")
        print("3. 使用命令: $env:HF_ENDPOINT='https://hf-mirror.com'; python your_script.py")
        raise

# 防止网络问题的稳健加载方式 - BERT模型
def safe_load_bert_model(model_name='bert-base-uncased', local_path=None):
    import os
    from requests.exceptions import ConnectionError, Timeout
    
    if local_path is None:
        local_path = f"./{model_name}-local"
    
    # 检查本地目录是否存在且包含模型权重文件
    model_files = ['pytorch_model.bin', 'model.safetensors', 'tf_model.h5', 'model.ckpt.index', 'flax_model.msgpack']
    has_model_file = False
    if os.path.exists(local_path):
        files_in_dir = os.listdir(local_path)
        has_model_file = any(f in files_in_dir for f in model_files)
    
    # 首先尝试从本地加载（只有当目录存在且包含模型文件时）
    if has_model_file:
        print(f"从本地加载BERT模型: {local_path}")
        return BertModel.from_pretrained(local_path)
    
    # 本地不存在或不完整，尝试在线下载
    try:
        print(f"从HuggingFace下载BERT模型: {model_name}")
        model = BertModel.from_pretrained(model_name)
        
        # 下载后保存到本地
        os.makedirs(local_path, exist_ok=True)
        model.save_pretrained(local_path)
        print(f"BERT模型已保存到本地: {local_path}")
        
        return model
        
    except (ConnectionError, Timeout) as e:
        print(f"网络连接失败: {e}")
        print("请尝试以下解决方案:")
        print(f"1. 手动下载模型到 {local_path} 目录")
        print("2. 设置环境变量: $env:HF_ENDPOINT='https://hf-mirror.com'")
        print("3. 使用命令: $env:HF_ENDPOINT='https://hf-mirror.com'; python your_script.py")
        raise

# 增强训练监控可视化函数
def plot_training_history(train_losses, val_metrics, all_predictions, all_actuals, save_path="training_history.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 训练损失
    axes[0, 0].plot(train_losses, 'b-', label='训练损失', marker='o')
    axes[0, 0].set_title('训练损失变化')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 验证指标
    epochs = range(1, len(val_metrics) + 1)
    mse_values = [m[1] for m in val_metrics]
    mae_values = [m[2] for m in val_metrics]
    
    axes[0, 1].plot(epochs, mse_values, 'r-', label='MSE', marker='o')
    axes[0, 1].plot(epochs, mae_values, 'g-', label='MAE', marker='s')
    axes[0, 1].set_title('验证指标变化')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # R²分数
    r2_values = [m[3] for m in val_metrics]
    axes[1, 0].plot(epochs, r2_values, 'purple', label='R² Score', marker='^')
    axes[1, 0].set_title('R²分数变化')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 预测vs实际散点图
    axes[1, 1].scatter(all_actuals, all_predictions, alpha=0.3, s=20)
    axes[1, 1].plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], 'r--', lw=2, label='完美预测线')
    axes[1, 1].set_title('预测值 vs 实际值')
    axes[1, 1].set_xlabel('实际评分')
    axes[1, 1].set_ylabel('预测评分')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加统计信息
    mse = mean_squared_error(all_actuals, all_predictions)
    mae = mean_absolute_error(all_actuals, all_predictions)
    r2 = r2_score(all_actuals, all_predictions)
    axes[1, 1].text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"训练历史图表已保存到 {save_path}")

# 主训练流程
def main():
    # 使用稳健的方式初始化tokenizer
    print("=" * 50)
    print("加载Tokenizer...")
    print("=" * 50)
    tokenizer = safe_load_tokenizer('bert-base-uncased')
    
    # 使用稳健的方式加载BERT模型
    print("\n" + "=" * 50)
    print("加载BERT模型...")
    print("=" * 50)
    bert_model = safe_load_bert_model('bert-base-uncased')
    
    # 创建数据集
    max_len = 256  # 可以根据需要调整
    batch_size = 16  # 可以根据GPU内存调整
    
    print("\n" + "=" * 50)
    print("创建数据集...")
    print("=" * 50)
    train_dataset = AnimeDataset(train_df, tokenizer, max_len)
    test_dataset = AnimeDataset(test_df, tokenizer, max_len)
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型（使用预加载的BERT模型）
    print("\n" + "=" * 50)
    print("初始化回归模型...")
    print("=" * 50)
    model = BERTRegressor(bert_model=bert_model).to(device)
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_data_loader) * 3  # 假设训练3个epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练循环
    epochs = 3
    train_losses = []
    val_metrics = []
    all_epoch_predictions = []
    all_epoch_actuals = []
    
    # 使用组合损失函数
    loss_fn = CombinedLoss(alpha=0.7)
    
    print("\n" + "=" * 50)
    print(f"开始训练，共 {epochs} 个Epoch")
    print("=" * 50)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # 训练
        train_loss = train_epoch(model, train_data_loader, optimizer, scheduler, device, loss_fn)
        train_losses.append(train_loss)
        
        # 评估
        val_loss, mse, mae, r2, predictions, actuals = evaluate(model, test_data_loader, device)
        val_metrics.append((val_loss, mse, mae, r2))
        
        # 保存每个epoch的预测结果
        all_epoch_predictions.extend(predictions)
        all_epoch_actuals.extend(actuals)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # 最终评估
    print("\n" + "=" * 50)
    print("最终评估")
    print("=" * 50)
    final_val_loss, final_mse, final_mae, final_r2, predictions, actuals = evaluate(model, test_data_loader, device)
    
    print("\n最终评估结果:")
    print(f"验证损失: {final_val_loss:.4f}")
    print(f"MSE: {final_mse:.4f}")
    print(f"MAE: {final_mae:.4f}")
    print(f"R2 Score: {final_r2:.4f}")
    
    # 保存模型
    print("\n" + "=" * 50)
    print("保存模型...")
    print("=" * 50)
    save_model(model, tokenizer, './bert_anime_model')
    
    # 使用增强的可视化函数
    print("\n" + "=" * 50)
    print("生成训练历史图表...")
    print("=" * 50)
    plot_training_history(train_losses, val_metrics, np.array(all_epoch_predictions), np.array(all_epoch_actuals), 'training_history.png')
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)
    
    return model, tokenizer

if __name__ == "__main__":
    try:
        model, tokenizer = main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束")