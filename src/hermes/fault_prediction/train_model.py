"""
Hermes故障预测模型训练脚本
训练LSTM模型并导出为ONNX格式
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 生成模拟数据
def generate_simulated_data(n_samples=10000):
    """生成GPU指标模拟数据"""
    np.random.seed(42)
    
    # 正常数据（故障概率低）
    normal_samples = int(n_samples * 0.8)
    temp_normal = np.random.normal(65, 8, normal_samples)
    power_normal = np.random.normal(140, 40, normal_samples)
    util_normal = np.random.normal(75, 15, normal_samples)
    mem_normal = np.random.normal(70, 18, normal_samples)
    ecc_normal = np.random.poisson(0.3, normal_samples)
    fan_normal = np.random.normal(55, 15, normal_samples)
    
    # 故障数据（故障概率高）
    fault_samples = n_samples - normal_samples
    temp_fault = np.random.normal(90, 5, fault_samples)
    power_fault = np.random.normal(250, 30, fault_samples)
    util_fault = np.random.normal(95, 3, fault_samples)
    mem_fault = np.random.normal(95, 3, fault_samples)
    ecc_fault = np.random.poisson(5, fault_samples)
    fan_fault = np.random.normal(95, 3, fault_samples)
    
    # 合并数据
    X = np.vstack([
        np.column_stack([temp_normal, power_normal, util_normal, mem_normal, ecc_normal, fan_normal]),
        np.column_stack([temp_fault, power_fault, util_fault, mem_fault, ecc_fault, fan_fault])
    ])
    
    # 标签：距离故障的时间（归一化到0-1）
    y = np.zeros(n_samples)
    y[:normal_samples] = np.random.uniform(0, 0.3, normal_samples)
    y[normal_samples:] = np.random.uniform(0.7, 1.0, fault_samples)
    
    return X, y

class FaultLSTM(nn.Module):
    """故障预测LSTM模型"""
    def __init__(self, input_size=6, hidden_size=32, num_layers=2):
        super(FaultLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

def train_model():
    """训练模型"""
    print("生成模拟数据...")
    X, y = generate_simulated_data()
    
    # 归一化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 转换为LSTM输入格式 (samples, seq_len, features)
    X_scaled = X_scaled.reshape(-1, 1, 6)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # 创建模型
    model = FaultLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    print("开始训练...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                print(f"Epoch [{epoch+1}/100], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    # 保存模型为ONNX格式
    print("导出ONNX模型...")
    dummy_input = torch.randn(1, 1, 6)
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    # 验证ONNX模型
    session = ort.InferenceSession("model.onnx")
    test_input = X_test[0].numpy()
    onnx_output = session.run(None, {'input': test_input})[0]
    torch_output = model(X_test[0:1]).detach().numpy()
    
    print(f"PyTorch输出: {torch_output[0][0]:.4f}")
    print(f"ONNX输出: {onnx_output[0][0]:.4f}")
    print("模型训练和导出完成！")
    
    # 保存归一化参数
    np.savez("scaler_params.npz", mean=scaler.mean_, scale=scaler.scale_)
    
    return model, scaler

if __name__ == "__main__":
    train_model()
