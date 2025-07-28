# models/custom_models/my_lstm_imputer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..base_imputer import BaseImputer

# 定义一个简单的LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # 我们只关心插补，所以直接用lstm的输出
        lstm_out, _ = self.lstm(x)
        imputation = self.fc(lstm_out)
        return imputation

class MyLSTMImputer(BaseImputer):
    """
    一个使用简单LSTM实现的自定义插补模型示例。
    """
    def __init__(self, n_features: int, hidden_dim: int, n_layers: int, 
                 lr: float, epochs: int, device: str = 'cpu', **kwargs):
        self.model = LSTMModel(n_features, hidden_dim, n_layers)
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)
        self.model.to(self.device)

    def fit(self, train_data: np.ndarray):
        print("Training custom LSTM model...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        # 将NaN替换为0进行训练
        train_data_tensor = torch.from_numpy(np.nan_to_num(train_data)).float().to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            imputed_sequence = self.model(train_data_tensor)
            
            # 只在非缺失值上计算损失
            mask = ~torch.isnan(torch.from_numpy(train_data).float().to(self.device))
            loss = loss_fn(imputed_sequence[mask], train_data_tensor[mask])
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.6f}")

    def impute(self, data: np.ndarray) -> np.ndarray:
        print("Imputing with custom LSTM model...")
        self.model.eval()
        
        data_tensor = torch.from_numpy(np.nan_to_num(data)).float().to(self.device)
        
        with torch.no_grad():
            imputed_values = self.model(data_tensor).cpu().numpy()

        imputed_data = data.copy()
        missing_mask = np.isnan(data)
        imputed_data[missing_mask] = imputed_values[missing_mask]
        
        return imputed_data

    def save(self, path: str):
        """ 保存自定义模型的权重 """
        print(f"Saving custom LSTM model state to {path}...")
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """ 加载自定义模型的权重 """
        print(f"Loading custom LSTM model state from {path}...")
        # map_location确保模型可以被加载到当前指定的device上
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
