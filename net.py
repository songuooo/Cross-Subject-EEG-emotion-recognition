import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py

class EEGDataset(Dataset):
    def __init__(self, eeg_data, label, window_size=250, stride=125, sfreq=250):
        """
        从长EEG序列创建滑动窗口Dataset
        
        参数:
            eeg_data: EEG数据,形状为 (n_samples, n_channels)
            window_size: 窗口大小（采样点数）
            stride: 滑动步长
            labels: 如果提供，应与窗口数匹配
            sfreq: 采样频率
        """
        self.eeg_data = eeg_data
        self.n_samples, self.n_channels = eeg_data.shape
        self.window_size = window_size
        self.stride = stride
        self.sfreq = sfreq
        
        # 计算可以生成的窗口数
        self.num_windows = (self.n_samples - window_size) // stride + 1
        
        if label is not None:
            # assert len(labels) == self.num_windows, "标签数量必须与窗口数匹配"

            self.labels = torch.full((self.num_windows,),label,dtype=torch.long)
        else:
            self.labels = torch.zeros(self.num_windows, dtype=torch.long)
        
        # print(f"创建滑动窗口数据集:")
        # print(f"  原始数据形状: {eeg_data.shape}")
        # print(f"  窗口大小: {window_size} ({window_size/sfreq:.2f}秒)")
        # print(f"  步长: {stride} ({stride/sfreq:.2f}秒)")
        # print(f"  生成窗口数: {self.num_windows}")
    
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        # 计算窗口起始位置
        start = idx * self.stride
        end = start + self.window_size
        
        # 提取窗口
        window = self.eeg_data[start:end,:]
        
        # 转换为tensor
        window_tensor = torch.FloatTensor(window)
        
        return window_tensor, self.labels[idx]
    
    def get_window_time(self, idx):
        """获取窗口的时间信息"""
        start = idx * self.stride
        end = start + self.window_size
        return start/self.sfreq, end/self.sfreq


def getDataset(path):
    with h5py.File(path,'r') as f:
        EEG_data_neu = np.array(f['EEG_data_neu']) # 中性 0 (50000=250Hz*4*50s,30)
        EEG_data_pos = np.array(f['EEG_data_pos']) # 积极 1
    return [EEGDataset(EEG_data_pos,1),EEGDataset(EEG_data_neu,0)]



class EEGConvNet(nn.Module):
    """
    专门针对EEG数据的卷积网络
    结合了空间和时间特征
    """
    
    def __init__(self, input_channels=30, seq_len=250, num_classes=2):
        super(EEGConvNet, self).__init__()
        
        # 第一层：空间滤波器（处理通道间关系）
        self.spatial_conv = nn.Conv2d(
            1,  # 输入通道数（将30个通道视为1个深度）
            8,  # 输出通道数
            kernel_size=(input_channels, 1),  # 卷积核覆盖所有通道
            padding=0
        )
        
        # 第二层：时间滤波器
        self.temporal_conv1 = nn.Conv1d(
            8, 16,
            kernel_size=5,
            padding=2
        )
        
        self.temporal_conv2 = nn.Conv1d(
            16, 32,
            kernel_size=5,
            padding=2
        )
        
        # 批标准化
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        
        # 池化
        self.pool = nn.MaxPool1d(2)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # 计算全连接层输入大小
        # 空间卷积不改变时间维度
        # 两次池化: 250 -> 125 -> 62
        fc_input_size = 32 * 62
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # 输入形状: (batch, 30, 250)
        
        # 添加通道维度: (batch, 1, 30, 250)
        x = x.unsqueeze(1)
        
        # 空间卷积: 学习通道间的关系
        x = self.spatial_conv(x)  # (batch, 8, 1, 250)
        x = x.squeeze(2)  # (batch, 8, 250)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 时间卷积1
        x = self.temporal_conv1(x)  # (batch, 16, 250)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 16, 125)
        x = self.dropout(x)
        
        # 时间卷积2
        x = self.temporal_conv2(x)  # (batch, 32, 125)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 32, 62)
        x = self.dropout(x)
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch, 32 * 62)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    

if __name__=="__main__":
    
    pass