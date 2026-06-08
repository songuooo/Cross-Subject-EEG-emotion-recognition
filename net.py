import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
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
    
    def __init__(self, input_channels=30, seq_len=250, num_classes=1):
        super(EEGConvNet, self).__init__()
        
        # 第一块：时间卷积 (kernel 沿时间维度)
        # 输入: (batch, 1, channels, time)
        # 输出: (batch, 8, channels, time)  因为 groups=1 默认，卷积核 (1,25) 只沿时间方向
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=(1, 25),
            padding=(0, 12),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(8)
        
        # 第二块：空间卷积 (深度卷积，沿通道方向)
        # 输入: (batch, 8, channels, time)
        # kernel_size=(channels, 1) 会在空间维度上覆盖所有通道，groups=8 表示每个输入通道独立卷积
        # 输出: (batch, 8, 1, time)
        self.spatial_conv = nn.Conv2d(
            in_channels=8,
            out_channels=8,
            kernel_size=(input_channels, 1),
            groups=8,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(8)
        
        # 第三块：可分离卷积 (先逐通道再逐点，这里用普通 Conv2d 模拟可分离效果)
        # 输入: (batch, 8, 1, time)
        # 输出: (batch, 16, 1, time)
        self.sep_conv = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(1, 10),
            padding=(0, 4),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(16)
        
        # 全局平均池化，将时间维度压缩为 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # 输出 (batch, 16, 1, 1)
        
        self.dropout = nn.Dropout(0.5)
        
        # 分类层
        self.classifier = nn.Linear(16, num_classes)   # num_classes=1 输出 logits
        
    def forward(self, x):
        # 输入形状: (batch, 250, 30)
        # x shape: (batch, time, channels)  例如 (32, 250, 30)
        # 转换为 (batch, 1, channels, time)
        x = x.permute(0, 2, 1).unsqueeze(1)   # (batch, 1, channels, time)
        
        # 时间卷积 + BN + 激活
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = F.elu(x)      # EEGNet 常用 ELU
        
        # 空间卷积 + BN + 激活
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        
        # 可分离卷积 + BN + 激活
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        
        # 全局平均池化
        x = self.avgpool(x)          # (batch, 16, 1, 1)
        x = x.view(x.size(0), -1)    # (batch, 16)
        
        x = self.dropout(x)
        logits = self.classifier(x)  # (batch, 1)
        
        return logits
    

if __name__=="__main__":
    
    pass