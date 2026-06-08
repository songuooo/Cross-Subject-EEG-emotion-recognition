import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from net import EEGDataset,getDataset,EEGConvNet
from torch.utils.data import ConcatDataset,DataLoader,random_split
import os
from sklearn.metrics import accuracy_score


def split_dataset(dataset):
    train_rio=0.7
    val_rio=0.15
    test_rio=0.15
    train_len = train_rio * len(dataset)
    val_len = val_rio * len(dataset)
    test_len = len(dataset)-train_len-val_len
    return random_split(
        dataset, [int(train_len), int(val_len), int(test_len)],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子确保可复现
    )
    pass


# 7. 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets.unsqueeze(1).float())
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录指标
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs)>=0.5).long().squeeze(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# 8. 验证函数
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets.unsqueeze(1).float())
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs)>=0.5).long().squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def train(model,train_loader,val_loader,test_loader,criterion,optimizer,device,epochs=30):
    best_val_acc = 0
    for epoch in range(epochs):
        train_loss,train_acc = train_epoch(model,train_loader,criterion,optimizer,device)
        val_loss,val_acc = validate(model,val_loader,criterion,device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),'best_model.pth')
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 50)
    
    print("\n测试集评估:")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')


if __name__=="__main__":
    depPath = './data/train/DEP/'
    hcPath = './data/train/HC/'
    # rpath = os.listdir('./data/train/DEP')
    datasets=[]
    # for p in os.listdir(hcPath):
    #     datasets+=getDataset(hcPath+p)
    for p in os.listdir(depPath):
        datasets+=getDataset(depPath+p)
    datasets=ConcatDataset(datasets)
    (train_dataset,val_dataset,test_dataset) = split_dataset(datasets)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGConvNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model,train_loader,val_loader,test_loader,criterion,optimizer,device,epochs=100)
    # print()
    # for 

    pass


