import h5py
from net import EEGDataset,getDataset
from torch.utils.data import ConcatDataset,DataLoader
import os

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
    dataLoader = DataLoader(
        dataset=datasets,
        shuffle=True,
        batch_size=32,
    )
        
    # print()
    # for 

    pass


