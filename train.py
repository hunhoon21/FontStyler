from src.models.model import AE_base
from src.data.common.dataset import FontDataset, PickledImageProvider

import torch
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from ignite.engine import Events, Engine
from ignite.metrics import Loss, MeanSquaredError, RunningAverage

import numpy as np
from tqdm import tqdm

from maplotlib import pyplot as plt

if __name__ == '__main__':
    
    '''
    Configuration: 
    TODO - parse.args 활용
    '''
    batch_size = 64
    validation_split = .15
    test_split = .05
    shuffle_dataset = True
    random_seed = 42
    
    lr = 0.001
    
    log_interval = 10
    epochs = 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    '''
    Dataset Loaders
    '''
    
    # get Dataset
    data_dir = 'src/data/dataset/integrated/'
    serif = PickledImageProvider(data_dir+'train_0.obj')
    dataset = FontDataset(serif)
    
    # get idx samplers
    dataset_size = len(dataset)
    idxs = list(range(dataset_size))
    split_test = int(np.floor(test_split * dataset_size))
    split_valid = int(np.floor((test_split + validation_split) * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(idxs)
    train_idxs, val_idxs, test_idxs = idxs[split_valid:], idxs[split_test:split_valid], idxs[:split_test]
    
    train_sampler = SubsetRandomSampler(train_idxs)
    valid_sampler = SubsetRandomSampler(val_idxs)
    test_sampler = SubsetRandomSampler(test_idxs)
    
    # get data_loaders
    train_loader = DataLoader(dataset, 
                          batch_size=batch_size,
                          sampler=train_sampler
                          )
    valid_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=valid_sampler
                            )
    test_loader = DataLoader(dataset,
                            batch_size=split_test,
                            sampler=test_sampler
                            )
    
    '''
    Modeling
    '''
    model = AE_base(category_size=0, 
                    alpha_size=52, 
                    font_size=256*256, 
                    z_size=64)
    
    '''
    Optimizer
    TODO - 옵티마이저도 모델 안으로 넣기
    Abstract model 만들기?
    '''
    optimizer = Adam(model.parameters(), lr=lr)
    '''
    엔진 구축
    '''
    
    def train_process(engine, batch):
        model.to(device).train()
        optimizer.zero_grad()
        #  = batch