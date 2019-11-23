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

from matplotlib import pyplot as plt

if __name__ == '__main__':
    
    '''
    Configuration: 
    TODO - parse.args 활용
    '''
    batch_size = 8
    validation_split = .15
    test_split = .05
    shuffle_dataset = True
    random_seed = 42
    
    lr = 0.0001
    
    log_interval = 10
    epochs = 200
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    '''
    Dataset Loaders
    '''
    
    # get Dataset
    data_dir = 'src/data/dataset/integrated/'
    serif = PickledImageProvider(data_dir+'train.obj')
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
                    font_size=128*128, 
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
    
    # Training 시 process_function
    def train_process(engine, batch):
        model.float().to(device).train()
        optimizer.zero_grad()
        vectors, font  = batch
        alpha_vector = vectors['alphabet']
        
        font, alpha_vector = font.float().to(device), alpha_vector.float().to(device)
        
        font_hat, _ = model(font, alpha_vector, None)
        
        loss = F.mse_loss(font_hat, font)
        loss.backward()
        
        optimizer.step()
        
        return loss.item()
    
    # Evaluating 시 process_function
    def evaluate_process(engine, batch):
        model.float().to(device).eval()
        with torch.no_grad():
            vectors, font = batch
            alpha_vector = vectors['alphabet']
            
            font, alpha_vector = font.float().to(device), alpha_vector.float().to(device)
            
            font_hat, _ = model(font, alpha_vector, None)
            
            return font, font_hat
        
        
    trainer = Engine(train_process)
    evaluator = Engine(evaluate_process)
    
    
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'mse')
    
    Loss(F.mse_loss, output_transform=lambda x: [x[1], x[0]]).attach(evaluator, 'mse')
    
    desc = "ITERATION - loss: {:.5f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    train_history = []
    valid_history = []
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        
        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        mse_loss = metrics['mse']
        # kld_loss = metrics['kld']
        tqdm.write(
            "Training Result - Epoch: {} MSE: {:.7f}"
            .format(engine.state.epoch, mse_loss)
        )
        global train_history
        train_history += [metrics['mse']]
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        mse_loss = metrics['mse']
        # kld_loss = metrics['kld']
        tqdm.write(
            "Validation Results - Epoch: {} MSE: {:.7f}"
            .format(engine.state.epoch, mse_loss)
        )
        global valid_history
        valid_history += [metrics['mse']]
        
    @trainer.on(Events.COMPLETED)
    def plot_history_results(engine):
        train_epoch = len(train_history)
        valid_epoch = len(valid_history)
        plt.plot(list(range(1, train_epoch+1)), train_history, label='train_history')
        plt.plot(list(range(1, valid_epoch+1)), valid_history, label='valid_history')
        plt.legend()
        plt.savefig('history.png')
        plt.close()
        
    @trainer.on(Events.COMPLETED)
    def plot_font_results(engine):
        evaluator.run(test_loader)
        real_font, fake_font = evaluator.state.output
        plt.figure(figsize=(50, 100))
        for i, (real, fake) in enumerate(zip(real_font, fake_font)):
            plt.subplot(43, 22, 2*i+1)
            plt.imshow(real.cpu().detach().numpy())
            plt.subplot(43, 22, 2*i+2)
            plt.imshow(fake.cpu().detach().numpy())
        plt.savefig('real_fake_fonts')
        plt.close()
        
        
    trainer.run(train_loader, max_epochs=epochs)