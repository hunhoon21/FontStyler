from src.models.model import Convolutional_AE_z
from src.data.common.dataset import KoreanFontDataset_with_Embedding, PickledImageProvider

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
import pickle

if __name__ == '__main__':
    
    '''
    Configuration: 
    TODO - parse.args 활용
    '''
    batch_size = 16
    validation_split = .1
    test_split = .05
    shuffle_dataset = True
    random_seed = 42
    
    lr = 0.003
    
    log_interval = 10
    epochs = 100
    
    print(torch.cuda.is_available())
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    conv_dim = 32
    
    '''
    Dataset Loaders
    '''
    
    # get Dataset
    data_dir = 'src/data/dataset/kor/'
    category_path = 'src/data/dataset/embedding/category_emb.pkl'
    letter_path = 'src/data/dataset/embedding/letter_emb.pkl'
    train_set = KoreanFontDataset_with_Embedding(PickledImageProvider(data_dir+'train.obj'),
                                                 category_emb_path=category_path,
                                                 letter_emb_path=letter_path
                                                 )
    valid_set = KoreanFontDataset_with_Embedding(PickledImageProvider(data_dir+'val.obj'), 
                                                 category_emb_path=category_path,
                                                 letter_emb_path=letter_path
                                                 )
    test_set = KoreanFontDataset_with_Embedding(PickledImageProvider(data_dir+'test.obj'),
                                                category_emb_path=category_path,
                                                letter_emb_path=letter_path
                                                )
    
    # get idx samplers
    train_size = len(train_set)
    valid_size = len(valid_set)
    test_size = len(test_set)
    
    train_idx = list(range(train_size))
    val_idx = list(range(valid_size))
    
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)

    # get data_loaders
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size,
                              sampler=train_sampler
                              )
    valid_loader = DataLoader(valid_set,
                              batch_size=batch_size,
                              sampler=valid_sampler
                              )
    test_loader = DataLoader(test_set,
                             batch_size=test_size
                             )
    '''
    Modeling
    '''
    model = Convolutional_AE_z(img_dim=1, conv_dim=conv_dim).to(device)
    
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
        
        _, font, category, letter  = batch
        
        font = font.float().to(device)
        category = category.float().to(device)
        letter = letter.float().to(device)
        
        font_hat, z= model(font, category, letter)
        
        mse = F.mse_loss(font_hat, font)
        loss = mse
        
        loss.backward()
        
        optimizer.step()
        
        return loss.item()
    
    # Evaluating 시 process_function
    def evaluate_process(engine, batch):
        model.float().to(device).eval()
        with torch.no_grad():
            _, font, category, letter = batch
            
            font = font.float().to(device)
            category = category.float().to(device)
            letter = letter.float().to(device)
            
            font_hat, z = model(font, category, letter)
            
            return font, font_hat, z
        
        
    trainer = Engine(train_process)
    evaluator = Engine(evaluate_process)
    
    
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    
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
            outputs = engine.state.output
            pbar.desc = desc.format(outputs)
            pbar.update(log_interval)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        mse_loss = metrics['mse']
        tqdm.write(
            "Training Result - Epoch: {} MSE: {:.4f}"
            .format(engine.state.epoch, mse_loss)
        )
        global train_history
        train_history += [metrics['mse']]
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        mse_loss = metrics['mse']
        tqdm.write(
            "Validation Results - Epoch: {} MSE: {:.4f}"
            .format(engine.state.epoch, mse_loss)
        )
        global valid_history
        valid_history += [metrics['mse']]

    history_path = 'history_convAE_Z_epoch_{}_dim_{}.png'
    @trainer.on(Events.COMPLETED)
    def plot_history_results(engine):
        train_epoch = len(train_history)
        valid_epoch = len(valid_history)
        plt.figure()
        plt.plot(list(range(1, train_epoch+1)), train_history, label='train_history')
        plt.plot(list(range(1, valid_epoch+1)), valid_history, label='valid_history')
        plt.legend()
        global history_path, epochs, conv_dim
        print(history_path.format(epochs, conv_dim))
        plt.savefig(history_path.format(epochs, conv_dim))
        plt.close()
    
    result_path = 'real_fake_convAE_Z_epoch_{}_dim_{}.png'
    @trainer.on(Events.COMPLETED)
    def plot_font_results(engine):
        evaluator.run(valid_loader)
        real_font, fake_font, _ = evaluator.state.output
        # print(real_font.shape)
        # print(fake_font)
        real_font, fake_font = real_font[:200], fake_font[:200]
        n = len(real_font)
        plt.figure(figsize=(10, 30))
        for i, (real, fake) in enumerate(zip(real_font, fake_font)):
            plt.subplot(40, 10, 2*i+1)
            plt.imshow(real.cpu().detach().numpy())
            plt.tick_params(axis='both', labelsize=0, length = 0)
            plt.subplot(40, 10, 2*i+2)
            plt.imshow(fake.cpu().detach().numpy())
            plt.tick_params(axis='both', labelsize=0, length = 0)
        global result_path, epochs, conv_dim
        plt.savefig(result_path.format(epochs, conv_dim))
        plt.close()
    
    latent_path = 'latent_convAE_z_epoch_{}_dim_{}.pkl'
    @trainer.on(Events.COMPLETED)
    def plot_latent_vectors(engine):
        evaluator.run(test_loader)
        real, fake, latent_vectors = evaluator.state.output
        print(latent_vectors.shape)
        # plt.figure()
        real = real.cpu().detach().numpy()
        fake = fake.cpu().detach().numpy()
        latent_vectors = latent_vectors.cpu().detach().numpy()
        data = {'real': real,
                'fake': fake,
                'latent': latent_vectors}
        # for i in range(len(latent_vectors)):
        #     plt.plot(latent_vectors[i, 0], latent_vectors[i, 1], marker='o')
        # plt.plot(latent_vectors[:, 0], latent_vectors[:, 1], marker='.')
        # plt.savefig('latent_vectors_for_category_layers.png')
        # plt.close()
        global latent_path, epochs, conv_dim
        with open(latent_path.format(epochs, conv_dim), 'wb') as f:
            pickle.dump(data, f)
    trainer.run(train_loader, max_epochs=epochs)
