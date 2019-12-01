import torch
from torch import nn
from torch.nn import functional as F

'''
TODO : 
1. How to deal with difference size of kinds of input
2. Make a FC module
3. How to apply to CNN Layers
4. Search Another models
5. Consider Our structures
'''
class Encoder_base(nn.Module):
    def __init__(self, 
                 input_category_size=5, input_alpha_size=52, input_font_size=128*128,
                 z_size=2):
        super(Encoder_base, self).__init__()
        input_size = input_category_size + input_alpha_size + input_font_size
        
        self.efc1 = nn.Linear(input_size, 8192)
        self.efc2 = nn.Linear(8192, 4096)
        self.efc3 = nn.Linear(4096, 2048)
        self.efc4 = nn.Linear(2048, 1024)
        self.efc5 = nn.Linear(1024, 256)
        self.efc6 = nn.Linear(256, 64)
        self.efc7 = nn.Linear(64, z_size)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.efc1(x))
        x = F.relu(self.efc2(x))
        x = F.relu(self.efc3(x))
        x = F.relu(self.efc4(x))
        x = F.relu(self.efc5(x))
        x = F.relu(self.efc6(x))
        z = self.efc7(x)
        return z
    
class Decoder_base(nn.Module):
    def __init__(self, z_latent_size, z_category_size, z_alpha_size, output_font_size=128*128):
        super(Decoder_base, self).__init__()
        
        z_size = z_latent_size + z_category_size + z_alpha_size
        self.dfc1 = nn.Linear(z_size, 64)
        self.dfc2 = nn.Linear(64, 256)
        self.dfc3 = nn.Linear(256, 1024)
        self.dfc4 = nn.Linear(1024, 2048)
        self.dfc5 = nn.Linear(2048, 4096)
        self.dfc6 = nn.Linear(4096, 8192)
        self.dfc7 = nn.Linear(8192, output_font_size)
        
    def forward(self, z):
        
        z = F.relu(self.dfc1(z))
        z = F.relu(self.dfc2(z))
        z = F.relu(self.dfc3(z))
        z = F.relu(self.dfc4(z))
        z = F.relu(self.dfc5(z))
        z = F.relu(self.dfc6(z))
        x_hat = self.dfc7(z)
        return x_hat