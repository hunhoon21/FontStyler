import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder_base, Decoder_base
from .layers import Encoder_category, Decoder_category

class AE_base(nn.Module):
    def __init__(self, 
                 category_size=5, 
                 alpha_size=52, 
                 font_size=256*256,
                 z_size=64,
                 ):
        super(AE_base, self).__init__()
        self.Encoder = Encoder_base(input_category_size=category_size,
                                    input_alpha_size=alpha_size,
                                    input_font_size=font_size,
                                    z_size=z_size)
        self.Decoder = Decoder_base(z_latent_size=z_size,
                                    z_category_size=category_size,
                                    z_alpha_size=alpha_size,
                                    output_font_size=font_size)
    
    def forward(self, x_font, alpha_vector, category_vector=None):
        origin_shape = x_font.shape
        x_font = x_font.view(x_font.shape[0], -1)
        alpha_vector = alpha_vector.view(alpha_vector.shape[0], -1)
        if category_vector is not None:
            category_vector = category_vector.view(category_vector.shape[0], -1)
        
        if category_vector is not None:
            x = torch.cat([x_font, category_vector, alpha_vector], dim=1)
        else:
            x = torch.cat([x_font, alpha_vector], dim=1)
        
        z_latent = self.Encoder(x)
        
        z_latent = z_latent.view(z_latent.shape[0], -1)
        if category_vector is not None:    
            z = torch.cat([z_latent, category_vector, alpha_vector], dim=1)
        else:
            z = torch.cat([z_latent, alpha_vector], dim=1)
        
        x_hat = self.Decoder(z)
        x_hat = x_hat.view(origin_shape)
        
        return x_hat, z_latent
    
    
class AE_category(nn.Module):
    def __init__(self, 
                 font_size=128*128,
                 z_size=64,
                 ):
        super(AE_category, self).__init__()
        self.Encoder = Encoder_category(input_font_size=font_size,
                                        z_size=z_size)
        self.Decoder = Decoder_category(z_latent_size=z_size,
                                        output_font_size=font_size)
    
    def forward(self, x_font):
        origin_shape = x_font.shape
        x = x_font.view(x_font.shape[0], -1)
        
        z_latent = self.Encoder(x)
        
        z = z_latent.view(z_latent.shape[0], -1)

        x_hat = self.Decoder(z)
        x_hat = x_hat.view(origin_shape)
        
        return x_hat, z_latent