import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder_base, Decoder_base
from .layers import Encoder_category, Decoder_category
from .layers import Encoder_conv, Decoder_conv, FC_conv_en, FC_conv_de
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
    
class AE_conv(nn.Module):
    def __init__(self, img_dim=1, conv_dim=64):
        super(AE_conv, self).__init__()
        self.Encoder = Encoder_conv(img_dim=img_dim, conv_dim=conv_dim)
        self.Decoder = Decoder_conv(img_dim=img_dim, embedded_dim=conv_dim*8, conv_dim=conv_dim)
    
    def forward(self, x):
        z = self.Encoder(x)
        x_hat = self.Decoder(z)
        
        return x_hat, z

class CA_conv(nn.Module):
    def __init__(self, img_dim=8, conv_dim=64):
        super(CA_conv, self).__init__()
        self.Encoder = Encoder_category(img_dim=img_dim, conv_dim=conv_dim)
        self.Decoder = Decoder_category(img_dim=img_dim, embedded_dim=conv_dim*8, conv_dim=conv_dim)
    
    def forward(self, x):
        z = self.Encoder(x)
        x_hat = self.Decoder(z)
        
        return x_hat, z
        
class AE_conv2(nn.Module):
    def __init__(self, img_dim=1, conv_dim=64):
        super(AE_conv2, self).__init__()
        self.Encoder = Encoder_conv(img_dim=img_dim, conv_dim=conv_dim)
        self.FC_en = FC_conv_en(conv_dim*8)
        self.FC_de = FC_conv_de(conv_dim*8)
        self.Decoder = Decoder_conv(img_dim=img_dim, embedded_dim=conv_dim*8, conv_dim=conv_dim)
    
    def forward(self, x):
        z_en = self.Encoder(x)
        z = self.FC_en(z_en)
        z_de = self.FC_de(z)       
        x_hat = self.Decoder(z_de)
        
        return x_hat, z

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