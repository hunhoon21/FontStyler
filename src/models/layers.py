import torch
from torch import nn
from torch.nn import functional as F
from .function import conv2d, deconv2d, batch_norm, lrelu, dropout

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
                 input_category_size=5, input_alpha_size=52, input_font_size=128 * 128,
                 z_size=64):
        super(Encoder_base, self).__init__()
        input_size = input_category_size + input_alpha_size + input_font_size

        self.efc1 = nn.Linear(input_size, 8192)
        self.efc2 = nn.Linear(8192, 4096)
        self.efc3 = nn.Linear(4096, 2048)
        self.efc4 = nn.Linear(2048, 1024)
        self.efc5 = nn.Linear(1024, 256)
        self.efc6 = nn.Linear(256, 128)
        self.efc7 = nn.Linear(128, z_size)

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
    def __init__(self, z_latent_size, z_category_size, z_alpha_size, output_font_size=128 * 128):
        super(Decoder_base, self).__init__()

        z_size = z_latent_size + z_category_size + z_alpha_size
        self.dfc1 = nn.Linear(z_size, 128)
        self.dfc2 = nn.Linear(128, 256)
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

class Encoder_conv(nn.Module):
    
    def __init__(self, img_dim=1, conv_dim=64):
        super(Encoder_conv, self).__init__()
        self.conv1 = conv2d(img_dim, conv_dim, k_size=5, stride=2, pad=2, dilation=2, lrelu=False, bn=False)
        self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=4, pad=2, dilation=2)
        self.conv3 = conv2d(conv_dim*2, conv_dim*4, k_size=4, stride=4, pad=1, dilation=1)
        self.conv4 = conv2d(conv_dim*4, conv_dim*8)
        self.conv5 = conv2d(conv_dim*8, conv_dim*8)
    
    def forward(self, images):
        # |images| = (batch, img, img)
        # print(images.shape)
        images = images.unsqueeze(dim=1)
        # |images| = (batch, 1, 128, 128)
        # print(images.shape)
        e1 = self.conv1(images)
        # |e1| = (batch, conv_dim, 64, 64)
        # print(e1.shape)
        e2 = self.conv2(e1)
        # |e2| = (batch, conv_dim*2, 16, 16)
        # print(e2.shape)
        e3 = self.conv3(e2)
        # |e3| = (batch, conv_dim*4, 4, 4)
        # print(e3.shape)
        e4 = self.conv4(e3)
        # |e4| = (batch, conv_dim*8, 2, 2)
        # print(e4.shape)
        encoded_source = self.conv5(e4)
        # |encoded_source| = (batch, conv_dim*8, 1, 1)
        # print(encoded_source.shape)
        
        return encoded_source

class FC_conv_en(nn.Module):
    def __init__(self, z_dim):
        super(FC_conv_en, self).__init__()
        self.fc1 = nn.Linear(z_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)
    def forward(self, z):
        # |z| = (batch, conv_dim*8, 1, 1)
        z = F.relu(z.squeeze(dim=3).squeeze(dim=2))
        # |z| = (batch, conv_dim*8)
        z = F.relu(self.fc1(z))
        # |z| = (batch, 32)
        z = F.relu(self.fc2(z))
        # |z| = (batch, 2)
        z = self.fc3(z)
        return z
        
class FC_conv_de(nn.Module):
    def __init__(self, z_dim):
        super(FC_conv_de, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 64)
        self.fc3 = nn.Linear(64, z_dim)
    def forward(self, z):
        # |z| = (batch, 2)
        z = F.relu(self.fc1(z))
        # |z| = (batch, 32)
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        # |z| = (batch, conv_dim*8)
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        return z
    
class Decoder_conv(nn.Module):
    
    def __init__(self, img_dim=1, embedded_dim=640, conv_dim=64):
        super(Decoder_conv, self).__init__()
        self.deconv1 = deconv2d(conv_dim*8, conv_dim*8, k_size=4, dilation=2, stride=2)
        self.deconv2 = deconv2d(conv_dim*8, conv_dim*4, k_size=4, dilation=2, stride=2)
        self.deconv3 = deconv2d(conv_dim*4, conv_dim*2, k_size=6, dilation=2, stride=4)
        self.deconv4 = deconv2d(conv_dim*2, conv_dim*1, k_size=6, dilation=2, stride=4)
        self.deconv5 = deconv2d(conv_dim*1, img_dim, k_size=4, dilation=2, stride=2, bn=False)
    
    def forward(self, embedded):
        # |embedded| = (batch, conv_dim*8, 1, 1)
        d1 = self.deconv1(embedded)
        # |d1| = (batch, conv_dim*8, 2, 2)
        # print( 1.shape)
        d2 = self.deconv2(d1)
        # |d2| = (batch, conv_dim*4, 4, 4)
        # print(d2.s hape)
        d3 = self.deconv3(d2)
        # |d3| = (batch, conv_dim*2, 16, 16)
        # print(d3. shape)
        d4 = self.deconv4(d3)
        # |d4| = (batch, conv_dim*1, 64, 64)
        # print(d4 .shape)
        d5 = self.deconv5(d4)        
        # |d5| = (batch, 1, 128, 128)
        # print(d5. shape)
        fake_target = d5
        # |fake_target| = (batch_size, 1, img, img)
        fake_target = fake_target.squeeze(dim=1)
        # |fake_target| = (batch_size, img, img)
        
        return fake_target
    

class Encoder_category(nn.Module):
    def __init__(self, 
                 input_font_size=128*128,
                 z_size=2):
        super(Encoder_category, self).__init__()
        input_size = input_font_size
        
        self.efc1 = nn.Linear(input_size, 8192)
        self.efc2 = nn.Linear(8192, 2048)
        self.efc3 = nn.Linear(2048, 1024)
        self.efc4 = nn.Linear(1024, 256)
        self.efc5 = nn.Linear(256, z_size)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.efc1(x))
        x = F.relu(self.efc2(x))
        x = F.relu(self.efc3(x))
        x = F.relu(self.efc4(x))
        z = self.efc5(x)
        return z
    
class Decoder_category(nn.Module):
    def __init__(self, z_latent_size, output_font_size=128*128):
        super(Decoder_category, self).__init__()
        
        z_size = z_latent_size
        self.dfc1 = nn.Linear(z_size, 256)
        self.dfc2 = nn.Linear(256, 1024)
        self.dfc3 = nn.Linear(1024, 2048)
        self.dfc4 = nn.Linear(2048, 8192)
        self.dfc5 = nn.Linear(8192, output_font_size)
        
    def forward(self, z):
        
        z = F.relu(self.dfc1(z))
        z = F.relu(self.dfc2(z))
        z = F.relu(self.dfc3(z))
        z = F.relu(self.dfc4(z))
        x_hat = self.dfc5(z)
        return x_hat
    
    
class Encoder_category(nn.Module):
    
    def __init__(self, img_dim=8, conv_dim=64):
        super(Encoder_category, self).__init__()
        self.conv1 = conv2d(img_dim, conv_dim, k_size=5, stride=2, pad=2, dilation=2, lrelu=False, bn=False)
        self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=4, pad=2, dilation=2)
        self.conv3 = conv2d(conv_dim*2, conv_dim*4, k_size=4, stride=4, pad=1, dilation=1)
        self.conv4 = conv2d(conv_dim*4, conv_dim*8)
        self.conv5 = conv2d(conv_dim*8, conv_dim*8)
    
    def forward(self, images):
        # |images| = (batch, 8, img, img)
        # print(images.shape)
        # images = images.unsqueeze(dim=1)
        # |images| = (batch, 1, 128, 128)
        # print(images.shape)
        e1 = self.conv1(images)
        # |e1| = (batch, conv_dim, 64, 64)
        # print(e1.shape)
        e2 = self.conv2(e1)
        # |e2| = (batch, conv_dim*2, 16, 16)
        # print(e2.shape)
        e3 = self.conv3(e2)
        # |e3| = (batch, conv_dim*4, 4, 4)
        # print(e3.shape)
        e4 = self.conv4(e3)
        # |e4| = (batch, conv_dim*8, 2, 2)
        # print(e4.shape)
        encoded_source = self.conv5(e4)
        # |encoded_source| = (batch, conv_dim*8, 1, 1)
        # print(encoded_source.shape)
        
        return encoded_source
    
class Decoder_category(nn.Module):
    
    def __init__(self, img_dim=8, embedded_dim=640, conv_dim=64):
        super(Decoder_category, self).__init__()
        self.deconv1 = deconv2d(conv_dim*8, conv_dim*8, k_size=4, dilation=2, stride=2)
        self.deconv2 = deconv2d(conv_dim*8, conv_dim*4, k_size=4, dilation=2, stride=2)
        self.deconv3 = deconv2d(conv_dim*4, conv_dim*2, k_size=6, dilation=2, stride=4)
        self.deconv4 = deconv2d(conv_dim*2, conv_dim*1, k_size=6, dilation=2, stride=4)
        self.deconv5 = deconv2d(conv_dim*1, img_dim, k_size=4, dilation=2, stride=2, bn=False)
    
    def forward(self, embedded):
        # |embedded| = (batch, conv_dim*8, 1, 1)
        d1 = self.deconv1(embedded)
        # |d1| = (batch, conv_dim*8, 2, 2)
        # print( 1.shape)
        d2 = self.deconv2(d1)
        # |d2| = (batch, conv_dim*4, 4, 4)
        # print(d2.s hape)
        d3 = self.deconv3(d2)
        # |d3| = (batch, conv_dim*2, 16, 16)
        # print(d3. shape)
        d4 = self.deconv4(d3)
        # |d4| = (batch, conv_dim*1, 64, 64)
        # print(d4 .shape)
        d5 = self.deconv5(d4)        
        # |d5| = (batch, 1, 128, 128)
        # print(d5. shape)
        fake_target = d5
        # |fake_target| = (batch_size, 8, img, img)
        # fake_target = fake_target.squeeze(dim=1)
        # |fake_target| = (batch_size, img, img)
        
        return fake_target
    
    
class Encoder_conv_variational(nn.Module):
    def __init__(self, img_dim=1, conv_dim=128):
        super(Encoder_conv_variational, self).__init__()
        
        self.conv1 = conv2d(img_dim+1, conv_dim, k_size=5, stride=2, pad=2, dilation=2, lrelu=False, bn=False)
        self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=4, pad=2, dilation=2)
        self.conv3 = conv2d(conv_dim*2, conv_dim*4, k_size=4, stride=4, pad=1, dilation=1)
        self.conv4 = conv2d(conv_dim*4, conv_dim*8)
        self.conv5 = conv2d(conv_dim*8, conv_dim*8)
        self.fc2mu = nn.Linear(conv_dim*8, conv_dim*4)
        self.fc2logvar = nn.Linear(conv_dim*8, conv_dim*4)
    
    def forward(self, images):
        # |images| = (batch, 2, img, img)
        # print(images.shape)
        # images = images.unsqueeze(dim=1)
        # |images| = (batch, 2, 128, 128)
        # print(images.shape)
        e1 = self.conv1(images)
        # |e1| = (batch, conv_dim, 64, 64)
        # print(e1.shape)
        e2 = self.conv2(e1)
        # |e2| = (batch, conv_dim*2, 16, 16)
        # print(e2.shape)
        e3 = self.conv3(e2)
        # |e3| = (batch, conv_dim*4, 4, 4)
        # print(e3.shape)
        e4 = self.conv4(e3)
        # |e4| = (batch, conv_dim*8, 2, 2)
        # print(e4.shape)
        e5 = F.leaky_relu(self.conv5(e4), 0.2)
        # |e5| = (batch, conv_dim*8, 1, 1)
        # print(encoded_source.shape)
        e5 = e5.view(e5.shape[0], -1)
        # |e5| = (batch, conv_dim*8)
        mu, logvar = self.fc2mu(e5), self.fc2logvar(e5)
        # |mu|, |logvar| = (batch, conv_dim*4)
        z = self._reparameterize(mu, logvar)
        # |z| = (batch, conv_dim*4)
        
        return z, mu, logvar
    
    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(), device=mu.device)
        
        z = mu + std * esp
        
        return z
    
class Decoder_conv_variational(nn.Module):
    
    def __init__(self, img_dim=1, embedded_dim=640, conv_dim=128):
        super(Decoder_conv_variational, self).__init__()
        
        self.fc2conv = nn.Linear(conv_dim*4 + 128 + 128, conv_dim*8)
        self.deconv1 = deconv2d(conv_dim*8, conv_dim*8, k_size=4, dilation=2, stride=2)
        self.deconv2 = deconv2d(conv_dim*8, conv_dim*4, k_size=4, dilation=2, stride=2)
        self.deconv3 = deconv2d(conv_dim*4, conv_dim*2, k_size=6, dilation=2, stride=4)
        self.deconv4 = deconv2d(conv_dim*2, conv_dim*1, k_size=6, dilation=2, stride=4)
        self.deconv5 = deconv2d(conv_dim*1, img_dim, k_size=4, dilation=2, stride=2, bn=False)
    
    def forward(self, embedded):
        # |embedded| = (batch, conv_dim*4 + 128 + 128)
        embedded = (self.fc2conv(embedded)).unsqueeze(dim=2).unsqueeze(dim=3)
        # |embedded| = (batch, conv_dim*8, 1, 1)
        d1 = self.deconv1(embedded)
        # |d1| = (batch, conv_dim*8, 2, 2)
        # print( 1.shape)
        d2 = self.deconv2(d1)
        # |d2| = (batch, conv_dim*4, 4, 4)
        # print(d2.s hape)
        d3 = self.deconv3(d2)
        # |d3| = (batch, conv_dim*2, 16, 16)
        # print(d3. shape)
        d4 = self.deconv4(d3)
        # |d4| = (batch, conv_dim*1, 64, 64)
        # print(d4 .shape)
        d5 = self.deconv5(d4)        
        # |d5| = (batch, 1, 128, 128)
        # print(d5. shape)
        fake_target = d5
        # |fake_target| = (batch_size, 1, img, img)
        fake_target = fake_target.squeeze(dim=1)
        # |fake_target| = (batch_size, img, img)
        
        return fake_target
    
    
class Encoder_conv_base(nn.Module):
    def __init__(self, img_dim=1, conv_dim=128):
        super(Encoder_conv_base, self).__init__()
        
        self.conv1 = conv2d(img_dim, conv_dim, k_size=5, stride=2, pad=2, dilation=2, lrelu=False, bn=False)
        self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=4, pad=2, dilation=2)
        self.conv3 = conv2d(conv_dim*2, conv_dim*4, k_size=4, stride=4, pad=1, dilation=1)
        self.conv4 = conv2d(conv_dim*4, conv_dim*8)
        self.conv5 = conv2d(conv_dim*8, conv_dim*8)
    
    def forward(self, images):
        # |images| = (batch, img, img)
        if len(images.size())==3:
            images = images.unsqueeze(dim=1)
        # |images| = (batch, 1, 128, 128)
        # print(images.shape)
        e1 = self.conv1(images)
        # |e1| = (batch, conv_dim, 64, 64)
        # print(e1.shape)
        e2 = self.conv2(e1)
        # |e2| = (batch, conv_dim*2, 16, 16)
        # print(e2.shape)
        e3 = self.conv3(e2)
        # |e3| = (batch, conv_dim*4, 4, 4)
        # print(e3.shape)
        e4 = self.conv4(e3)
        # |e4| = (batch, conv_dim*8, 2, 2)
        # print(e4.shape)
        e5 = self.conv5(e4)
        # |e5| = (batch, conv_dim*8, 1, 1)
        
        return e5
    
    
class Decoder_conv_base(nn.Module):
    
    def __init__(self, img_dim=1, embedded_dim=128*8, conv_dim=128):
        super(Decoder_conv_base, self).__init__()
        
        self.deconv1 = deconv2d(conv_dim*8, conv_dim*8, k_size=4, dilation=2, stride=2)
        self.deconv2 = deconv2d(conv_dim*8, conv_dim*4, k_size=4, dilation=2, stride=2)
        self.deconv3 = deconv2d(conv_dim*4, conv_dim*2, k_size=6, dilation=2, stride=4)
        self.deconv4 = deconv2d(conv_dim*2, conv_dim*1, k_size=6, dilation=2, stride=4)
        self.deconv5 = deconv2d(conv_dim*1, img_dim, k_size=4, dilation=2, stride=2, bn=False)
    
    def forward(self, embedded):
        # |embedded| = (batch, conv_dim*8, 1, 1)
        d1 = self.deconv1(embedded)
        # |d1| = (batch, conv_dim*8, 2, 2)
        # print( 1.shape)
        d2 = self.deconv2(d1)
        # |d2| = (batch, conv_dim*4, 4, 4)
        # print(d2.s hape)
        d3 = self.deconv3(d2)
        # |d3| = (batch, conv_dim*2, 16, 16)
        # print(d3. shape)
        d4 = self.deconv4(d3)
        # |d4| = (batch, conv_dim*1, 64, 64)
        # print(d4 .shape)
        d5 = self.deconv5(d4)        
        # |d5| = (batch, 1, 128, 128)
        # print(d5. shape)
        fake_target = d5
        # |fake_target| = (batch_size, 1, img, img)
        fake_target = fake_target.squeeze(dim=1)
        # |fake_target| = (batch_size, img, img)
        
        return fake_target
    
class Encoder_conv_z(nn.Module):
    def __init__(self, img_dim=1, conv_dim=128):
        super(Encoder_conv_z, self).__init__()
        
        self.conv1 = conv2d(img_dim, conv_dim, k_size=5, stride=2, pad=2, dilation=2, lrelu=False, bn=False)
        self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=4, pad=2, dilation=2)
        self.conv3 = conv2d(conv_dim*2, conv_dim*4, k_size=4, stride=4, pad=1, dilation=1)
        self.conv4 = conv2d(conv_dim*4, conv_dim*8)
        self.conv5 = conv2d(conv_dim*8, conv_dim*8)
        self.fc = nn.Linear(conv_dim*8, conv_dim)
    
    def forward(self, images):
        # |images| = (batch, img, img)
        if len(images.size())==3:
            images = images.unsqueeze(dim=1)
        # |images| = (batch, 1, img, img)
        e1 = self.conv1(images)
        # |e1| = (batch, conv_dim, 64, 64)
        # print(e1.shape)
        e2 = self.conv2(e1)
        # |e2| = (batch, conv_dim*2, 16, 16)
        # print(e2.shape)
        e3 = self.conv3(e2)
        # |e3| = (batch, conv_dim*4, 4, 4)
        # print(e3.shape)
        e4 = self.conv4(e3)
        # |e4| = (batch, conv_dim*8, 2, 2)
        # print(e4.shape)
        e5 = F.leaky_relu(self.conv5(e4), 0.2)
        # |e5| = (batch, conv_dim*8, 1, 1)
        # print(encoded_source.shape)
        e5 = e5.view(e5.shape[0], -1)
        # |e5| = (batch, conv_dim*8)
        z = self.fc(e5)
        # |z| = (batch, conv_dim)
        
        return z
    
    
class Decoder_conv_z(nn.Module):
    
    def __init__(self, img_dim=1, embedded_dim=128, conv_dim=128):
        super(Decoder_conv_z, self).__init__()
        
        self.fc2conv = nn.Linear(conv_dim+128+128, conv_dim*8)
        self.deconv1 = deconv2d(conv_dim*8, conv_dim*8, k_size=4, dilation=2, stride=2)
        self.deconv2 = deconv2d(conv_dim*8, conv_dim*4, k_size=4, dilation=2, stride=2)
        self.deconv3 = deconv2d(conv_dim*4, conv_dim*2, k_size=6, dilation=2, stride=4)
        self.deconv4 = deconv2d(conv_dim*2, conv_dim*1, k_size=6, dilation=2, stride=4)
        self.deconv5 = deconv2d(conv_dim*1, img_dim, k_size=4, dilation=2, stride=2, bn=False)
    
    def forward(self, embedded):
        # |embedded| = (batch, conv_dim + 128 + 128)
        embedded = (self.fc2conv(embedded)).unsqueeze(dim=2).unsqueeze(dim=3)
        # |embedded| = (batch, conv_dim*8, 1, 1)
        d1 = self.deconv1(embedded)
        # |d1| = (batch, conv_dim*8, 2, 2)
        # print( 1.shape)
        d2 = self.deconv2(d1)
        # |d2| = (batch, conv_dim*4, 4, 4)
        # print(d2.s hape)
        d3 = self.deconv3(d2)
        # |d3| = (batch, conv_dim*2, 16, 16)
        # print(d3. shape)
        d4 = self.deconv4(d3)
        # |d4| = (batch, conv_dim*1, 64, 64)
        # print(d4 .shape)
        d5 = self.deconv5(d4)        
        # |d5| = (batch, 1, 128, 128)
        # print(d5. shape)
        fake_target = d5
        # |fake_target| = (batch_size, 1, img, img)
        fake_target = fake_target.squeeze(dim=1)
        # |fake_target| = (batch_size, img, img)
        
        return fake_target
    