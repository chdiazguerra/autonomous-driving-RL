import torch
from torch import nn
import pytorch_lightning as pl

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                            kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)
    
class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0, last=False):
        super().__init__()
        if not last:
            self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                       kernel_size, stride, padding,
                                                       output_padding),
                                        nn.BatchNorm2d(out_channels),
                                        nn.ReLU(inplace=True))
        else:
            self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                         kernel_size, stride, padding,
                                                            output_padding))
    def forward(self, x):
        return self.deconv(x)
    
class EncoderVAE(nn.Module):
    def __init__(self, input_size=(256, 256), emb_size=256):
        super().__init__()
        self.conv1 = ConvBlock(4, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = ConvBlock(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = ConvBlock(256, 64, kernel_size=4, stride=2, padding=1)

        self.y_final = input_size[0] // (2 ** 5)
        self.x_final = input_size[1] // (2 ** 5)

        self.fc = nn.Linear(64*self.x_final*self.y_final, emb_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
    
class DecoderVae(nn.Module):
    def __init__(self, input_size=(256, 256), emb_size=256, out_ch=4):
        super().__init__()
        self.y_inicial = input_size[0] // (2 ** 5)
        self.x_inicial = input_size[1] // (2 ** 5)

        self.fc = nn.Linear(emb_size, 64*self.x_inicial*self.y_inicial)

        self.deconv1 = DeConvBlock(64, 256, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv2 = DeConvBlock(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv3 = DeConvBlock(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv4 = DeConvBlock(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv5 = DeConvBlock(32, out_ch, kernel_size=4, stride=2, padding=1, output_padding=0, last=True)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, self.y_inicial, self.x_inicial)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = torch.sigmoid(self.deconv5(x)).squeeze()
        return x
    
class VAE(pl.LightningModule):
    def __init__(self, input_size=(256, 256), emb_size=256, lr=1e-3, out_ch=4):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = EncoderVAE(input_size, emb_size)
        self.decoder = DecoderVae(input_size, emb_size, out_ch)
        self.fc_mu = nn.Linear(emb_size, emb_size)
        self.fc_var = nn.Linear(emb_size, emb_size)
        self.lr = lr
        
    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.sample()
        return z
    
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded 
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = nn.functional.mse_loss(x_hat, y)

        # kl
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = kld_loss + recon_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded 
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = nn.functional.mse_loss(x_hat, y)

        # kl
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = kld_loss + recon_loss

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
