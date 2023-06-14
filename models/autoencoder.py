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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super().__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                       kernel_size, stride, padding,
                                                       output_padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))
    def forward(self, x):
        return self.deconv(x)

class Encoder(nn.Module):
    def __init__(self, input_size=(256, 256), emb_size=200):
        super().__init__()
        self.conv1 = ConvBlock(4, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = ConvBlock(256, 64, kernel_size=3, stride=2, padding=1)

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
    
class Decoder(nn.Module):
    def __init__(self, input_size=(256, 256), num_classes=23):
        super().__init__()
        self.y_inicial = input_size[0] // (2 ** 5)
        self.x_inicial = input_size[1] // (2 ** 5)

        self.fc = nn.Linear(200, 64*self.x_inicial*self.y_inicial)

        self.deconv1 = DeConvBlock(64, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = DeConvBlock(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = DeConvBlock(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv4 = DeConvBlock(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv5 = DeConvBlock(32, 16, kernel_size=6, stride=2, padding=2, output_padding=0)
        self.convfinal = ConvBlock(16, num_classes, kernel_size=3, stride=1, padding=1)

        self.fc_data = nn.Linear(200, 3)
        self.fc_junction = nn.Linear(200, 1)

    def forward(self, x):
        data = self.fc_data(x)
        junction = self.fc_junction(x)
        x = self.fc(x)
        x = x.view(-1, 64, self.y_inicial, self.x_inicial)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.convfinal(x)
        return x, data, junction
    

class Autoencoder(pl.LightningModule):
    def __init__(self, input_size=(256, 256), emb_size=200, num_classes=29, lr=1e-3, weights=(0.6, 0.2, 0.2)):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(input_size, emb_size)
        self.decoder = Decoder(input_size, num_classes)
        self.lr = lr
        self.weights = weights

    def forward(self, x):
        x = self.encoder(x)
        x, data, junction = self.decoder(x)
        return x, data, junction
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, sem, data, junction = batch

        sem_hat, data_hat, junction_hat = self(x)
        loss_sem = nn.functional.cross_entropy(sem_hat, sem)
        loss_data = nn.functional.mse_loss(data_hat, data)
        loss_junction = nn.functional.binary_cross_entropy_with_logits(junction_hat, junction)
        loss = self.weights[0]*loss_sem + self.weights[1]*loss_data + self.weights[2]*loss_junction
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, sem, data, junction = batch

        sem_hat, data_hat, junction_hat = self(x)
        loss_sem = nn.functional.cross_entropy(sem_hat, sem)
        loss_data = nn.functional.mse_loss(data_hat, data)
        loss_junction = nn.functional.binary_cross_entropy_with_logits(junction_hat, junction)
        loss = self.weights[0]*loss_sem + self.weights[1]*loss_data + self.weights[2]*loss_junction
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss