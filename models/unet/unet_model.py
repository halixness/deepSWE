""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

class R_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(R_UNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # n_channels x 3_tiles x 16x16_img
        self.recurrent = nn.LSTM(512*48*48,1024)
        # 512 channels * width / factor * n_convolutions
        # self.projection = nn.Linear(512*256*3//(factor**4), 512*256//(factor**4))
        # problem: can't crop with a UNet structure

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        original_shape = x5.shape

        print(x5.shape)
        x5 = torch.flatten(x5, start_dim=1)
        print(x5.shape)
        x5 = torch.unsqueeze(x5, 0)
        print(x5.shape)
        x5 = self.recurrent(x5)
        print("LSTMed")
        x5 = torch.reshape(x5, original_shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
