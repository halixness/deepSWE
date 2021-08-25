""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

class RecNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size, bilinear=True):
        super(RecNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.img_size = img_size

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.factor = factor
        self.down4 = Down(512, 1024 // factor)

        h_conv_size = 512 * (img_size//(factor**4))**2
        self.recurrent = nn.LSTM(h_conv_size, h_conv_size, batch_first=True)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def single_downConv(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5

    def single_upConv(self, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        hidden = self.outc(x)

        return hidden

    def forward(self, x):
        
        X = None

        # Encodes single frames
        for i in range(x.shape[1]):
            # all i-th frames across the batch
            h = self.single_downConv(x[:,i,:,:,:])
            
            if X == None: X = h.unsqueeze(1)
            else: X = torch.cat((X, h.unsqueeze(1)), 1)

        # Recurrent level: out, (h,c)
        X = self.recurrent(X.flatten(2))[0]
        X = X[:, -1] # many-to-one

        X = torch.reshape(
            X, 
            (X.shape[0], 1, 512, self.img_size//(self.factor**4), (self.img_size//(self.factor**4)))
        )

        # Decoding
        X = self.single_upConv(X)

        # PROBLEM: UNet needs the encoder layers' output for the decoder skip connections.
        #          It is difficult to memorize these information for each forward pass (frame).
    
        return X

    
