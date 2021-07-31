
import numpy as np
import torch as th
import torch.nn as nn

class ResNetBlock(nn.Module):

    def __init__(self, in_filters, filters, stride, kernel_size, padding):
        super(ResNetBlock, self).__init__()

        self.activation = nn.ReLU()
        self.c1 = nn.Conv2d(in_filters, filters, kernel_size, stride, padding=padding)
        self.c2 = nn.Conv2d(filters, filters, kernel_size, padding=padding)
        self.c3 = nn.Conv2d(in_filters, filters, (1, 1), stride)

        # self.bn = nn.BatchNorm3d(num_features=filters)

    def forward(self, x, ):
        residual = x

        y = self.c1(x)

        # y = self.bn(y)
        y = self.activation(y)
        y = self.c2(y)
        # y = self.bn(y)

        # reshape
        if residual.shape != y.shape:
            residual = self.c3(residual)
            # residual = self.bn(residual)

        return self.activation(residual + y)

# ---------------------------------------------------------------------------------

class ResNetAE(nn.Module):

    def __init__(self, channels):
        super(ResNetAE, self).__init__()

        kernel_size = 3
        padding = 1

        # ------ Encoder
        self.encoder = nn.ModuleList([
            nn.Conv2d(channels, 8, kernel_size=kernel_size, stride=1, padding=padding),
            nn.AvgPool2d(2),

            ResNetBlock(in_filters=8, filters=8, stride=1, kernel_size=kernel_size, padding=padding),

            nn.Conv2d(8, 16, 1),
            ResNetBlock(in_filters=16, filters=16, stride=1, kernel_size=kernel_size, padding=padding),

            nn.Conv2d(16, 32, 1),
            ResNetBlock(in_filters=32, filters=32, stride=1, kernel_size=kernel_size, padding=padding),

            nn.Conv2d(32, 64, 1),
            ResNetBlock(in_filters=64, filters=64, stride=1, kernel_size=kernel_size, padding=padding),
        ])

        # ------ Latent
        self.recurrent = nn.LSTM(100, 1024)


        # ------ Decoder
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(64, 32, 1, stride=2),
            nn.ConvTranspose2d(32, 16, 1, stride=(1, 1)),
            nn.ConvTranspose2d(16, 8, 1, stride=(1, 1)),
            nn.ConvTranspose2d(8, 8, 1, stride=(1, 1)),
            nn.ConvTranspose2d(8, 3, 1, stride=(1, 1)),
        ])

    def forward(self, x, summary=False):

        if summary:
            print("==== Model Summary ====")
            print("{:<15s}{:>4s}".format("Block", "Output shape"))

        # ------ Encoder
        for i, l in enumerate(self.encoder):
            x = l(x)
            if summary:
                print("{:<20s}{:>4s}".format(
                    str(l).split("(")[0],
                    str(x.shape).split("[")[1].split("]")[0]
                ))

        # ------ Latent
        print(x.shape)

        # ------ Decoder
        for i, l in enumerate(self.decoder):
            x = l(x)
            if summary:
                print("{:<20s}{:>4s}".format(
                    str(l).split("(")[0],
                    str(x.shape).split("[")[1].split("]")[0]
                ))

        return x

# ---------------------------------------------------------------------------------

def xavier_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        th.nn.init.xavier_uniform(m.weight.data)


def sqrt_weights(m):
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)