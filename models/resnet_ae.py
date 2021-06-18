
import numpy as np
import torch as th
import torch.nn as nn

class ResNetBlock(nn.Module):

    def __init__(self, in_filters, filters, stride, kernel_size, padding):
        super(ResNetBlock, self).__init__()

        self.activation = nn.ReLU()
        self.c1 = nn.Conv3d(in_filters, filters, kernel_size, stride, padding=padding)
        self.c2 = nn.Conv3d(filters, filters, kernel_size, padding=padding)
        self.c3 = nn.Conv3d(in_filters, filters, (1, 1, 1), stride)

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

        self.layers = nn.ModuleList([
            nn.Conv3d(channels, 8, kernel_size=kernel_size, stride=1, padding=padding),
            nn.AvgPool3d((1, 2, 2)),

            ResNetBlock(in_filters=8, filters=8, stride=2, kernel_size=kernel_size, padding=padding),

            nn.Conv3d(8, 16, (1, 1, 1)),
            ResNetBlock(in_filters=16, filters=16, stride=2, kernel_size=kernel_size, padding=padding),

            nn.Conv3d(16, 32, (1, 1, 1)),
            ResNetBlock(in_filters=32, filters=32, stride=(2, 1, 1), kernel_size=kernel_size, padding=padding),

            nn.Conv3d(32, 64, (1, 1, 1)),
            ResNetBlock(in_filters=64, filters=64, stride=1, kernel_size=kernel_size, padding=padding),

            # ----------------------

            nn.ConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 2, 2)),
            # nn.BatchNorm3d(num_features=32),

            nn.ConvTranspose3d(32, 16, (2, 2, 2), stride=(2, 2, 2)),
            # nn.BatchNorm3d(num_features=16),

            nn.ConvTranspose3d(16, 8, (2, 2, 2), stride=(2, 2, 2)),
            # nn.BatchNorm3d(num_features=8),

            nn.ConvTranspose3d(8, 8, (1, 1, 1), stride=(1, 1, 1)),
            # nn.BatchNorm3d(num_features=8),

            nn.ConvTranspose3d(8, 2, (1, 1, 1), stride=(1, 1, 1)),
        ])

    def forward(self, x, summary=False):

        if summary:
            print("==== Model Summary ====")
            print("{:<15s}{:>4s}".format("Block", "Output shape"))

        for i, l in enumerate(self.layers):
            x = l(x)

            if summary:
                print("{:<20s}{:>4s}".format(
                    str(l).split("(")[0],
                    str(x.shape).split("[")[1].split("]")[0]
                ))

        return x

# ---------------------------------------------------------------------------------

def xavier_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        th.nn.init.xavier_uniform(m.weight.data)


def sqrt_weights(m):
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)