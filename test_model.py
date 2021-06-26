# %%
import os
from datetime import datetime
from utils.dataloader import DataPartitions, DataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib as mat

from models.resnet_ae import ResNetAE
import torch as th
import torch.optim as optim
import torch.nn as nn
import pytorch_ssim
from torch.autograd import Variable

mat.use("Agg") # headless mode
#mat.rcParams['text.color'] = 'w'

# -------------- Functions

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def reverse_ssim(y, y_true):
    tot_ssim = 0
    # for each sequence
    for i, y_pred in enumerate(y):
        # for each frame
        for j, frame in enumerate(y_pred[0]):
            y_h_pred = Variable(th.unsqueeze(th.unsqueeze(frame, 0), 0))
            y_h_true = Variable(th.unsqueeze(th.unsqueeze(y_true[i, 0, j], 0), 0))

            if th.cuda.is_available():
                y_h_pred = y_h_pred.cuda()
                y_h_true = y_h_true.cuda()

            tot_ssim += pytorch_ssim.ssim(y_h_pred, y_h_true)

    return 1 / (tot_ssim / len(y))

# Torch over CUDA
if th.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = th.device(dev)

# -------------- Setting up the run

num_run = len(os.listdir("runs/")) + 1
now = datetime.now()
foldername = "{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
os.mkdir("runs/" + foldername)

# -------------- Data definition

plotsize = 15

partitions = DataPartitions(
    past_frames=6,
    future_frames=4,
    root="../datasets/baganza/",
    partial=0.4
)

arda_ds = DataGenerator(
    root="../datasets/baganza/",
    dataset_partitions=partitions.get_partitions(),
    past_frames=partitions.past_frames,
    future_frames=partitions.future_frames,
    input_dim=(partitions.past_frames, 256, 256, 3),
    output_dim=(partitions.future_frames, 256, 256, 2),
    batch_size=4,
    n_channels=1,
    buffer_size=1e3,
    buffer_memory=100,
    downsampling=False,
)

X = arda_ds.get_X()
Y = arda_ds.get_Y()

X[X > 10e5] = 0
Y[Y > 10e5] = 0


# -------------- Data preprocessing

# Shuffle batches
X, Y = unison_shuffled_copies(X, Y)

print("DEP min: {}\nVEL min: {}\nBTM min: {}".format(
    np.min(X[:, :, :, :, :, 0]),
    np.min(X[:, :, :, :, :, 1]),
    np.min(X[:, :, :, :, :, 2])
))

# Load on GPU and convert to channel-first (for Torch)
X = th.Tensor(X).to(device)
Y = th.Tensor(Y).to(device)

X = X.permute(0, 1, 5, 2, 3, 4)
Y = Y.permute(0, 1, 5, 2, 3, 4)

# -------------- Model

net = ResNetAE(channels=3).to(device)

# Verbose network forward pass to diplay the architecture
print(net(th.Tensor(np.random.random((16, 3, 6, 128, 128))).to(device), True).shape)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Loading model weights from previous training
net.load_state_dict(th.load("runs/10_18_06_2021_14_20_18/model.weights"))
net.eval()

losses = []
errors = []
test_errors = []

epochs = 50

print(net.eval(X[0]))

'''
plt.clf()
for i, batch in enumerate(X):

    optimizer.zero_grad()

    outputs = net(batch)

    loss = criterion(outputs, Y[i])
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()

    losses.append(loss.item())

    # print("batch {} - loss {}".format(i, loss.item()))

    if i == len(X) - 1:
        # randomly pick a test batch and compute it
        i = np.random.randint(len(Y), size=1)

        train_out = net(X[i][0])
        test_out = net(X[i][0])

        # train_err = reverse_ssim(train_out, Y[i]).item()
        # test_err = reverse_ssim(test_out, Y[i]).item()
        train_err = 0
        test_err = 0

        test_errors.append(test_err)
        errors.append(train_err)

        print('[%d, %5d] train_err: %.3f \t  test_err: %.3f \t avg_loss: %.3f' %
              (epoch, i, train_err, test_err, np.mean(losses)))

    if epoch == epochs-1:
        i = np.random.randint(len(X))
        outputs = net(X[i])

        # ------------------------------
        fig, axs = plt.subplots(1, outputs[0, 0].shape[0], figsize=(plotsize, plotsize))

        for ax in axs:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        for i, frame in enumerate(outputs[0, 0]):
            axs[i].matshow(frame.cpu().detach().numpy())

        plt.show()
        plt.savefig("runs/" + foldername + "/last_epoch_prediction.png")
        # ------------------------------

        # if epoch % 10 == 0:
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 2000))
        #    running_loss = 0.0

th.save(net.state_dict(), "runs/" + foldername + "/model.weights")
print("[!] Training completed, model weights saved")

# loss plot
plt.clf()
plt.title("loss")
plt.plot(range(len(losses)), losses)
plt.show()
plt.savefig("runs/" + foldername + "/loss.png")

# ssim plot
plt.clf()
plt.title("relative error")
plt.plot(range(len(errors)), errors, label="train")
plt.plot(range(len(test_errors)), test_errors, label="test")
plt.legend()
plt.show()
plt.savefig("runs/" + foldername + "/ssim_error.png")
'''
