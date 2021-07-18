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

print("[x] Benchmark initialized, loading dataset...")

# -------------- Setting up the run

num_run = len(os.listdir("runs/")) + 1
now = datetime.now()
foldername = "eval_{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
os.mkdir("runs/" + foldername)

# -------------- Data definition

plotsize = 15

partitions = DataPartitions(
    past_frames=6,
    future_frames=4,
    root="../datasets/arda/mini/",
    partial=0.1
)

arda_ds = DataGenerator(
    root="../datasets/arda/mini/",
    dataset_partitions=partitions.get_partitions(),
    past_frames=partitions.past_frames,
    future_frames=partitions.future_frames,
    input_dim=(partitions.past_frames, 256, 256, 3),
    output_dim=(partitions.future_frames, 256, 256, 2),
    batch_size=4,
    buffer_size=1e3,
    buffer_memory=100,
    downsampling=False,
)

X = arda_ds.get_X()
Y = arda_ds.get_Y()

X[X > 10e5] = 0
Y[Y > 10e5] = 0


# -------------- Data preprocessing

print("[x] Preprocessing started...")

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

print("[x] Initializing model...")

net = ResNetAE(channels=3).to(device)

# Verbose network forward pass to diplay the architecture
print(net(th.Tensor(np.random.random((16, 3, 6, 128, 128))).to(device), True).shape)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

# Loading model weights from previous training
weights_path = "runs/10_18_06_2021_14_20_18/model.weights"
net.load_state_dict(th.load(weights_path))
net.eval()

losses = []
errors = []
test_errors = []

epochs = 50

j = np.random.randint(len(X))       # random batch
k = np.random.randint(len(X[j]))    # random datapoint
outputs = net(X[j])

print("[!] Successfully loaded weights from {}".format(weights_path))

# ------------------------------
fig, axs = plt.subplots(1, outputs[k, 0].shape[0], figsize=(plotsize, plotsize))

for ax in axs:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

for i, frame in enumerate(outputs[k, 0]):
    axs[i].matshow(frame.cpu().detach().numpy())

plt.show()
plt.savefig("runs/" + foldername + "/eval_prediction_{}.png".format(j))

# ------------------------------

print("[x] Starting inference...")
true_means = []
predicted_means = []
for i, frame in enumerate(X[j][k]):
    true_means.append(frame.cpu().detach().numpy().mean())
    predicted_means.append(outputs[k, 0, i].cpu().detach().numpy().mean())

plt.clf()
plt.plot(range(len(true_means)), true_means,  "-b", label="True frames mean")
plt.plot(range(len(true_means)), true_means,  "*")

plt.plot(range(len(predicted_means)), predicted_means,  "-g", label="Predicted frames mean")
plt.plot(range(len(predicted_means)), predicted_means,  "*")
plt.grid()
plt.legend()
plt.savefig("runs/" + foldername + "/eval_means_{}.png".format(j))
