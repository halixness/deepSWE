# %%
import os
from datetime import datetime
from utils.data_lightning.otf import SWEDataset
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.patches as patches

import torch as th
from models_lightning.ae.main import deepSWE
from torch.autograd import Variable
import pytorch_ssim

import argparse
import pytorch_ssim
import time

mat.use("Agg")  # headless mode


# -------------- Functions

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# -------------------------------

parser = argparse.ArgumentParser(description='Tests a train model against a given dataset')

parser.add_argument('-root', dest='root', required=True,
                    help='root path with the simulation files (cropped and stored in folders)')
parser.add_argument('-weights', dest='weights', required=True,
                    help='path for network weights')
parser.add_argument('-past_frames', dest='past_frames', default=4, type=int,
                    help='number of past frames')
parser.add_argument('-future_frames', dest='future_frames', default=1, type=int,
                    help='number of future frames')
parser.add_argument('-partial', dest='partial', default=None, type=float,
                    help='percentage of portion of dataset (to load partial, lighter chunks)')
parser.add_argument('-batch_size', dest='batch_size', default=4, type=int,
                    help='Batch size')
parser.add_argument('-workers', dest='workers', default=4, type=int,
                    help='Number of cpu threads for the data loader')
parser.add_argument('-filtering', dest='filtering', default=False, type=str2bool,
                    help='Apply sequence dynamicity filtering')
parser.add_argument('-gpus', dest='gpus', default=1, type=int,
                    help='Number of GPUs')
parser.add_argument('-tests', dest='tests', default=3, type=int,
                    help='Number of tests to do')

args = parser.parse_args()

print("[~] Benchmark initialized, loading dataset...")

# -------------- Setting up the run

num_run = len(os.listdir("runs/")) + 1
now = datetime.now()
foldername = "eval_{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
os.mkdir("runs/" + foldername)

# -------------- Data definition
plotsize = 15

# -------------- Model
print("[x] Loading model weights")
net = deepSWE.load_from_checkpoint(args.weights, nf=24, in_chan=4, out_chan=3)
net.summarize(mode="full")

net.eval()
net.freeze()

# ------------------------------

inference_times = []

ssim = pytorch_ssim.SSIM()
l1 = th.nn.L1Loss()
l2 = th.nn.MSELoss()

ssim_score = 0
l1_score = 0
l2_score = 0

dataset = SWEDataset(
    root=args.root,
    past_frames=args.past_frames,
    future_frames=args.future_frames,
    partial=args.partial
)

for t in range(args.tests):

    print("-- Test {} running...".format(t), end="")

    i = np.random.randint(len(dataset))  # random batch
    datapoint = dataset[i]

    while datapoint is None:
        i = np.random.randint(len(dataset))  # random batch
        datapoint = dataset[i]

    print("\t data loaded!")

    # b, t, c, h, w
    x, y = datapoint
    x = th.FloatTensor(x)
    y = th.FloatTensor(y)

    start = time.time()
    # 1, t, c, h, w
    outputs = net(x, 1)
    end = time.time()
    inference_times.append(end - start)

    # 1, c, h, w
    img1 = Variable(outputs[0, :, 0, :, :].unsqueeze(0), requires_grad=False)
    img2 = Variable(y[0, 0, :, :, :].unsqueeze(0), requires_grad=True)

    curr_ssim = ssim(img1, img2)
    curr_l1 = l1(img1, img2)
    curr_l2 = l2(img1, img2)

    ssim_score += curr_ssim
    l1_score += curr_l1
    l2_score += curr_l2

    # ------------- Plotting
    test_dir = "runs/" + foldername + "/test_{}".format(t)
    os.mkdir(test_dir)

    fig, axs = plt.subplots(1, x.shape[1] + 2, figsize=(plotsize, plotsize))

    for ax in axs:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # Past frames
    for i, frame in enumerate(x[0]):
        axs[i].title.set_text('t={}'.format(i))
        axs[i].matshow(frame[0].cpu().detach().numpy())
        rect = patches.Rectangle((256, 256), 256, 256, linewidth=1, edgecolor='r', facecolor='none')
        axs[i].add_patch(rect)

    # Prediction
    axs[i + 1].matshow(outputs[0, 0, 0, :, :].cpu().detach().numpy())
    rect = patches.Rectangle((256, 256), 256, 256, linewidth=1, edgecolor='r', facecolor='none')
    axs[i + 1].add_patch(rect)
    axs[i + 1].title.set_text('Predicted')

    # Ground truth
    axs[i + 2].matshow(y[0, 0, 0, :, :].cpu().detach().numpy())
    rect = patches.Rectangle((256, 256), 256, 256, linewidth=1, edgecolor='r', facecolor='none')
    axs[i + 2].add_patch(rect)
    axs[i + 2].title.set_text('Ground Truth')

    plt.savefig(test_dir + "/sequence.png")

    # Saving single images
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    for i, frame in enumerate(x[0]):
        plt.matshow(frame[0].cpu().detach().numpy())
        plt.savefig(test_dir + "/{}.png".format(i))

    plt.matshow(outputs[0, 0, 0, :, :].cpu().detach().numpy())
    plt.savefig(test_dir + "/predicted.png")

    plt.matshow(y[0, 0, 0, :, :].cpu().detach().numpy())
    plt.savefig(test_dir + "/ground_truth.png")

    # Write stats
    text_file = open("runs/" + foldername + "/test_{}/scores.txt".format(t), "w")
    n = text_file.write("SSIM: {}\nL1: {}\nMSE:{}".format(curr_ssim, curr_l1, curr_l2))
    text_file.close()

    # ----------------

ssim_score = ssim_score / args.tests
l1_score = l1_score / args.tests
l2_score = l2_score / args.tests

stats = "SSIM: {}\nL1: {}\nMSE:{}\nAvg.Inference Time: {}".format(ssim_score, l1_score, l2_score,
                                                                  np.mean(inference_times))

text_file = open("runs/" + foldername + "/avg_score.txt", "w")
n = text_file.write("SSIM: {}\nL1: {}\nMSE:{}".format(ssim_score, l1_score, l2_score))
text_file.close()