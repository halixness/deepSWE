# %%
import os
from datetime import datetime
from utils.data_lightning.otf import SWEDataset
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.patches as patches

import torch as th
from torch.autograd import Variable
import torch.nn as nn

from models.experiments.nfnets import seq2seq_NFLSTM
from models.ae import seq2seq_ConvLSTM

import argparse
import pytorch_ssim
import time

mat.use("Agg") # headless mode
#mat.rcParams['text.color'] = 'w'

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


parser.add_argument('-network', dest='network', default = "conv",
                    help='Type of architecture: conv/nfnet/...')
parser.add_argument('-test_size', dest='test_size', default = None,
                    help='Test size for the split')
parser.add_argument('-shuffle', dest='shuffle', default=True, type=str2bool,
                    help='Shuffle the dataset')
parser.add_argument('-multigpu', dest='multigpu', default=False, type=str2bool,
                    help='Supports multi-gpu models')
parser.add_argument('-filters', dest='filters', default=4, type=int,
                    help='number of hidden layers')
parser.add_argument('-in_channels', dest='in_channels', default=4, type=int,
                    help='number of input channels')
parser.add_argument('-out_channels', dest='out_channels', default=3, type=int,
                    help='number of input channels')                    
parser.add_argument('-tests', dest='n_tests', default=10, type=int,
                    help='number of tests to perform')
parser.add_argument('-weights', dest='weights_path', required=True,
                    help='model weights for testing')
parser.add_argument('-dset', dest='dataset_path',
                    help='path to a npy stored dataset')
parser.add_argument('-root', dest='root', required=True,
                    help='root path with the simulation files (cropped and stored in folders)')
parser.add_argument('-partial', dest='partial', default=None, type=float,
                    help='percentage of portion of dataset (to load partial, lighter chunks)')
parser.add_argument('-image_size', dest='image_size', default=256, type=int,
                    help='image size (width = height)')
parser.add_argument('-batch_size', dest='batch_size', default=4, type=int,
                    help='batch size')

parser.add_argument('-p', dest='past_frames', default=4, type=int,
                    help='number of past frames')       
parser.add_argument('-f', dest='future_frames', default=1, type=int, 
                    help='number of future frames')
parser.add_argument('-d', dest='dynamicity', default=1e-3, type=float,
                    help='dynamicity rate (to filter out "dynamic" sequences)')                                                                                                  
parser.add_argument('-bs', dest='buffer_size', default=1e3, type=float,
                    help='size of the cache memory (in entries)')
parser.add_argument('-t', dest='buffer_memory', default=100, type=int,
                    help='temporal length of the cache memory (in iterations)')                                                                                                  

args = parser.parse_args()

print("[~] Benchmark initialized, loading dataset...")

# -------------- Setting up the run

num_run = len(os.listdir("runs/")) + 1
now = datetime.now()
foldername = "eval_{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
os.mkdir("runs/" + foldername)

# -------------- Data definition
if th.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = th.device(dev)

plotsize = 15

# -------------- Model
# Loading model weights from previous training
print("[x] Loading model weights")

net = seq2seq_ConvLSTM.EncoderDecoderConvLSTM(nf=args.filters, in_chan=args.in_channels, out_chan=args.out_channels)

if args.multigpu:
    net = nn.DataParallel(net)

net.load_state_dict(
    th.load(args.weights_path)
)

net = net.to(device)
net.eval() # evaluation mode

print("[!] Successfully loaded weights from {}".format(args.weights_path))

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
    partial=args.partial,
    dynamicity=1e-1
)

for t in range(args.n_tests):

    print("-- Test {} running...".format(t), end="", flush=True)

    i = np.random.randint(len(dataset))       # random batch
    datapoint = dataset[i]

    while datapoint is None:
        i = np.random.randint(len(dataset))       # random sequence
        datapoint = dataset[i]

    print("\t data loaded!", flush=True)
    
    # b, t, c, h, w 
    x, y = datapoint

    x = x.to(device)
    y = y.to(device)

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
    axs[i + 1].matshow(outputs[0,0,0,:,:].cpu().detach().numpy())
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

    plt.matshow(outputs[0,0,0,:,:].cpu().detach().numpy())
    plt.savefig(test_dir + "/predicted.png")

    plt.matshow(y[0, 0, 0, :, :].cpu().detach().numpy())
    plt.savefig(test_dir + "/ground_truth.png")

    # Write stats
    text_file = open("runs/" + foldername + "/test_{}/scores.txt".format(t), "w")
    n = text_file.write("SSIM: {}\nL1: {}\nMSE:{}".format(curr_ssim, curr_l1, curr_l2))
    text_file.close()

    del x
    del y
    del img1
    del img2
    th.cuda.empty_cache()
    # ----------------

ssim_score = ssim_score/args.n_tests
l1_score = l1_score/args.n_tests
l2_score = l2_score/args.n_tests

stats = "SSIM: {}\nL1: {}\nMSE:{}\nAvg.Inference Time: {}".format(ssim_score, l1_score, l2_score, np.mean(inference_times))

text_file = open("runs/" + foldername + "/avg_score.txt", "w")
n = text_file.write("SSIM: {}\nL1: {}\nMSE:{}".format(ssim_score, l1_score, l2_score))
text_file.close()
