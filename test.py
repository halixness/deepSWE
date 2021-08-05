# %%
import os
from datetime import datetime
from utils.dataloader import DataPartitions, DataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib as mat

import torch as th
import torch.optim as optim
import torch.nn as nn
import pytorch_ssim
from torch.autograd import Variable

import argparse
import pytorch_ssim

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

    # -------------------------------

parser = argparse.ArgumentParser(description='Tests a train model against a given dataset')


parser.add_argument('-tf', dest='test_flight',
                    help='Test flight. Avoids creating a train folder for this session.')
parser.add_argument('-hidden_layers', dest='hidden_layers', default=4, type=int,
                    help='number of hidden layers')
parser.add_argument('-in_channels', dest='in_channels', default=4, type=int,
                    help='number of input channels')
parser.add_argument('-tests', dest='n_tests', default=10, type=int,
                    help='number of tests to perform')
parser.add_argument('-weights', dest='weights_path',
                    help='model weights for testing')
parser.add_argument('-dset', dest='dataset_path',
                    help='path to a npy stored dataset')
parser.add_argument('-r', dest='root',
                    help='root path with the simulation files (cropped and stored in folders)')
parser.add_argument('-p', dest='past_frames', default=1, type=int,
                    help='number of past frames')       
parser.add_argument('-f', dest='future_frames', default=1, type=int, 
                    help='number of future frames')       
parser.add_argument('-s', dest='partial', default=None, type=float,
                    help='percentage of portion of dataset (to load partial, lighter chunks)')                                                            
parser.add_argument('-i', dest='image_size', default=256, type=int,
                    help='image size (width = height)')
parser.add_argument('-b', dest='batch_size', default=4, type=int,
                    help='batch size') 
parser.add_argument('-d', dest='dynamicity', default=1e-3, type=float,
                    help='dynamicity rate (to filter out "dynamic" sequences)')                                                                                                  
parser.add_argument('-bs', dest='buffer_size', default=1e3, type=float,
                    help='size of the cache memory (in entries)')
parser.add_argument('-t', dest='buffer_memory', default=100, type=int,
                    help='temporal length of the cache memory (in iterations)')                                                                                                  
parser.add_argument('-ds', dest='downsampling', default=False, type=str2bool, nargs='?',
                    const=True, help='enable 2x downsampling (with gaussian filter)')  
parser.add_argument('-ls', dest='latent_size', default=1024, type=int,
                    help='latent size for the VAE')                                                                                                                                                      

args = parser.parse_args()

if args.root is None:
    print("required: please specify a root path: -r /path")
    exit()

if args.weights_path is None:
    print("required: please specify a path for the weights: -weights /path/model.weights")
    exit()

# -------------- 

if th.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = th.device(dev)

print("[x] Benchmark initialized, loading dataset...")

# -------------- Setting up the run

if args.test_flight is None:
    num_run = len(os.listdir("runs/")) + 1
    now = datetime.now()
    foldername = "eval_{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
    os.mkdir("runs/" + foldername)

# -------------- Data definition

plotsize = 15

# ---- No dataset file: load from dir
if args.dataset_path is None:

    partitions = DataPartitions(
        past_frames=args.past_frames,
        future_frames=args.future_frames,
        root=args.root,
        partial=args.partial
    )

    dataset = DataGenerator(
        root=args.root,
        dataset_partitions=partitions.get_partitions(),
        past_frames=partitions.past_frames, 
        future_frames=partitions.future_frames,
        input_dim=(partitions.past_frames, args.image_size, args.image_size, 4),
        output_dim=(partitions.future_frames, args.image_size, args.image_size, 3),
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        buffer_memory=args.buffer_memory,
        downsampling=False,
        dynamicity = args.dynamicity
    )


    X, Y, extra_batch = dataset.get_data()

    X[X > 10e5] = 0 
    Y[Y > 10e5] = 0

# ---- Otherwise load stored file
else:
    X, Y, extra_batch = np.load(args.dataset_path, allow_pickle=True)
    print("[!] Successfully loaded dataset from {} \nX.shape: {}\nY.shape: {}\n".format(
        args.dataset_path, X.shape, Y.shape
    ))

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

# b, s, t, h, w, c -> b, s, t, c, h, w
X = X.permute(0, 1, 2, 5, 3, 4)
Y = Y.permute(0, 1, 2, 5, 3, 4)

# -------------- Model

from models.ae import seq2seq_ConvLSTM
net = seq2seq_ConvLSTM.EncoderDecoderConvLSTM(nf=args.hidden_layers, in_chan=args.in_channels).to(device) # False: many to one

# Loading model weights from previous training
print("[x] Loading model weights")
net.load_state_dict(th.load(args.weights_path))
net.eval() # evaluation mode

print("[!] Successfully loaded weights from {}".format(args.weights_path))

# ------------------------------
ssim = pytorch_ssim.SSIM()
l1 = th.nn.L1Loss()
l2 = th.nn.MSELoss()

ssim_score = 0
l1_score = 0
l2_score = 0

for t in range(args.n_tests):

    j = np.random.randint(len(X))       # random batch
    k = np.random.randint(len(X[j]))    # random datapoint

    outputs = net(X[j], 1)

    img1 = Variable(outputs[k, 0, :, :].unsqueeze(0), requires_grad=False)
    img2 = Variable(Y[j, k, 0, 0].unsqueeze(0).unsqueeze(0), requires_grad=True)

    ssim_score += ssim(img1, img2)
    l1_score += l1(img1, img2)
    l2_score += l2(img1, img2)

    fig, axs = plt.subplots(1, X.shape[2] + 2, figsize=(plotsize, plotsize))

    for ax in axs:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    # pick random datapoint from batch
    x = np.random.randint(X[j].shape[0])

    for i, frame in enumerate(X[j, k]):
        axs[i].title.set_text('t={}'.format(i))
        axs[i].matshow(frame[0].cpu().detach().numpy())

    axs[i + 1].matshow(outputs[k][0][0].cpu().detach().numpy())
    axs[i + 1].title.set_text('Predicted')

    axs[i + 2].matshow(Y[j, k][0][0].cpu().detach().numpy())
    axs[i + 2].title.set_text('Ground Truth')

    plt.show()
    if args.test_flight is None:
        plt.savefig("runs/" + foldername + "/eval_prediction_{}.png".format(t))


ssim_score = ssim_score/args.n_tests
l1_score = l1_score/args.n_tests
l2_score = l2_score/args.n_tests

print("SSIM: {}\nL1: {}\nMSE:{}".format(ssim_score, l1_score, l2_score))

text_file = open("runs/" + foldername + "/score.txt", "w")
n = text_file.write("SSIM: {}\nL1: {}\nMSE:{}".format(ssim_score, l1_score, l2_score))
text_file.close()

# ------------------------------
'''
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
'''
