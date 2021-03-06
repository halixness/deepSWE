# %%
import os
from datetime import datetime
from utils.data_lightning.otf import SWEDataset
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mat

import torch as th
from torch.autograd import Variable

from models.experiments.nfnets import seq2seq_NFLSTM
from models.ae import seq2seq_ConvLSTM

import argparse
import pytorch_ssim
import time

mat.use("Agg") # headless mode
#mat.rcParams['text.color'] = 'w'

# -------------- Functions

def accuracy(prediction, target, threshold = 1e-2):

    total = (target * prediction).cpu().detach().numpy()
    total = np.array(total > 0).astype(int) # TP + TN + FP + FN

    diff = np.abs((target - prediction).cpu().detach().numpy())
    correct_cells = (diff < threshold).astype(int)
    correct_cells = correct_cells*total # TP + TN

    accuracy = np.sum(correct_cells)/np.sum(total)
    return accuracy

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

parser.add_argument('-accuracy_threshold', dest='accuracy_threshold', default = 1e-1, type=float,
                    help='Delta threshold to consider true positives, [0,1] ')
parser.add_argument('-blur_radius', dest='blur_radius', default = 3, type=int,
                    help='Blur radius for downsampling')
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
parser.add_argument('-dynamicity', dest='dynamicity', default=1e-1, type=float,
                    help='dynamicity rate (to filter out "dynamic" sequences)')
parser.add_argument('-downsampling', dest='downsampling', default=False, type=str2bool,
                    help='Use 4xdownsampling')
parser.add_argument('-future_frames', dest='future_frames', default=1, type=int, 
                    help='number of future frames')

parser.add_argument('-p', dest='past_frames', default=4, type=int,
                    help='number of past frames')       
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
dev = "cpu"
device = th.device(dev)

plotsize = 15

# -------------- Model
net = seq2seq_ConvLSTM.EncoderDecoderConvLSTM(nf=args.filters, in_chan=args.in_channels, out_chan=args.out_channels)

if args.multigpu:
    net = nn.DataParallel(net)

net.load_state_dict(
    th.load(args.weights_path, map_location=device)
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
acc_score = 0

dataset = SWEDataset(
    root=args.root,  
    past_frames=args.past_frames, 
    future_frames=args.future_frames, 
    partial=args.partial,
    dynamicity=args.dynamicity,
    downsampling=args.downsampling,
    blur_radius=args.blur_radius
)


i = np.random.randint(len(dataset))       # random batch
datapoint = dataset[i]

while datapoint is None:
    i = np.random.randint(len(dataset))       # random batch
    datapoint = dataset[i]

print("[!] Datapoint loaded!")

# b, t, c, h, w 
x_in, y = datapoint

# ------------- Plotting
test_dir = "runs/" + foldername + "/"

for i, frame in enumerate(x_in[0]):
    plt.matshow(frame[0].cpu().detach().numpy())
    plt.savefig(test_dir + "/{}.png".format(i))
    plt.clf()

# predicting
for s in range(args.future_frames):

    start = time.time()
    # 1, t, c, h, w 
    outputs = net(x_in, 1)
    end = time.time()
    inference_times.append(end - start)

    # Saves plot
    plt.matshow(outputs[0,0,0].cpu().detach().numpy())
    plt.savefig(test_dir + "/pred_{:02d}.png".format(s))
    plt.clf()
    
    plt.matshow(y[0,s,0].cpu().detach().numpy())
    plt.savefig(test_dir + "/true_{:02d}.png".format(s))
    plt.clf()

    # 1, c, h, w
    img1 = Variable(outputs[0,0,0].unsqueeze(0).unsqueeze(0), requires_grad=False)
    img2 = Variable(y[0,s,0].unsqueeze(0).unsqueeze(0), requires_grad=True)

    acc_score += accuracy(img1, img2, threshold=1e-1)
    ssim_score += ssim(img1, img2)
    l1_score += l1(img1, img2)
    l2_score += l2(img1, img2)

    # Builds a new sequence with its own prediction
    tmp = th.empty(x_in.shape)
    tmp[0, :(args.past_frames-1), :, :, :] = x_in[0, 1:, :, :, :] # get last n-1 frames
    
    # output + btm
    new_frame = th.cat((
        outputs[:,:,0,:,:], 
        x_in[:, 0, 3, :, :].unsqueeze(0)
    ), dim=1) 

    tmp[0, -1, :, :, :] = new_frame
    x_in = tmp

    print(". ", end="", flush=True)

acc_score = acc_score/args.future_frames
ssim_score = ssim_score/args.future_frames
l1_score = l1_score/args.future_frames
l2_score = l2_score/args.future_frames

stats = "Accuracy: {}\nSSIM: {}\nL1: {}\nMSE:{}\nAvg.Inference Time: {}".format(acc_score, ssim_score, l1_score, l2_score, np.mean(inference_times))
print(stats, flush=True)

text_file = open("runs/" + foldername + "/avg_score.txt", "w")
n = text_file.write(stats)
text_file.close()

