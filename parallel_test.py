from utils.data_legacy.dataloader import DataPartitions, DataGenerator
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mat

import argparse

import torch as th
import torch.nn as nn
import os
from datetime import datetime
import matplotlib as mpl
import torch.optim as optim
import time

from models.ae import seq2seq_NFLSTM
from models.ae import seq2seq_ConvLSTM

from threading import Thread

mat.use("Agg") # headless mode

# -------------- Functions

def mass_conservation_loss(output, target):
    # output: b, h, w
    diff = 0
    for i,datapoint in enumerate(output):
        diff += th.abs(
            th.sum(
                th.abs(output[i])
            ) -
            th.sum(
                th.abs(target[i])
            )
        )
    return diff**(1/2)

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

parser = argparse.ArgumentParser(description='Trains a given model on a given dataset')

parser.add_argument('-save', dest="save_dataset", default = False, type=str2bool,
                    help='Save the dataset loaded from disk to a npy file')
parser.add_argument('-network', dest="network", default = "conv",
                    help='Network type: conv/nfnet/(...)')
parser.add_argument('-test_size', dest='test_size', default = 0.2,
                    help='Test size for the split')
parser.add_argument('-shuffle', dest='shuffle', default=True, type=str2bool,
                    help='Shuffle the dataset')
parser.add_argument('-tf', dest='test_flight',
                    help='Test flight. Avoids creating a train folder for this session.')
parser.add_argument('-npy', dest='numpy_file',
                    help='path to a npy stored dataset')
parser.add_argument('-root', dest='root',
                    help='root path with the simulation files (cropped and stored in folders)')
parser.add_argument('-p', dest='past_frames', default=4, type=int,
                    help='number of past frames')       
parser.add_argument('-f', dest='future_frames', default=1, type=int, 
                    help='number of future frames')       
parser.add_argument('-partial', dest='partial', default=None, type=float,
                    help='percentage of portion of dataset (to load partial, lighter chunks)')                                                            
parser.add_argument('-image_size', dest='image_size', default=256, type=int,
                    help='image size (width = height)')
parser.add_argument('-b', dest='batch_size', default=4, type=int,
                    help='batch size') 
parser.add_argument('-d', dest='dynamicity', default=1e-1, type=float,
                    help='dynamicity rate (to filter out "dynamic" sequences)')                                                                                                  
parser.add_argument('-bs', dest='buffer_size', default=1e4, type=float,
                    help='size of the cache memory (in entries)')
parser.add_argument('-t', dest='buffer_memory', default=1000, type=int,
                    help='temporal length of the cache memory (in iterations)')                                                                                                  
parser.add_argument('-ds', dest='downsampling', default=False, type=str2bool, nargs='?',
                    const=True, help='enable 2x downsampling (with gaussian filter)')  
parser.add_argument('-lr', dest='learning_rate', default=0.0001, type=float,
                    help='learning rate')                                              
parser.add_argument('-epochs', dest='epochs', default=100, type=int,
                    help='training iterations')
parser.add_argument('-ls', dest='latent_size', default=1024, type=int,
                    help='latent size for the VAE')
parser.add_argument('-hidden_layers', dest='hidden_layers', default=4, type=int,
                    help='number of hidden layers')
parser.add_argument('-in_channels', dest='in_channels', default=4, type=int,
                    help='number of input channels')
parser.add_argument('-out_channels', dest='out_channels', default=1, type=int,
                    help='number of input channels')
parser.add_argument('-checkpoint', dest='checkpoint', default=0.1, type=float,
                    help='Percentage of dataset at which saving the network weights')


args = parser.parse_args()

if args.root is None and args.numpy_file is None:
    print("required: please specify a root path: -r /path")
    exit()
   
# -------------- Setting up the run

if args.test_flight is None:
    num_run = len(os.listdir("runs/")) + 1
    now = datetime.now()
    foldername = "train_otf_{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
    os.mkdir("runs/" + foldername)

# -------------------------------
if th.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = th.device(dev) 

plotsize = 15

partitions = DataPartitions(
    past_frames=args.past_frames, 
    future_frames=args.future_frames, 
    root=args.root, 
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

# ---- Model
if args.network == "conv":
    net = seq2seq_ConvLSTM.EncoderDecoderConvLSTM(nf=args.hidden_layers, in_chan=args.in_channels, out_chan=args.out_channels).to(device) # False: many to one
elif args.network == "nfnet":
    net = seq2seq_NFLSTM.EncoderDecoderConvLSTM(nf=args.hidden_layers, in_chan=args.in_channels, out_chan=args.out_channels).to(device)
else:
    raise Exception("Unkown network type given.")

criterion = nn.MSELoss(reduction='sum') # reduction='sum'
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

# ---- Training time!
losses = []
avg_losses = []
errors = []
test_errors = []
training_times = []
print("\n[!] It's training time!")

epochs = args.epochs

parts = partitions.get_partitions()

# Random indexes
random_accesses = []
for i, area in enumerate(parts):

    # random shuffle within the area only
    sequences = area[1]
    np.random.shuffle(sequences)
    
    for j, sequence in enumerate(sequences):
        random_accesses.append([i, sequence, None])

print("[~] Virtually shuffled the dataset, total sequences: {}".format(len(random_accesses)))

# Initializing batches
batch_x = np.empty((args.batch_size, args.past_frames, args.image_size, args.image_size, args.in_channels))
batch_y = np.empty((args.batch_size, args.future_frames, args.image_size, args.image_size, args.out_channels))
k = 0

checkpoint = int(args.checkpoint * len(random_accesses))
print("[~] Network weights will be saved each {} samples".format(checkpoint))

results = []
def get_datapoint(access, check=False):
    results.append(
        dataset.get_datapoint(access[0], access[1], check=check)
    )
    print("done")

# Training loop
for epoch in range(epochs):  # loop over the dataset multiple times

    i = 0
    while i < len(random_accesses):
        
        threads = []

        # Start x parallel threads
        for x in range(2):
            access = random_accesses[i] 
            process = Thread(target=get_datapoint, args=(access,))
            threads.append(process)
            process.start()
            i += 1
            print("Started thread {}...".format(x))

        # Waiting for each to end
        print("waiting threads...")
        start = time.time()
        for process in threads:
            process.join()
        end = time.time()
        print("load time: {}".format(end - start))

        print(len(results))
        raise KeyboardInterrupt


    for i, access in enumerate(random_accesses):

        # Checkpoint weights saving
        if i % checkpoint == 0:
            weights_path = "runs/" + foldername + "/epoch_{}_chk_{}.weights".format(epoch, i)
            th.save(net.state_dict(), weights_path)

        
        
        # False mark -> invalid datapoint
        if access[2] != False:
            
            # True mark -> already checked, valid datapoint
            if access[2] == True: check = False
            else: check = True
    
            datapoint = dataset.get_datapoint(access[0], access[1], check=check)
            
            # If the sequence is invalid -> mark it
            if datapoint == None:
                random_accesses[i][2] = False
                print("=", end="", flush=True)

            # Sequence valid
            else:
                # Forward pass and empty the batch               
                if k >= args.batch_size:

                    # b, s, t, h, w, c -> b, s, t, c, h, w
                    X = th.Tensor(batch_x).to(device)
                    Y = th.Tensor(batch_y).to(device)
                    X = X.permute(0, 1, 4, 2, 3)
                    Y = Y.permute(0, 1, 4, 2, 3)

                    optimizer.zero_grad()

                    # ---- Predicting
                    outputs = net(X, 1) # 0 for layer index, 0 for h index

                    # ---- Batch Loss
                    xstart=256
                    xend=xstart+256
                    ystart=256
                    yend=ystart+256
                    
                    loss = criterion(outputs[0,:,0,ystart:yend,xstart:xend], Y[0,0,:,ystart:yend,xstart:xend]) 

                    loss.backward()
                    optimizer.step()

                    # print statistics
                    losses.append(loss.item())
                    batch_x = np.empty((args.batch_size, args.past_frames, args.image_size, args.image_size, args.in_channels))
                    batch_y = np.empty((args.batch_size, args.future_frames, args.image_size, args.image_size, args.out_channels))
                    k = 0

                    print("-- avg.loss: {}".format(np.mean(losses)), flush=True)

                # Accumulates datapoints in batch 
                X, Y = datapoint
                batch_x[k] = X
                batch_y[k] = Y
                k += 1

                print("x", end="", flush=True)

print('[!] Finished Training, storing weights...')
if args.test_flight is None:
    weights_path = "runs/" + foldername + "/model.weights"
    th.save(net.state_dict(), weights_path)

# Loss plot
mpl.rcParams['text.color'] = 'k'

plt.title("average loss")
plt.plot(range(len(avg_losses)), avg_losses)
if args.test_flight is None:
    plt.savefig("runs/" + foldername + "/avg_loss.png")
plt.clf()

print("Avg.training time: {}".format(np.mean(training_times)))
