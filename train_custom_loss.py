
# coding: utf-8

from utils.dataloader import DataLoader, DataPartitions, DataGenerator
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.patches as patches

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

mat.use("Agg") # headless mode

# -------------- Functions

def residual_partial_mse(prior, output, target, threshold=0):
    
    # t, h, w
    # compares the difference between n and n+1 frame with ground truth and prediction
    # masking prediction to force predicting water levels (not houses)
    #output = output[:,:,0] * output[:,:,1]

    #plt.matshow(target_residual[0,0])
    #plt.savefig("runs/" + foldername + "/residual.png", vmin=0, vmax=10)
    #plt.close()

    #plt.matshow(output[0,0,0].detach().numpy())
    #plt.savefig("runs/" + foldername + "/prediction.png", vmin=0, vmax=5)
    #plt.close()

    n_target_wet_cells = th.flatten(target[target > threshold]).shape[0]
    n_pred_wet_cells = th.flatten(output[output > threshold]).shape[0]

    # Intuition: MSE loss averages on a way higher number of cells, but many need to be ignored
    #print("\nEffective cells: {} %".format(
    #    (th.flatten(target[target > 0]).shape[0]/ th.flatten(target).shape[0]) * 100)
    #)

    n_cells = max(n_target_wet_cells, n_pred_wet_cells)

    # MSE but I'm not using th.mean()
    if args.residual_loss:
        pred_residual = th.abs(output - prior)
        target_residual = th.abs(target[:,:,0] - prior)
        
        loss = th.sum(
            (pred_residual - target_residual)**2
        ) / n_cells
    else:
        loss = th.sum(
            (output - target)**2
        ) / n_cells

    return loss

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
parser.add_argument('-past_frames', dest='past_frames', default=4, type=int,
                    help='number of past frames')       
parser.add_argument('-future_frames', dest='future_frames', default=1, type=int, 
                    help='number of future frames')       
parser.add_argument('-partial', dest='partial', default=None, type=float,
                    help='percentage of portion of dataset (to load partial, lighter chunks)')                                                            
parser.add_argument('-image_size', dest='image_size', default=256, type=int,
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
parser.add_argument('-residual_loss', dest='residual_loss', default=False, type=str2bool,
                    help='number of input channels')


args = parser.parse_args()

if args.root is None and args.numpy_file is None:
    print("required: please specify a root path: -r /path")
    exit()
   
# -------------- Setting up the run

if args.test_flight is None:
    num_run = len(os.listdir("runs/")) + 1
    now = datetime.now()
    foldername = "train_{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
    os.mkdir("runs/" + foldername)

# -------------------------------
if th.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = th.device(dev) 

plotsize = 15

dataset = DataLoader(
    root=args.root, 
    numpy_file=args.numpy_file, 
    past_frames=args.past_frames, 
    future_frames=args.future_frames, 
    image_size=args.image_size, 
    batch_size=args.batch_size, 
    buffer_size=args.buffer_size, 
    buffer_memory=args.buffer_memory, 
    dynamicity=args.dynamicity, 
    partial=args.partial, 
    test_size=args.test_size, 
    clipping_threshold=1e5, 
    shuffle=args.shuffle,
    device=device
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

X_train, y_train = dataset.get_train()
X_test, y_test = dataset.get_test()

# Stores the dataset on disk for faster loading
#if args.save_dataset is not None and is not False:
#    dataset.save_dataset("runs/" + foldername + "/dataset.npy")

epochs = args.epochs

for epoch in range(epochs):  # loop over the dataset multiple times

    print("********* Epoch {} *********".format(epoch))

    for i, batch in enumerate(X_train):

        optimizer.zero_grad()

        # ---- Predicting

        start = time.time()
        outputs = net(batch, args.future_frames)  # 0 for layer index, 0 for h index
        outputs = outputs.permute(0, 2, 1, 3, 4)

        # ---- Batch Loss
        # central square only 
        prior_frame = batch[:,-1,0,:,:].unsqueeze(1) # last frame
        output_dep = outputs[:,:,:,:,:]
        target_dep = y_train[i, :,:,:,:,:]

        loss = residual_partial_mse(prior_frame, output_dep, target_dep)
        #loss = criterion(outputs[:, :args.out_channels, 0, 256:512, 256:512], y_train[i, :, 0, :args.out_channels, 256:512, 256:512])

        loss.backward()
        optimizer.step()

        end = time.time()
        training_times.append(end - start)

        losses.append(loss.item())

        if i == 0: 
            print("batch {} - avg.loss {}".format(i, np.mean(losses)))
            avg_losses.append(np.mean(losses))

    if epoch % 2 == 0:

        k = np.random.randint(len(X_test))

        outputs = net(X_test[k], 1)  # 0 for layer index, 0 for h index

        #test_loss = criterion(outputs[0],  y_test[k,:,:,0,:,:])
        #print("test loss: {}".format(test_loss.item()))

        #------------------------------
        fig, axs = plt.subplots(1, X_test.shape[2] + 2, figsize=(plotsize, plotsize))

        for ax in axs:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        # pick random datapoint from batch
        x = np.random.randint(X_test[k].shape[0])

        for i, frame in enumerate(X_test[k, x]):
            axs[i].title.set_text('t={}'.format(i))
            axs[i].matshow(frame[0].cpu().detach().numpy())
            rect = patches.Rectangle((256, 256), 256, 256, linewidth=1, edgecolor='r', facecolor='none')
            axs[i].add_patch(rect)

        axs[i+1].matshow(outputs[x][0][0].cpu().detach().numpy())
        rect = patches.Rectangle((256, 256), 256, 256, linewidth=1, edgecolor='r', facecolor='none')
        axs[i+1].add_patch(rect)
        axs[i+1].title.set_text('Predicted')

        axs[i+2].matshow(y_test[k,x][0][0].cpu().detach().numpy())
        rect = patches.Rectangle((256, 256), 256, 256, linewidth=1, edgecolor='r', facecolor='none')
        axs[i+2].add_patch(rect)
        axs[i+2].title.set_text('Ground Truth')

        plt.show()

        if args.test_flight is None:
            plt.savefig("runs/" + foldername + "/{}_{}_plot.png".format(epoch, k))
            plt.close()
        
        plt.clf()
        
        #------------------------------

        #if epoch % 10 == 0:
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 2000))
        #    running_loss = 0.0

end = time.time()
print(end - start)

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
    plt.close()
plt.clf()

print("Avg.training time: {}".format(np.mean(training_times)))

