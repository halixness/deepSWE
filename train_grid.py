
# coding: utf-8

# In[20]:


from utils.data_lightning.preloading import SWEDataModule
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mat

import argparse

import torch as th
import os
from datetime import datetime
import matplotlib as mpl
import torch.optim as optim
import time

from models.ae import seq2seq_ConvLSTM

mat.use("Agg") # headless mode

# -------------- Functions

def sse_loss(input, target):
    return th.sum((target - input) ** 2)

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

parser.add_argument('-test_size', dest='test_size', default = 0,
                    help='Test size for the split')
parser.add_argument('-val_size', dest='val_size', default = 0,
                    help='Test size for the split')
parser.add_argument('-shuffle', dest='shuffle', default=True, type=str2bool,
                    help='Shuffle the dataset')
parser.add_argument('-root', dest='root',
                    help='root path with the simulation files (cropped and stored in folders)')
parser.add_argument('-past_frames', dest='past_frames', default=4, type=int,
                    help='number of past frames')
parser.add_argument('-future_frames', dest='future_frames', default=1, type=int,
                    help='number of future frames')
parser.add_argument('-partial', dest='partial', default=None, type=float,
                    help='percentage of portion of dataset (to load partial, lighter chunks)')
parser.add_argument('-filters', dest='filters', default=16, type=int,
                    help='Number of network kernels in hidden layers')
parser.add_argument('-image_size', dest='image_size', default=256, type=int,
                    help='Width=height of image')
parser.add_argument('-batch_size', dest='batch_size', default=4, type=int,
                    help='Batch size')
parser.add_argument('-lr', dest='learning_rate', default=0.0001, type=float,
                    help='learning rate')                                              
parser.add_argument('-epochs', dest='epochs', default=200, type=int,
                    help='training iterations')
parser.add_argument('-in_channels', dest='in_channels', default=4, type=int,
                    help='number of input channels')
parser.add_argument('-out_channels', dest='out_channels', default=3, type=int,
                    help='number of input channels')
parser.add_argument('-filtering', dest='filtering', default=True, type=str2bool,
                    help='Enable filtering dynamic sequences only')
parser.add_argument('-workers', dest='workers', default=4, type=int,
                    help='Dataloader threads')

args = parser.parse_args()
   
# -------------- Setting up the run

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

dataset = SWEDataModule(
    root=args.root,
    test_size=args.test_size,
    val_size=args.val_size,
    past_frames=args.past_frames,
    future_frames=args.future_frames,
    partial=args.partial,
    filtering=args.filtering,
    batch_size=args.batch_size,
    workers=args.workers,
    image_size=args.image_size,
    shuffle=False,
    dynamicity=100,
    caching=False
)
dataset.prepare_data()

# ---- Model
net = seq2seq_ConvLSTM.EncoderDecoderConvLSTM(nf=args.filters, in_chan=args.in_channels, out_chan=args.out_channels).to(device) # False: many to one
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

# ---- Training time!
losses = []
avg_losses = []
errors = []
test_errors = []
training_times = []
print("\n[!] It's training time!")

epochs = args.epochs

for epoch in range(epochs):  # loop over the dataset multiple times

    print("---- Epoch {}".format(epoch))
    for x,y in dataset.train_dataloader():

        optimizer.zero_grad()

        x = x.float().to(device)
        y = y.float().to(device)

        # ---- Predicting

        start = time.time()
        outputs = net(x, args.future_frames)  # 0 for layer index, 0 for h index

        # ---- Batch Loss
        loss = sse_loss(outputs[:, :args.out_channels, 0, :, :], y[:, 0, :args.out_channels, :, :])
 
        loss.backward()
        optimizer.step()

        end = time.time()
        training_times.append(end - start)

        losses.append(loss.item())

    print("epoch {} - avg.loss {}".format(epoch, np.mean(losses)))
    avg_losses.append(np.mean(losses))

    '''
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
        
        plt.clf()
        
        #------------------------------

        #if epoch % 10 == 0:
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 2000))
        #    running_loss = 0.0
    '''

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
plt.clf()

print("Avg.training time: {}".format(np.mean(training_times)))


# In[26]:
'''

plt.title("relative error")
plt.plot(range(len(errors)), errors, label="train")
plt.plot(range(len(test_errors)), test_errors, label="test")
plt.legend()
pass

# In[27]:


mpl.rcParams['text.color'] = 'w'
prep = Preprocessing()


# In[28]:


j = np.random.randint(len(y_test))
j = 3
m = 2
k = 4

print("k = {}".format(k))

#k = 5
input = th.unsqueeze(X_test[j, m], 0)
outputs = net(input)

#------------------------------
num_predicted_frames = outputs[0, 0].shape[0] # per allineare frames passati e futuri
fig, axs = plt.subplots(1, num_predicted_frames, figsize=(plotsize,plotsize))

for ax in axs:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

for i,frame in enumerate(input[0,0,-num_predicted_frames:]):
    frame = frame.cpu().detach().numpy()
    axs[i].matshow(frame)
    axs[i].set_title('t = {}'.format(i))

print("======== Past frames ========")
plt.show()

print("======== True Future vs Predicted frames ========")

#------------------------------
fig, axs = plt.subplots(1, num_predicted_frames, figsize=(plotsize,plotsize))

for ax in axs:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

for i,frame in enumerate(y_test[j,m,0]):
    axs[i].matshow(frame.cpu().detach().numpy())
    axs[i].set_title('t = {}'.format(i+num_predicted_frames))

plt.show()
#------------------------------
fig, axs = plt.subplots(1, outputs[0,0].shape[0], figsize=(plotsize,plotsize))

for ax in axs:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

for i,frame in enumerate(outputs[0,0]):
    attention_mask = outputs[0,1,i].cpu().detach().numpy()
    y_frame = frame.cpu().detach().numpy()
    y_true = y_test[j,m,0,i].cpu().detach().numpy()

    y_frame = np.dot(y_frame, attention_mask)
    y_true = np.dot(y_true, attention_mask)

    print(y_frame[:5,:5])
    print("#-------------")
    print(y_true[:5,:5])
    print("\n\n")

    ssim = prep.pytorch_ssim(y_true, y_frame).item()
    axs[i].matshow(y_frame)
    axs[i].set_title('ssim = {}'.format(ssim))

plt.show()
#------------------------------


# In[29]:


print("k = {}".format(k))
iterations = 4
#k = 5

input = th.unsqueeze(X_train[k,0], 0)
outputs = net(input)

#------------------------------
print("======== Past frames ========")
num_predicted_frames = outputs[0,0].shape[0] # per allineare frames passati e futuri
fig, axs = plt.subplots(1, num_predicted_frames, figsize=(plotsize,plotsize))

for ax in axs:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

for i,frame in enumerate(input[0,0,-num_predicted_frames:]):
    axs[i].matshow(frame.cpu().detach().numpy())
    axs[i].set_title('t = {}'.format(i))

plt.show()
#------------------------------
print("======== True vs Autoregressive Pred Frames  ========")
fig, axs = plt.subplots(1, num_predicted_frames, figsize=(plotsize,plotsize))

true_means = []

for ax in axs:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

for i,frame in enumerate(y_train[k,0,0]):
    axs[i].matshow(frame.cpu().detach().numpy())
    axs[i].set_title('t = {}'.format(i+num_predicted_frames))
    true_means.append(frame.cpu().detach().numpy().mean())

plt.show()
#------------------------------

#i = np.random.randint(len(X_test))
input = th.unsqueeze(X_train[k][0], 0)

fig, axs = plt.subplots(1, iterations, figsize=(plotsize,plotsize))

for ax in axs:
    ax.set_yticklabels([])
    ax.set_xticklabels([])

predicted_means = []
for x in range(iterations):
    # first predicted frame only
    output = th.unsqueeze(net(input)[:,:,0,:,:],2)
    # next frame = first predicted from output + btm map
    next_frame = output.detach()
    next_frame = th.cat((next_frame, th.unsqueeze(input[:,2:,0,:,:],2)), axis=1)
    # added on top of (input sequence - first frame)
    input = th.cat((next_frame, input[:,:,1:,:]), axis=2)

    axs[x].matshow(output[0,0,0].cpu().detach().numpy())
    axs[x].set_title('t = {}'.format(x+num_predicted_frames))
    predicted_means.append(output[0,0,0].cpu().detach().numpy().mean())
    #print(np.mean(output[0,0,0].cpu().detach().numpy()))

plt.show()


# In[30]:


mpl.rcParams['text.color'] = 'b'

plt.clf()
plt.plot(range(len(true_means)), true_means,  "-b", label="True frames mean")
plt.plot(range(len(true_means)), true_means,  "*")

plt.plot(range(len(predicted_means)), predicted_means,  "-g", label="Predicted frames mean")
plt.plot(range(len(predicted_means)), predicted_means,  "*")
plt.grid()
plt.legend()
pass


# In[31]:


print("{:<20s}{:<20s}{:<20s}{:<20s}{:<20s}".format("", "min", "max", "mean", "std"))
print("{:<20s}{:<20f}{:<20f}{:<20f}{:<20f}".format("prediction", th.min(outputs), th.max(outputs), th.mean(outputs), th.std(outputs)))
print("{:<20s}{:<20f}{:<20f}{:<20f}{:<20f}".format("true", th.min(y_test[0]), th.max(y_test[0]), th.mean(y_test[0]), th.std(y_test[0])))

'''
