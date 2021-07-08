import numpy as np
from tqdm import tqdm
import math

import torch as th
import pytorch_ssim
from torch.autograd import Variable

class Preprocessing():
    def __init__(self):
        pass

    ''' PyTorch SSIM metric '''
    def pytorch_ssim(self, frame1, frame2):
        frame1 = Variable(th.unsqueeze(th.unsqueeze(th.Tensor(frame1), 0),0))
        frame2 = Variable(th.unsqueeze(th.unsqueeze(th.Tensor(frame2),0),0))

        if th.cuda.is_available():
            frame1 = frame1.cuda()
            frame2 = frame2.cuda()

        return pytorch_ssim.ssim(frame1, frame2)

    ''' Returns false/true if the given sequence of frames is "sufficiently dynamic" '''
    def eval_datapoint(self, X, threshold):
        # shape: frames, h, w, channels
        max_diff = 0.0
        for i in range(len(X)-1):
            # velocity * dep
            curr_frame = np.dot(X[i,:,:,0], X[i,:,:,1])
            next_frame = np.dot(X[i+1,:,:,0], X[i+1,:,:,1])
            similarity = self.pytorch_ssim(curr_frame, next_frame).item()
            max_diff = max(max_diff, 1-similarity)

        return max_diff >= threshold
