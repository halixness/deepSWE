import numpy as np
import math

import torch as th
import pytorch_ssim
from torch.autograd import Variable


class Preprocessing():
    def __init__(self):
        pass

    ''' PyTorch SSIM metric '''

    def pytorch_ssim(self, frame1, frame2):
        frame1 = Variable(th.unsqueeze(th.unsqueeze(th.Tensor(frame1), 0), 0))
        frame2 = Variable(th.unsqueeze(th.unsqueeze(th.Tensor(frame2), 0), 0))

        if th.cuda.is_available():
            frame1 = frame1.cuda()
            frame2 = frame2.cuda()

        return pytorch_ssim.ssim(frame1, frame2)

    def eval_datapoint_ssim(self, seq, threshold):
        ''' Returns false/true if the given sequence of frames is "sufficiently dynamic" 
            threshold = [0,1]
        '''
        # shape: frames, h, w, channels
        max_diff = 0.0

        for i in range(len(seq) - 1):
            # velocity * dep
            # curr_frame = np.dot(X[i,:,:,0], X[i,:,:,1])
            # next_frame = np.dot(X[i+1,:,:,0], X[i+1,:,:,1])

            curr_frame = seq[i, :, :, 0]
            next_frame = seq[i + 1, :, :, 0]

            similarity = self.pytorch_ssim(curr_frame, next_frame).item()
            max_diff = max(max_diff, 1 - similarity)

        return max_diff, max_diff >= threshold

    def filter(self, tensor):
        tensor[np.isnan(tensor)] = 0
        tensor[tensor > 10e5] = 0
        return tensor

    def eval_datapoint_diff(self, X, Y, threshold, sensibility=0.1, grid_size=3):
        ''' Returns false/true if the given sequence of frames is "sufficiently dynamic"
        '''
        center = int(X.shape[1] / grid_size)
        start = center
        end = 2 * center

        # compares the % difference between last and first frame
        first_frame = self.filter(X[0, start:end, start:end, 0])

        # Difference between first frame and any in the output
        deltas = []

        for frame in Y:
            current_frame = self.filter(frame[start:end, start:end, 0])

            wet_cells = (current_frame * first_frame) # conjunction
            wet_cells = np.array(wet_cells > 0).astype(int) # total wet cells

            # totale celle bagnate del fiume nella previsione
            diff = np.abs((current_frame - first_frame))
            similar_cells = (diff < sensibility).astype(int)
            similar_cells = similar_cells*wet_cells # conjuction

            similarity = (np.sum(similar_cells)/np.sum(wet_cells))

            if np.isnan(similarity):
                similarity = 1

            distance = 1 - similarity

            # Only one has to satisfy the condition
            if distance >= threshold:
                return True

        return False

    def eval_datapoint(self, X, Y, threshold=1e-1):
        ''' Returns false/true if the given sequence of frames is "sufficiently dynamic"
            threshold = [0,1]
        '''
        return self.eval_datapoint_diff(X, Y, threshold)


