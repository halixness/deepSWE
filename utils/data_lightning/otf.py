import numpy as np
import os
import math
import cv2 as cv
import pandas as pd
from utils.preprocessing import Preprocessing

import torch as th
from torch.utils.data import Dataset, random_split, DataLoader
import pytorch_lightning as pl

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

# ------------------------------------------------------------------------------

class SWEDataModule(pl.LightningDataModule):

    def __init__(self, root, past_frames, future_frames, caching=False, dynamicity=100, shuffle=True, image_size=768, batch_size=4,
                 downsampling=False, workers=4, filtering=True, test_size=0.1, val_size=0.1, partial=None):
        super(SWEDataModule, self).__init__()

        self.test_size = test_size
        self.val_size = val_size
        self.partial = partial
        self.root = root
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.filtering = filtering
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_size
        self.shuffle = shuffle
        self.dynamicity = dynamicity
        self.caching = caching
        self.downsampling = downsampling
        
    def prepare_data(self):

        dataset = SWEDataset(
            root=self.root,
            past_frames=self.past_frames, 
            future_frames=self.future_frames,
            partial=self.partial,
            filtering = self.filtering,
            image_size=self.image_size,
            shuffle=self.shuffle,
            dynamicity=self.dynamicity,
            caching=self.caching,
            downsampling=self.downsampling
        )

        test_len = int(len(dataset) * self.test_size)
        val_len = int(len(dataset) * self.val_size)
        train_len = len(dataset) - test_len - val_len

        datasets = random_split(dataset, [train_len, test_len, val_len])

        self.train_loader = DataLoader(datasets[0], batch_size=self.batch_size, num_workers=self.workers)
        self.test_loader = DataLoader(datasets[1], batch_size=self.batch_size, num_workers=self.workers)
        self.val_loader = DataLoader(datasets[2], batch_size=self.batch_size, num_workers=self.workers)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

# ------------------------------------------------------------------------------

class SWEDataset(Dataset):
    def __init__(self, past_frames, future_frames, root, shuffle=True, caching=False, filtering=True, image_size=256, batch_size=4, dynamicity=1e-3,
                 buffer_memory=100, buffer_size=1000, downsampling=False, partial=None):
        ''' Initiates the dataloading process '''

        self.filtering = filtering

        self.partitions = DataPartitions(
            past_frames=past_frames,
            future_frames=future_frames,
            root=root,
            partial=partial
        )

        self.dataset = DataGenerator(
            root=root,
            dataset_partitions=self.partitions.get_partitions(),
            past_frames=self.partitions.past_frames,
            future_frames=self.partitions.future_frames,
            input_dim=(self.partitions.past_frames, image_size, image_size, 4),
            output_dim=(self.partitions.future_frames, image_size, image_size, 3),
            batch_size=batch_size,
            buffer_size=buffer_size,
            buffer_memory=buffer_memory,
            downsampling=downsampling,
            dynamicity=dynamicity,
            caching=caching
        )

        self.valid_datapoints = dict()

    # ------------------------------

    def pushValid(self, i, j):
        ''' Marks sequence j of area i as valid '''
        if i not in self.valid_datapoints:
            self.valid_datapoints[i] = [j]
        else:
            if j not in self.valid_datapoints[i]:
                self.valid_datapoints[i].append(j)

    def isValid(self, i, j):
        ''' True if area i and sequence j is marked as valid already '''
        return i in self.valid_datapoints and j in self.valid_datapoints[i]
            
    def __len__(self):
        size = 0
        for part in self.partitions.get_partitions(): # sum of num.seqs per area
            size += len(part[1])
        return size

    def __getitem__(self, idx):

        # De-linearization
        x = 0
        X = None
        Y = None
        for i, area in enumerate(self.partitions.get_partitions()):
            for j, seq in enumerate(area[1]):
                x += 1
                if x == idx: break
            if x == idx: break

        # Picks the datapoint on the fly
        if self.filtering: # if enabled
            X, Y = self.dataset.get_datapoint(i, j, check=(not self.isValid(i, j)))
        else:
            X, Y = self.dataset.get_datapoint(i, j, check=False)

        if X is not None:
            if self.filtering:
                self.pushValid(i, j)

            x = th.Tensor(X)
            y = th.Tensor(Y)

            # Channel-first conversion
            # b, s, t, h, w, c -> b, s, t, c, h, w
            x = x.permute(0, 1, 4, 2, 3)
            y = y.permute(0, 1, 4, 2, 3)
            
            return (x, y)
        else:
            return None
           

# ------------------------------------------------------------------------------

class DataPartitions():
    def __init__(self, past_frames, future_frames, root, partial=None):

        self.root = root
        self.past_frames = past_frames
        self.future_frames = future_frames

        self.dataset_partitions = []
        self.areas = []
        self.partial = partial

        self.create_partitions()

    def get_areas(self):
        ''' Returns list of areas (source folder name)
        '''
        return self.areas

    def get_partitions(self):
        ''' Returns test dataset partitions
            Partitions: [(ids_x(x, 10), ids_y(x, 4))]
        '''

        return self.dataset_partitions

    def create_partitions(self):
        ''' Creates the keras.DataLoader partitions (input-label)
            - file format: mini-xxxx.DEP/VVX/VVY/BTM
        '''

        self.areas = os.listdir(self.root)
        self.areas = [x for x in sorted(self.areas) if x.startswith("mini-") and os.path.isdir(self.root + x)]
        np.random.shuffle(self.areas)

        if self.partial is not None:
            self.areas = self.areas[:int(len(self.areas) * self.partial)]

        if len(self.areas) <= 0:
            raise Exception("Nessuna cartella area valida trovata.")

        # Applica intervalli di interesse
        for area in self.areas:

            n_frames = len([x for x in os.listdir(self.root + area) if x.endswith(".DEP")])

            # Fino ai frame che interessano
            size = n_frames - self.past_frames - self.future_frames

            partition_raw = []
            labels = dict()

            # -----------
            for i in range(size):
                partition_raw.append("id-{}".format(i))
                labels["id-{}".format(i)] = list(
                    range(i + (self.past_frames), i + (self.past_frames) + (self.future_frames)))

            # folder_name, x_frame_id, y_frames_id
            self.dataset_partitions.append((area, partition_raw, labels))


# ------------------------------------------------------------------------------
class DataGenerator():
    def __init__(self, root, dataset_partitions, past_frames, future_frames, input_dim, output_dim,
                 buffer_memory=1e2, buffer_size=1e3, batch_size=16, caching=True, downsampling=False, dynamicity=1e-3):
        '''
            Data Generator
            Inputs:

                - Path containing folders of frames
                - List of the names of these folders
                - Partitions: [(ids_x(x, 10), ids_y(x, 4))]
        '''

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dataset_partitions = dataset_partitions
        self.batch_size = np.min([len(x[1]) for x in self.dataset_partitions]) # minimo numero di sequenze per area

        self.past_frames = past_frames
        self.future_frames = future_frames
        self.caching = caching

        self.batch_size = batch_size
        self.blurry_filter_size = (3, 3)
        self.downsampling = downsampling

        self.root = root

        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_memory = buffer_memory
        self.buffer_hit_ratio = 0

        self.preprocessing = Preprocessing()
        self.dynamicity = dynamicity


    def get_datapoint(self, area_index, sequence_index, check=True):  
        '''
            Generates a single datapoint on the fly (cached)
            Inputs:
                - index of the area
                - index of the sequence
                - flag to check sequence validity
            Outputs:
                - case 1: valid sequence        ->  (X, Y)
                - case 2: non-valid sequence    ->  None
        '''

        # Initialization
        X = None
        Y = None

        area = self.dataset_partitions[area_index]
        sequence = self.dataset_partitions[area_index][1][sequence_index]

        # --- BTM
        btm_filenames = [x for x in os.listdir(self.root + self.dataset_partitions[area_index][0]) if x.endswith(".BTM")]
        if len(btm_filenames) == 0:
            raise Exception("No BTM map found for the area {}".format(self.dataset_partitions[area_index][0]))
        btm = pd.read_csv(self.root + self.dataset_partitions[area_index][0] + "/" + btm_filenames[0],' ',header=None).values

        # --- Preprocessing
        if self.downsampling:
            btm = cv.GaussianBlur(btm, self.blurry_filter_size, 0)
            btm = cv.pyrDown(btm)

        # riduzione valori il sottraendo minimo
        min_btm = np.min(btm)
        btm = btm - min_btm

        btm.resize(btm.shape[0], btm.shape[1], 1)
        btm_x = np.tile(btm, (self.past_frames, 1, 1, 1))

        deps = None
        vvx_s = None
        vvy_s = None

        framestart = int(sequence.replace("id-", ""))

        # Starts from the right frame
        for k in range(framestart, framestart + self.past_frames + self.future_frames):

            # id area -> id frame
            gid = "{}-{}-{}".format(area_index, sequence, k)

            # Parameters
            extensions = ["DEP", "VVX", "VVY"]
            matrices = []

            # Gets datapoint filename
            dep_filenames = [x for x in os.listdir(self.root + self.dataset_partitions[area_index][0]) if
                            x.endswith(".DEP")]

            if len(dep_filenames) == 0:
                raise Exception("No DEP maps found for the area {}".format(self.dataset_partitions[area_index][0]))

            # asserting that all maps are named with the same prefix
            dep_filename = dep_filenames[0].split(".")[0][:-4]

            # 1 frame -> 3 matrices (3 extensions)
            for i, ext in enumerate(extensions):
                
                global_id = "{}-{}".format(i, gid)  # indice linearizzato globale

                # ----- Cache
                if self.caching:
                    cache_frame = self.buffer_lookup(
                        global_id
                    )
                    if cache_frame is False:
                        frame = pd.read_csv(self.root + self.dataset_partitions[area_index][0] + "/{}{:04d}.{}".format(dep_filename,k, ext), ' ', header=None).values
                        self.buffer_push(global_id, frame)
                    else:
                        frame = cache_frame

                # ----- No cache
                else:
                    frame = pd.read_csv(self.root + self.dataset_partitions[area_index][0] + "/{}{:04d}.{}".format(dep_filename,k, ext), ' ', header=None).values

                # --- On-spot Gaussian Blurring
                if self.downsampling:
                    frame = cv.GaussianBlur(frame, self.blurry_filter_size, 0)
                    frame = cv.pyrDown(frame)
                matrices.append(frame)

            frame, vvx, vvy = matrices

            # ---

            if deps is None:
                deps = np.array([frame])
            else:
                deps = np.concatenate((deps, np.array([frame])))

            if vvx_s is None:
                vvx_s = np.array([vvx])
            else:
                vvx_s = np.concatenate((vvx_s, np.array([vvx])))

            if vvy_s is None:
                vvy_s = np.array([vvy])
            else:
                vvy_s = np.concatenate((vvy_s, np.array([vvy])))

        # ---------

        deps[deps > 10e5] = 0
        vvx_s[vvx_s > 10e5] = 0
        vvy_s[vvy_s > 10e5] = 0
        btm_x[btm_x > 10e5] = 0

        # --- X
        x_dep = deps[:self.past_frames]
        x_dep.resize((x_dep.shape[0], x_dep.shape[1], x_dep.shape[2], 1))

        x_vx = vvx_s[:self.past_frames]
        x_vx.resize((x_vx.shape[0], x_vx.shape[1], x_vx.shape[2], 1))

        x_vy = vvy_s[:self.past_frames]
        x_vy.resize((x_vy.shape[0], x_vy.shape[1], x_vy.shape[2], 1))

        x = np.concatenate((x_dep, x_vx, x_vy, btm_x), axis=3)

        # --- Y
        y_dep = deps[self.past_frames:]
        y_dep.resize((y_dep.shape[0], y_dep.shape[1], y_dep.shape[2], 1))

        y_vx = vvx_s[self.past_frames:]
        y_vx.resize((y_vx.shape[0], y_vx.shape[1], y_vx.shape[2], 1))

        y_vy = vvy_s[self.past_frames:]
        y_vy.resize((y_vy.shape[0], y_vy.shape[1], y_vy.shape[2], 1))

        y = np.concatenate((y_dep, y_vx, y_vy), axis=3)

        # filtering
        if check:
            score, valid = self.preprocessing.eval_datapoint(x[:,:,:,:3], y, self.dynamicity)

            if valid:

                if X is None: X = np.expand_dims(x,0)
                else: X = np.concatenate((X, np.expand_dims(x,0)))

                if Y is None: Y = np.expand_dims(y,0)
                else: Y = np.concatenate((Y, np.expand_dims(y,0)))

                return X, Y
            else:
                return (None, None)

        else:
            if X is None: X = np.expand_dims(x,0)
            else: X = np.concatenate((X, np.expand_dims(x,0)))

            if Y is None: Y = np.expand_dims(y,0)
            else: Y = np.concatenate((Y, np.expand_dims(y,0)))

            return X, Y


    # ------------------------------------

    def buffer_lookup(self, k):
        ''' Get sequence (datapoint) from cache given the start frame global id '''

        if self.caching:
            for i, x in enumerate(self.buffer):
                # Returns found record
                if x["global_id"] == k:
                    self.buffer[i]["fresh"] += 1
                    return x["value"]

                # Set any read record to 0 (second chance)
                elif self.buffer[i]["fresh"] != 0:
                    self.buffer[i]["fresh"] -= 1

        return False

    def buffer_push(self, k, x):
        ''' Add sequence (datapoint) to cache with start frame global id '''

        if self.caching:
            # Makes space
            if len(self.buffer) >= self.buffer_size:
                for i, j in enumerate(self.buffer):
                    if j["fresh"] == 0:
                        del self.buffer[i]
            # Push
            self.buffer.append({'fresh': self.buffer_memory, 'global_id': k, 'value': x})