import numpy as np
import os
from tqdm import tqdm
import math
import cv2 as cv
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

    def __init__(self, root, past_frames, future_frames, caching=False, dynamicity=100, shuffle=True, image_size=768, batch_size=4, workers=4, filtering=True, test_size=0.1, val_size=0.1, partial=None):
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

    def prepare_data(self):
        dataset = SWEDataset(
            root=self.root,
            past_frames=self.past_frames,
            future_frames=self.future_frames,
            partial=self.partial,
            filtering=self.filtering,
            image_size=self.image_size,
            shuffle=self.shuffle,
            dynamicity=self.dynamicity,
            caching=self.caching
        )

        test_len = int(max(1, len(dataset) * self.test_size))
        val_len = int(max(1, len(dataset) * self.val_size))
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
    def __init__(self, past_frames, future_frames, root, shuffle=True, filtering=True, caching=False, numpy_file=None, image_size=256, batch_size=4,
                 dynamicity=1e-3, buffer_memory=100, buffer_size=1000, partial=None):
        ''' Initiates the dataloading process '''

        if root is None and numpy_file is None:
            raise Exception("Please specify either a root path or a numpy file: -r /path -npy /dataset.py")

        self.filtering = filtering

        self.partitions = DataPartitions(
            past_frames=past_frames,
            future_frames=future_frames,
            root=root,
            partial=partial,
            shuffle=shuffle
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
            downsampling=False,
            dynamicity=dynamicity,
            caching=caching
        )

        self.X, self.Y = self.dataset.get_data()
        self.X = self.X.transpose(0, 1, 4, 2, 3)
        self.Y = self.Y.transpose(0, 1, 4, 2, 3)

    # ------------------------------

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

# ------------------------------------------------------------------------------

class DataPartitions():
    def __init__(self, past_frames, future_frames, root, partial=None, shuffle=True):

        self.root = root
        self.past_frames = past_frames
        self.future_frames = future_frames

        self.dataset_partitions = []
        self.areas = []
        self.partial = partial
        self.shuffle = shuffle

        self.create_partitions()

    def get_areas(self):
        ''' Returns list of areas (source folder name)
        '''
        return self.areas

    def get_partitions(self):
        ''' Returns test dataset partitions
            Partitions: [(ids_x(x, 10), ids_y(x, 4))]
        '''

        # Shuffle areas
        if self.shuffle:
            np.random.shuffle(self.dataset_partitions)
        return self.dataset_partitions

    def create_partitions(self):
        ''' Creates the keras.DataLoader partitions (input-label)
            - file format: mini-xxxx.DEP/VVX/VVY/BTM
        '''

        self.areas = os.listdir(self.root)
        self.areas = [x for x in sorted(self.areas) if x.startswith("mini-") and os.path.isdir(self.root + x)]

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

            # Shuffle sequences
            if self.shuffle:
                np.random.shuffle(partition_raw)

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
        self.batch_size = np.min([len(x[1]) for x in self.dataset_partitions])  # minimo numero di sequenze per area

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

    def get_data(self):
        'Generates batches of datapoints'
        X, Y = self.__data_generation()  # seq, t, h, w, c
        return X, Y

    def __data_generation(self):
        'Generates the raw sequence of datapoints (filtered)'

        # stats
        accesses = 0
        hits = 0

        # Initialization
        X = None
        Y = None

        print("[x] {} areas found".format(len(self.dataset_partitions)))

        # For each area
        for area_index, area in tqdm(enumerate(self.dataset_partitions)):
            # For each sequence
            loaded = 0

            print("Area {} ".format(area_index), end="", flush=True)
            for i, sequence in enumerate(area[1]):

                # --- BTM
                btm_filenames = [x for x in os.listdir(self.root + self.dataset_partitions[area_index][0]) if
                                 x.endswith(".BTM")]
                if len(btm_filenames) == 0:
                    raise Exception("No BTM map found for the area {}".format(self.dataset_partitions[area_index][0]))
                btm = iter_loadtxt(self.root + self.dataset_partitions[area_index][0] + "/" + btm_filenames[0], delimiter=" ")


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
                        raise Exception(
                            "No DEP maps found for the area {}".format(self.dataset_partitions[area_index][0]))

                    # asserting that all maps are named with the same prefix
                    dep_filename = dep_filenames[0].split(".")[0][:-4]

                    # 1 frame -> 3 matrices (3 extensions)
                    for i, ext in enumerate(extensions):
                        accesses += 1
                        global_id = "{}-{}".format(i, gid)  # indice linearizzato globale

                        cache = self.buffer_lookup(
                            global_id
                        )
                        if cache is False:
                            frame = iter_loadtxt(
                                self.root + self.dataset_partitions[area_index][0] + "/{}{:04d}.{}".format(dep_filename,
                                                                                                           k, ext), delimiter=" ")
                            # --- On-spot Gaussian Blurring
                            if self.downsampling:
                                frame = cv.GaussianBlur(frame, self.blurry_filter_size, 0)
                                frame = cv.pyrDown(frame)
                            # ----
                            self.buffer_push(global_id, frame)
                        else:
                            frame = cache
                            hits += 1
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
                score, valid = self.preprocessing.eval_datapoint(x[:, :, :, :3], y, self.dynamicity)

                if valid:
                    loaded += 1

                    if X is None:
                        X = np.expand_dims(x, 0)
                    else:
                        X = np.concatenate((X, np.expand_dims(x, 0)))

                    if Y is None:
                        Y = np.expand_dims(y, 0)
                    else:
                        Y = np.concatenate((Y, np.expand_dims(y, 0)))

                    print("x ", end="", flush=True)
                else:
                    print("- ", end="", flush=True)

            print(
                "\n[{}%] {} valid sequences loaded".format(round((area_index + 1) / len(self.dataset_partitions) * 100),
                                                           loaded))

        # Buffer ratio calculation
        if accesses != 0:
            self.buffer_hit_ratio = self.buffer_hit_ratio * 0.5 + 0.5 * (hits / accesses)

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