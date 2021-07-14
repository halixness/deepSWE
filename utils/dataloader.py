import numpy as np
import os
from tqdm import tqdm
import math
import cv2 as cv
from utils.preprocessing import Preprocessing

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
                 buffer_memory=1, buffer_size=100, batch_size=16, caching=True, downsampling=False, dynamicity=0.1):
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

    def get_data(self):
        'Generates batches of datapoints'
        X, Y = self.__data_generation()
        ex_X = None
        ex_Y = None

        batch_residual = X.shape[0] % self.batch_size
        n_batches = X.shape[0] // self.batch_size
        
        # In case of n_sequences =/= k*batch_size
        if batch_residual > 0:
            X_b = X[:-batch_residual].reshape((n_batches, self.batch_size, *self.input_dim))
            Y_b = Y[:-batch_residual].reshape((n_batches, self.batch_size, *self.output_dim))
            
            # extra batch with n < batch_size
            ex_X = np.array([[X[-batch_residual:]]])
            ex_Y = np.array([[Y[-batch_residual:]]])
            
        else:
            X_b = X.reshape((n_batches, self.batch_size, *self.input_dim))
            Y_b = Y.reshape((n_batches, self.batch_size, *self.output_dim))

        return X_b, Y_b, (ex_X, ex_Y)


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
        for area_index, area in enumerate(self.dataset_partitions):
            # For each sequence
            loaded = 0
            for i, sequence in enumerate(area[1]):

                # --- BTM
                btm_filenames = [x for x in os.listdir(self.root + self.dataset_partitions[area_index][0]) if x.endswith(".BTM")]
                if len(btm_filenames) == 0:
                    raise Exception("No BTM map found for the area {}".format(self.dataset_partitions[area_index][0]))
                btm = np.loadtxt(self.root + self.dataset_partitions[area_index][0] + "/" + btm_filenames[0])

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
                        accesses += 1
                        global_id = "{}-{}".format(i, gid)  # indice linearizzato globale

                        cache = self.buffer_lookup(
                            global_id
                        )
                        if cache is False:
                            frame = np.loadtxt(self.root + self.dataset_partitions[area_index][0] + "/{}{:04d}.{}".format(dep_filename, k, ext))
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
                    velocity = np.sqrt(vvx ** 2 + vvy ** 2)

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
                rx = deps[:self.past_frames]
                rx.resize((rx.shape[0], rx.shape[1], rx.shape[2], 1))

                rvx = vvx_s[:self.past_frames]
                rvx.resize((rvx.shape[0], rvx.shape[1], rvx.shape[2], 1))

                rvy = vvy_s[:self.past_frames]
                rvy.resize((rvy.shape[0], rvy.shape[1], rvy.shape[2], 1))

                # --- Y
                ry = deps[self.past_frames:]
                ry.resize((ry.shape[0], ry.shape[1], ry.shape[2], 1))

                rvx = vvx_s[self.past_frames:]
                rvx.resize((rvx.shape[0], rvx.shape[1], rvx.shape[2], 1))

                rvy = vvy_s[self.past_frames:]
                rvy.resize((rvy.shape[0], rvy.shape[1], rvy.shape[2], 1))
                
                # --- Datapoint
                x = np.concatenate((rx, rvx, rvy, btm_x), axis=3)
                y = np.concatenate((ry, rvx, rvy), axis=3)

                # filtering 
                sequence = np.concatenate((x[:,:,:,:3], y), axis=0)
                score, valid = self.preprocessing.eval_datapoint(sequence, self.dynamicity)

                if valid:
                    
                    loaded += 1

                    if X is None: X = x
                    else: X = np.concatenate((X, x))

                    if Y is None: Y = y
                    else: Y = np.concatenate((Y, y))
                    
                    print(". ", end = '')
            
            print("\n[{}%] {} valid sequences loaded".format(round((area_index+1)/len(self.dataset_partitions)*100), loaded))

        # Buffer ratio calculation
        if accesses is not 0:
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


