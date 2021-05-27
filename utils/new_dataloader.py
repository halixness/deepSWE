import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm


class DataPartitions():
    def __init__(self, past_frames, future_frames, frame_distance, root, partial=None):

        self.root = root
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.frame_distance = frame_distance

        self.dataset_partitions = []
        self.areas = []
        self.partial = partial

        self.create_partitions()

    def get_areas(self):
        ''' Returns list of areas (source folder name)
        '''

        return self.areas

    def get_partitions(self):
        ''' Returns dataset partitions
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
            size = n_frames - (self.past_frames * self.frame_distance) - (self.future_frames * self.frame_distance)

            train_sequence = dict()
            target_sequence = dict()

            # -----------
            for i in range(size):
                past_stop = i + (self.past_frames * self.frame_distance)
                future_stop = i + (self.past_frames * self.frame_distance) + (self.future_frames * self.frame_distance)

                train_sequence["id-{}".format(i)] = list(range(i, past_stop, self.frame_distance))
                target_sequence["id-{}".format(i)] = list(range(past_stop, future_stop, self.frame_distance))

            self.dataset_partitions.append((train_sequence, target_sequence))


# ------------------------------------------------------------------------------
class FrameSequence:
    # area: index of area dataset
    # frame_start: index of local starting frame
    def __init__(self, area, frame_start):
        self.area = area
        self.frame_start = frame_start

    def get_global_frame_start(self):
        return self.area * self.frame_start


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, root, filenames, dataset_partitions, past_frames, future_frames, frame_distance, input_dim, output_dim,
                 buffer_memory=1, buffer_size=100, batch_size=16, n_channels=1, shuffle=True):
        '''
            Data Generator
            Inputs:  
                
                - Path containing folders of frames
                - List of the names of these folders
                - Partitions: [(ids_x(x, 10), ids_y(x, 4))]
        '''

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.batch_size = batch_size

        self.dataset_partitions = dataset_partitions

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.frame_distance = frame_distance

        self.root = root
        self.filenames = filenames

        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_memory = buffer_memory

        # Shuffle and initialize
        self.on_epoch_end()

    def get_X(self):
        num_batches = self.get_total_batches()
        X = np.zeros((num_batches, self.batch_size, *self.input_dim))

        for i in tqdm(range(num_batches)):
            x, _ = self.__getitem__(i)
            X[i] = x

        return X

    def get_Y(self):
        num_batches = self.get_total_batches()
        Y = np.zeros((num_batches, self.batch_size, *self.output_dim))

        for i in tqdm(range(num_batches)):
            _, y = self.__getitem__(i)
            Y[i] = y

        return Y

    def buffer_lookup(self, k):
        ''' Get sequence (datapoint) from cache given the start frame global id '''

        for i, x in enumerate(self.buffer):
            # Returns found record
            if x["global_id"] == k:
                self.buffer[i]["fresh"] += 1
                return x["value"]

            # Set any read record to 0 (second chance)
            elif self.buffer[i]["fresh"] is not 0:
                self.buffer[i]["fresh"] -= 1

        return False

    def buffer_push(self, k, x):
        ''' Add sequence (datapoint) to cache with start frame global id '''
        # Makes space
        if len(self.buffer) >= self.buffer_size:
            for i, j in enumerate(self.buffer):
                if j["fresh"] == 0:
                    del self.buffer[i]
        # Push
        self.buffer.append({'fresh': self.buffer_memory, 'global_id': k, 'value': x})

    def get_total_frames(self):
        'Sum of #frames for each folder'

        total_frames = 0
        for area in self.filenames:
            total_frames += len([x for x in os.listdir(self.root + area) if x.endswith(".DEP")]) - 1

        return total_frames

    def get_frame_indices(self):
        'Sum of #frames for each folder'

        indexes = list(range(self.get_total_frames())) # all frame indexes (global)
        considerable_indexes = []

        x = 0 # indice fino alla fine dell'area precedente
        for i, area in enumerate(self.filenames):
            num_frames = (len([x for x in os.listdir(self.root + area) if x.endswith(".DEP")]) - 1)
            considerable_frames = num_frames - (self.past_frames * self.frame_distance) + (self.future_frames * self.frame_distance)

            # takes only the K starting frames so that sliding windows won't overflow
            considerable_indexes += indexes[x:considerable_frames]
            x += num_frames

        return considerable_indexes

    def get_total_batches(self):
        'Sum of #batches for each folder'

        batches = 0

        # sum of batches per dataset
        for p in self.dataset_partitions:
            batches += int(np.floor(len(p[0]) / self.batch_size))

        return batches

    def __len__(self):
        'Denotes the number of batches per epoch'

        return self.get_total_batches()

    def __getitem__(self, index):
        'Generate one batch of data'

        # Seleziona batch_size sequenze, ciascuna indicata dal frame di inizio
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Shuffles indexes for each epoch, determines total indexes'

        # Shuffle di tutti gli id frames globali (attinge da più datasets a random)
        self.indexes = self.get_frame_indices()

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_dataset_id(self, index):
        'Get #folder by #global_frame'

        curmax = 0

        # Get index of which dataset it belongs to
        for i, p in enumerate(self.dataset_partitions):
            curmax += len(p[0])
            if index < curmax:
                return i
        return -1

    def get_local_id(self, global_id, dataset_id):
        'Get #local_frame by #global_frame'

        # remove lens of all the previous ones
        for i in range(dataset_id):
            global_id -= len(self.dataset_partitions[i][0])

        return global_id

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size sequences'

        # Initialization
        X = np.empty((self.batch_size, *self.input_dim))
        y = np.empty((self.batch_size, *self.output_dim))

        # For each sequence (denoted by starting frame id)
        for i, global_id in enumerate(list_IDs_temp):

            # Id dataset da cui attingere
            # Id frame start nel dataset
            dataset_id = self.get_dataset_id(global_id)
            local_id = self.get_local_id(global_id, dataset_id)

            # -1 per il BTM che non conta
            size = len([x for x in os.listdir(self.root + self.filenames[dataset_id]) if x.endswith(".DEP")]) - 1

            # Carica file BTM
            btm = np.loadtxt(self.root + self.filenames[dataset_id] + "/mini-decoded.BTM")
            btm.resize(btm.shape[0], btm.shape[1], 1)

            btm_x = np.tile(btm, (self.past_frames, 1, 1, 1))

            # Dalla cartella dataset scelta, leggo tutti i frames
            deps = None
            velocities = None

            # ------ DEP, Velocities train + pred
            for j in range(size):

                # ----
                i_frame = "1-{:04d}".format(global_id)
                cache = self.buffer_lookup(i_frame)

                if cache is False:
                    frame = np.loadtxt(self.root + self.filenames[dataset_id] + "/mini-decoded-{:04d}.DEP".format(j))
                    self.buffer_push(i_frame, frame)
                else:
                    frame = cache

                # ----
                i_vvx = "2-{:04d}".format(global_id)
                cache = self.buffer_lookup(i_vvx)

                if cache is False:
                    vvx = np.loadtxt(self.root + self.filenames[dataset_id] + "/mini-decoded-{:04d}.VVX".format(j))
                    self.buffer_push(i_vvx, vvx)
                else:
                    vvx = cache

                # ----
                i_vvy = "3-{:04d}".format(global_id)
                cache = self.buffer_lookup(i_vvy)

                if cache is False:
                    vvy = np.loadtxt(self.root + self.filenames[dataset_id] + "/mini-decoded-{:04d}.VVY".format(j))
                    self.buffer_push(i_vvy, vvy)
                else:
                    vvy = cache

                # ---
                velocity = np.sqrt(vvx ** 2 + vvy ** 2)

                if deps is None:
                    deps = np.array([frame])
                else:
                    deps = np.concatenate((deps, np.array([frame])))

                if velocities is None:
                    velocities = np.array([velocity])
                else:
                    velocities = np.concatenate((velocities, np.array([velocity])))

            deps[deps > 10e5] = 0
            velocities[velocities > 10e5] = 0

            # partition[id] -> start position for train window
            # ID
            # channels, frames, height, width
            r = deps[self.dataset_partitions[dataset_id][0]["id-{}".format(local_id)]]
            r.resize((r.shape[0], r.shape[1], r.shape[2], 1))

            r2 = velocities[self.dataset_partitions[dataset_id][0]["id-{}".format(local_id)]]
            r2.resize((r2.shape[0], r2.shape[1], r2.shape[2], 1))

            # frame, channel, width, height
            X[i, :, ] = np.concatenate((r, r2, btm_x), axis=3)

            # y: lista indici già specificata nelle partitions
            r = deps[self.dataset_partitions[dataset_id][1]["id-{}".format(local_id)]]
            r.resize((r.shape[0], r.shape[1], r.shape[2], 1))

            # we predict only DEP
            y[i, :, ] = r

        return X, y
