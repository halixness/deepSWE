import numpy as np
import os
from tqdm import tqdm
import random


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
        # self.areas = ["mini-211-466-421-676"]

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
                 buffer_memory=1, buffer_size=100, batch_size=16, n_channels=1, shuffle=True, deep_shuffle=True, caching=True):
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
        self.deep_shuffle = shuffle and deep_shuffle
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.caching = caching

        self.root = root

        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_memory = buffer_memory
        self.buffer_hit_ratio = 0

        # Shuffle and initialize
        if self.shuffle:
            self.shuffle_sequences()

    def get_total_sequences(self):
        ' Numero totale di sequenze frames '
        return sum([len(a[1]) for a in self.dataset_partitions])

    def __len__(self):
        ' Numero totale di batches '
        return  \
            int(self.get_total_sequences() / self.batch_size) + (self.get_total_sequences() % self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'

        return self.__data_generation(index)

    def shuffle_sequences(self):
        ' Shuffle delle sequenze solo quando viene creato il dataloader '

        # Shuffle sequenze esterne
        random.shuffle(self.dataset_partitions)

        # Shuffle sequenze interne
        if self.deep_shuffle:
            for i in range(len(self.dataset_partitions)):
                n_sequences = list(range(len(self.dataset_partitions[i][1])))
                random.shuffle(n_sequences)  # shuffle indici sequenze X (le Y dipendono dall'accesso alla X)
                self.dataset_partitions[i] = (
                    self.dataset_partitions[i][0],
                    [self.dataset_partitions[i][1][x] for x in n_sequences], # rialloca le sequenze X randomizzate
                    self.dataset_partitions[i][2]
                )

    def get_X(self):
        'Generates only the X split array'
        num_batches = self.__len__()
        X = np.zeros((num_batches, self.batch_size, *self.input_dim))

        for i in tqdm(range(num_batches)):
            x, _ = self.__getitem__(i)
            X[i] = x

        return X

    def get_Y(self):
        'Generates only the Y split array'
        num_batches = self.__len__()
        Y = np.zeros((num_batches, self.batch_size, *self.output_dim))

        for i in tqdm(range(num_batches)):
            _, y = self.__getitem__(i)
            Y[i] = y

        return Y

    def __data_generation(self, batch_index):
        'Generates 1 batch with batch_size samples'

        # stats
        accesses = 0
        hits = 0

        # Initialization
        X = np.empty((self.batch_size, *self.input_dim))
        y = np.empty((self.batch_size, *self.output_dim))

        seq_start = batch_index * self.batch_size

        area_start = int(seq_start / len(self.dataset_partitions[0][1]))
        seq_start = int(seq_start % len(self.dataset_partitions[0][1]))
        cnt = self.batch_size

        i = area_start
        x = 0
        # Dall'area di partenza
        while i < len(self.dataset_partitions) and cnt > 0:

            if i == area_start: j = seq_start
            else: j = 0

            # Dalla sequenza di partenza
            seq_len = len(self.dataset_partitions[area_start][1])
            while j < seq_len and cnt > 0:

                # --- BTM
                #btm = np.loadtxt(self.root + self.dataset_partitions[i][0] + "/mini-decoded.BTM")
                #btm.resize(btm.shape[0], btm.shape[1], 1)
                #btm_x = np.tile(btm, (self.past_frames, 1, 1, 1))

                deps = None
                velocities = None

                # Scorre frames nella sequenza
                for k in range(j, j+self.past_frames+self.future_frames):

                    gid = i * self.batch_size * seq_len + k

                    # ---- DEP
                    accesses += 1
                    global_id = "1-{:04d}".format(gid) # indice linearizzato globale
                    cache = self.buffer_lookup(
                        global_id
                    )
                    if cache is False:
                        frame = np.loadtxt(self.root + self.dataset_partitions[i][0] + "/mini-decoded-{:04d}.DEP".format(k))
                        self.buffer_push(global_id, frame)
                    else:
                        frame = cache
                        hits += 1

                    # ---- VVX
                    accesses += 1
                    global_id = "2-{:04d}".format(gid)  # indice linearizzato globale
                    cache = self.buffer_lookup(
                        global_id
                    )
                    if cache is False:
                        vvx = np.loadtxt(
                            self.root + self.dataset_partitions[i][0] + "/mini-decoded-{:04d}.VVX".format(k))
                        self.buffer_push(global_id, vvx)
                    else:
                        vvx = cache
                        hits += 1

                    # ---- VVY
                    accesses += 1
                    global_id = "2-{:04d}".format(gid)  # indice linearizzato globale
                    cache = self.buffer_lookup(
                        global_id
                    )
                    if cache is False:
                        vvy = np.loadtxt(
                            self.root + self.dataset_partitions[i][0] + "/mini-decoded-{:04d}.VVY".format(k))
                        self.buffer_push(global_id, vvy)
                    else:
                        vvy = cache
                        hits += 1

                    # ---
                    velocity = np.sqrt(vvx ** 2 + vvy ** 2)

                    if deps is None: deps = np.array([frame])
                    else: deps = np.concatenate((deps, np.array([frame])))

                    if velocities is None: velocities = np.array([velocity])
                    else: velocities = np.concatenate((velocities, np.array([velocity])))

                # ---------

                deps[deps > 10e5] = 0
                velocities[velocities > 10e5] = 0
                #btm_x[btm_x > 10e5] = 0

                # --- X
                rx = deps[:self.past_frames]
                rx.resize((rx.shape[0], rx.shape[1], rx.shape[2], 1))

                rvx = velocities[:self.past_frames]
                rvx.resize((rvx.shape[0], rvx.shape[1], rvx.shape[2], 1))

                # x has 3 channels: dep, velocities, btm
                #X[x, :, ] = np.concatenate((rx, rvx, btm_x), axis=3)
                X[x, :, ] = np.concatenate((rx, rvx), axis=3)

                # --- Y
                ry = deps[self.past_frames:]
                ry.resize((ry.shape[0], ry.shape[1], ry.shape[2], 1))
                y[x, :, ] = ry # y has 1 channel: dep

                # Scorre sequenza
                cnt -= 1
                j += 1
                x += 1 # sequenza da zero

            # Scorre area
            i += 1

        if accesses != 0:
            self.buffer_hit_ratio = hits / accesses;

        return X, y

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


