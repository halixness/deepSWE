import numpy as np
import os
from tqdm import tqdm
import math
import cv2 as cv
from utils.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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

class DataLoader():
    def __init__(self, numpy_file, past_frames, future_frames, root, buffer_size, image_size, batch_size, buffer_memory, dynamicity, downsampling=False, partial=None, clipping_threshold=1e5, test_size=None, shuffle=True):
        ''' Initiates the dataloading process '''

        if root is None and numpy_file is None:
            raise Exception("Please specify either a root path or a numpy file: -r /path -npy /dataset.py")

        # Loads from disk
        if numpy_file is None:
            print("[~] Loading from disk...")

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
                downsampling=False,
                dynamicity=dynamicity
            )
            self.X, self.Y, self.extra_batch = self.dataset.get_data()

            self.X[self.X > clipping_threshold] = 0
            self.Y[self.Y > clipping_threshold] = 0

        # Loads from stored file
        else:
            print("[~] Loading from npy file...")

            self.X, self.Y, self.extra_batch = np.load(numpy_file, allow_pickle=True)
            print("[!] Successfully loaded dataset from {} \nX.shape: {}\nY.shape: {}\n".format(
                numpy_file, self.X.shape, self.Y.shape
            ))

        #  Shuffling the dataset
        if shuffle:
            self.X, self.Y = unison_shuffled_copies(self.X, self.Y)

        print("[~] Performing train/test split...")

        # Splitting
        if test_size is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)      

        print("[!] Preprocessing completed!")

    def get_dataset(self):
        ''' Get complete dataset '''
        return self.X, self.Y

    def save_dataset(self, filename):
        ''' Store the dataset -> npy '''
        #try:
        np.save(filename, (self.X, self.Y), dtype=object)
        print("[~] Dataset stored to -> {}".format(filename))
        #    return True
        #except:
        #    print("[!] Error while saving the npy file")
        #    return False

    def get_train(self):
        ''' Get train dataset '''
        return self.X_train, self.y_train

    def get_test(self):
        ''' Get test dataset '''
        return self.X_test, self.y_test

    def print_stats(self):
        ''' Prints dataset statistics'''

        print("DEP min: {}\nVEL min: {}\nBTM min: {}".format(
            np.min(self.X_train[:, :, :, :, :, 0]),
            np.min(self.X_train[:, :, :, :, :, 1]),
            np.min(self.X_train[:, :, :, :, :, 2])
        ))

        print("DEP max: {}\nVEL max: {}\nBTM max: {}".format(
            np.max(self.X_train[:, :, :, :, :, 0]),
            np.max(self.X_train[:, :, :, :, :, 1]),
            np.max(self.X_train[:, :, :, :, :, 2])
        ))                

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
                 shuffle=True, buffer_memory=1, buffer_size=100, batch_size=16, caching=True, downsampling=False, dynamicity=0.1):
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

        if shuffle:
            np.random.shuffle(self.dataset_partitions)

    def get_data(self):
        'Generates batches of datapoints'
        X, Y = self.__data_generation() # seq, t, h, w, c
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
                #btm = np.loadtxt(self.root + self.dataset_partitions[area_index][0] + "/" + btm_filenames[0])
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
                            #frame = np.loadtxt(self.root + self.dataset_partitions[area_index][0] + "/{}{:04d}.{}".format(dep_filename, k, ext))
                            frame = iter_loadtxt(self.root + self.dataset_partitions[area_index][0] + "/{}{:04d}.{}".format(dep_filename, k, ext), delimiter=" ")
                            
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
                sequence = np.concatenate((x[:,:,:,:3], y), axis=0)
                score, valid = self.preprocessing.eval_datapoint(sequence, self.dynamicity)

                if valid:

                    loaded += 1

                    if X is None: X = np.expand_dims(x,0)
                    else: X = np.concatenate((X, np.expand_dims(x,0)))

                    if Y is None: Y = np.expand_dims(y,0)
                    else: Y = np.concatenate((Y, np.expand_dims(y,0)))

                    print("o ", end='', flush=True)
                else:
                    print("x ", end='', flush=True)

            print("\n[{}%] {} valid sequences loaded".format(round((area_index+1)/len(self.dataset_partitions)*100), loaded))

        # Buffer ratio calculation
        if accesses != 0:
            self.buffer_hit_ratio = self.buffer_hit_ratio * 0.5 + 0.5 * (hits / accesses)

        return X, Y

    # ------------------------------------------

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
        sequence = sequence_index

        # --- BTM
        btm_filenames = [x for x in os.listdir(self.root + self.dataset_partitions[area_index][0]) if x.endswith(".BTM")]
        if len(btm_filenames) == 0:
            raise Exception("No BTM map found for the area {}".format(self.dataset_partitions[area_index][0]))
        #btm = np.loadtxt(self.root + self.dataset_partitions[area_index][0] + "/" + btm_filenames[0])
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
                raise Exception("No DEP maps found for the area {}".format(self.dataset_partitions[area_index][0]))

            # asserting that all maps are named with the same prefix
            dep_filename = dep_filenames[0].split(".")[0][:-4]

            # 1 frame -> 3 matrices (3 extensions)
            for i, ext in enumerate(extensions):
                
                global_id = "{}-{}".format(i, gid)  # indice linearizzato globale

                cache = self.buffer_lookup(
                    global_id
                )
                if cache is False:
                    #frame = np.loadtxt(self.root + self.dataset_partitions[area_index][0] + "/{}{:04d}.{}".format(dep_filename, k, ext))
                    frame = iter_loadtxt(self.root + self.dataset_partitions[area_index][0] + "/{}{:04d}.{}".format(dep_filename, k, ext), delimiter=" ")
                    
                    # --- On-spot Gaussian Blurring
                    if self.downsampling:
                        frame = cv.GaussianBlur(frame, self.blurry_filter_size, 0)
                        frame = cv.pyrDown(frame)
                    # ----
                    self.buffer_push(global_id, frame)
                else:
                    frame = cache

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

        deps[deps > 10e10] = 0
        vvx_s[vvx_s > 10e10] = 0
        vvy_s[vvy_s > 10e10] = 0
        btm_x[btm_x > 10e10] = 0

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
            sequence = np.concatenate((x[:,:,:,:3], y), axis=0)
            score, valid = self.preprocessing.eval_datapoint(sequence, self.dynamicity)

            if valid:

                if X is None: X = np.expand_dims(x,0)
                else: X = np.concatenate((X, np.expand_dims(x,0)))

                if Y is None: Y = np.expand_dims(y,0)
                else: Y = np.concatenate((Y, np.expand_dims(y,0)))

                return X, Y
            else:
                return None

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
