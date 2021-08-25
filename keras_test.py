from tensorflow import keras
from tensorflow.keras import layers
from utils.data_legacy.dataloader_keras import DataLoader
import matplotlib as mat
import argparse
import os
from datetime import datetime

mat.use("Agg") # headless mode

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

parser.add_argument('-root', dest='root',
                    help='root path with the simulation files (cropped and stored in folders)')
parser.add_argument('-p', dest='past_frames', default=4, type=int,
                    help='number of past frames')       
parser.add_argument('-f', dest='future_frames', default=1, type=int, 
                    help='number of future frames')     
parser.add_argument('-image_size', dest='image_size', default=256, type=int,
                    help='image size (width = height)')
parser.add_argument('-b', dest='batch_size', default=4, type=int,
                    help='batch size') 
parser.add_argument('-bs', dest='buffer_size', default=1e3, type=float,
                    help='size of the cache memory (in entries)')
parser.add_argument('-t', dest='buffer_memory', default=100, type=int,
                    help='temporal length of the cache memory (in iterations)')
parser.add_argument('-partial', dest='partial', default=None, type=float,
                    help='percentage of portion of dataset (to load partial, lighter chunks)')                                                                                                                                                              
parser.add_argument('-test_size', dest='test_size', default = 0.2,
                    help='Test size for the split')
parser.add_argument('-shuffle', dest='shuffle', default=True, type=str2bool,
                    help='Shuffle the dataset')
parser.add_argument('-d', dest='dynamicity', default=1e-3, type=float,
                    help='dynamicity rate (to filter out "dynamic" sequences)')                                                                                                  
parser.add_argument('-epochs', dest='epochs', default=100, type=float,
                    help='# of epochs')                                                                                                  

# -------------------------------
  

args = parser.parse_args()

if args.root is None and args.numpy_file is None:
    print("required: please specify a root path: -r /path")
    exit()
   
# -------------- Setting up the run

num_run = len(os.listdir("runs/")) + 1
now = datetime.now()
foldername = "train_{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
os.mkdir("runs/" + foldername)

dataset = DataLoader(
    root=args.root, 
    numpy_file=None, 
    past_frames=args.past_frames, 
    future_frames=args.future_frames, 
    image_size=args.image_size, 
    batch_size=1, 
    buffer_size=args.buffer_size, 
    buffer_memory=args.buffer_memory, 
    dynamicity=args.dynamicity, 
    partial=args.partial, 
    test_size=args.test_size, 
    clipping_threshold=1e5, 
    shuffle=args.shuffle
)

x_train, y_train = dataset.get_train()
x_test, y_test = dataset.get_test()

x_train = x_train[:,0,:,:,:]
y_train = y_train[:,0,:,:,:]

x_test = x_test[:,0,:,:,:]
y_test = y_test[:,0,:,:,:]


# Construct the input layer with no definite frame size.
inp = layers.Input(shape=(None, *x_train.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(),
)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = args.epochs

# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=args.batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr],
)

model.save("runs/" + foldername + "/model")