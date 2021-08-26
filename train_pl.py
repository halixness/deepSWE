import pytorch_lightning as pl
from utils.data_lightning.preloading import SWEDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from models_lightning.ae.main import deepSWE
import os
from datetime import datetime

# ----------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Trains a given model on a given dataset')

parser.add_argument('-test_size', dest='test_size', default = 0.1,
                    help='Test size for the split')
parser.add_argument('-val_size', dest='val_size', default = 0.1,
                    help='Validation size for the split')
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
parser.add_argument('-workers', dest='workers', default=4, type=int,
                    help='Number of cpu threads for the data loader')
parser.add_argument('-filtering', dest='filtering', default=False, type=str2bool,
                    help='Apply sequence dynamicity filtering')
parser.add_argument('-gpus', dest='gpus', default=1, type=int,
                    help='Number of GPUs')

args = parser.parse_args()

if args.root is None:
    print("required: please specify a root path: -r /path")
    exit()

num_run = len(os.listdir("runs/")) + 1
now = datetime.now()
dest = "runs/train_{}_{}".format(num_run, now.strftime("%d_%m_%Y_%H_%M_%S"))
os.mkdir(dest)

# ----------------------------------------------------

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
    shuffle=False
)

model = deepSWE(
    nf=args.filters,
    in_chan=4,
    out_chan=3,
    future_frames=args.future_frames,
    image_size=args.image_size
)
logger = TensorBoardLogger("tb_logs", name="deepSWE")

trainer = pl.Trainer(
    gpus=args.gpus,
    logger=logger,
    log_every_n_steps=1,
    precision=64,
    default_root_dir=dest
)
trainer.fit(model, dataset)