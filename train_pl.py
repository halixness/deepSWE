import torch as th
import pytorch_lightning as pl
from utils.dataloader_pl import SWEDataModule
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from models_lightning.ae.main import deepSWE

# ----------------------------------------------------

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

args = parser.parse_args()

if args.root is None:
    print("required: please specify a root path: -r /path")
    exit()

# ----------------------------------------------------

dataset = SWEDataModule(
    root=args.root, 
    test_size=args.test_size,
    val_size=args.val_size,
    past_frames=args.past_frames,
    future_frames=args.future_frames,
    partial=args.partial
)

model = deepSWE(nf=4, in_chan=4)
logger = TensorBoardLogger("tb_logs", name="deepSWE")

trainer = pl.Trainer(logger=logger, log_every_n_steps=10)
trainer.fit(model, dataset)