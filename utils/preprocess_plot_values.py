import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt

# --------------
parser = argparse.ArgumentParser(description='Tests a train model against a given dataset')
parser.add_argument('-path', dest='path', required=True,
                    help='Path to plot files')
args = parser.parse_args()
# --------------

df = pd.read_csv(args.path)

# Cleaning and smoothing
TSBOARD_SMOOTHING = 0.9
# = df["Step"]
#df = df.drop(columns=["Wall time", "Step"])
df = df.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

# Saving to file
parts = args.path.split("/")
parts[-1] = "smooth_" + parts[-1]
df.to_csv("/".join(parts))

#df.plot()
#plt.show()
