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

TSBOARD_SMOOTHING = 0.7
#df.ewm(alpha=(1 - TSBOARD_SMOOTHING))
print(df)

parts = args.path.split("/")
parts[-1] = "smooth_" + parts[-1]
df.to_csv("/".join(parts))

df.plot()
plt.show()

while True:
    pass
