import argparse
import numpy as np
import matplotlib.pyplot as plt

# --------------
parser = argparse.ArgumentParser(description='Preprocessing slurm training times')
parser.add_argument('-path', dest='path', required=True,
                    help='Path to slurm file')
args = parser.parse_args()
# --------------

file = open(args.path, "r")
line = file.readline()
times = []
hours = 0

while line:
    if "fwd_time" in line:
        parts = line.split("it [")[1]
        time = parts.split(",")[0]
        time = datetime.datetime.strptime(time, "%M:%S")
        print(time)
        break
        times.append(time)
        hours += time / 3600
    line = file.readline()
file.close() 

print("Avg. epoch time: {:.2f}".format(np.mean(times)))
print("Training time (hours): {:.2f}".format(hours))