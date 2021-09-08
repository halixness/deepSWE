import imageio
from datetime import datetime
import os
import argparse

# --------------
parser = argparse.ArgumentParser(description='Tests a train model against a given dataset')
parser.add_argument('-path', dest='path', required=True,
                    help='Path to plot files')
parser.add_argument('-prefix', dest='prefix', required=True,
                    help='File prefix')
parser.add_argument('-name', dest='name', help='Animation name')
args = parser.parse_args()

if args.name is None:
    now = datetime.now()
    filename = now.strftime("%d_%m_%Y_%H_%M_%S") + ".gif"
else:
    filename = args.name + ".gif"
# --------------

# Build GIF
print("[~] Building gif")
with imageio.get_writer(filename, mode='I') as writer:
    # Reading the frames
    root = args.path
    files = os.listdir(args.path)
    frames = [x for x in sorted(files) if x.startswith(args.prefix) and os.path.isfile(root + x)]
    for filename in frames:
        image = imageio.imread(root + filename)
        # Building gif
        writer.append_data(image)
        print(". ", end="", flush=True)

