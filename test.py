
from utils.dataloader import DataPartitions, DataGenerator
import numpy as np

partitions = DataPartitions(
    past_frames=4,
    future_frames=8,
    root="../datasets/arda/04_21_full/"
)

dataset = DataGenerator(
    root="../datasets/arda/04_21_full/",
    dataset_partitions=partitions.get_partitions(),
    past_frames=partitions.past_frames,
    future_frames=partitions.future_frames,
    input_dim=(partitions.past_frames, 256, 256, 3),
    output_dim=(partitions.future_frames, 256, 256, 1),
    batch_size=16,
    n_channels=1,
    shuffle=True,
    buffer_size=1e3,
    buffer_memory=100
)

print(np.mean(dataset.__getitem__(0)[0][0]))
print(np.mean(dataset.__getitem__(0)[0][1]))
print(np.mean(dataset.__getitem__(0)[0][2]))

