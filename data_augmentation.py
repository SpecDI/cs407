import numpy as np
import os
from glob import glob

# Path to action tubes
path_tubes = './action-tubes/training/all/completed/*/'
path_aug_tubes

# Determine average tubes per class
tube_counts = []
for action_class in glob(path_tubes):
    tube_counts.append(sum([len(files) for r, d, files in os.walk(action_class)]))

# Augment classes with less than average
for action_class in glob(path_tubes):
