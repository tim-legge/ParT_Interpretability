# Compile distributions from persistent volume storage

import numpy as np
import matplotlib.pyplot as plt
import os

# Load TL files from /batched_hists

stem = 'tl_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_hist_distribution_batch_0.npy'):
    f = np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_hist_distribution_batch_0.npy', allow_pickle=True)
    compiled = f

print(compiled.shape)
        