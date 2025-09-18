### Pulls feature set from the persistent volume and graphs, leaving as histogram .npy files.

import os
import numpy as np
import matplotlib.pyplot as plt
from graph_jc_features import mask_out, sort_feats, compile_histograms

if __name__ == "__main__":

    features_dir = '/part-vol-3/timlegge-ParT-trained/data_from_tl_train/'
    output_dir = '/part-vol-3/timlegge-ParT-trained/histograms_tl_feats/'
    assert os.path.exists(features_dir), f"Features directory {features_dir} does not exist."
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."
    #print(f"Reading features from {features_dir} and saving histograms to {output_dir}")

    feats_dict = {
        'part_pt_log': [],
        'part_e_log': [],
        'part_log_ptrel': [],
        'part_log_erel': [],
        'part_deltaR': [],
        'part_deta': [],
        'part_dphi': [],
    }

    compile_histograms(data_dir=features_dir, output_dir=output_dir, feats_dict=feats_dict, vecs_dict=None)