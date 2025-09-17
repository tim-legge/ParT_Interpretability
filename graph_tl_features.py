### Pulls feature set from the persistent volume and graphs, leaving as histogram .npy files.

import os
import numpy as np
import matplotlib
from graph_jc_features import mask_out, compile_histograms

def sort_feats(masked_feats, masked_vecs):

    feats_dict = {
        'part_pt_log': [],
        'part_e_log': [],
        'part_log_ptrel': [],
        'part_log_erel': [],
        'part_deltaR': [],
        'part_deta': [],
        'part_dphi': [],
    }

    for jet in masked_feats:
        for idx, key in enumerate(feats_dict.keys()):
            feats_dict[key].extend(jet[idx])
        
    vecs_dict = {
        'part_px': [],
        'part_py': [],
        'part_pz': [],
        'part_energy': [],
    }

    for jet in masked_vecs:
        for idx, key in enumerate(vecs_dict.keys()):
            vecs_dict[key].extend(jet[idx])

    return feats_dict, vecs_dict

if __name__ == "__main__":

    features_dir = '/part-vol-3/timlegge-ParT-trained/data_from_tl_train/'
    output_dir = '/part-vol-3/timlegge-ParT-trained/histograms_tl_feats/'
    assert os.path.exists(features_dir), f"Features directory {features_dir} does not exist."
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."
    print(f"Reading features from {features_dir} and saving histograms to {output_dir}")

    feature_files = [f for f in sorted(os.listdir(features_dir)) if 'pf_features' in f and f.endswith('.npy')]
    mask_files = [f for f in sorted(os.listdir(features_dir)) if 'pf_mask' in f and f.endswith('.npy')]
    label_files = [f for f in sorted(os.listdir(features_dir)) if 'labels' in f and f.endswith('.npy')]
    vector_files = [f for f in sorted(os.listdir(features_dir)) if 'pf_vectors' in f and f.endswith('.npy')]

    compile_histograms(data_dir=features_dir, output_dir=output_dir)