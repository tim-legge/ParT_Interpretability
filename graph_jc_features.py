### Pulls feature set from the persistent volume and graphs, leaving as histogram .npy files.

import os
import numpy as np
import matplotlib

def mask_out(feats, vectors, mask):
    masked_feats = []
    masked_vecs = []
    #masked_feats, masked_vecs = mask_out(features, vectors, masks)
    for jet_idx, jet in enumerate(feats):
        masked_feats.append(jet[:, :np.sum(mask[jet_idx].astype('bool'))])
    for jet_idx, jet in enumerate(vectors):
        masked_vecs.append(jet[:, :np.sum(mask[jet_idx].astype('bool'))])
    
    return masked_feats, masked_vecs

def sort_feats(masked_feats, masked_vecs):

    feats_dict = {
        'part_pt_log': [],
        'part_e_log': [],
        'part_log_ptrel': [],
        'part_log_erel': [],
        'part_deltaR': [],
        'part_charge': [],
        'part_isChargedHadron': [],
        'part_isNeutralHadron': [],
        'part_isPhoton': [],
        'part_isElectron': [],
        'part_isMuon': [],
        'part_d0': [],
        'part_d0err': [],
        'part_dz': [],
        'part_dzerr': [],
        'part_deta': [],
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

def compile_histograms(data_dir, output_dir):
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."
    print(f"Processing data from {data_dir} and saving histograms to {output_dir}")

    labels_list = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']

    # look at the counter in the output dir
    if not os.path.exists(os.path.join(output_dir, 'hist_counter.txt')):
        with open(os.path.join(output_dir, 'hist_counter.txt'), "w") as f:
            f.write("0")
    with open(os.path.join(output_dir, 'hist_counter.txt'), "r") as f:
            hist_counter = int(f.read())
    while hist_counter < 50:
        print(f"Starting histogram collection from batch {hist_counter}")
        feats_hists = None
        vecs_hists = None
        for feat_file, mask_file, label_file, vec_file in zip(feature_files, mask_files, label_files, vector_files)[hist_counter*20:(hist_counter+1)*20]:
            feats = np.load(os.path.join(data_dir, feat_file))
            masks = np.load(os.path.join(data_dir, mask_file))
            labels = np.load(os.path.join(data_dir, label_file))
            vecs = np.load(os.path.join(data_dir, vec_file))

            # determine label
            if np.argmax(labels[0], axis=1) == np.argmax(labels[-1], axis=1):
                label = np.argmax(labels[0], axis=1)
                label_name = labels_list[label]
            else:
                print("Warning: Inconsistent labels in the batch, skipping this batch.")
                continue

            masked_feats, masked_vecs = mask_out(feats, vecs, masks)
            feats_dict, vecs_dict = sort_feats(masked_feats, masked_vecs)

            if feats_hists is None:
                feats_hists = [np.zeros(50) for key in feats_dict.keys()]
                vecs_hists = [np.zeros(50) for key in vecs_dict.keys()]

            # TODO: Figure out the right ranges for each feature (use 100k example jets probably)
            feats_hists += [np.histogram(feats_dict[key], bins=50) for key in feats_dict.keys()]
            vecs_hists += [np.histogram(vecs_dict[key], bins=50) for key in vecs_dict.keys()]
        print(f"Completed processing batch {hist_counter}, saving histograms.")
        for idx, key in enumerate(feats_dict.keys()):
            np.save(os.path.join(output_dir, f"{label_name}_hist_{key}.npy"), feats_hists[idx])
        for idx, key in enumerate(vecs_dict.keys()):
            np.save(os.path.join(output_dir, f"{label_name}_hist_{key}.npy"), vecs_hists[idx])
        hist_counter += 1
        with open(os.path.join(output_dir, 'hist_counter.txt'), "w") as f:
            f.write(str(hist_counter))
        print(f"Updated histogram counter to {hist_counter}")

if __name__ == "__main__":

    features_dir = '/part-vol-3/timlegge-ParT-trained/data_from_jc_train/'
    output_dir = '/part-vol-3/timlegge-ParT-trained/histograms_jc_feats/'
    assert os.path.exists(features_dir), f"Features directory {features_dir} does not exist."
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."
    print(f"Reading features from {features_dir} and saving histograms to {output_dir}")

    feature_files = [f for f in sorted(os.listdir(features_dir)) if 'pf_features' in f and f.endswith('.npy')]
    mask_files = [f for f in sorted(os.listdir(features_dir)) if 'pf_mask' in f and f.endswith('.npy')]
    label_files = [f for f in sorted(os.listdir(features_dir)) if 'labels' in f and f.endswith('.npy')]
    vector_files = [f for f in sorted(os.listdir(features_dir)) if 'pf_vectors' in f and f.endswith('.npy')]

    compile_histograms(data_dir=features_dir, output_dir=output_dir)