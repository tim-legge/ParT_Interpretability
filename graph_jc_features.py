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
    
    print(masked_feats[0][0])
    
    return masked_feats, masked_vecs

def sort_feats(masked_feats, masked_vecs, feats_dict=None, vecs_dict=None):

    if feats_dict is None:
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
            'part_dphi': [],
        }
    else:
        feats_dict = feats_dict
    
    for jet_idx, jet in enumerate(masked_feats):
        for idx, key in enumerate(feats_dict.keys()):
        #if idx >= 13 and np.random.rand() < 0.01:
        #    print(f'jet_idx: {jet_idx}, idx: {idx}, key: {key}')
            try:
                feats_dict[key].extend(masked_feats[jet_idx][idx].flatten().tolist())
                
            except IndexError as e:
                print(f"IndexError for jet_idx {jet_idx}, idx {idx}, key {key}: {e}")
                continue
    if vecs_dict is None:    
        vecs_dict = {
            'part_px': [],
            'part_py': [],
            'part_pz': [],
            'part_energy': [],
        }
    else:
        vecs_dict = vecs_dict

    for jet_idx, jet in enumerate(masked_vecs):
        for idx, key in enumerate(vecs_dict.keys()):
            try:
                vecs_dict[key].extend(masked_feats[jet_idx][idx].flatten().tolist())
            except IndexError as e:
                print(f"IndexError for jet_idx {jet_idx}, idx {idx}, key {key}: {e}")
                continue

    return feats_dict, vecs_dict

def compile_histograms(data_dir, output_dir, feats_dict=None, vecs_dict=None, 
                        labelstype='jc'):
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."
    print(f"Processing data from {data_dir} and saving histograms to {output_dir}")
    if labelstype == 'jc' or None:
        labels_list = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
    elif labelstype == 'tl':
        labels_list = ['QCD', 'Tbqq']

    feat_ranges = {
        'part_pt_log': (-2.4, 6.3),
        'part_e_log': (-0.9, 7.0),
        'part_log_ptrel': (-8.6, -0.1),
        'part_log_erel': (-7.7, -0.2),
        'part_deltaR': (0, 3.0),
        'part_charge': (-1.0, 1.0),
        'part_isChargedHadron': (0.0, 1.0),
        'part_isNeutralHadron': (0.0, 1.0),
        'part_isPhoton': (0.0, 1.0),
        'part_isElectron': (0.0, 1.0),
        'part_isMuon': (0.0, 1.0),
        'part_d0': (-1.0, 1.0),
        'part_d0err': (0.0, 0.7),
        'part_dz': (-1.0, 1.0),
        'part_dzerr': (0.0, 2.2),
        'part_deta': (-2.9, 2.5),
        'part_dphi': (-0.9, 0.9),
    }

    vec_ranges = {
        'part_px': (-2.4, 6.3),
        'part_py': (-0.9, 7.0),
        'part_pz': (-8.7, 0.2),
        'part_energy': (-7.7, -0.2),
    }

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

        feature_files = [f for f in sorted(os.listdir(data_dir)) if 'pf_features' in f and f.endswith('.npy')]
        mask_files = [f for f in sorted(os.listdir(data_dir)) if 'pf_mask' in f and f.endswith('.npy')]
        label_files = [f for f in sorted(os.listdir(data_dir)) if 'labels' in f and f.endswith('.npy')]
        vector_files = [f for f in sorted(os.listdir(data_dir)) if 'pf_vectors' in f and f.endswith('.npy')]
        i = 0
        for feat_file, mask_file, label_file, vec_file in list(zip(feature_files, mask_files, label_files, vector_files))[hist_counter*20:(hist_counter+1)*20]:
            feats = np.load(os.path.join(data_dir, feat_file))
            masks = np.load(os.path.join(data_dir, mask_file))
            labels = np.load(os.path.join(data_dir, label_file))
            vecs = np.load(os.path.join(data_dir, vec_file))

            # determine label
            if labelstype == 'jc':
                if np.argmax(labels[0]) == np.argmax(labels[-1]):
                    label = np.argmax(labels[0])
                    label_name = labels_list[label]
                else:
                    print("Warning: Inconsistent labels in the batch, skipping this batch.")
                    continue
            elif labelstype == 'tl':
                if labels[0] == labels[-1]:
                    label = labels[0]
                    label_name = labels_list[label]
                else:
                    print("Warning: Inconsistent labels in the batch, skipping this batch.")
                    continue
            masked_feats, masked_vecs = mask_out(feats, vecs, masks)
            feats_dict, vecs_dict = sort_feats(masked_feats, masked_vecs, feats_dict=feats_dict, vecs_dict=vecs_dict)


            # TODO: Figure out the right ranges for each feature (use 100k example jets probably)
            #feats_hists += [np.histogram(feats_dict[key], bins=50, range=feat_ranges[key]) for key in feats_dict.keys()]
            if i == 0:
                feats_hists = [np.histogram(feats_dict[key], bins=50) for key in feats_dict.keys()]
                vecs_hists = [np.histogram(vecs_dict[key], bins=50, range=vec_ranges[key]) for key in vecs_dict.keys()]
            else:
                feats_hists += [np.histogram(feats_dict[key], bins=50) for key in feats_dict.keys()]
            #assert np.sum(feats_hists[0]) != 0, "Histogram bins is zero despite good data input"
                vecs_hists += [np.histogram(vecs_dict[key], bins=50, range=vec_ranges[key]) for key in vecs_dict.keys()]
            i += 1
        print(f"Completed processing batch {hist_counter}, saving histograms.")
        for idx, key in enumerate(feats_dict.keys()):
            np.save(os.path.join(output_dir, f"{label_name}_{hist_counter}_hist_{key}.npy"), feats_hists[idx])
        for idx, key in enumerate(vecs_dict.keys()):
            np.save(os.path.join(output_dir, f"{label_name}_{hist_counter}_hist_{key}.npy"), vecs_hists[idx])
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