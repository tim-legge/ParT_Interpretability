# reads data features from JetClass and TopLandscape and compares

import numpy as np
import awkward as ak
import uproot
import os


def _clip(a, min_value, max_value):
    assert isinstance(a, ak.Array), "expected awkward array"
    main_list = []
    for i in range(len(a)):
        sublist = ak.to_list(a[i])
        sublist = np.clip(sublist, min_value, max_value)
        main_list.append(sublist)
    return ak.from_iter(main_list)

def _pad(a, maxlen=128, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

def build_features_and_labels_tl(tree, transform_features=True):

    """Build features for TopLandscape dataset based on top_kin.yaml"""
    # load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label'])

    # compute new features (same as QG)
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])

    # apply standardization based on top_kin.yaml (same as QG)
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0

    # Feature list for TopLandscape (same kinematic features as QG)
    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'],
        'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel', 
            'part_logerel',
            'part_deltaR',
            'part_deta',
            'part_dphi',
        ],
        'pf_vectors': [
            'part_px',
            'part_py',
            'part_pz',
            'part_energy',
        ],
        'pf_mask': ['part_mask']
    }

    def _pad(a, maxlen=128, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)

    # Labels for TopLandscape (binary classification) 
    out['label'] = a['label'].to_numpy().astype('int')

    return out

def get_tl_features(dir_path='/part-vol-3/timlegge-ParT-trained/tl_dataset/TopLandscape/',
                          counter_path='/part-vol-3/timlegge-ParT-trained/collect_tl_features_counter.txt', tree_name='tree', batch_size=2000):
    assert os.path.exists(dir_path), f"Directory {dir_path} does not exist."
    print("Creating a counter file outside of the directory if it doesn't exist.")
    if not os.path.exists(counter_path):
        with open(counter_path, "w") as f:
            f.write("0")
    with open(counter_path, "r") as f:
            counter = int(f.read())
    print(f'Beginning on file number {counter}')
    for i, file in enumerate(sorted(os.listdir(dir_path))):    
        print("Reading file in directory:", file)
        if file.endswith('.root') and 'train' in file:
            file_path = os.path.join(dir_path, file)
            while counter < 500:
                print(f"Processing segment {counter} from file {file}")
                with uproot.open(file_path) as f:
                    tree = f[tree_name]
                    data = build_features_and_labels_tl(tree)
                    data = {
                            'pf_points': data['pf_points'][batch_size*counter:batch_size*(counter+1)],
                            'pf_features': data['pf_features'][batch_size*counter:batch_size*(counter+1)],
                            'pf_vectors': data['pf_vectors'][batch_size*counter:batch_size*(counter+1)],
                            'pf_mask': data['pf_mask'][batch_size*counter:batch_size*(counter+1)],
                            'labels': data['label'][batch_size*counter:batch_size*(counter+1)]
                            }
                    for key, item in data.items():
                        np.save(f"/part-vol-3/timlegge-ParT-trained/data_from_tl_train/{key}_{counter}.npy", data[key])
                counter += 1
                with open(counter_path, "w") as f:
                    f.write(str(counter))
                print(f"Processed segment {counter-1}, updated counter to {counter}")      
    print("All files have been processed.")
if __name__ == "__main__":
    get_tl_features()