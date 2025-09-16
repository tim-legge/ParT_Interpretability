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

def build_features_and_labels(tree, transform_features=True):

    # load arrays from the tree
    a = tree.arrays(filter_name=['part_*', 'jet_pt', 'jet_energy', 'label_*'])

    # compute new features
    a['part_mask'] = ak.ones_like(a['part_energy'])
    a['part_pt'] = np.hypot(a['part_px'], a['part_py'])
    a['part_pt_log'] = np.log(a['part_pt'])
    a['part_e_log'] = np.log(a['part_energy'])
    a['part_logptrel'] = np.log(a['part_pt']/a['jet_pt'])
    a['part_logerel'] = np.log(a['part_energy']/a['jet_energy'])
    a['part_deltaR'] = np.hypot(a['part_deta'], a['part_dphi'])
    a['part_d0'] = np.tanh(a['part_d0val'])
    a['part_dz'] = np.tanh(a['part_dzval'])

    # apply standardization
    if transform_features:
        a['part_pt_log'] = (a['part_pt_log'] - 1.7) * 0.7
        a['part_e_log'] = (a['part_e_log'] - 2.0) * 0.7
        a['part_logptrel'] = (a['part_logptrel'] - (-4.7)) * 0.7
        a['part_logerel'] = (a['part_logerel'] - (-4.7)) * 0.7
        a['part_deltaR'] = (a['part_deltaR'] - 0.2) * 4.0
        a['part_d0err'] = _clip(a['part_d0err'], 0, 1)
        a['part_dzerr'] = _clip(a['part_dzerr'], 0, 1)

    feature_list = {
        'pf_points': ['part_deta', 'part_dphi'], # not used in ParT
        'pf_features': [
            'part_pt_log',
            'part_e_log',
            'part_logptrel',
            'part_logerel',
            'part_deltaR',
            'part_charge',
            'part_isChargedHadron',
            'part_isNeutralHadron',
            'part_isPhoton',
            'part_isElectron',
            'part_isMuon',
            'part_d0',
            'part_d0err',
            'part_dz',
            'part_dzerr',
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

    out = {}
    for k, names in feature_list.items():
        out[k] = np.stack([_pad(a[n], maxlen=128).to_numpy() for n in names], axis=1)

    label_list = ['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']
    out['label'] = np.stack([a[n].to_numpy().astype('int') for n in label_list], axis=1)

    return out

def get_jetclass_features(dir_path='/part-vol-3/weaver-core/particle_transformer/datasets/JetClass/Pythia/train_100M/',
                          counter_path='/part-vol-3/timlegge-ParT-trained/collect_features_counter.txt', tree_name='tree', batch_size=2000):
    assert os.path.exists(dir_path), f"Directory {dir_path} does not exist."
    print("Creating a counter file outside of the directory if it doesn't exist.")
    if not os.path.exists(counter_path):
        with open(dir_path+"/collect_features_counter.txt", "w") as f:
            f.write("0")
    with open(counter_path, "r") as f:
            counter = int(f.read().strip())
    print(f'Beginning on file number {counter}')
    for i, file in enumerate(sorted(os.listdir(dir_path))):
        if i < counter:
            continue
        if i==0:
            print("Reading first file in the directory:", file)
        if file.endswith('.root'):
            file_path = os.path.join(dir_path, file)
            with uproot.open(file_path) as f:
                tree = f[tree_name]
                data = build_features_and_labels(tree)
                data = {
                            'pf_points': data['pf_points'][:batch_size],
                            'pf_features': data['pf_features'][:batch_size],
                            'pf_vectors': data['pf_vectors'][:batch_size],
                            'pf_mask': data['pf_mask'][:batch_size],
                            'labels': data['label'][:batch_size]
                        }
                for key, item in data.items():
                    data[key] = data[key].to_numpy()
                    np.save(counter_path + f"./data_from_train/{key}_{i}.npy", data[key])           
            counter += 1
            with open(counter_path, "w") as f:
                f.write(str(counter))
            print(f"Processed file {i+1}: {file}")
    print("All files have been processed.")

get_jetclass_features()