import os
import numpy as np
import matplotlib.pyplot as plt
import uproot
from graph_jc_features import mask_out
import awkward as ak

stem = '/part-vol-3/timlegge-ParT-trained/example_jc_feat_dists/'

def _clip(a, min_value, max_value):
    assert isinstance(a, ak.Array), "expected awkward array"
    main_list = []
    #print("This line is ok - #4")
    for i in range(len(a)):
        #if np.random.rand() < 0.01:
        #    print(f"This line is ok - #7, i={i}")
        sublist = ak.to_list(a[i])
        sublist = np.clip(sublist, min_value, max_value)
        main_list.append(sublist)
    #print("This line is ok - #9")
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

#if not os.path.exists('/part-vol-3/timlegge-ParT-trained/example_jc_feat_dists/'):
#    os.makedirs('/part-vol-3/timlegge-ParT-trained/example_jc_feat_dists/')

datapath = '/part-vol-3/timlegge-ParT-trained/JetClass_example_100k.root'
with uproot.open(datapath) as f:
    tree = f['tree']
    data = build_features_and_labels(tree)
    features = data['pf_features'][:4000]
    vectors = data['pf_vectors'][:4000]
    masks = data['pf_mask'][:4000]

print("Features, vectors, and masks loaded.")

print(masks[0])
num_particles = [np.sum(masks[i].astype('int')) for i in range(masks.shape[0])]
print(num_particles[:10])

masked_feats, masked_vecs = mask_out(features, vectors, masks)

print(f"Shapes of first 10 items in masked_feats and masked_vecs:")
for i in range(10):
    print(f"masked_feats[{i}]: {masked_feats[i].shape}, masked_vecs[{i}]: {masked_vecs[i].shape}")

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
feats_idx_map = [key for key in feats_dict.keys()]
print(f'feats_idx_map: {feats_idx_map}')
vecs_dict = {
    'part_px': [],
    'part_py': [],
    'part_pz': [],
    'part_energy': [],
}
vecs_idx_map = [key for key in vecs_dict.keys()]
print(f'vecs_idx_map: {vecs_idx_map}')

print(len(masked_feats), len(masked_vecs))

for jet_idx, jet in enumerate(masked_feats):
    for idx, key in enumerate(feats_dict.keys()):
        #if idx >= 13 and np.random.rand() < 0.01:
        #    print(f'jet_idx: {jet_idx}, idx: {idx}, key: {key}')
        try:
            feats_dict[key].extend(masked_feats[jet_idx][idx].flatten().tolist())
            # we need to check that padding is actually being removed
            #if np.random.rand() < 0.01:
            #    print(f'Num of particles: {len(masked_feats[jet_idx][idx])} for jet_idx {jet_idx}, idx {idx}, key {key}')
        except IndexError as e:
            print(f"IndexError for jet_idx {jet_idx}, idx {idx}, key {key}: {e}")
            continue

for jet_idx, jet in enumerate(masked_vecs):
     for idx, key in enumerate(vecs_dict.keys()):
        try:
            vecs_dict[key].extend(masked_feats[jet_idx][idx].flatten().tolist())
        except IndexError as e:
            print(f"IndexError for jet_idx {jet_idx}, idx {idx}, key {key}: {e}")
            continue

print("Features and vectors sorted.")

# before plotting, get maximum and minimum values for each feature
feat_ranges = {}
for key in feats_dict.keys():
    feat_ranges[key] = (np.min(feats_dict[key]), np.max(feats_dict[key]))

vec_ranges = {}
for key in vecs_dict.keys():
    vec_ranges[key] = (np.min(vecs_dict[key]), np.max(vecs_dict[key]))

if os.path.exists(stem+'ranges.txt'):
    with open(stem+'ranges.txt', 'w') as f:
        f.write("Feature Ranges:\n")
        for key, (min_val, max_val) in feat_ranges.items():
            f.write(f"{key}: min={min_val}, max={max_val}\n")
        f.write("\nVector Ranges:\n")
        for key, (min_val, max_val) in vec_ranges.items():
            f.write(f"{key}: min={min_val}, max={max_val}\n")

print("Feature and vector ranges saved.")

# now plot histograms for each feature
for key, values in feats_dict.items():
    hist, bin_edges = np.histogram(values, bins=50, range=feat_ranges[key])
    fig, ax = plt.subplots()
    ax.step(bin_edges[:-1], hist, where='pre')
    ax.set_title(f'{key} distribution - JetClass 100k Sample')
    ax.set_xlabel(key)
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    plt.savefig(stem+f'{key}_hist.png')
    plt.close()

# for each vector
for key, values in vecs_dict.items():
    hist, bin_edges = np.histogram(values, bins=50, range=vec_ranges[key])
    fig, ax = plt.subplots()
    ax.step(bin_edges[:-1], hist, where='pre')
    ax.set_title(f'{key} distribution - JetClass 100k Sample')
    ax.set_xlabel(key)
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    plt.savefig(stem+f'{key}_hist.png')
    plt.close()

print("Feature distributions saved.")