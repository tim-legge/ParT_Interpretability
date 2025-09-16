# reads data features from JetClass and TopLandscape and compares

import numpy as np
import awkward as ak
import uproot
import os
from TL_Inference_tops_only import build_features_and_labels_tl
from JC_full_inference import _clip, _pad, build_features_and_labels

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