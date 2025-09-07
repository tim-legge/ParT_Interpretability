
import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
from typing import Dict, List, Any

def read_h5(h5_file_path: str) -> None:
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        keys = h5_file.keys()
        jets_data = h5_file['jets'][:]
        particle_data = h5_file['jetConstituentList'][:]
        particle_feat_names = h5_file['particleFeatureNames'][:]
        print(h5_file.keys())


    #Print particle feature names
    print(f'Particle Feature Names:\n{particle_feat_names}')

read_h5("/mnt/c/Interpretability_Data/HLS4ML/LHCjet_150p_train_h5storage/jetImage_0_150p_0_10000.h5")
        