import os
from hepconvert import parquet_to_root

if not os.path.exists('/home/javyon/ParT_Interpretability/datasets/hls4ml/HLS4ML/val/val-root'):
    os.makedirs('/home/javyon/ParT_Interpretability/datasets/hls4ml/HLS4ML/val/val-root')

if not os.path.exists('/home/javyon/ParT_Interpretability/datasets/hls4ml/HLS4ML/train/train-root'):
    os.makedirs('/home/javyon/ParT_Interpretability/datasets/hls4ml/HLS4ML/train/train-root')

for f in os.listdir('/home/javyon/ParT_Interpretability/datasets/hls4ml/HLS4ML'):
    if f.endswith('.parquet'):
        if 'val' in f:
            print(f'Converting {f} to ROOT format...')
            parquet_to_root('/home/javyon/ParT_Interpretability/datasets/hls4ml/HLS4ML/val/val-root/' + f.split('/')[-1].split('.p')[0] + '.root',
                            f, name='tree', force=True)
        if 'train' in f:
            print(f'Converting {f} to ROOT format...')
            parquet_to_root('/home/javyon/ParT_Interpretability/datasets/hls4ml/HLS4ML/train/train-root/' + f.split('/')[-1].split('.p')[0] + '.root',
                            f, name='tree', force=True)