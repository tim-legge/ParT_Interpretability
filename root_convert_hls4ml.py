import os
from hepconvert import parquet_to_root

if not os.path.exists('datasets/hls4ml/HLS4ML/val/val-root'):
    os.makedirs('datasets/hls4ml/HLS4ML/val/val-root')

if not os.path.exists('datasets/hls4ml/HLS4ML/train/train-root'):
    os.makedirs('datasets/hls4ml/HLS4ML/train/train-root')

train_parquet_dir = 'datasets/hls4ml/HLS4ML/train/train-parquet'
val_parquet_dir = 'datasets/hls4ml/HLS4ML/val/val-parquet'

for f in os.listdir(val_parquet_dir):
    if f.endswith('.parquet'):
        print(f'Converting {f} to ROOT format...')
        parquet_to_root('datasets/hls4ml/HLS4ML/val/val-root/' + os.path.split(os.path.splitext(f)[0])[-1] + '.root',
                            os.path.join(val_parquet_dir, f), name='tree', force=True)

for f in os.listdir(train_parquet_dir):
    if f.endswith('.parquet'):
        print(f'Converting {f} to ROOT format...')
        parquet_to_root('datasets/hls4ml/HLS4ML/train/train-root/' + os.path.split(os.path.splitext(f)[0])[-1] + '.root',
                            os.path.join(train_parquet_dir, f), name='tree', force=True)