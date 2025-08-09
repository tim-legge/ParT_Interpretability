#https://stackoverflow.com/questions/46157709/converting-hdf5-to-parquet-without-loading-into-memory/46158785

import os
import sys
import warnings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py

#reference .parquet file at '/mnt/c/Interpretability_Data/QuarkGluon/test_file_0.parquet'

def convert_hdf5_to_parquet(h5_file, parquet_file, chunksize=10000):

    with h5py.File(h5_file, 'r') as f:
        h5_group = f['Jets']

    df = pd.read_hdf(h5_file, key='Jets', mode='r')

    stream = pd.read_hdf(h5_file, chunksize=chunksize)

    for i, chunk in enumerate(stream):
        print("Chunk {}".format(i))

        if i == 0:
            # Infer schema and open parquet file on first chunk
            parquet_schema = pa.Table.from_pandas(df=chunk).schema
            parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')

        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)

    parquet_writer.close()

#"C:\Interpretability_Data\HLS4ML\hls4ml_LHCjet_100p_train.tar.gz"

def main():
    h5_prefix = "/mnt/c/Interpretability_Data/HLS4ML/LHCjet_150p_train_h5storage/jetImage"
    parquet_prefix = "/mnt/c/Interpretability_Data/HLS4ML/hls4ml_LHCjet_150p_train_parquet/train/jetImage"

    num_files = 6
    for i in range(num_files):
        for j in range(9):
            h5_file = f"{h5_prefix}_{i}_150p_{j*10**4}_{(j+1)*10**4}.h5"

            parquet_file = f"{parquet_prefix}_{i}_150p_{j*10**4}_{(j+1)*10**4}.parquet"

            # Create parquet files (only if they do not exist)
            if not os.path.exists(parquet_file):
                f = open(parquet_file, 'wb')
                f.close()
            
            convert_hdf5_to_parquet(h5_file, parquet_file)

with h5py.File("/mnt/c/Interpretability_Data/HLS4ML/LHCjet_150p_train_h5storage/jetImage_0_150p_0_10000.h5", 'r') as f:
    for key in f.keys():
        print(f"{key}: {f[key].shape} {f[key].dtype}")
    print(f['jetFeatureNames'][:])
    print(f['particleFeatureNames'][:])

#if __name__ == "__main__":
#    main()