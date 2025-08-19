#!/usr/bin/env python3
"""
Batch conversion script for HDF5 to Parquet conversion.
Usage: python batch_convert.py [input_dir] [output_dir]
"""

import os
import sys
from convert_h5_to_parquet import convert_h5_to_parquet, extract_particle_features

def batch_convert(input_dir, output_dir, max_files=None):
    """
    Convert all HDF5 files in input_dir to Parquet format in output_dir.
    
    Args:
        input_dir: Directory containing HDF5 files
        output_dir: Directory to save Parquet files
        max_files: Maximum number of files to convert (None for all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files
    h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    
    if max_files:
        h5_files = h5_files[:max_files]
    
    print(f"Found {len(h5_files)} HDF5 files to convert")
    
    success_count = 0
    error_count = 0
    
    for i, filename in enumerate(h5_files):
        h5_path = os.path.join(input_dir, filename)
        parquet_filename = filename.replace('.h5', '.parquet')
        parquet_path = os.path.join(output_dir, parquet_filename)
        
        # Skip if output file already exists
        if os.path.exists(parquet_path):
            print(f"[{i+1}/{len(h5_files)}] Skipping {filename} - output already exists")
            continue
        
        try:
            print(f"[{i+1}/{len(h5_files)}] Converting {filename}...")
            convert_h5_to_parquet(h5_path, parquet_path)
            success_count += 1
            print(f"  ✓ Success")
        except Exception as e:
            error_count += 1
            print(f"  ✗ Error: {e}")
    
    print(f"\nBatch conversion completed:")
    print(f"  Successful: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Skipped: {len(h5_files) - success_count - error_count}")

if __name__ == "__main__":
    # Default directories
    default_input = "/mnt/c/Interpretability_Data/HLS4ML/LHCjet_150p_train_h5storage"
    default_output = "/mnt/c/Interpretability_Data/HLS4ML/LHCjet_150p_train_parquet"
    
    if len(sys.argv) >= 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) == 2:
        input_dir = sys.argv[1]
        output_dir = input_dir.replace('h5storage', 'parquet')
    else:
        input_dir = default_input
        output_dir = default_output
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # For testing, limit to first 3 files
    batch_convert(input_dir, output_dir)
