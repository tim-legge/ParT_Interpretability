#!/usr/bin/env python3
"""
Minimal script to convert HDF5 files to Parquet format.
Creates a schema that mimics the reference QuarkGluon parquet structure.
"""

import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py
from typing import Dict, List, Any


def extract_particle_features(particle_data: np.ndarray) -> Dict[str, List[List[float]]]:
    """
    Extract essential particle features from the jetConstituentList array.
    
    Args:
        particle_data: Array of shape (n_jets, n_particles, n_features)
    
    Returns:
        Dictionary with particle feature lists for each jet
    """
    n_jets, n_particles, n_features = particle_data.shape
    
    # Initialize lists
    particle_features = {
        'part_px': [],
        'part_py': [],
        'part_pz': [],
        'part_energy': [],
        'part_deta': [],
        'part_dphi': []
    }
    
    for jet_idx in range(n_jets):
        jet_particles = particle_data[jet_idx]
        
        # Extract momentum components (px, py, pz, energy)
        particle_features['part_px'].append(jet_particles[:, 0].tolist())      # j1_px
        particle_features['part_py'].append(jet_particles[:, 1].tolist())      # j1_py
        particle_features['part_pz'].append(jet_particles[:, 2].tolist())      # j1_pz
        particle_features['part_energy'].append(jet_particles[:, 3].tolist())  # j1_e
        
        # Extract relative eta and phi
        particle_features['part_deta'].append(jet_particles[:, 7].tolist())    # j1_etarel
        particle_features['part_dphi'].append(jet_particles[:, 10].tolist())   # j1_phirel
    
    return particle_features


def create_parquet_schema() -> pa.Schema:
    """Create PyArrow schema matching QG parquet format."""
    return pa.schema([
        pa.field("label", pa.float64()),
        pa.field("jet_pt", pa.float32()),
        pa.field("jet_eta", pa.float32()),
        pa.field("jet_phi", pa.float32()),
        pa.field("jet_energy", pa.float32()),
        pa.field("jet_mass", pa.float32()),
        pa.field("jet_nparticles", pa.int64()),
        pa.field("part_px", pa.list_(pa.float32())),
        pa.field("part_py", pa.list_(pa.float32())),
        pa.field("part_pz", pa.list_(pa.float32())),
        pa.field("part_energy", pa.list_(pa.float32())),
        pa.field("part_deta", pa.list_(pa.float32())),
        pa.field("part_dphi", pa.list_(pa.float32())),
    ])


def convert_h5_to_parquet(h5_file_path: str, parquet_file_path: str) -> None:
    """
    Convert a single HDF5 file to Parquet format.
    
    Args:
        h5_file_path: Path to input HDF5 file
        parquet_file_path: Path to output Parquet file
    """
    print(f"Converting {h5_file_path} to {parquet_file_path}")
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Read jet-level features
        jets_data = h5_file['jets'][:]
        particle_data = h5_file['jetConstituentList'][:]
        
        # Extract jet features (using indices based on feature names)
        jet_pt = jets_data[:, 1].astype(np.float32)    # j_pt
        jet_eta = jets_data[:, 2].astype(np.float32)   # j_eta
        jet_mass = jets_data[:, 3].astype(np.float32)  # j_mass
        
        # Calculate jet phi from particle data (using first particle's phi as approximation)
        # You may want to compute a proper weighted average
        jet_phi = np.zeros(len(jets_data), dtype=np.float32)
        jet_energy = np.sum(particle_data[:, :, 3], axis=1).astype(np.float32)  # sum of particle energies
        
        # Extract labels from jet classification features
        # Indices for: j_g=53, j_q=54, j_w=55, j_z=56, j_t=57 (based on feature names)
        j_g = jets_data[:, 53]   # gluon
        j_q = jets_data[:, 54]   # quark (non-top)
        j_w = jets_data[:, 55]   # W boson
        j_z = jets_data[:, 56]   # Z boson
        j_t = jets_data[:, 57]   # top quark
        
        # Create labels based on the highest score
        # 0=gluon, 1=quark, 2=W, 3=Z, 4=top
        label_probs = np.stack([j_g, j_q, j_w, j_z, j_t], axis=1)
        labels = np.argmax(label_probs, axis=1).astype(np.float64)
        
        # Count particles per jet (non-zero energy particles)
        jet_nparticles = np.sum(particle_data[:, :, 3] > 0, axis=1).astype(np.int64)
        
        # Extract particle features
        particle_features = extract_particle_features(particle_data)
        
        # Create DataFrame with essential features only
        data = {
            'label': labels,
            'jet_pt': jet_pt,
            'jet_eta': jet_eta,
            'jet_phi': jet_phi,
            'jet_energy': jet_energy,
            'jet_mass': jet_mass,
            'jet_nparticles': jet_nparticles,
            **particle_features
        }
        
        # Create PyArrow table with proper schema
        schema = create_parquet_schema()
        table = pa.Table.from_pydict(data, schema=schema)
        
        # Write to parquet
        pq.write_table(table, parquet_file_path)
        print(f"Successfully converted to {parquet_file_path}")


def main():
    """Main conversion function."""
    # Configuration
    input_dir = "/mnt/c/Interpretability_Data/HLS4ML/LHCjet_150p_train_h5storage"
    output_dir = "/mnt/c/Interpretability_Data/HLS4ML/LHCjet_150p_train_parquet"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert all HDF5 files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.h5'):
            h5_path = os.path.join(input_dir, filename)
            parquet_filename = filename.replace('.h5', '.parquet')
            parquet_path = os.path.join(output_dir, parquet_filename)
            
            # Skip if output file already exists
            if os.path.exists(parquet_path):
                print(f"Skipping {filename} - output already exists")
                continue
            
            try:
                convert_h5_to_parquet(h5_path, parquet_path)
            except Exception as e:
                print(f"Error converting {filename}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage: python convert_h5_to_parquet.py input.h5 [output.parquet]
        h5_file = sys.argv[1]
        parquet_file = sys.argv[2] if len(sys.argv) > 2 else h5_file.replace('.h5', '.parquet')
        
        if os.path.exists(h5_file):
            convert_h5_to_parquet(h5_file, parquet_file)
            print(f"Conversion completed. Output: {parquet_file}")
        else:
            print(f"Error: Input file {h5_file} not found.")
    else:
        # Default: convert test file or run batch conversion
        h5_file = "/mnt/c/Interpretability_Data/HLS4ML/LHCjet_150p_train_h5storage/jetImage_0_150p_0_10000.h5"
        parquet_file = "/tmp/test_conversion.parquet"
        
        if os.path.exists(h5_file):
            convert_h5_to_parquet(h5_file, parquet_file)
            print(f"Test conversion completed. Output: {parquet_file}")
        else:
            print(f"Test file {h5_file} not found. Running batch conversion...")
            main()
