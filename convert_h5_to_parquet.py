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


def extract_particle_features(particle_data: np.ndarray, jets_data: np.ndarray) -> Dict[str, List[List[float]]]:
    """
    Extract particle features with proper filtering.
    """
    n_jets, n_particles, n_features = particle_data.shape
    
    particle_features = {
        'part_px': [],
        'part_py': [],
        'part_pz': [],
        'part_energy': [],
        'part_erel': [],
        'part_pt': [],
        'part_ptrel': [],
        'part_eta': [],
        'part_etarel': [],
        'part_etarot': [],
        'part_phi': [],
        'part_phirel': [],
        'part_phirot': [],
        'part_deltaR': [],
        'part_costheta': [],
        'part_costhetarel': [],
    }
    
    for jet_idx in range(n_jets):
        jet_particles = particle_data[jet_idx]
        
        # Filter out zero-energy particles (padding)
        valid_mask = jet_particles[:, 3] > 0  # energy > 0
        valid_particles = jet_particles[valid_mask]
        
        if len(valid_particles) == 0:
            # Handle empty jets by adding a single zero particle
            valid_particles = np.zeros((1, n_features))
        
        particle_features['part_px'].append(valid_particles[:, 0].tolist())
        particle_features['part_py'].append(valid_particles[:, 1].tolist())
        particle_features['part_pz'].append(valid_particles[:, 2].tolist())
        particle_features['part_energy'].append(valid_particles[:, 3].tolist())
        particle_features['part_erel'].append(valid_particles[:, 4].tolist())
        particle_features['part_pt'].append(valid_particles[:, 5].tolist())
        particle_features['part_ptrel'].append(valid_particles[:, 6].tolist())
        particle_features['part_eta'].append(valid_particles[:, 7].tolist())
        particle_features['part_etarel'].append(valid_particles[:, 8].tolist())
        particle_features['part_etarot'].append(valid_particles[:, 9].tolist())
        particle_features['part_phi'].append(valid_particles[:, 10].tolist())
        particle_features['part_phirel'].append(valid_particles[:, 11].tolist())
        particle_features['part_phirot'].append(valid_particles[:, 12].tolist())
        particle_features['part_deltaR'].append(valid_particles[:, 13].tolist())
        particle_features['part_costheta'].append(valid_particles[:, 14].tolist())
        particle_features['part_costhetarel'].append(valid_particles[:, 15].tolist())

    return particle_features


def create_parquet_schema() -> pa.Schema:
    """Create PyArrow schema matching QG parquet format."""
    return pa.schema([
        pa.field("jet_pt", pa.float32(), nullable = False),
        pa.field("jet_eta", pa.float32(), nullable = False),
        pa.field("jet_phi", pa.float32(), nullable = False),
        pa.field("jet_energy", pa.float32(), nullable = False),
        pa.field("jet_mass", pa.float32(), nullable = False),
        pa.field("jet_nparticles", pa.int64(), nullable = False),
        pa.field("jet_isGluon", pa.int64(), nullable = False),
        pa.field("jet_isQuark", pa.int64(), nullable = False),
        pa.field("jet_isW", pa.int64(), nullable = False),
        pa.field("jet_isZ", pa.int64(), nullable = False),
        pa.field("jet_isTop", pa.int64(), nullable = False),
        pa.field("part_px", pa.list_(pa.float32()), nullable = False),
        pa.field("part_py", pa.list_(pa.float32()), nullable = False),
        pa.field("part_pz", pa.list_(pa.float32()), nullable = False),
        pa.field("part_energy", pa.list_(pa.float32()), nullable = False),
        pa.field("part_erel", pa.list_(pa.float32()), nullable = False),
        pa.field("part_pt", pa.list_(pa.float32()), nullable = False),
        pa.field("part_ptrel", pa.list_(pa.float32()), nullable = False),
        pa.field("part_eta", pa.list_(pa.float32()), nullable = False),
        pa.field("part_etarel", pa.list_(pa.float32()), nullable = False),
        pa.field("part_etarot", pa.list_(pa.float32()), nullable = False),
        pa.field("part_phi", pa.list_(pa.float32()), nullable = False),
        pa.field("part_phirel", pa.list_(pa.float32()), nullable = False),
        pa.field("part_phirot", pa.list_(pa.float32()), nullable = False),
        pa.field("part_deltaR", pa.list_(pa.float32()), nullable = False),
        pa.field("part_costheta", pa.list_(pa.float32()), nullable = False),
        pa.field("part_costhetarel", pa.list_(pa.float32()), nullable = False),
    ])


def convert_h5_to_parquet(h5_file_path: str, parquet_file_path: str) -> None:
    """Convert with proper particle filtering."""
    print(f"Converting {h5_file_path} to {parquet_file_path}")
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        jets_data = h5_file['jets'][:]
        particle_data = h5_file['jetConstituentList'][:]
        
        # Extract jet features
        jet_pt = jets_data[:, 1].astype(np.float32)
        jet_eta = jets_data[:, 2].astype(np.float32) 
        jet_mass = jets_data[:, 3].astype(np.float32)
        
        # Compute jet phi and energy from valid particles only
        jet_phi = np.zeros(len(jets_data), dtype=np.float32)
        jet_energy = np.zeros(len(jets_data), dtype=np.float32)
        jet_nparticles = np.zeros(len(jets_data), dtype=np.int64)
        
        for jet_idx in range(len(jets_data)):
            jet_particles = particle_data[jet_idx]
            valid_mask = jet_particles[:, 3] > 0  # energy > 0
            valid_particles = jet_particles[valid_mask]
            
            if len(valid_particles) > 0:
                # Compute phi as energy-weighted average
                px_sum = np.sum(valid_particles[:, 0])
                py_sum = np.sum(valid_particles[:, 1])
                jet_phi[jet_idx] = np.arctan2(py_sum, px_sum)
                jet_energy[jet_idx] = np.sum(valid_particles[:, 3])
                jet_nparticles[jet_idx] = len(valid_particles)
            else:
                jet_phi[jet_idx] = 0.0
                jet_energy[jet_idx] = 0.0
                jet_nparticles[jet_idx] = 0
        
        # Extract labels (same as before)
        j_g = jets_data[:, 53]
        j_q = jets_data[:, 54]
        j_w = jets_data[:, 55]
        j_z = jets_data[:, 56]
        j_t = jets_data[:, 57]
        
        #label_probs = np.stack([j_g, j_q, j_w, j_z, j_t], axis=1)
        #labels = np.argmax(label_probs, axis=1).astype(np.float64)
        
        # Extract particle features with filtering
        particle_features = extract_particle_features(particle_data, jets_data)
        
        # Create data dictionary
        data = {
            'jet_pt': jet_pt,
            'jet_eta': jet_eta, 
            'jet_phi': jet_phi,
            'jet_energy': jet_energy,
            'jet_mass': jet_mass,
            'jet_nparticles': jet_nparticles,
            'jet_isGluon': j_g,
            'jet_isQuark': j_q,
            'jet_isW': j_w,
            'jet_isZ': j_z,
            'jet_isTop': j_t,
            **particle_features
        }

        # Validate and clean data before writing
        for key, values in data.items():
            if isinstance(values, np.ndarray):
                if np.isnan(values).any() or np.isinf(values).any():
                    print(f"Warning: NaN/inf values found in {key}")
                    # Replace with safe defaults
                    if key.startswith('jet_'):
                        if key in ['jet_pt', 'jet_energy']:
                            data[key] = np.nan_to_num(values, nan=1.0, posinf=1000.0, neginf=0.0)
                        else:
                            data[key] = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
                    elif key.startswith('jet_is'):
                        data[key] = np.nan_to_num(values, nan=0).astype(np.int32)
            elif isinstance(values, list):
                # Validate particle feature lists
                for i, sublist in enumerate(values):
                    if any(np.isnan(sublist)) or any(np.isinf(sublist)):
                        print(f"Warning: NaN/inf values found in {key}[{i}]")
                        values[i] = [np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0) for x in sublist]

        # Write to parquet
        schema = create_parquet_schema()
        table = pa.Table.from_pydict(data, schema=schema)
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
