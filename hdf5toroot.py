#!/usr/bin/env python3
"""
HDF5 to ROOT Converter for HLS4ML Dataset

This script converts HDF5 dataset files from the HLS4ML dataset to ROOT format
for use in analyzing PyTorch models. It uses modern libraries (uproot, awkward)
instead of the deprecated root_numpy.

Credits and Inspiration:
- Original concept from CYGNUS-RD/hdf2root: https://github.com/CYGNUS-RD/hdf2root/blob/master/hdf2root.py
- Data structure analysis from ParT_Interpretability project conversion scripts
- Uses modern Python ecosystem: uproot (Scikit-HEP), awkward-array (Scikit-HEP)

Key Features:
- Direct HDF5 to ROOT conversion without intermediate formats
- Handles jagged particle data using awkward arrays
- Preserves HLS4ML dataset structure and labels
- Compatible with PyTorch model analysis workflows
- Command-line interface with batch processing support

Author: Generated for ParT_Interpretability project
Date: August 18, 2025
"""

import os
import sys
import argparse
import numpy as np
import h5py
import uproot
import awkward as ak
from typing import Dict, List, Optional, Tuple


def read_hdf5_structure(h5_file_path: str) -> Dict:
    """
    Read and analyze the structure of an HDF5 file.
    
    Args:
        h5_file_path: Path to HDF5 file
        
    Returns:
        Dictionary containing dataset information
    """
    with h5py.File(h5_file_path, 'r') as f:
        structure = {
            'datasets': list(f.keys()),
            'jet_features': None,
            'particle_features': None,
            'n_jets': None,
            'n_particles': None,
            'n_jet_features': None,
            'n_particle_features': None
        }
        
        # Get feature names if available
        if 'jetFeatureNames' in f:
            structure['jet_features'] = [
                x.decode('utf-8') if isinstance(x, bytes) else x 
                for x in f['jetFeatureNames'][:]
            ]
            structure['n_jet_features'] = len(structure['jet_features'])
            
        if 'particleFeatureNames' in f:
            structure['particle_features'] = [
                x.decode('utf-8') if isinstance(x, bytes) else x 
                for x in f['particleFeatureNames'][:]
            ]
            structure['n_particle_features'] = len(structure['particle_features'])
            
        # Get data dimensions
        if 'jets' in f:
            structure['n_jets'] = f['jets'].shape[0]
            structure['n_jet_features'] = f['jets'].shape[1]
            
        if 'jetConstituentList' in f:
            jet_constituents_shape = f['jetConstituentList'].shape
            structure['n_particles'] = jet_constituents_shape[1]
            if len(jet_constituents_shape) > 2:
                structure['n_particle_features'] = jet_constituents_shape[2]
    
    return structure


def extract_jet_features(jets_data: np.ndarray, feature_names: List[str]) -> Dict[str, np.ndarray]:
    """
    Extract jet-level features from HDF5 data.
    
    Args:
        jets_data: Raw jet data array
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to arrays
    """
    jet_features = {}
    
    for i, name in enumerate(feature_names):
        if i < jets_data.shape[1]:
            jet_features[name] = jets_data[:, i].astype(np.float32)
        else:
            print(f"Warning: Feature {name} index {i} exceeds data dimensions")
            
    return jet_features


def extract_particle_features(particle_data: np.ndarray, feature_names: List[str]) -> Dict[str, ak.Array]:
    """
    Extract particle-level features from HDF5 data.
    
    Args:
        particle_data: Raw particle data array (n_jets, n_particles, n_features)
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to awkward arrays
    """
    n_jets, n_particles, n_features = particle_data.shape
    particle_features = {}
    
    for i, name in enumerate(feature_names):
        if i < n_features:
            # Extract feature for all jets and particles
            feature_data = particle_data[:, :, i]
            
            # Convert to jagged arrays by filtering out zero-energy particles
            jagged_data = []
            for jet_idx in range(n_jets):
                jet_particles = feature_data[jet_idx]
                
                # For energy feature, use it to determine valid particles
                if name in ['j1_e', 'part_energy'] or i == 3:  # Assume energy is at index 3
                    valid_mask = jet_particles > 0
                else:
                    # For other features, use energy from the same jet to determine validity
                    energy_data = particle_data[jet_idx, :, 3]  # Assume energy at index 3
                    valid_mask = energy_data > 0
                
                valid_particles = jet_particles[valid_mask]
                jagged_data.append(valid_particles.astype(np.float32))
                
            particle_features[name] = ak.Array(jagged_data)
        else:
            print(f"Warning: Particle feature {name} index {i} exceeds data dimensions")
            
    return particle_features


def convert_hdf5_to_root(h5_file_path: str, root_file_path: str, 
                        tree_name: str = "tree", verbose: bool = True) -> None:
    """
    Convert HDF5 file to ROOT format.
    
    Args:
        h5_file_path: Input HDF5 file path
        root_file_path: Output ROOT file path
        tree_name: Name of the ROOT tree
        verbose: Print progress information
    """
    if verbose:
        print(f"Converting {h5_file_path} to {root_file_path}")
    
    # Read HDF5 structure
    structure = read_hdf5_structure(h5_file_path)
    if verbose:
        print(f"Found {structure['n_jets']} jets with {structure['n_jet_features']} jet features")
        print(f"Found {structure['n_particles']} particles with {structure['n_particle_features']} particle features")
    
    # Read data from HDF5
    with h5py.File(h5_file_path, 'r') as f:
        jets_data = f['jets'][:]
        particle_data = f['jetConstituentList'][:]
        
        # Extract features
        jet_features = extract_jet_features(jets_data, structure['jet_features'])
        particle_features = extract_particle_features(particle_data, structure['particle_features'])
        
        # Combine all data
        root_data = {**jet_features, **particle_features}
        
        # Add derived features commonly used in jet physics
        if 'j_pt' in jet_features and 'j_eta' in jet_features and 'j_mass' in jet_features:
            # Compute jet phi if not present
            if 'j_phi' not in jet_features:
                # Try to compute from particle data
                if 'j1_px' in particle_features and 'j1_py' in particle_features:
                    jet_px = ak.sum(particle_features['j1_px'], axis=1)
                    jet_py = ak.sum(particle_features['j1_py'], axis=1)
                    root_data['j_phi'] = np.arctan2(ak.to_numpy(jet_py), ak.to_numpy(jet_px)).astype(np.float32)
                else:
                    root_data['j_phi'] = np.zeros(structure['n_jets'], dtype=np.float32)
            
            # Compute jet energy if not present
            if 'j_energy' not in jet_features:
                if 'j1_e' in particle_features:
                    root_data['j_energy'] = ak.to_numpy(ak.sum(particle_features['j1_e'], axis=1)).astype(np.float32)
                else:
                    # Estimate from pt, eta, mass
                    pt = jet_features['j_pt']
                    eta = jet_features['j_eta']
                    mass = jet_features['j_mass']
                    root_data['j_energy'] = np.sqrt(pt**2 * np.cosh(eta)**2 + mass**2).astype(np.float32)
        
        # Add particle count
        if 'j1_e' in particle_features:
            root_data['j_nparticles'] = ak.to_numpy(ak.num(particle_features['j1_e'])).astype(np.int32)
        
        # Extract labels (HLS4ML specific: j_g, j_q, j_w, j_z, j_t at indices 53-57)
        if structure['n_jet_features'] > 57:
            label_features = {
                'j_isGluon': (jets_data[:, 53] > 0.5).astype(np.int32),
                'j_isQuark': (jets_data[:, 54] > 0.5).astype(np.int32), 
                'j_isW': (jets_data[:, 55] > 0.5).astype(np.int32),
                'j_isZ': (jets_data[:, 56] > 0.5).astype(np.int32),
                'j_isTop': (jets_data[:, 57] > 0.5).astype(np.int32)
            }
            root_data.update(label_features)
            
            # Create a single categorical label
            labels = np.zeros(structure['n_jets'], dtype=np.int32)
            labels[jets_data[:, 53] > 0.5] = 0  # Gluon
            labels[jets_data[:, 54] > 0.5] = 1  # Quark  
            labels[jets_data[:, 55] > 0.5] = 2  # W
            labels[jets_data[:, 56] > 0.5] = 3  # Z
            labels[jets_data[:, 57] > 0.5] = 4  # Top
            root_data['label'] = labels
    
    # Write to ROOT file using uproot
    if verbose:
        print(f"Writing {len(root_data)} branches to ROOT file...")
        
    with uproot.recreate(root_file_path) as root_file:
        root_file[tree_name] = root_data
    
    if verbose:
        print(f"Successfully converted to {root_file_path}")


def convert_directory(input_dir: str, output_dir: str, pattern: str = "*.h5",
                     tree_name: str = "tree", verbose: bool = True) -> None:
    """
    Convert all HDF5 files in a directory to ROOT format.
    
    Args:
        input_dir: Input directory containing HDF5 files
        output_dir: Output directory for ROOT files
        pattern: File pattern to match (default: "*.h5")
        tree_name: Name of the ROOT tree
        verbose: Print progress information
    """
    import glob
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    
    h5_files = glob.glob(os.path.join(input_dir, pattern))
    
    if not h5_files:
        print(f"No files matching {pattern} found in {input_dir}")
        return
    
    if verbose:
        print(f"Found {len(h5_files)} files to convert")
    
    for h5_file in h5_files:
        basename = os.path.basename(h5_file)
        root_filename = basename.replace('.h5', '.root')
        root_file_path = os.path.join(output_dir, root_filename)
        
        # Skip if output already exists
        if os.path.exists(root_file_path):
            if verbose:
                print(f"Skipping {basename} - output already exists")
            continue
            
        try:
            convert_hdf5_to_root(h5_file, root_file_path, tree_name, verbose=False)
            if verbose:
                print(f"✓ Converted {basename}")
        except Exception as e:
            print(f"✗ Error converting {basename}: {e}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 files to ROOT format for HLS4ML dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert single file
    python hdf5toroot.py input.h5 output.root
    
    # Convert all files in directory
    python hdf5toroot.py -d /path/to/h5/files -o /path/to/root/files
    
    # Convert with custom tree name
    python hdf5toroot.py input.h5 output.root --tree-name events
    
    # Analyze file structure without converting
    python hdf5toroot.py --analyze input.h5
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input HDF5 file or directory')
    parser.add_argument('output', nargs='?', help='Output ROOT file or directory')
    parser.add_argument('-d', '--input-dir', help='Input directory containing HDF5 files')
    parser.add_argument('-o', '--output-dir', help='Output directory for ROOT files')
    parser.add_argument('--pattern', default='*.h5', help='File pattern to match (default: *.h5)')
    parser.add_argument('--tree-name', default='tree', help='Name of ROOT tree (default: tree)')
    parser.add_argument('--analyze', action='store_true', help='Analyze file structure without converting')
    parser.add_argument('-v', '--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet mode (minimal output)')
    
    args = parser.parse_args()
    
    if args.quiet:
        args.verbose = False
    
    # Analyze mode
    if args.analyze:
        if not args.input:
            print("Error: Input file required for analysis mode")
            sys.exit(1)
        if not os.path.exists(args.input):
            print(f"Error: File {args.input} not found")
            sys.exit(1)
            
        structure = read_hdf5_structure(args.input)
        print(f"Analysis of {args.input}:")
        print(f"  Datasets: {structure['datasets']}")
        print(f"  Number of jets: {structure['n_jets']}")
        print(f"  Number of particles per jet: {structure['n_particles']}")
        print(f"  Jet features ({structure['n_jet_features']}): {structure['jet_features'][:10]}{'...' if len(structure['jet_features']) > 10 else ''}")
        print(f"  Particle features ({structure['n_particle_features']}): {structure['particle_features']}")
        return
    
    # Directory mode
    if args.input_dir and args.output_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory {args.input_dir} not found")
            sys.exit(1)
        convert_directory(args.input_dir, args.output_dir, args.pattern, args.tree_name, args.verbose)
        return
    
    # Single file mode
    if args.input and args.output:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            sys.exit(1)
        convert_hdf5_to_root(args.input, args.output, args.tree_name, args.verbose)
        return
    
    # No valid arguments
    print("Error: Must specify either single file conversion or directory conversion")
    print("Use --help for usage information")
    sys.exit(1)


if __name__ == '__main__':
    main()
