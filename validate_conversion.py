#!/usr/bin/env python3
"""
HDF5 to ROOT Conversion Validation Script

This script analyzes the comparison between HDF5-converted ROOT files and reference
ROOT files to identify potential conversion errors and suggest fixes.

Author: Generated for ParT_Interpretability project  
Date: August 18, 2025
"""

import os
import sys
import numpy as np
import uproot
import h5py
from compare_root_files import ROOTFileComparator


def analyze_conversion_issues(hdf5_file: str, converted_root: str, reference_root: str):
    """
    Analyze potential issues in HDF5 to ROOT conversion by comparing with reference.
    
    Args:
        hdf5_file: Original HDF5 file
        converted_root: ROOT file converted from HDF5  
        reference_root: Reference ROOT file (e.g., TopLandscape)
    """
    
    print("=" * 80)
    print("HDF5 TO ROOT CONVERSION VALIDATION ANALYSIS")
    print("=" * 80)
    print(f"HDF5 source: {hdf5_file}")
    print(f"Converted ROOT: {converted_root}")
    print(f"Reference ROOT: {reference_root}")
    print()
    
    # 1. Analyze HDF5 structure vs converted ROOT
    print("1. ANALYZING HDF5 vs CONVERTED ROOT CONSISTENCY")
    print("-" * 50)
    
    with h5py.File(hdf5_file, 'r') as h5f:
        with uproot.open(converted_root) as rootf:
            tree = rootf['tree']
            
            # Check data preservation
            jets_data = h5f['jets'][:]
            particles_data = h5f['jetConstituentList'][:]
            
            print(f"HDF5 jets shape: {jets_data.shape}")
            print(f"HDF5 particles shape: {particles_data.shape}")
            print(f"ROOT entries: {tree.num_entries}")
            print(f"ROOT branches: {len(tree.keys())}")
            
            # Check if jet counts match
            if jets_data.shape[0] == tree.num_entries:
                print("✓ Entry counts match between HDF5 and ROOT")
            else:
                print("✗ Entry count mismatch!")
                
            # Check some basic features
            try:
                root_jet_pt = tree['j_pt'].array()
                hdf5_jet_pt = jets_data[:, 1]  # j_pt is at index 1
                
                pt_diff = np.abs(root_jet_pt[:1000] - hdf5_jet_pt[:1000])
                max_diff = np.max(pt_diff)
                mean_diff = np.mean(pt_diff)
                
                print(f"Jet pT preservation check:")
                print(f"  Max difference: {max_diff:.6f}")
                print(f"  Mean difference: {mean_diff:.6f}")
                
                if max_diff < 1e-5:
                    print("✓ Jet pT values correctly preserved")
                else:
                    print("⚠ Potential precision loss in jet pT")
                    
            except Exception as e:
                print(f"✗ Error checking jet pT: {e}")
                
            # Check particle data structure
            try:
                root_particle_energy = tree['j1_e'].array()
                hdf5_particle_energy = particles_data[:, :, 3]  # energy at index 3
                
                # Check first few jets
                n_check = min(10, len(root_particle_energy))
                issues = 0
                
                for i in range(n_check):
                    root_particles = root_particle_energy[i]
                    hdf5_particles = hdf5_particle_energy[i]
                    
                    # Filter non-zero particles from HDF5
                    hdf5_valid = hdf5_particles[hdf5_particles > 0]
                    
                    if len(root_particles) != len(hdf5_valid):
                        issues += 1
                        
                if issues == 0:
                    print("✓ Particle filtering appears correct")
                else:
                    print(f"⚠ Particle count issues in {issues}/{n_check} jets")
                    
            except Exception as e:
                print(f"✗ Error checking particle data: {e}")
    
    print()
    
    # 2. Compare with reference using existing comparator
    print("2. COMPARING WITH REFERENCE DATASET")
    print("-" * 50)
    
    try:
        comparator = ROOTFileComparator(converted_root, reference_root)
        structure = comparator.compare_structure()
        
        # Analyze scale differences
        similar_comparison = comparator.compare_similar_branches(max_entries=1000)
        
        print("Scale analysis for similar features:")
        
        for category, results in similar_comparison.items():
            if category == 'jet_features' and results:
                print(f"\n{category.upper()}:")
                for result in results:
                    if 'data1_stats' in result and 'data2_stats' in result:
                        stats1 = result['data1_stats']
                        stats2 = result['data2_stats']
                        
                        if 'mean' in stats1 and 'mean' in stats2:
                            ratio = stats1['mean'] / stats2['mean'] if stats2['mean'] != 0 else float('inf')
                            print(f"  {result['description']}: {stats1['mean']:.2f} vs {stats2['mean']:.2f} (ratio: {ratio:.2f})")
                            
                            # Identify potential unit issues
                            if ratio > 10 or ratio < 0.1:
                                print(f"    ⚠ Large scale difference - possible unit conversion issue")
                            elif 0.9 < ratio < 1.1:
                                print(f"    ✓ Similar scales")
                            else:
                                print(f"    ⚠ Moderate scale difference")
        
        # Check label distribution
        print("\nLabel distribution analysis:")
        
        with uproot.open(converted_root) as f1, uproot.open(reference_root) as f2:
            try:
                labels1 = f1['tree']['label'].array()[:1000]
                labels2 = f2['tree']['label'].array()[:1000]
                
                unique1, counts1 = np.unique(labels1, return_counts=True)
                unique2, counts2 = np.unique(labels2, return_counts=True)
                
                print(f"  Converted file labels: {dict(zip(unique1, counts1))}")
                print(f"  Reference file labels: {dict(zip(unique2, counts2))}")
                
                # Check if label ranges are compatible
                if len(unique1) > len(unique2):
                    print("  ⚠ Converted file has more label categories than reference")
                elif len(unique1) < len(unique2):
                    print("  ⚠ Converted file has fewer label categories than reference")
                else:
                    print("  ✓ Similar number of label categories")
                    
            except Exception as e:
                print(f"  ✗ Error analyzing labels: {e}")
                
    except Exception as e:
        print(f"Error in reference comparison: {e}")
    
    print()
    
    # 3. Recommendations
    print("3. RECOMMENDATIONS FOR CONVERSION IMPROVEMENT")
    print("-" * 50)
    
    recommendations = []
    
    # Based on the comparison, suggest improvements
    try:
        comparator = ROOTFileComparator(converted_root, reference_root)
        similar_comparison = comparator.compare_similar_branches(max_entries=1000)
        
        for category, results in similar_comparison.items():
            if category == 'jet_features':
                for result in results:
                    if 'data1_stats' in result and 'data2_stats' in result:
                        stats1 = result['data1_stats']
                        stats2 = result['data2_stats']
                        
                        if 'mean' in stats1 and 'mean' in stats2:
                            ratio = stats1['mean'] / stats2['mean'] if stats2['mean'] != 0 else float('inf')
                            
                            if 'energy' in result['description'] and ratio > 2:
                                recommendations.append(
                                    f"Consider checking energy units - HLS4ML might be in GeV while reference is in different units"
                                )
                            elif 'momentum' in result['description'] and ratio > 2:
                                recommendations.append(
                                    f"Consider checking momentum units - scale difference detected in {result['description']}"
                                )
                            elif 'pt' in result['description'] and ratio > 2:
                                recommendations.append(
                                    f"Consider checking pT units or selection criteria - significant scale difference"
                                )
                                
        # Add general recommendations
        recommendations.extend([
            "Verify that particle filtering (energy > 0) is correctly implemented",
            "Check if coordinate systems are consistent (e.g., detector vs physics coordinates)",
            "Ensure label mappings are correct between different dataset conventions",
            "Consider adding validation against known physics constraints (e.g., energy-momentum relation)",
            "Add checks for special values (NaN, inf) in the conversion process"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
            
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        print("General recommendations:")
        print("1. Verify data type consistency between HDF5 and ROOT")
        print("2. Check for proper handling of jagged arrays")
        print("3. Ensure coordinate system consistency")
        print("4. Validate label encoding/decoding")
    
    print()
    print("=" * 80)


def main():
    """Main function for validation analysis."""
    
    if len(sys.argv) != 4:
        print("Usage: python validate_conversion.py <hdf5_file> <converted_root> <reference_root>")
        print()
        print("Example:")
        print("  python validate_conversion.py jetImage_6_150p_0_10000.h5 jetImage_6_test.root tl_test_file_0.root")
        sys.exit(1)
    
    hdf5_file = sys.argv[1]
    converted_root = sys.argv[2]
    reference_root = sys.argv[3]
    
    # Check if files exist
    for filepath in [hdf5_file, converted_root, reference_root]:
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found")
            sys.exit(1)
    
    analyze_conversion_issues(hdf5_file, converted_root, reference_root)


if __name__ == '__main__':
    main()
