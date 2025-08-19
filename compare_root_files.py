#!/usr/bin/env python3
"""
ROOT File Structure and Data Comparison Tool

This script compares the structure and data values between two ROOT files,
specifically designed to validate HDF5 to ROOT conversions by comparing
against reference datasets like TopLandscape.

Author: Generated for ParT_Interpretability project
Date: August 18, 2025
"""

import os
import sys
import argparse
import numpy as np
import uproot
from typing import Dict, List, Tuple, Optional, Any

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class ROOTFileComparator:
    """Class to compare structure and data between ROOT files."""
    
    def __init__(self, file1_path: str, file2_path: str, tree_name: str = "tree"):
        """
        Initialize comparator with two ROOT files.
        
        Args:
            file1_path: Path to first ROOT file (e.g., converted HLS4ML)
            file2_path: Path to second ROOT file (e.g., reference TopLandscape)
            tree_name: Name of the tree in both files
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.tree_name = tree_name
        
        # Load files
        try:
            self.file1 = uproot.open(file1_path)
            self.tree1 = self.file1[tree_name]
            self.name1 = os.path.basename(file1_path).replace('.root', '')
        except Exception as e:
            raise ValueError(f"Cannot open {file1_path}: {e}")
            
        try:
            self.file2 = uproot.open(file2_path)
            self.tree2 = self.file2[tree_name]
            self.name2 = os.path.basename(file2_path).replace('.root', '')
        except Exception as e:
            raise ValueError(f"Cannot open {file2_path}: {e}")
    
    def compare_structure(self) -> Dict[str, Any]:
        """Compare the structural properties of both ROOT files."""
        
        keys1 = set(self.tree1.keys())
        keys2 = set(self.tree2.keys())
        
        structure_comparison = {
            'file1_info': {
                'path': self.file1_path,
                'name': self.name1,
                'n_entries': self.tree1.num_entries,
                'n_branches': len(keys1),
                'branches': sorted(list(keys1))
            },
            'file2_info': {
                'path': self.file2_path,
                'name': self.name2,
                'n_entries': self.tree2.num_entries,
                'n_branches': len(keys2),
                'branches': sorted(list(keys2))
            },
            'common_branches': sorted(list(keys1.intersection(keys2))),
            'file1_only': sorted(list(keys1 - keys2)),
            'file2_only': sorted(list(keys2 - keys1)),
            'similar_branches': self._find_similar_branches(keys1, keys2)
        }
        
        return structure_comparison
    
    def _find_similar_branches(self, keys1: set, keys2: set) -> Dict[str, List[str]]:
        """Find branches with similar names/purposes between files."""
        
        # Define mapping patterns between different naming conventions
        mapping_patterns = {
            # Jet-level features
            'jet_features': {
                'patterns': [
                    (['j_pt', 'jet_pt'], 'transverse momentum'),
                    (['j_eta', 'jet_eta'], 'pseudorapidity'),
                    (['j_phi', 'jet_phi'], 'azimuthal angle'),
                    (['j_mass', 'jet_mass'], 'mass'),
                    (['j_energy', 'jet_energy'], 'energy'),
                    (['j_nparticles', 'jet_nparticles'], 'number of particles')
                ]
            },
            # Particle-level features  
            'particle_features': {
                'patterns': [
                    (['j1_px', 'part_px'], 'particle px'),
                    (['j1_py', 'part_py'], 'particle py'),
                    (['j1_pz', 'part_pz'], 'particle pz'),
                    (['j1_e', 'part_energy'], 'particle energy'),
                    (['j1_pt', 'part_pt'], 'particle pt'),
                    (['j1_eta', 'part_eta'], 'particle eta'),
                    (['j1_phi', 'part_phi'], 'particle phi'),
                    (['j1_etarel', 'part_deta'], 'relative eta'),
                    (['j1_phirel', 'part_dphi'], 'relative phi'),
                    (['j1_deltaR', 'part_deltaR'], 'delta R')
                ]
            },
            # Labels
            'labels': {
                'patterns': [
                    (['label', 'label'], 'classification label'),
                    (['j_isGluon', 'label_QCD'], 'gluon/QCD label'),
                    (['j_isQuark', 'label_Quark'], 'quark label'),
                    (['j_isW', 'label_W'], 'W boson label'),
                    (['j_isZ', 'label_Z'], 'Z boson label'),
                    (['j_isTop', 'label_Top'], 'top quark label')
                ]
            }
        }
        
        similar_branches = {}
        
        for category, info in mapping_patterns.items():
            similar_branches[category] = []
            for pattern_list, description in info['patterns']:
                matches = []
                for pattern in pattern_list:
                    if pattern in keys1:
                        matches.append(('file1', pattern))
                    if pattern in keys2:
                        matches.append(('file2', pattern))
                
                if len(matches) >= 2:
                    similar_branches[category].append({
                        'description': description,
                        'matches': matches,
                        'pattern': pattern_list
                    })
        
        return similar_branches
    
    def compare_common_branches(self, max_entries: int = 1000) -> Dict[str, Any]:
        """Compare data values for branches that exist in both files."""
        
        common_keys = set(self.tree1.keys()).intersection(set(self.tree2.keys()))
        
        if not common_keys:
            return {'message': 'No common branches found'}
        
        comparison_results = {}
        
        for key in sorted(common_keys):
            try:
                # Read limited number of entries for comparison
                data1 = self.tree1[key].array(entry_stop=min(max_entries, self.tree1.num_entries))
                data2 = self.tree2[key].array(entry_stop=min(max_entries, self.tree2.num_entries))
                
                comparison_results[key] = self._compare_branch_data(key, data1, data2)
                
            except Exception as e:
                comparison_results[key] = {'error': str(e)}
        
        return comparison_results
    
    def _compare_branch_data(self, branch_name: str, data1, data2) -> Dict[str, Any]:
        """Compare data from a single branch between two files."""
        
        import awkward as ak
        
        result = {
            'branch_name': branch_name,
            'data1_type': str(type(data1)),
            'data2_type': str(type(data2)),
            'data1_shape': getattr(data1, 'shape', 'N/A'),
            'data2_shape': getattr(data2, 'shape', 'N/A')
        }
        
        try:
            # Handle awkward arrays (jagged)
            if hasattr(data1, 'type') and 'var' in str(data1.type):
                # Jagged array
                result['data1_type'] = 'jagged_array'
                result['data1_lengths'] = ak.to_numpy(ak.num(data1))
                result['data1_mean_length'] = np.mean(result['data1_lengths'])
                result['data1_flattened_stats'] = self._get_array_stats(ak.flatten(data1))
            else:
                # Regular array
                result['data1_type'] = 'regular_array'
                result['data1_stats'] = self._get_array_stats(data1)
            
            if hasattr(data2, 'type') and 'var' in str(data2.type):
                # Jagged array
                result['data2_type'] = 'jagged_array'
                result['data2_lengths'] = ak.to_numpy(ak.num(data2))
                result['data2_mean_length'] = np.mean(result['data2_lengths'])
                result['data2_flattened_stats'] = self._get_array_stats(ak.flatten(data2))
            else:
                # Regular array
                result['data2_type'] = 'regular_array'
                result['data2_stats'] = self._get_array_stats(data2)
                
        except Exception as e:
            result['comparison_error'] = str(e)
        
        return result
    
    def _get_array_stats(self, arr) -> Dict[str, float]:
        """Get statistical summary of an array."""
        import awkward as ak
        
        try:
            if hasattr(arr, 'type'):
                # Awkward array
                arr_np = ak.to_numpy(arr)
            else:
                arr_np = np.array(arr)
            
            # Remove any non-finite values
            finite_mask = np.isfinite(arr_np)
            arr_clean = arr_np[finite_mask]
            
            if len(arr_clean) == 0:
                return {'error': 'No finite values'}
            
            stats = {
                'count': len(arr_clean),
                'mean': float(np.mean(arr_clean)),
                'std': float(np.std(arr_clean)),
                'min': float(np.min(arr_clean)),
                'max': float(np.max(arr_clean)),
                'median': float(np.median(arr_clean)),
                'q25': float(np.percentile(arr_clean, 25)),
                'q75': float(np.percentile(arr_clean, 75))
            }
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def compare_similar_branches(self, max_entries: int = 1000) -> Dict[str, Any]:
        """Compare data between branches that serve similar purposes."""
        
        structure_info = self.compare_structure()
        similar_branches = structure_info['similar_branches']
        
        comparison_results = {}
        
        for category, matches in similar_branches.items():
            comparison_results[category] = []
            
            for match_info in matches:
                try:
                    # Extract file1 and file2 branch names
                    file1_branches = [m[1] for m in match_info['matches'] if m[0] == 'file1']
                    file2_branches = [m[1] for m in match_info['matches'] if m[0] == 'file2']
                    
                    if file1_branches and file2_branches:
                        # Compare the first matching branch from each file
                        branch1 = file1_branches[0]
                        branch2 = file2_branches[0]
                        
                        data1 = self.tree1[branch1].array(entry_stop=min(max_entries, self.tree1.num_entries))
                        data2 = self.tree2[branch2].array(entry_stop=min(max_entries, self.tree2.num_entries))
                        
                        comparison = self._compare_branch_data(f"{branch1} vs {branch2}", data1, data2)
                        comparison['description'] = match_info['description']
                        comparison['file1_branch'] = branch1
                        comparison['file2_branch'] = branch2
                        
                        comparison_results[category].append(comparison)
                        
                except Exception as e:
                    comparison_results[category].append({
                        'error': str(e),
                        'description': match_info['description']
                    })
        
        return comparison_results
    
    def generate_report(self, output_file: Optional[str] = None, max_entries: int = 1000) -> str:
        """Generate comprehensive comparison report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ROOT FILE COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated on: {pd.Timestamp.now()}")
        report_lines.append("")
        
        # Structure comparison
        structure = self.compare_structure()
        report_lines.append("STRUCTURE COMPARISON")
        report_lines.append("-" * 40)
        report_lines.append(f"File 1: {structure['file1_info']['name']}")
        report_lines.append(f"  Path: {structure['file1_info']['path']}")
        report_lines.append(f"  Entries: {structure['file1_info']['n_entries']:,}")
        report_lines.append(f"  Branches: {structure['file1_info']['n_branches']}")
        report_lines.append("")
        report_lines.append(f"File 2: {structure['file2_info']['name']}")
        report_lines.append(f"  Path: {structure['file2_info']['path']}")
        report_lines.append(f"  Entries: {structure['file2_info']['n_entries']:,}")
        report_lines.append(f"  Branches: {structure['file2_info']['n_branches']}")
        report_lines.append("")
        
        # Common branches
        if structure['common_branches']:
            report_lines.append(f"Common branches ({len(structure['common_branches'])}): {', '.join(structure['common_branches'])}")
        else:
            report_lines.append("No common branches found")
        report_lines.append("")
        
        # File-specific branches
        if structure['file1_only']:
            report_lines.append(f"Branches only in {self.name1} ({len(structure['file1_only'])}):")
            for i in range(0, len(structure['file1_only']), 5):
                report_lines.append(f"  {', '.join(structure['file1_only'][i:i+5])}")
        report_lines.append("")
        
        if structure['file2_only']:
            report_lines.append(f"Branches only in {self.name2} ({len(structure['file2_only'])}):")
            for i in range(0, len(structure['file2_only']), 5):
                report_lines.append(f"  {', '.join(structure['file2_only'][i:i+5])}")
        report_lines.append("")
        
        # Similar branches analysis
        report_lines.append("SIMILAR BRANCHES ANALYSIS")
        report_lines.append("-" * 40)
        
        similar_comparison = self.compare_similar_branches(max_entries)
        for category, results in similar_comparison.items():
            if results:
                report_lines.append(f"\n{category.upper()}:")
                for result in results:
                    if 'error' in result:
                        report_lines.append(f"  ERROR in {result.get('description', 'unknown')}: {result['error']}")
                    else:
                        report_lines.append(f"  {result['description']}:")
                        report_lines.append(f"    {result['file1_branch']} vs {result['file2_branch']}")
                        
                        # Compare statistics if available
                        if 'data1_stats' in result and 'data2_stats' in result:
                            stats1 = result['data1_stats']
                            stats2 = result['data2_stats']
                            if 'mean' in stats1 and 'mean' in stats2:
                                report_lines.append(f"    Mean: {stats1['mean']:.4f} vs {stats2['mean']:.4f}")
                                report_lines.append(f"    Std:  {stats1['std']:.4f} vs {stats2['std']:.4f}")
                                report_lines.append(f"    Range: [{stats1['min']:.4f}, {stats1['max']:.4f}] vs [{stats2['min']:.4f}, {stats2['max']:.4f}]")
                        
                        elif 'data1_flattened_stats' in result and 'data2_flattened_stats' in result:
                            stats1 = result['data1_flattened_stats']
                            stats2 = result['data2_flattened_stats']
                            if 'mean' in stats1 and 'mean' in stats2:
                                report_lines.append(f"    Mean length: {result.get('data1_mean_length', 'N/A'):.2f} vs {result.get('data2_mean_length', 'N/A'):.2f}")
                                report_lines.append(f"    Flattened mean: {stats1['mean']:.4f} vs {stats2['mean']:.4f}")
                                report_lines.append(f"    Flattened std:  {stats1['std']:.4f} vs {stats2['std']:.4f}")
        
        # Common branches data comparison
        if structure['common_branches']:
            report_lines.append("\n\nCOMMON BRANCHES DATA COMPARISON")
            report_lines.append("-" * 40)
            
            common_comparison = self.compare_common_branches(max_entries)
            for branch, result in common_comparison.items():
                if 'error' in result:
                    report_lines.append(f"{branch}: ERROR - {result['error']}")
                else:
                    report_lines.append(f"\n{branch}:")
                    report_lines.append(f"  Type: {result['data1_type']} vs {result['data2_type']}")
                    
                    if 'data1_stats' in result and 'data2_stats' in result:
                        stats1 = result['data1_stats']
                        stats2 = result['data2_stats']
                        if 'mean' in stats1 and 'mean' in stats2:
                            report_lines.append(f"  Mean: {stats1['mean']:.6f} vs {stats2['mean']:.6f}")
                            if stats1['mean'] != 0:
                                diff_pct = abs(stats1['mean'] - stats2['mean']) / abs(stats1['mean']) * 100
                                report_lines.append(f"  Difference: {diff_pct:.2f}%")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Compare structure and data between two ROOT files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two ROOT files
    python compare_root_files.py file1.root file2.root
    
    # Save comparison report
    python compare_root_files.py file1.root file2.root -o comparison_report.txt
    
    # Use different tree name
    python compare_root_files.py file1.root file2.root --tree-name events
    
    # Limit entries for faster comparison
    python compare_root_files.py file1.root file2.root --max-entries 500
        """
    )
    
    parser.add_argument('file1', help='First ROOT file (e.g., converted HLS4ML file)')
    parser.add_argument('file2', help='Second ROOT file (e.g., reference TopLandscape file)')
    parser.add_argument('-o', '--output', help='Output file for comparison report')
    parser.add_argument('--tree-name', default='tree', help='Name of tree in ROOT files (default: tree)')
    parser.add_argument('--max-entries', type=int, default=1000, 
                       help='Maximum entries to compare for performance (default: 1000)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if files exist
    for filepath in [args.file1, args.file2]:
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found")
            sys.exit(1)
    
    try:
        # Create comparator
        comparator = ROOTFileComparator(args.file1, args.file2, args.tree_name)
        
        # Generate report
        report = comparator.generate_report(args.output, args.max_entries)
        
        if not args.output:
            print(report)
            
    except Exception as e:
        print(f"Error during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
