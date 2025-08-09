#!/usr/bin/env python3
"""
Test script to check model availability and paths
"""

import os
import sys

# Model paths from QGandTLAttention.py
qgtrained_modelpath = '/home/tim_legge/save_qg_model/on-qg_final_best_epoch_state.pt'
tltrained_modelpath = '/home/tim_legge/save_tl_model/on-tl_final_best_epoch_state.pt'

def check_file_existence():
    """Check if the model files exist"""
    print("=== Model File Existence Check ===")
    
    print(f"QG Model Path: {qgtrained_modelpath}")
    print(f"QG Model Exists: {os.path.exists(qgtrained_modelpath)}")
    
    print(f"TL Model Path: {tltrained_modelpath}")
    print(f"TL Model Exists: {os.path.exists(tltrained_modelpath)}")
    
    # Check for alternative paths
    print("\n=== Alternative Paths ===")
    alt_paths = [
        'models/ParT_full.pt',
        'models/ParT_kin.pt', 
        'models/ParT_kinpid.pt'
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            print(f"Found: {path}")
            try:
                import torch
                state = torch.load(path, map_location='cpu')
                print(f"  - Loadable: Yes")
                print(f"  - Type: {type(state)}")
                if isinstance(state, dict):
                    print(f"  - Keys: {list(state.keys())}")
            except Exception as e:
                print(f"  - Error loading: {e}")

def check_weaver():
    """Check if weaver is available"""
    print("\n=== Weaver Availability Check ===")
    try:
        from weaver.nn.model.ParticleTransformer import ParticleTransformer
        print("Weaver ParticleTransformer: Available")
        return True
    except ImportError as e:
        print(f"Weaver ParticleTransformer: Not available - {e}")
        return False

def check_data():
    """Check if data files are available"""
    print("\n=== Data File Check ===")
    data_files = [
        'JetClass_example_100k.root',
        'data/JetClass_example_100k.root'
    ]
    
    for data_file in data_files:
        exists = os.path.exists(data_file)
        print(f"{data_file}: {'Found' if exists else 'Not found'}")
        if exists:
            try:
                import uproot
                f = uproot.open(data_file)
                print(f"  - Keys: {f.keys()}")
                if 'tree' in f:
                    tree = f['tree']
                    print(f"  - Tree entries: {len(tree)}")
                    print(f"  - Tree branches: {list(tree.keys())[:10]}...")  # First 10 branches
            except Exception as e:
                print(f"  - Error reading: {e}")

if __name__ == "__main__":
    check_file_existence()
    check_weaver()
    check_data()
    
    print("\n=== Summary ===")
    print("If models exist at the specified paths, the main script should work.")
    print("If weaver is not available, fallback attention generation will be used.")
    print("Real data will be used if JetClass_example_100k.root is available.")
