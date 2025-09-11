import uproot
import numpy as np

with uproot.open("/part-vol-3/weaver-core/particle-transformer/datasets/JetClass/Pythia/train_100M/TTBar_000.root") as file:
    tree = file["tree"]
    features = tree.arrays(filter_name="*part_*")
    if "part_d0" in features:
        part_d0 = features["part_d0"].to_numpy()
        print("part_d0 exists:", part_d0[:20])
    else:
        print("part_d0 does not exist")