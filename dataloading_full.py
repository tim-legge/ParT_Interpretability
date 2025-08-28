from hepconvert import parquet_to_root
import os

for i, f in enumerate(os.listdir("./datasets/qg_dataset/QuarkGluon/")):
    if f.startswith("test") and f.endswith(".parquet"):
        qg_datapath = os.path.join('./datasets/qg_dataset/QuarkGluon/', f'{f.replace('.parquet', '.root')}')
    parquet_to_root(f"./datasets/qg_dataset/QuarkGluon/qg_test_file_{i}.root", qg_datapath, name="tree", force=True)

parquet_to_root("./datasets/tl_dataset/TopLandscape/tl_test_file.root", "./datasets/tl_dataset/TopLandscape/test_file.parquet", name="tree", force=True)