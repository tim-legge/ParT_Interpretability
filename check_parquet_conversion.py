import pandas as pd
df = pd.read_parquet("/home/jovyan/ParT_Interpretability/datasets/hls4ml/HLS4ML/train/train-parquet/jetImage_0_150p_0_10000.parquet")
print(f"Jet nparticles range: {df['jet_nparticles'].min()} - {df['jet_nparticles'].max()}")
print(f"Any zero-energy particles: {any(any(energies) == 0 for energies in df['part_energy'][:10])}")
print(f"Sample particle counts per jet: {[len(particles) for particles in df['part_energy'][:5]]}")