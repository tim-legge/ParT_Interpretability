# Compile distributions from persistent volume storage

import numpy as np
import matplotlib.pyplot as plt
import os
import mplhep as hep
plt.style.use(hep.style.ROOT)

# Load QG files from /batched_hists

print('Loading QG hists...')

stem = 'qg_attention_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/qg_attention_distribution_batch_0.npy'):
    print('QG Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/qg_attention_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

# normalize compiled
compiled /= compiled.sum()

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/qg_compiled_hist.npy', compiled)

print('Saved! Now plotting...')

num_bins = len(compiled)
bin_edges = np.linspace(0, 1, num_bins + 1)

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, compiled, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title('QG-trained Attention Distribution', fontsize=16)
plt.savefig('Compiled_QG_attentionDist.pdf', bbox_inches="tight")

print('QG plot saved! Now for TL...')

# Load TL topsonly files from /batched_hists

stem = 'tl_topsonly_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_hist_distribution_batch_0.npy'):
    print('TL Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_hist_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

# normalize compiled
compiled /= compiled.sum()

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/tl_topsonly_compiled_hist.npy', compiled)

print('Saved! Now plotting...')

num_bins = len(compiled)
bin_edges = np.linspace(0, 1, num_bins + 1)

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, compiled, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title(r'TL $t\rightarrow bqq\prime$ Attention Distribution', fontsize=16)
plt.savefig('Compiled_TL_topsonly_attentionDist.pdf', bbox_inches="tight")

print('TL Tops only plot saved! Now for TL QCD only...')

# Load TL qcdonly files from /batched_hists

stem = 'tl_qcdonly_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_qcdonly_hist_distribution_batch_0.npy'):
    print('TL Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_qcdonly_hist_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

# normalize compiled
compiled /= compiled.sum()

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/tl_qcdonly_compiled_hist.npy', compiled)

print('TL QCD Dist saved! Now plotting...')

num_bins = len(compiled)
bin_edges = np.linspace(0, 1, num_bins + 1)

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, compiled, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title(r'TL-trained $q/g$ Attention Distribution', fontsize=16)
plt.savefig('Compiled_TL_QCDonly_attentionDist.pdf', bbox_inches="tight")

print('TL QCD only plot saved! All done.')

# Load TL run4 files from /batched_hists

stem = 'tl_run4_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_run4_hist_distribution_batch_0.npy'):
    print('TL Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_run4_hist_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

# normalize compiled
compiled /= compiled.sum()

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/tl_run4_compiled_hist.npy', compiled)

print('Saved! Now plotting...')

num_bins = len(compiled)
bin_edges = np.linspace(0, 1, num_bins + 1)

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, compiled, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title('TL-trained Attention Distribution', fontsize=16)
plt.savefig('Compiled_TL_run4_attentionDist.pdf', bbox_inches="tight")

print('TL plot saved!')

# Load JC kin top only hists from /batched_hists

stem = 'jc_kin_topsonly_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_kin_hist_distribution_batch_0.npy'):
    print('TL Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_kin_topsonly_hist_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

# normalize compiled
compiled /= compiled.sum()

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/jc_kin_topsonly_compiled_hist.npy', compiled)

print('Saved! Now plotting...')

num_bins = len(compiled)
bin_edges = np.linspace(0, 1, num_bins + 1)

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, compiled, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title(r'JC Kinematic $t\rightarrow bqq \prime$ Attention Distribution', fontsize=16)
plt.savefig('Compiled_JC_kin_topsonly_attentionDist.pdf', bbox_inches="tight")

print('JetClass kin top-only plot saved!')

# Load JC kin qcd only hists from /batched_hists

stem = 'jc_kin_qcdonly_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_kin_qcdonly_hist_distribution_batch_0.npy'):
    print('JC Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_kin_qcdonly_hist_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

# normalize compiled
compiled /= compiled.sum()

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/jc_kin_qcdonly_compiled_hist.npy', compiled)

print('Saved! Now plotting...')

num_bins = len(compiled)
bin_edges = np.linspace(0, 1, num_bins + 1)

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, compiled, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title(r'JC Kinematic $q/g$ Attention Distribution', fontsize=16)
plt.savefig('Compiled_JC_kin_qcdonly_attentionDist.pdf', bbox_inches="tight")

print('JetClass kin QCD-only plot saved!')

# JetClass full top only

stem = 'jc_full_topsonly_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_full_hist_distribution_batch_0.npy'):
    print('TL Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_full_topsonly_hist_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

# normalize compiled
compiled /= compiled.sum()

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/jc_full_topsonly_compiled_hist.npy', compiled)

print('Saved! Now plotting...')

num_bins = len(compiled)
bin_edges = np.linspace(0, 1, num_bins + 1)

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, compiled, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title(r'JC Full $t\rightarrow bqq\prime$ Attention Distribution', fontsize=16)
plt.savefig('Compiled_JC_full_topsonly_attentionDist.pdf', bbox_inches="tight")

print('JetClass kin top-only plot saved!')

# Load JC kin qcd only hists from /batched_hists

stem = 'jc_full_qcdonly_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_full_qcdonly_hist_distribution_batch_0.npy'):
    print('JC Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_full_qcdonly_hist_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

# normalize compiled
compiled /= compiled.sum()

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/jc_full_qcdonly_compiled_hist.npy', compiled)

print('Saved! Now plotting...')

num_bins = len(compiled)
bin_edges = np.linspace(0, 1, num_bins + 1)

# Manually set the bin centers to have equal bar spacing
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the centers of each bin
equal_width = bin_edges[1] - bin_edges[0]  # Set equal width for all bars based on bin spacing

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

# Plot bar graph with equal-width bars
ax.bar(bin_centers, compiled, width=equal_width, log=False)  # No log scale for clearer visualization

# Set custom x-tick locations and labels (optional)
#ax.set_xticks(bin_centers, 0.1)
#ax.set_xticklabels([f'{edge:.2f}' for edge in bin_centers], fontsize=10, fontweight='bold')

# Set x and y axis labels
ax.set_xlabel('Attention Score', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
plt.yscale('log')

# Add a title
ax.set_title(r'JC Full $q/g$ Attention Distribution', fontsize=16)
plt.savefig('Compiled_JC_full_qcdonly_attentionDist.pdf', bbox_inches="tight")

print('JetClass full QCD-only plot saved!')