# Compile distributions from persistent volume storage

import numpy as np
import matplotlib.pyplot as plt
import os

CMS = {
    # "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Heros",
    "mathtext.bf": "TeX Gyre Heros:bold",
    "mathtext.sf": "TeX Gyre Heros",
    "mathtext.it": "TeX Gyre Heros:italic",
    "mathtext.tt": "TeX Gyre Heros",
    "mathtext.cal": "TeX Gyre Heros",
    "mathtext.default": "regular",
    "figure.figsize": (8.0, 8.0),
    "font.size": 14,
    #"text.usetex": True,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    # Make legends smaller
    "legend.fontsize": "x-small",  # Adjusted to a smaller size
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "savefig.transparent": False,
}
plt.style.use(CMS)

# Load QG files from /batched_hists

print('Loading QG hists...')

stem = 'qg_attention_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/qg_attention_distribution_batch_0.npy'):
    print('QG Files detected')

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

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

# Load TL files from /batched_hists

stem = 'tl_hist_distribution_batch_'
num_files = 50

if os.path.exists('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_hist_distribution_batch_0.npy'):
    print('TL Files detected')
    compiled = np.zeros_like(np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/tl_hist_distribution_batch_0.npy', allow_pickle=True))

for f in os.listdir('/part-vol-3/timlegge-ParT-trained/batched_hists/'):
    if stem in f:
        print(f'Loading {f}...')
        arr = np.load(f'/part-vol-3/timlegge-ParT-trained/batched_hists/{f}', allow_pickle=True)
        compiled += arr

print('All files loaded, now saving...')

np.save('/part-vol-3/timlegge-ParT-trained/tl_compiled_hist.npy', compiled)

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
plt.savefig('Compiled_TL_attentionDist.pdf', bbox_inches="tight")

print('TL plot saved! All done.')