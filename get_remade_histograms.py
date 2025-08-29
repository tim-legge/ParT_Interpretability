import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
plt.style.use(hep.style.ROOT)

fontsize = 20

hist = np.load('/part-vol-3/timlegge-ParT-trained/batched_hists/jc_full_qcdonly_hist_distribution_batch_40.npy')

import os
import numpy as np
import matplotlib.pyplot as plt

# 🔧 Folder with your precomputed histograms
folder = "/part-vol-3/timlegge-ParT-trained/batched_hists"

# Collect and aggregate all histograms
all_hist = None

for fname in sorted(os.listdir(folder)):
    if fname.startswith("jc_full_topsonly_hist_distribution_batch_") and fname.endswith(".npy"):
        fpath = os.path.join(folder, fname)
        print(f"Loading {fpath} ...")
        hist = np.load(fpath)

        if all_hist is None:
            all_hist = hist.astype(np.float64)
        else:
            all_hist += hist  # elementwise sum

if all_hist is None:
    print("No matching histogram files found.")
else:
    print(f"Aggregated histogram shape: {all_hist.shape}")

    # Plot aggregated histogram
    plt.figure(figsize=(8,5))
    plt.bar(np.arange(len(all_hist)), all_hist, width=1.0, edgecolor="black")
    plt.xlabel("Bin Index")
    plt.ylabel("Counts")
    plt.title("Aggregated Attention Score Histogram")
    plt.show()

#FULL JC QCD ONLY

import os
import numpy as np
import matplotlib.pyplot as plt

# 🔧 Folder with precomputed histograms
folder = "/part-vol-3/timlegge-ParT-trained/batched_hists"

# Aggregate all histograms
all_hist = None
for fname in sorted(os.listdir(folder)):
    if fname.startswith("jc_full_qcdonly_hist_distribution_batch_") and fname.endswith(".npy"):
        fpath = os.path.join(folder, fname)
        hist = np.load(fpath)
        if all_hist is None:
            all_hist = hist.astype(np.float64)
        else:
            all_hist += hist

if all_hist is None:
    print("No histograms found.")
else:
    # Normalize to probability distribution
    probabilities = all_hist / all_hist.sum()

    # Create bin edges (assuming scores are between 0 and 1)
    num_bins = len(probabilities)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    equal_width = bin_edges[1] - bin_edges[0]

    # Plot in your preferred format
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    #ax.bar(bin_centers, probabilities, width=equal_width, log=False, edgecolor="black")
    #ax.hist(probabilities, bins=num_bins, edgecolor="black", log=True)

    counts = all_hist
    bins = bin_edges

    ax.hist(bins[:-1], bins, weights=counts)
    ax.set_xlabel("Attention Score", fontsize=fontsize)
    ax.set_ylabel("Probability", fontsize=fontsize)
    plt.yscale("log")
    plt.show()
    # Save if needed:
    plt.savefig("JC_FULL_QCDONLY_AttentionDist.pdf", bbox_inches="tight")

## FULL JC Hadronic TOPSONLY

import os
import numpy as np
import matplotlib.pyplot as plt

# 🔧 Folder with precomputed histograms
folder = "/part-vol-3/timlegge-ParT-trained/batched_hists"

# Aggregate all histograms
all_hist = None
for fname in sorted(os.listdir(folder)):
    if fname.startswith("jc_full_topsonly_hist_distribution_batch_") and fname.endswith(".npy"):
        fpath = os.path.join(folder, fname)
        hist = np.load(fpath)
        if all_hist is None:
            all_hist = hist.astype(np.float64)
        else:
            all_hist += hist

if all_hist is None:
    print("No histograms found.")
else:
    # Normalize to probability distribution
    probabilities = all_hist / all_hist.sum()

    # Create bin edges (assuming scores are between 0 and 1)
    num_bins = len(probabilities)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    equal_width = bin_edges[1] - bin_edges[0]

    # Plot in your preferred format
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    #ax.bar(bin_centers, probabilities, width=equal_width, log=False, edgecolor="black")
    #ax.hist(probabilities, bins=num_bins, edgecolor="black", log=False)

    counts = all_hist
    bins = bin_edges

    ax.hist(bins[:-1], bins, weights=counts)
    ax.set_xlabel("Attention Score", fontsize=fontsize)
    ax.set_ylabel("Probability", fontsize=fontsize)

    plt.yscale("log")
    plt.savefig("./JetClass_Full_TBQQAttentionDist.pdf", bbox_inches="tight")

    plt.show()
    # Save if needed:

import os
import numpy as np
import matplotlib.pyplot as plt

# 🔧 Folder with precomputed histograms
folder = "/part-vol-3/timlegge-ParT-trained/batched_hists"

# Aggregate all histograms
all_hist = None
for fname in sorted(os.listdir(folder)):
    if fname.startswith("tl_topsonly_hist_distribution_batch_") and fname.endswith(".npy"):
        fpath = os.path.join(folder, fname)
        hist = np.load(fpath)
        if all_hist is None:
            all_hist = hist.astype(np.float64)
        else:
            all_hist += hist

if all_hist is None:
    print("No histograms found.")
else:
    # Normalize to probability distribution
    probabilities = all_hist / all_hist.sum()

    # Create bin edges (assuming scores are between 0 and 1)
    num_bins = len(probabilities)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    equal_width = bin_edges[1] - bin_edges[0]

    # Plot in your preferred format
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    #ax.bar(bin_centers, probabilities, width=equal_width, log=False, edgecolor="black")
    #ax.bar(probabilities, bins=num_bins, edgecolor="black", log=False)
    counts = all_hist
    bins = bin_edges

    ax.hist(bins[:-1], bins, weights=counts)
    ax.set_xlabel("Attention Score", fontsize=fontsize)
    ax.set_ylabel("Probability", fontsize=fontsize)

    plt.yscale("log")
    plt.savefig("./TL_TOPSONLY_AttentionDist.pdf", bbox_inches="tight")

    plt.show()

    # Save if needed:

import os
import numpy as np
import matplotlib.pyplot as plt

# 🔧 Folder with precomputed histograms
folder = "/part-vol-3/timlegge-ParT-trained/batched_hists"

# Aggregate all histograms
all_hist = None
for fname in sorted(os.listdir(folder)):
    if fname.startswith("qg") and fname.endswith(".npy"):
        fpath = os.path.join(folder, fname)
        hist = np.load(fpath)
        if all_hist is None:
            all_hist = hist.astype(np.float64)
        else:
            all_hist += hist

if all_hist is None:
    print("No histograms found.")
else:
    # Normalize to probability distribution
    probabilities = all_hist / all_hist.sum()

    # Create bin edges (assuming scores are between 0 and 1)
    num_bins = len(probabilities)
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    equal_width = bin_edges[1] - bin_edges[0]

    # Plot in your preferred format
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    #ax.bar(bin_centers, probabilities, width=equal_width, log=False, edgecolor="black")
    #ax.hist(probabilities, bins=num_bins, edgecolor="black", log=False)
    counts = all_hist
    bins = bin_edges

    ax.hist(bins[:-1], bins, weights=counts)
    ax.set_xlabel("Attention Score", fontsize=fontsize)
    ax.set_ylabel("Probability", fontsize=fontsize)
    plt.yscale("log")
    plt.savefig("./QG_overall_AttentionDist.pdf", bbox_inches="tight")

    plt.show()
    # Save if needed:
