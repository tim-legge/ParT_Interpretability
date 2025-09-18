import os
import numpy as np
import matplotlib.pyplot as plt

### Pool the histograms in persistent volume, plot as separate distributions

jc_histograms_dir = '/part-vol-3/timlegge-ParT-trained/histograms_jc_feats/'
tl_histograms_dir = '/part-vol-3/timlegge-ParT-trained/histograms_tl_feats/'
output_dir = '/part-vol-3/timlegge-ParT-trained/plot_jc_tl_features/'

def pool_hists(dataset):
    if dataset == 'jc':
        histograms_dir = jc_histograms_dir
        
    elif dataset == 'tl':
        histograms_dir = tl_histograms_dir
    else:
        raise ValueError("dataset must be 'jc' or 'tl'")
    
    hists_dict = {
            'part_pt_log': [],
            'part_e_log': [],
            'part_log_ptrel': [],
            'part_log_erel': [],
            'part_deltaR': [],
            'part_deta': [],
            'part_dphi': [],
        }
    
    histograms = [f for f in sorted(os.listdir(histograms_dir)) if 'hist' in f and f.endswith('.npy')]
    for key in hists_dict.keys():
        feature_histograms = [f for f in histograms if key in f]
        #print(f"elements of feature_histograms as filenames {key}: {feature_histograms}")
        feature_histograms = [np.load(os.path.join(histograms_dir,f)) for f in feature_histograms]
        print(f'first two elements of feature_histograms from {histograms_dir} as arrays {key}: {feature_histograms[:2]}')
        feature_histograms = sum(feature_histograms)
        assert isinstance(feature_histograms[0], np.ndarray), "Histogram bins is not a numpy array"
        assert isinstance(feature_histograms, np.ndarray), "Summed histograms is not a numpy array"
        feature_histograms = feature_histograms/np.sum(feature_histograms.flatten())
        hists_dict[key] = feature_histograms

    for key in hists_dict.keys():
        np.save(os.path.join(output_dir, f'pooled_{dataset}_{key}_hists.npy'), hists_dict[key])

    print('Saved pooled data for all features!')
    return hists_dict

jc_pooled_hists = pool_hists('jc')
tl_pooled_hists = pool_hists('tl')

def plot_individual_features(pooled_hists):

    feat_ranges = {
        'part_pt_log': (-2.4, 6.3),
        'part_e_log': (-0.9, 7.0),
        'part_log_ptrel': (-8.6, -0.1),
        'part_log_erel': (-7.7, -0.2),
        'part_deltaR': (0, 3.0),
        'part_charge': (-1.0, 1.0),
        'part_isChargedHadron': (0.0, 1.0),
        'part_isNeutralHadron': (0.0, 1.0),
        'part_isPhoton': (0.0, 1.0),
        'part_isElectron': (0.0, 1.0),
        'part_isMuon': (0.0, 1.0),
        'part_d0': (-1.0, 1.0),
        'part_d0err': (0.0, 0.7),
        'part_dz': (-1.0, 1.0),
        'part_dzerr': (0.0, 2.2),
        'part_deta': (-2.9, 2.5),
        'part_dphi': (-0.9, 0.9),
    }
    placeholder = np.zeros(1)
    for key in pooled_hists.keys():
        fig, ax = plt.subplots(figsize=(8,6))
        _, bin_edges = np.histogram(placeholder, bins=50, range=feat_ranges[key])
        ax.step(bin_edges[:-1], pooled_hists[key], where='post', label=key)
        ax.set_xlabel(key)
        ax.set_ylabel('Probability')
        ax.set_title(f'Histogram of {key}')
        plt.savefig(os.path.join(output_dir, f'{key}_hist.png'))
        plt.close()
        print(f'Saved plot for {key}')

print('Plotting individual JC features')
plot_individual_features(jc_pooled_hists)
print('Plotting individual TL features')
plot_individual_features(tl_pooled_hists)

def plot_comparative_features(jc_pooled_hists, tl_pooled_hists):
    feat_ranges = {
        'part_pt_log': (-2.4, 6.3),
        'part_e_log': (-0.9, 7.0),
        'part_log_ptrel': (-8.6, -0.1),
        'part_log_erel': (-7.7, -0.2),
        'part_deltaR': (0, 3.0),
        'part_charge': (-1.0, 1.0),
        'part_isChargedHadron': (0.0, 1.0),
        'part_isNeutralHadron': (0.0, 1.0),
        'part_isPhoton': (0.0, 1.0),
        'part_isElectron': (0.0, 1.0),
        'part_isMuon': (0.0, 1.0),
        'part_d0': (-1.0, 1.0),
        'part_d0err': (0.0, 0.7),
        'part_dz': (-1.0, 1.0),
        'part_dzerr': (0.0, 2.2),
        'part_deta': (-2.9, 2.5),
        'part_dphi': (-0.9, 0.9),
    }
    placeholder = np.zeros(1)
    for key in tl_pooled_hists.keys():
        fig, ax = plt.subplots(figsize=(8,6))
        _, bin_edges = np.histogram(placeholder, bins=50, range=feat_ranges[key])
        ax.step(bin_edges[:-1], jc_pooled_hists[key], where='post', label='JetClass', color='blue', alpha=0.7)
        ax.step(bin_edges[:-1], tl_pooled_hists[key], where='post', label='TopLandscape', color='orange', alpha=0.7)
        ax.set_xlabel(key)
        ax.set_ylabel('Probability')
        ax.set_title(f'Comparative Histogram of {key}')
        ax.legend()
        plt.savefig(os.path.join(output_dir, f'comparative_{key}_hist.png'))
        plt.close()
        print(f'Saved comparative plot for {key}')

print('Plotting comparative features')
plot_comparative_features(jc_pooled_hists, tl_pooled_hists)