"""Plotting functions for jet substructure distributions."""

import numpy as np
import matplotlib.pyplot as plt

from .observables import OBS_NAMES, OBS_LABELS, OBS_TITLES


def plot_substructure(ran_results, of_results, ibu_results, norm_params,
                      output_dir, n_bins=50):
    """Plot 6 substructure distributions with error bands."""
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 12,
        'axes.grid': True, 'axes.grid.which': 'both',
    })

    fig, axes = plt.subplots(6, 1, figsize=(8, 30))

    for obs_i, var in enumerate(OBS_NAMES):
        ax_main = axes[obs_i]

        ran_r = ran_results[0]
        of_r = of_results[0]

        truth = ran_r['data_set_1'][:, obs_i]
        gen = ran_r['data_set_0'][:, obs_i]

        mu, sig = norm_params[var]

        truth_raw = truth * sig + mu
        gen_raw = gen * sig + mu

        lo = min(truth_raw.min(), gen_raw.min())
        hi = max(truth_raw.max(), gen_raw.max())
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        h_truth, _ = np.histogram(truth_raw, bins=edges, density=True)

        h_gen, _ = np.histogram(gen_raw, bins=edges, density=True)

        ran_hists = []
        for r in ran_results:
            h, _ = np.histogram(r['data_set_0'][:, obs_i] * sig + mu,
                                bins=edges, weights=r['weights'], density=True)
            ran_hists.append(h)
        h_ran = np.mean(ran_hists, axis=0)
        h_ran_err = np.std(ran_hists, axis=0)

        of_hists = []
        for r in of_results:
            h, _ = np.histogram(r['data_set_0'][:, obs_i] * sig + mu,
                                bins=edges, weights=r['weights'], density=True)
            of_hists.append(h)
        h_of = np.mean(of_hists, axis=0)
        h_of_err = np.std(of_hists, axis=0)

        ax_main.step(centers, h_truth, where='mid', color='black',
                     linewidth=1.5, label='Truth')
        ax_main.step(centers, h_gen, where='mid', color='orange',
                     linewidth=1, linestyle='--', label='Generation')
        ax_main.errorbar(centers, h_of, yerr=h_of_err, fmt='none',
                         color='red', capsize=2, label='OmniFold')
        ax_main.step(centers, h_of, where='mid', color='red',
                     linewidth=1, alpha=0.7)
        ax_main.errorbar(centers, h_ran, yerr=h_ran_err, fmt='none',
                         color='blue', capsize=2, label='RAN')
        ax_main.step(centers, h_ran, where='mid', color='blue',
                     linewidth=1, alpha=0.7)

        ax_main.set_ylabel('Density')
        ax_main.set_title(f"{OBS_TITLES[var]}: Particle level")
        ax_main.legend(fontsize=9)
        ax_main.set_xlabel(OBS_LABELS[var])

    fig.tight_layout()
    fig.savefig(output_dir / 'substructure.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'substructure.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved substructure plots to {output_dir}/substructure.pdf")


def plot_substructure_full(ran_results, of_results, norm_params,
                           output_dir, n_bins=50):
    """Plot 6 substructure distributions (extended layout)."""
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
    })

    fig, all_axes = plt.subplots(6, 2, figsize=(14, 24),
                                  gridspec_kw={'width_ratios': [1, 1],
                                               'hspace': 0.4, 'wspace': 0.3})

    for obs_i, var in enumerate(OBS_NAMES):
        col = obs_i % 2
        row = obs_i // 2

        ax = all_axes[obs_i // 2 * 2 if obs_i < 4 else obs_i - 2, obs_i % 2 if obs_i < 4 else obs_i - 4]

    # Simpler layout: 3 rows x 2 columns
    fig, axes = plt.subplots(6, 1, figsize=(10, 36))

    for obs_i, var in enumerate(OBS_NAMES):
        ax = axes[obs_i]

        ran_r = ran_results[0]
        truth = ran_r['data_set_1'][:, obs_i]
        gen = ran_r['data_set_0'][:, obs_i]

        mu, sig = norm_params[var]
        truth_raw = truth * sig + mu
        gen_raw = gen * sig + mu

        lo = min(truth_raw.min(), gen_raw.min())
        hi = max(truth_raw.max(), gen_raw.max())
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        h_truth, _ = np.histogram(truth_raw, bins=edges, density=True)
        h_gen, _ = np.histogram(gen_raw, bins=edges, density=True)

        ran_hists = []
        for r in ran_results:
            h, _ = np.histogram(r['data_set_0'][:, obs_i] * sig + mu,
                                bins=edges, weights=r['weights'], density=True)
            ran_hists.append(h)
        h_ran = np.mean(ran_hists, axis=0)
        h_ran_err = np.std(ran_hists, axis=0)

        of_hists = []
        for r in of_results:
            h, _ = np.histogram(r['data_set_0'][:, obs_i] * sig + mu,
                                bins=edges, weights=r['weights'], density=True)
            of_hists.append(h)
        h_of = np.mean(of_hists, axis=0)
        h_of_err = np.std(of_hists, axis=0)

        ax.step(centers, h_truth, where='mid', color='black', lw=1.5, label='Truth')
        ax.step(centers, h_gen, where='mid', color='orange', lw=1, ls='--', label='Generation')
        ax.errorbar(centers, h_of, yerr=h_of_err, fmt='none', color='red', capsize=2)
        ax.step(centers, h_of, where='mid', color='red', lw=1, alpha=0.7, label='OmniFold')
        ax.errorbar(centers, h_ran, yerr=h_ran_err, fmt='none', color='blue', capsize=2)
        ax.step(centers, h_ran, where='mid', color='blue', lw=1, alpha=0.7, label='RAN')

        ax.set_ylabel('Density')
        ax.set_xlabel(OBS_LABELS[var])
        ax.set_title(OBS_TITLES[var])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'substructure_full.pdf', bbox_inches='tight')
    fig.savefig(output_dir / 'substructure_full.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
