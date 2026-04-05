#!/usr/bin/env python
"""
Jet Substructure Experiment: RAN vs OmniFold vs IBU
Reproduces the jet substructure comparison from the paper.
"""

from pathlib import Path
import argparse
import pickle

import numpy as np
import torch

from unfold import OBS_NAMES, load_data, compute_metrics
from unfold.ran import run_ran
from unfold.omnifold import run_omnifold_6d
from unfold.ibu import run_ibu
from unfold.plotting import plot_substructure, plot_substructure_full


def save_binned_histograms(ran_results, of_results, norm_params, output_dir,
                           n_plot_bins=50):
    """Save binned histogram data for substructure plots (all runs)."""
    hist_data = {}
    for obs_i, var in enumerate(OBS_NAMES):
        mu, sig = norm_params[var]

        # Truth and generation from first run (same across runs)
        truth_raw = ran_results[0]['data_set_1'][:, obs_i] * sig + mu
        gen_raw = ran_results[0]['data_set_0'][:, obs_i] * sig + mu

        lo = min(truth_raw.min(), gen_raw.min())
        hi = max(truth_raw.max(), gen_raw.max())
        edges = np.linspace(lo, hi, n_plot_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        h_truth, _ = np.histogram(truth_raw, bins=edges, density=True)
        h_gen, _ = np.histogram(gen_raw, bins=edges, density=True)

        # RAN: all per-run histograms, shape (n_runs, n_bins)
        ran_hists = []
        for r in ran_results:
            h, _ = np.histogram(r['data_set_0'][:, obs_i] * sig + mu,
                                bins=edges, weights=r['weights'], density=True)
            ran_hists.append(h)

        hist_data[f'{var}_edges'] = edges
        hist_data[f'{var}_centers'] = centers
        hist_data[f'{var}_truth'] = h_truth
        hist_data[f'{var}_gen'] = h_gen
        hist_data[f'{var}_ran_hists'] = np.array(ran_hists)  # (n_runs, n_bins)

        # OmniFold: all per-run histograms, shape (n_runs, n_bins)
        if of_results:
            of_hists = []
            for r in of_results:
                h, _ = np.histogram(r['data_set_0'][:, obs_i] * sig + mu,
                                    bins=edges, weights=r['weights'], density=True)
                of_hists.append(h)
            hist_data[f'{var}_omnifold_hists'] = np.array(of_hists)

    # Also store normalization params and observable names
    hist_data['obs_names'] = np.array(OBS_NAMES, dtype=object)
    for var in OBS_NAMES:
        mu, sig = norm_params[var]
        hist_data[f'{var}_norm_mu'] = np.array(mu)
        hist_data[f'{var}_norm_sig'] = np.array(sig)

    hist_path = output_dir / 'histograms.npz'
    np.savez_compressed(hist_path, **hist_data)
    print(f"Saved binned histograms to {hist_path}")


def save_full_distributions(ran_results, of_results, norm_params, n_runs,
                            output_dir):
    """Save full unbinned distributions and raw per-event weights."""
    full_data = {}

    # RAN: save per-run unbinned truth-level distributions + raw weights
    for run_idx, r in enumerate(ran_results):
        for obs_i, var in enumerate(OBS_NAMES):
            mu, sig = norm_params[var]
            full_data[f'ran_run{run_idx}_{var}_herwig_gen'] = (
                r['data_set_0'][:, obs_i] * sig + mu)
            full_data[f'ran_run{run_idx}_{var}_pythia_gen'] = (
                r['data_set_1'][:, obs_i] * sig + mu)
        full_data[f'ran_run{run_idx}_weights'] = r['weights']

    # OmniFold: save per-run unbinned truth-level distributions + raw weights
    if of_results:
        for run_idx, r in enumerate(of_results):
            for obs_i, var in enumerate(OBS_NAMES):
                mu, sig = norm_params[var]
                full_data[f'of_run{run_idx}_{var}_herwig_gen'] = (
                    r['data_set_0'][:, obs_i] * sig + mu)
                full_data[f'of_run{run_idx}_{var}_pythia_gen'] = (
                    r['data_set_1'][:, obs_i] * sig + mu)
            full_data[f'of_run{run_idx}_weights'] = r['weights']

    full_data['n_runs'] = np.array(n_runs)
    full_data['obs_names'] = np.array(OBS_NAMES, dtype=object)
    for var in OBS_NAMES:
        mu, sig = norm_params[var]
        full_data[f'{var}_norm_mu'] = np.array(mu)
        full_data[f'{var}_norm_sig'] = np.array(sig)

    full_path = output_dir / 'full_distributions.npz'
    np.savez_compressed(full_path, **full_data)
    print(f"Saved full unbinned distributions to {full_path}")


def print_tables(methods):
    """Print Table I (Wasserstein distance) and Table II (VLC divergence)."""
    header = f"{'Method':<12}" + "".join(f"{v:>16}" for v in OBS_NAMES)

    print("\n--- Table I: Wasserstein Distance ---")
    print(header)
    print("-" * len(header))
    for name, m in methods.items():
        row = f"{name:<12}"
        for i in range(len(OBS_NAMES)):
            mean_val = m['wd_mean'][i]
            std_val = m['wd_std'][i]
            row += f"{mean_val:7.4f}\u00b1{std_val:.4f}  "
        print(row)

    print("\n--- Table II: VLC Divergence ---")
    print(header)
    print("-" * len(header))
    for name, m in methods.items():
        row = f"{name:<12}"
        for i in range(len(OBS_NAMES)):
            mean_val = m['vlc_mean'][i]
            std_val = m['vlc_std'][i]
            row += f"{mean_val:7.6f}\u00b1{std_val:.6f}  "
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Jet Substructure: RAN vs OmniFold vs IBU (Section V)")
    parser.add_argument('--data_file', type=str, default='rawdata_omnifold.npz',
                        help='Path to prepared data file')
    parser.add_argument('--max_events', type=int, default=150000)
    parser.add_argument('--ran_epochs', type=int, default=1000,
                        help='RAN training epochs (default: 1000, matching SixVar.py)')
    parser.add_argument('--of_niter', type=int, default=5,
                        help='OmniFold iterations (default: 5)')
    parser.add_argument('--of_epochs', type=int, default=10)
    parser.add_argument('--ibu_niter', type=int, default=10)
    parser.add_argument('--n_runs', type=int, default=20,
                        help='Independent runs for error bars (default: 20)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./jet_results')
    parser.add_argument('--skip_omnifold', action='store_true')
    parser.add_argument('--skip_ibu', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Jet Substructure: RAN vs OmniFold vs IBU")
    print("=" * 70)
    print(f"Data:       {args.data_file} (max {args.max_events} events)")
    print(f"RAN:        {args.ran_epochs} epochs, {args.n_runs} runs")
    print(f"OmniFold:   niter={args.of_niter}, epochs={args.of_epochs}")
    print(f"IBU:        niter={args.ibu_niter}")
    print(f"Device:     {device}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    arrays, norm_params = load_data(args.data_file, args.max_events)
    N = len(arrays['m_true'])
    print(f"Loaded {N} events per stream, {len(OBS_NAMES)} observables")

    # --- RAN ---
    print("\n" + "=" * 70)
    print("Training RAN...")
    print("=" * 70)
    ran_results = run_ran(
        arrays, device,
        n_epochs=args.ran_epochs,
        n_runs=args.n_runs,
        verbose=args.verbose,
    )

    with open(output_dir / 'ran_results.pkl', 'wb') as f:
        pickle.dump(ran_results, f)

    # --- OmniFold ---
    of_results = None
    if not args.skip_omnifold:
        print("\n" + "=" * 70)
        print("Running OmniFold...")
        print("=" * 70)

        of_results = run_omnifold_6d(
            arrays, device,
            niter=args.of_niter,
            epochs=args.of_epochs,
            n_runs=args.n_runs,
            verbose=args.verbose,
        )

        with open(output_dir / 'of_results.pkl', 'wb') as f:
            pickle.dump(of_results, f)

    # --- IBU ---
    ibu_results = None
    if not args.skip_ibu:
        print("\n" + "=" * 70)
        print("Running IBU...")
        print("=" * 70)
        ibu_results = run_ibu(
            arrays, niter=args.ibu_niter,
            n_runs=args.n_runs, verbose=args.verbose,
        )

        with open(output_dir / 'ibu_results.pkl', 'wb') as f:
            pickle.dump(ibu_results, f)

    # --- Metrics ---
    print("\n" + "=" * 70)
    print("Computing metrics...")
    print("=" * 70)

    methods = {}

    # Generation baseline
    gen_wd_all, gen_vlc_all = [], []
    for r in ran_results:
        wd, vlc = compute_metrics(r['data_set_1'], r['data_set_0'])
        gen_wd_all.append(wd)
        gen_vlc_all.append(vlc)
    methods['Generation'] = {
        'wd_mean': np.mean(gen_wd_all, axis=0),
        'wd_std': np.std(gen_wd_all, axis=0),
        'vlc_mean': np.mean(gen_vlc_all, axis=0),
        'vlc_std': np.std(gen_vlc_all, axis=0),
    }

    # RAN
    ran_wd_all, ran_vlc_all = [], []
    for r in ran_results:
        wd, vlc = compute_metrics(r['data_set_1'], r['data_set_0'],
                                   weights=r['weights'])
        ran_wd_all.append(wd)
        ran_vlc_all.append(vlc)
    methods['RAN'] = {
        'wd_mean': np.mean(ran_wd_all, axis=0),
        'wd_std': np.std(ran_wd_all, axis=0),
        'vlc_mean': np.mean(ran_vlc_all, axis=0),
        'vlc_std': np.std(ran_vlc_all, axis=0),
    }

    # OmniFold
    if of_results:
        of_wd_all, of_vlc_all = [], []
        for r in of_results:
            wd, vlc = compute_metrics(r['data_set_1'], r['data_set_0'],
                                       weights=r['weights'])
            of_wd_all.append(wd)
            of_vlc_all.append(vlc)
        methods['OmniFold'] = {
            'wd_mean': np.mean(of_wd_all, axis=0),
            'wd_std': np.std(of_wd_all, axis=0),
            'vlc_mean': np.mean(of_vlc_all, axis=0),
            'vlc_std': np.std(of_vlc_all, axis=0),
        }

    print_tables(methods)

    # Save metrics
    with open(output_dir / 'metrics.pkl', 'wb') as f:
        pickle.dump(methods, f)

    # --- Save data ---
    print("\nSaving binned histograms...")
    save_binned_histograms(ran_results, of_results, norm_params, output_dir)

    print("Saving full unbinned distributions and weights...")
    save_full_distributions(ran_results, of_results, norm_params,
                            args.n_runs, output_dir)

    # --- Plots ---
    if of_results:
        print("\nCreating substructure plots...")
        plot_substructure(ran_results, of_results, ibu_results, norm_params,
                          output_dir)
        plot_substructure_full(ran_results, of_results, norm_params,
                               output_dir)

    print(f"\nAll results saved to {output_dir}/")
    print("Done.")


if __name__ == '__main__':
    main()
