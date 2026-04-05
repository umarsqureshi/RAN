"""IBU: Iterative Bayesian Unfolding (per-observable, binned).
"""

import numpy as np
from sklearn.model_selection import train_test_split

from .observables import OBS_NAMES


def run_ibu(arrays, n_bins=50, niter=10, n_runs=20, verbose=True):
    """Run IBU independently per observable."""
    n_vars = len(OBS_NAMES)
    N = len(arrays['m_true'])

    gen_truth = np.column_stack([arrays[f'{v}_true_alt'] for v in OBS_NAMES])
    sim_reco = np.column_stack([arrays[f'{v}_reco_alt'] for v in OBS_NAMES])
    pythia_gen = np.column_stack([arrays[f'{v}_true'] for v in OBS_NAMES])
    data_reco = np.column_stack([arrays[f'{v}_reco'] for v in OBS_NAMES])

    all_results = []

    if verbose:
        print(f"  IBU config: n_bins={n_bins}, niter={niter}, n_runs={n_runs}")

    for run_idx in range(n_runs):
        if verbose:
            print(f"\n--- IBU Run {run_idx+1}/{n_runs} ---")

        (gen_tr, gen_te, sim_tr, sim_te,
         data_tr, data_te, pyth_tr, pyth_te) = train_test_split(
            gen_truth, sim_reco, data_reco, pythia_gen,
            test_size=0.25, random_state=42 + run_idx
        )

        if verbose:
            print(f"  Train: {len(gen_tr)} MC, {len(data_tr)} data  |  "
                  f"Test: {len(gen_te)} MC, {len(pyth_te)} truth")

        # IBU per observable
        weights_per_obs = []
        for obs_i in range(n_vars):
            if verbose:
                print(f"  Unfolding {OBS_NAMES[obs_i]}...", end=" ")
            gen_col = gen_tr[:, obs_i]
            sim_col = sim_tr[:, obs_i]
            data_col = data_tr[:, obs_i]

            # Determine bin edges from combined range
            lo = min(gen_col.min(), sim_col.min(), data_col.min())
            hi = max(gen_col.max(), sim_col.max(), data_col.max())
            edges = np.linspace(lo, hi, n_bins + 1)

            # Build response matrix: R[j,i] = P(reco bin j | truth bin i)
            gen_bins = np.digitize(gen_col, edges) - 1
            sim_bins = np.digitize(sim_col, edges) - 1
            gen_bins = np.clip(gen_bins, 0, n_bins - 1)
            sim_bins = np.clip(sim_bins, 0, n_bins - 1)

            R = np.zeros((n_bins, n_bins), dtype=np.float64)
            for t_bin, r_bin in zip(gen_bins, sim_bins):
                R[r_bin, t_bin] += 1

            # Normalize columns (P(reco|truth))
            col_sums = R.sum(axis=0)
            col_sums[col_sums == 0] = 1
            R = R / col_sums[np.newaxis, :]

            # Data histogram
            data_hist = np.histogram(data_col, bins=edges)[0].astype(np.float64)

            # Prior (flat or from gen)
            prior = np.histogram(gen_col, bins=edges)[0].astype(np.float64)
            prior[prior == 0] = 1e-10
            prior = prior / prior.sum()

            # Iterate
            unfolded = prior.copy()
            for _ in range(niter):
                # E-step: expected reco from current unfolded
                expected = R @ unfolded
                expected[expected == 0] = 1e-10

                # M-step: correction factors
                correction = data_hist / expected

                # Update: element-wise via Bayes' theorem
                for i in range(n_bins):
                    if unfolded[i] > 0:
                        factor = np.sum(R[:, i] * correction) / np.sum(R[:, i] + 1e-10)
                        unfolded[i] *= factor

                unfolded[unfolded < 0] = 0

            weights_per_obs.append(unfolded)
            if verbose:
                print(f"done (sum={unfolded.sum():.1f})")

        if verbose:
            print(f"  Run {run_idx+1} complete.")

        all_results.append({
            'unfolded_hists': weights_per_obs,
            'data_set_0': gen_te,
            'data_set_1': pyth_te,
            'edges': edges,  # Last observable edges (for reference)
        })

    return all_results
