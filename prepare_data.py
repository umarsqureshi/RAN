#!/usr/bin/env python
"""Prepare jet substructure data."""

import argparse
import numpy as np


def get_var(dset, var, ptype):
    """Extract observable from EnergyFlow dataset.

    Args:
        dset: EnergyFlow dataset dict
        var: Observable name (m, M, w, tau21, zg, sdm)
        ptype: Particle type prefix ('gen' or 'sim')
    """
    if var == 'm':
        return dset[ptype + '_jets'][:, 3]
    elif var == 'M':
        return dset[ptype + '_mults']
    elif var == 'w':
        return dset[ptype + '_widths']
    elif var == 'tau21':
        return dset[ptype + '_tau2s'] / (dset[ptype + '_widths'] + 1e-50)
    elif var == 'zg':
        return dset[ptype + '_zgs']
    elif var == 'sdm':
        jet_pt_sq = dset[ptype + '_jets'][:, 0] ** 2
        eps = 1e-12 * np.mean(jet_pt_sq)
        return np.log(dset[ptype + '_sdms'] ** 2 / np.maximum(jet_pt_sq, eps) + eps)
    else:
        raise ValueError(f"Unknown variable '{var}'")


def main():
    parser = argparse.ArgumentParser(description="Prepare jet substructure data")
    parser.add_argument('--output', type=str, default='rawdata_omnifold.npz')
    parser.add_argument('--max_events', type=int, default=150000,
                        help='Max events per dataset (default: 150000)')
    args = parser.parse_args()

    import energyflow as ef

    print("Loading Pythia 8.243 (simulation) dataset...")
    pythia = ef.zjets_delphes.load(
        'Pythia26', num_data=args.max_events,
        which='all', exclude_keys=['particles']
    )

    print("Loading Herwig 7.1.5 (truth) dataset...")
    herwig = ef.zjets_delphes.load(
        'Herwig', num_data=args.max_events,
        which='all', exclude_keys=['particles']
    )

    obs_names = ['m', 'M', 'w', 'tau21', 'zg', 'sdm']
    out_data = {}

    for var in obs_names:
        out_data[var + '_true'] = get_var(pythia, var, 'gen')
        out_data[var + '_reco'] = get_var(pythia, var, 'sim')
        out_data[var + '_true_alt'] = get_var(herwig, var, 'gen')
        out_data[var + '_reco_alt'] = get_var(herwig, var, 'sim')
        print(f"  {var}: true={out_data[var+'_true'].shape[0]}, "
              f"reco={out_data[var+'_reco'].shape[0]}, "
              f"true_alt={out_data[var+'_true_alt'].shape[0]}, "
              f"reco_alt={out_data[var+'_reco_alt'].shape[0]}")

    np.savez_compressed(args.output, **out_data)
    print(f"\nSaved to {args.output}")
    print(f"Variables: {obs_names}")
    print(f"Streams: _true (Pythia gen), _reco (Pythia sim), "
          f"_true_alt (Herwig gen), _reco_alt (Herwig sim)")


if __name__ == '__main__':
    main()
