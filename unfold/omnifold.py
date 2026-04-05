"""OmniFold: iterative DCTR reweighting with binary classifiers."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance

from .observables import OBS_NAMES


class Classifier(nn.Module):
    """Simple MLP binary classifier for OmniFold."""

    def __init__(self, input_dim: int = 1, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [50, 50, 50]

        layers = []
        in_features = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def weighted_bce(y_pred, y_true, weights):
    """Weighted binary cross-entropy loss matching OmniFold convention."""
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    loss = -weights * (y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
    return loss.mean()


def _dctr_reweight(classifier, events_tensor, device):
    """Density ratio reweighting: w = f(x) / (1 - f(x))"""
    classifier.eval()
    with torch.no_grad():
        f = classifier(events_tensor).cpu().numpy()
    w = f / (1.0 - f + 1e-10)
    return np.nan_to_num(w.squeeze())


def _train_classifier(classifier, train_loader, val_loader,
                      device, epochs=10, patience=5, lr=5e-5):
    """Train a binary classifier with early stopping."""
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    classifier.to(device)

    for epoch in range(epochs):
        # Training
        classifier.train()
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            pred = classifier(xb)
            loss = weighted_bce(pred, yb, wb)
            loss.backward()
            optimizer.step()

        # Validation
        classifier.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                pred = classifier(xb)
                val_loss += weighted_bce(pred, yb, wb).item()
                n_batches += 1
        val_loss /= max(n_batches, 1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best weights
    if best_state is not None:
        classifier.load_state_dict(best_state)
    return classifier


def _make_loaders(x, y, w, device, batch_size=2048, val_frac=0.25):
    """Create train/val DataLoaders with tensors pre-moved to device."""
    x_tr, x_val, y_tr, y_val, w_tr, w_val = train_test_split(
        x, y, w, test_size=val_frac
    )

    def _to_ds(xa, ya, wa):
        return TensorDataset(
            torch.tensor(xa, dtype=torch.float32).to(device),
            torch.tensor(ya, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(wa, dtype=torch.float32).unsqueeze(1).to(device),
        )

    train_loader = DataLoader(_to_ds(x_tr, y_tr, w_tr),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_to_ds(x_val, y_val, w_val),
                            batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_omnifold_6d(arrays, device, niter=5, epochs=10, batch_size=512,
                    lr=5e-5, patience=5, n_runs=20, verbose=True):
    """Run OmniFold on 6D jet data."""
    n_vars = len(OBS_NAMES)
    N = len(arrays['m_true'])

    # Build arrays
    gen_truth = np.column_stack([arrays[f'{v}_true_alt'] for v in OBS_NAMES])
    sim_reco = np.column_stack([arrays[f'{v}_reco_alt'] for v in OBS_NAMES])
    data_reco = np.column_stack([arrays[f'{v}_reco'] for v in OBS_NAMES])
    pythia_gen = np.column_stack([arrays[f'{v}_true'] for v in OBS_NAMES])

    all_results = []

    for run_idx in range(n_runs):
        if verbose:
            print(f"\n--- OmniFold Run {run_idx+1}/{n_runs} ---")

        # Split
        (gen_tr, gen_te, sim_tr, sim_te,
         data_tr, data_te, pyth_tr, pyth_te) = train_test_split(
            gen_truth, sim_reco, data_reco, pythia_gen,
            test_size=0.25, random_state=42 + run_idx
        )

        n_mc = len(gen_tr)
        n_data = len(data_tr)
        weights_push = np.ones(n_mc, dtype=np.float32)

        if verbose:
            print(f"  MC samples: {n_mc}, Data samples: {n_data}")
            print(f"  niter={niter}, epochs={epochs}, batch_size={batch_size}, "
                  f"lr={lr}, patience={patience}")

        for it in range(niter):
            if verbose:
                print(f"\n  --- Iteration {it+1}/{niter} ---")

            # Step 1: detector-level classifier (sim vs data)
            if verbose:
                print(f"    Step 1 (Pull): training reco-level classifier "
                      f"({n_mc} sim + {n_data} data)...")
            x1 = np.concatenate([sim_tr, data_tr], axis=0)
            y1 = np.concatenate([np.zeros(n_mc), np.ones(n_data)])
            w1 = np.concatenate([weights_push, np.ones(n_data)])

            train_ld, val_ld = _make_loaders(
                x1, y1, w1, device, batch_size=batch_size, val_frac=0.25)
            clf1 = Classifier(input_dim=n_vars)
            clf1 = _train_classifier(clf1, train_ld, val_ld, device=device,
                                      epochs=epochs, patience=patience, lr=lr)

            # DCTR reweight on sim_tr
            sim_tr_t = torch.tensor(sim_tr, dtype=torch.float32).to(device)
            weights_pull = weights_push * _dctr_reweight(clf1, sim_tr_t, device)
            del train_ld, val_ld, clf1, sim_tr_t

            if verbose:
                print(f"    Pull weights: mean={weights_pull.mean():.4f} "
                      f"std={weights_pull.std():.4f} "
                      f"min={weights_pull.min():.4f} max={weights_pull.max():.4f}")

            # Step 2: particle-level classifier (gen vs gen reweighted)
            if verbose:
                print(f"    Step 2 (Push): training truth-level classifier "
                      f"({n_mc} gen vs {n_mc} reweighted gen)...")
            x2 = np.concatenate([gen_tr, gen_tr], axis=0)
            y2 = np.concatenate([np.zeros(n_mc), np.ones(n_mc)])
            w2 = np.concatenate([np.ones(n_mc), weights_pull])

            train_ld, val_ld = _make_loaders(
                x2, y2, w2, device, batch_size=batch_size, val_frac=0.25)
            clf2 = Classifier(input_dim=n_vars)
            clf2 = _train_classifier(clf2, train_ld, val_ld, device=device,
                                      epochs=epochs, patience=patience, lr=lr)

            # DCTR reweight on gen_tr
            gen_tr_t = torch.tensor(gen_tr, dtype=torch.float32).to(device)
            weights_push = _dctr_reweight(clf2, gen_tr_t, device)

            if verbose:
                # Compute WD per observable to track convergence
                wd_per_obs = [
                    wasserstein_distance(pyth_tr[:, i], gen_tr[:, i],
                                         v_weights=weights_push)
                    for i in range(n_vars)
                ]
                wd_str = " ".join(f"{OBS_NAMES[i]}={wd_per_obs[i]:.4f}"
                                  for i in range(n_vars))
                print(f"    Push weights: mean={weights_push.mean():.4f} "
                      f"std={weights_push.std():.4f} "
                      f"min={weights_push.min():.4f} max={weights_push.max():.4f}")
                print(f"    WD per obs: {wd_str}")
                print(f"    Mean WD: {np.mean(wd_per_obs):.6f}")
            del train_ld, val_ld, gen_tr_t

            # Keep clf2 for reweighting test set on last iteration
            if it < niter - 1:
                del clf2

        # Transfer weights to test set via reweight classifier
        if verbose:
            print(f"\n  Transferring weights to test set ({len(gen_te)} samples)...")
        n_te = len(gen_te)
        x_rw = np.concatenate([gen_tr, gen_tr], axis=0)
        y_rw = np.concatenate([np.zeros(n_mc), np.ones(n_mc)])
        w_rw = np.concatenate([np.ones(n_mc), weights_push])

        train_ld, val_ld = _make_loaders(
            x_rw, y_rw, w_rw, device, batch_size=batch_size, val_frac=0.25)
        clf_test = Classifier(input_dim=n_vars)
        clf_test = _train_classifier(clf_test, train_ld, val_ld, device=device,
                                      epochs=epochs, patience=patience, lr=lr)
        gen_te_t = torch.tensor(gen_te, dtype=torch.float32).to(device)
        of_weights = _dctr_reweight(clf_test, gen_te_t, device)
        del train_ld, val_ld, clf_test, gen_te_t

        if verbose:
            wd_test = [
                wasserstein_distance(pyth_te[:, i], gen_te[:, i],
                                     v_weights=of_weights)
                for i in range(n_vars)
            ]
            wd_str = " ".join(f"{OBS_NAMES[i]}={wd_test[i]:.4f}"
                              for i in range(n_vars))
            print(f"  Test weights: mean={of_weights.mean():.4f} "
                  f"std={of_weights.std():.4f}")
            print(f"  Test WD per obs: {wd_str}")
            print(f"  Test mean WD: {np.mean(wd_test):.6f}")

        all_results.append({
            'weights': of_weights,
            'data_set_0': gen_te,   # Herwig gen test (to reweight)
            'data_set_1': pyth_te,  # Pythia gen test (truth target)
        })

    return all_results
