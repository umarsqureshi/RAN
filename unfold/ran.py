"""RAN: Reweighting Adversarial Network training loop."""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split

from .observables import OBS_NAMES
from .models import RANGenerator, RANCritic
from .losses import WGANCriticLoss, WGANGenLoss


def run_ran(arrays, device, n_epochs=100, n_runs=20, val_frac=0.25,
            gen_width=100, gen_depth=3, disc_width=50, disc_depth=3,
            lr_gen=2e-4, lr_disc=2e-4,
            m_gen=2, n_disc=3, verbose=True,
            gp_lambda=10.0, lambda_mean=1.0, lambda_var=0.1,
            batch_size=8192, dropout=0.0):
    """Train RAN on 6D jet substructure data using WGAN-GP.
      - Generator: 1+log(softplus) output, LeakyReLU, BN, Dropout(0.2)
      - Critic: Dense (no spectral norm), LeakyReLU, LayerNorm, linear output
      - RMSProp optimizer, batch_size=256
      - 3 critic : 2 generator update ratio
      - Gradient penalty lambda=10
      - Model selection: detector-level Wasserstein distance between
        reweighted Herwig reco and Pythia reco on the held-out split (no
        truth-level information is used to pick the checkpoint)
    """
    n_vars = len(OBS_NAMES)
    N = len(arrays['m_true'])

    # Build combined arrays: alt (y=0) + true (y=1)
    xvals_truth = np.column_stack(
        [np.concatenate([arrays[f'{v}_true_alt'], arrays[f'{v}_true']]) for v in OBS_NAMES]
    )
    xvals_reco = np.column_stack(
        [np.concatenate([arrays[f'{v}_reco_alt'], arrays[f'{v}_reco']]) for v in OBS_NAMES]
    )
    yvals = np.concatenate([np.zeros(N, dtype=np.float32),
                            np.ones(N, dtype=np.float32)])

    all_weights = []

    for run_idx in range(n_runs):
        if verbose:
            print(f"\n--- RAN Run {run_idx+1}/{n_runs} ---")

        # Train/test split (75/25, matching sklearn default)
        (X_train_truth, X_test_truth,
         X_train_reco, X_test_reco,
         Y_train, Y_test) = train_test_split(
            xvals_truth, xvals_reco, yvals,
            test_size=0.25, random_state=42 + run_idx
        )

        # Convert to tensors
        truth_t = torch.tensor(X_train_truth, dtype=torch.float32)
        reco_t = torch.tensor(X_train_reco, dtype=torch.float32)
        y_t = torch.tensor(Y_train.reshape(-1, 1), dtype=torch.float32)

        dataset = TensorDataset(reco_t, truth_t, y_t)
        # Paper uses shuffle for mini-batch SGD
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Build models matching the paper architecture
        generator = RANGenerator(n_vars, gen_width, gen_depth, dropout).to(device)
        critic = RANCritic(n_vars, disc_width, disc_depth, dropout).to(device)

        # Paper Sec III B c: "Pretraining the Generator to the Identity"
        # Pretrain generator so g(z) ~ 1 for all inputs before adversarial training.
        # This ensures the initial reweighting is close to no reweighting.
        pretrain_opt = optim.Adam(generator.parameters(), lr=1e-3)
        pretrain_epochs = 50
        generator.train()
        alt_truth_train = torch.tensor(
            X_train_truth[Y_train == 0], dtype=torch.float32).to(device)
        for _ in range(pretrain_epochs):
            idx = torch.randperm(len(alt_truth_train))[:min(4096, len(alt_truth_train))]
            batch = alt_truth_train[idx]
            pretrain_opt.zero_grad()
            w_out = generator(batch)
            pretrain_loss = ((w_out - 1.0) ** 2).mean()
            pretrain_loss.backward()
            pretrain_opt.step()
        if verbose:
            generator.eval()
            with torch.no_grad():
                w_check = generator(alt_truth_train[:4096]).cpu()
            print(f"  Pretrained generator: mean={w_check.mean():.4f}, "
                  f"std={w_check.std():.4f}")

        # Paper: "We use the RMSProp optimizer, as it has been found to
        # outperform Adam for WGANs"
        opt_disc = optim.RMSprop(critic.parameters(), lr=lr_disc)
        opt_gen = optim.RMSprop(generator.parameters(), lr=lr_gen)

        loss_fn_critic = WGANCriticLoss(gp_lambda=gp_lambda)
        loss_fn_gen = WGANGenLoss(lambda_mean=lambda_mean, lambda_var=lambda_var)

        # Held-out sets. Truth-level arrays are used only for reporting;
        # reco-level arrays drive model selection (observable in a real
        # measurement: Pythia reco plays the role of "data").
        data_set_1 = X_test_truth[Y_test == 1]  # Pythia gen (test)
        data_set_0 = X_test_truth[Y_test == 0]  # Herwig gen (test)
        reco_set_1 = X_test_reco[Y_test == 1]   # Pythia reco (test) -- "data"
        reco_set_0 = X_test_reco[Y_test == 0]   # Herwig reco (test)

        best_metric = float('inf')
        best_gen_state = None
        best_epoch = -1

        n_batches = len(loader)
        print(f"  Training: {n_epochs} epochs, {n_batches} batches/epoch, "
              f"batch_size={batch_size}")
        print(f"  Generator: {gen_depth}x Dense({gen_width}) [LeakyReLU->BN->Dropout({dropout})], "
              f"output=1+log(sp+1/e), lr={lr_gen}")
        print(f"  Critic:    {disc_depth}x Dense({disc_width}) [no SpectralNorm, LeakyReLU->Dropout->LayerNorm], "
              f"lr={lr_disc}, GP_lambda={gp_lambda}")
        print(f"  Optimizer: RMSProp, updates per batch: {n_disc} disc, {m_gen} gen, "
              f"\u03bb_mean={lambda_mean}, \u03bb_var={lambda_var}")
        print(f"  Train samples: {len(X_train_truth)} "
              f"(alt={int((Y_train==0).sum())}, true={int((Y_train==1).sum())})")
        print(f"  Test samples:  {len(X_test_truth)} "
              f"(alt={len(data_set_0)}, true={len(data_set_1)})")

        epoch_pbar = tqdm(range(n_epochs), desc=f"  Run {run_idx+1} epochs",
                          leave=False)
        for epoch in epoch_pbar:
            critic.train()
            d_losses, g_losses = [], []

            for X_reco_batch, X_truth_batch, Y_batch in loader:
                X_reco_batch = X_reco_batch.to(device)
                X_truth_batch = X_truth_batch.to(device)
                Y_batch = Y_batch.to(device)

                alt_mask = (Y_batch.squeeze() == 0)
                if alt_mask.sum() == 0:
                    continue

                # --- Critic update (n_disc times) ---
                for _ in range(n_disc):
                    generator.eval()
                    with torch.no_grad():
                        W_batch = generator(X_truth_batch)
                    # Batch-level normalization (paper: "normalized at the batch level")
                    alt_w = W_batch[alt_mask]
                    if alt_w.numel() > 0:
                        W_batch = W_batch / (alt_w.mean() + 1e-8)
                    W_batch = torch.where(Y_batch == 1, torch.ones_like(W_batch), W_batch)

                    critic.train()
                    opt_disc.zero_grad()
                    d_loss = loss_fn_critic(critic, X_reco_batch, Y_batch, W_batch, device)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    opt_disc.step()
                    d_losses.append(d_loss.item())

                # --- Generator update (m_gen times) ---
                for _ in range(m_gen):
                    X_truth_alt = X_truth_batch[alt_mask]
                    X_reco_alt = X_reco_batch[alt_mask]

                    generator.train()
                    opt_gen.zero_grad()
                    W_alt = generator(X_truth_alt)

                    # Critic scores (frozen)
                    critic.eval()
                    for p in critic.parameters():
                        p.requires_grad_(False)
                    scores_alt = critic(X_reco_alt)
                    for p in critic.parameters():
                        p.requires_grad_(True)

                    # Loss uses detached-mean normalization internally
                    g_loss = loss_fn_gen(scores_alt, W_alt)
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                    opt_gen.step()
                    g_losses.append(g_loss.item())

            # Update progress bar with current losses
            avg_d = np.mean(d_losses[-n_batches*n_disc:]) if d_losses else 0
            avg_g = np.mean(g_losses[-n_batches*m_gen:]) if g_losses else 0
            epoch_pbar.set_postfix(D=f"{avg_d:.4f}", G=f"{avg_g:.4f}")

            # Per-epoch metrics -- evaluate every 10 epochs
            eval_interval = max(1, min(10, n_epochs // 20))
            if (epoch + 1) % eval_interval == 0:
                generator.eval()
                with torch.no_grad():
                    alt_truth_test = torch.tensor(
                        data_set_0, dtype=torch.float32).to(device)
                    w_test_raw = generator(alt_truth_test).cpu().numpy().flatten()
                    w_test = w_test_raw / (w_test_raw.mean() + 1e-8)

                # Selection metric: detector-level WD (reweighted Herwig reco
                # vs Pythia reco). Weights are per-event, so the same w_test
                # applies to the reco row of each Herwig event.
                reco_wd_per_obs = [
                    wasserstein_distance(reco_set_1[:, i], reco_set_0[:, i],
                                         v_weights=w_test)
                    for i in range(n_vars)
                ]
                mean_reco_wd = np.mean(reco_wd_per_obs)

                # Diagnostic only: truth-level WD (not used for selection)
                wd_per_obs = [
                    wasserstein_distance(data_set_1[:, i], data_set_0[:, i],
                                         v_weights=w_test)
                    for i in range(n_vars)
                ]
                mean_wd = np.mean(wd_per_obs)

                if mean_reco_wd < best_metric:
                    best_metric = mean_reco_wd
                    best_epoch = epoch + 1
                    best_gen_state = {
                        k: v.clone() for k, v in generator.state_dict().items()
                    }

                if verbose and (epoch + 1) % max(1, min(50, n_epochs // 10)) == 0:
                    wd_str = " ".join(f"{OBS_NAMES[i]}={wd_per_obs[i]:.4f}"
                                      for i in range(n_vars))
                    reco_wd_str = " ".join(
                        f"{OBS_NAMES[i]}={reco_wd_per_obs[i]:.4f}"
                        for i in range(n_vars))
                    print(f"\n  Epoch {epoch+1:4d}/{n_epochs}: "
                          f"D={avg_d:.4f}  G={avg_g:.4f}  "
                          f"reco_WD={mean_reco_wd:.6f} (selection)  "
                          f"truth_WD={mean_wd:.6f}"
                          f"{'  *best*' if best_epoch == epoch + 1 else ''}")
                    print(f"    Reco WD per obs:  {reco_wd_str}")
                    print(f"    Truth WD per obs: {wd_str}")
                    print(f"    Gen weights: "
                          f"mean={w_test.mean():.4f} std={w_test.std():.4f} "
                          f"min={w_test.min():.4f} max={w_test.max():.4f}")

        epoch_pbar.close()

        # Restore best
        if best_gen_state is not None:
            generator.load_state_dict(best_gen_state)
            if verbose:
                print(f"\n  Restored best epoch {best_epoch} "
                      f"(reco mean_WD={best_metric:.6f})")

        # Extract final weights for Herwig gen test set (normalized to mean=1)
        generator.eval()
        with torch.no_grad():
            alt_truth_test = torch.tensor(
                data_set_0, dtype=torch.float32).to(device)
            weights_raw = generator(alt_truth_test).cpu().numpy().flatten()
            weights = weights_raw / (weights_raw.mean() + 1e-8)

        if verbose:
            wd_final = [
                wasserstein_distance(data_set_1[:, i], data_set_0[:, i],
                                     v_weights=weights)
                for i in range(n_vars)
            ]
            wd_str = " ".join(f"{OBS_NAMES[i]}={wd_final[i]:.4f}"
                              for i in range(n_vars))
            print(f"  Final test weights: mean={weights.mean():.4f} "
                  f"std={weights.std():.4f} "
                  f"min={weights.min():.4f} max={weights.max():.4f}")
            print(f"  Final test WD: {wd_str}")
            print(f"  Final mean WD: {np.mean(wd_final):.6f}")

        all_weights.append({
            'weights': weights,
            'data_set_0': data_set_0,  # Herwig gen (to be reweighted)
            'data_set_1': data_set_1,  # Pythia gen (truth target)
            'Y_test': Y_test,
            'X_test_truth': X_test_truth,
        })

    return all_weights
