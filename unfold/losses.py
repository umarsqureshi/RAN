"""WGAN loss functions for RAN training."""

import torch
import torch.nn as nn


class WGANCriticLoss(nn.Module):
    """WGAN critic loss with gradient penalty.

    L_D = E_w[D(alt_reco)] - E[D(truth_reco)] + lambda * GP

    where E_w[.] = sum(w*.) / sum(w) is the expectation under the reweighted
    distribution.  Normalizing by sum(w) ensures both terms have equal scale
    regardless of the current mean weight.
    """
    def __init__(self, gp_lambda=10.0):
        super().__init__()
        self.gp_lambda = gp_lambda

    def forward(self, critic, x_reco, target, weights, device):
        scores = critic(x_reco)  # (N, 1), linear output

        truth_mask = (target == 1).squeeze()
        alt_mask = (target == 0).squeeze()

        scores_truth = scores[truth_mask]
        scores_alt = scores[alt_mask]
        w_alt = weights[alt_mask]

        # Proper expectation under reweighted distribution: sum(w*D)/sum(w)
        # This keeps the two terms at equal scale even when mean(w) != 1
        w_sum = w_alt.sum() + 1e-8
        loss_w = torch.sum(w_alt * scores_alt) / w_sum - torch.mean(scores_truth)

        # Gradient penalty on interpolated samples
        gp = self._gradient_penalty(critic, x_reco[truth_mask], x_reco[alt_mask], device)

        return loss_w + self.gp_lambda * gp

    def _gradient_penalty(self, critic, x_real, x_fake, device):
        n = min(len(x_real), len(x_fake))
        x_real = x_real[:n]
        x_fake = x_fake[:n]
        alpha = torch.rand(n, 1, device=device)
        x_interp = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)
        d_interp = critic(x_interp)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=x_interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True
        )[0]
        grad_norm = grads.norm(2, dim=1)
        return torch.mean((grad_norm - 1.0) ** 2)


class WGANGenLoss(nn.Module):
    """WGAN generator loss with detached-mean normalization + regularization.

    L_G = -sum(w * D) / sum(w).detach() + lambda_m*(mean(w)-1)^2 + lambda_v*var(w)

    Uses DETACHED mean normalization: gradient flows through numerator only,
    preventing the positive feedback loop that causes weight explosion when
    gradient flows through both numerator and denominator.

    Regularization keeps weights well-behaved:
      - lambda_mean: penalizes deviation from mean=1
      - lambda_var: penalizes weight variance (prevents a few events
        from dominating with extreme weights)
    """
    def __init__(self, lambda_mean=1.0, lambda_var=0.1):
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_var = lambda_var

    def forward(self, scores_alt, weights_alt):
        # Detached-mean normalization: gradient only through numerator
        w_mean = weights_alt.mean().detach() + 1e-8
        w_norm = weights_alt / w_mean
        wgan_loss = -torch.mean(w_norm * scores_alt)

        # Regularization
        mean_reg = (weights_alt.mean() - 1.0) ** 2
        var_reg = weights_alt.var()

        return wgan_loss + self.lambda_mean * mean_reg + self.lambda_var * var_reg
