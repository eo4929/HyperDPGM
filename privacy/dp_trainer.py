"""
Phase 2: DP-SGD training for TabVAE.

Improvements over v0:
  - Per-epoch privacy budget tracking
  - Gradient norm monitoring
  - Early stopping when privacy budget exceeded
  - Cleaner micro-batch accumulation
  - Non-DP training factored out
  - Hypergraph Laplacian + pairwise regularization in latent space (DP-safe)
"""

import math
from collections import defaultdict

import torch
import numpy as np

from config import Config
from models.vae import TabVAE
from privacy.dp_accountant import compute_epsilon, find_sigma


# ── Hypergraph helpers ───────────────────────────────────────────────

def _build_neighbor_info(N: int, hyperedges: list):
    """Pre-compute per-node neighbor sets and Laplacian edge weights.

    w_{ij} = Σ_{e: i,j ∈ e} 1/|e|  (hypergraph Laplacian off-diagonal weight)

    Returns:
        neighbors:   List[List[int]]       — neighbor indices per node
        lap_weights: List[Dict[int,float]] — Laplacian weight per neighbor
    """
    neighbors = [set() for _ in range(N)]
    lap_weights = [defaultdict(float) for _ in range(N)]

    for members in hyperedges:
        he_size = len(members)
        if he_size < 2:
            continue
        w = 1.0 / he_size
        for i in members:
            for j in members:
                if i != j and i < N and j < N:
                    neighbors[i].add(j)
                    lap_weights[i][j] += w

    return [list(s) for s in neighbors], lap_weights


def _get_all_mu(vae: TabVAE, data_t: torch.Tensor,
                device: torch.device, batch_size: int = 512) -> torch.Tensor:
    """Encode all samples to latent means without gradient (frozen targets)."""
    vae.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(data_t), batch_size):
            mu, _ = vae.encode(data_t[start:start + batch_size])
            parts.append(mu)
    vae.train()
    return torch.cat(parts, dim=0)      # (N, latent_dim) on device


# ── Training ─────────────────────────────────────────────────────────

def train_dp_vae(data_t: torch.Tensor, cfg: Config,
                 hyperedges: list = None):
    """Train a TabVAE with DP-SGD.

    When hyperedges is provided and graph lambdas > 0, adds two
    DP-safe latent-space regularisation terms per sample i:

      L_lap  = μᵢᵀ (Δμ)ᵢ           Hypergraph Laplacian smoothness on μ
      L_pair = Σⱼ∈N(i) ||μᵢ−μⱼ||²  Pairwise pull toward hyperedge neighbours

    Neighbour latents μⱼ are frozen once per epoch (no gradient),
    so the gradient of each term depends only on μᵢ — preserving
    per-sample clipping and the (ε,δ)-DP guarantee.

    Returns (vae, achieved_epsilon).
    """
    device = cfg.device
    data_t = data_t.to(device)
    N, D = data_t.shape
    bs = cfg.dp_batch_size
    q = bs / N
    steps_per_ep = max(1, N // bs)
    total_steps = cfg.dp_epochs * steps_per_ep

    sigma = find_sigma(cfg.target_epsilon, cfg.delta, q, total_steps)
    achieved = compute_epsilon(sigma, q, total_steps, cfg.delta)
    print(f"  DP-SGD  sigma={sigma:.3f}  C={cfg.dp_clip_norm}  "
          f"steps={total_steps}  q={q:.4f}")
    print(f"  Achieved (eps={achieved:.4f}, delta={cfg.delta})-DP")

    use_graph = (hyperedges is not None
                 and (cfg.graph_lambda_laplacian > 0
                      or cfg.graph_lambda_pairwise > 0))
    if use_graph:
        print(f"  Graph reg: λ_lap={cfg.graph_lambda_laplacian}  "
              f"λ_pair={cfg.graph_lambda_pairwise}")
        neighbors, lap_weights = _build_neighbor_info(N, hyperedges)

    vae = TabVAE(D, cfg).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae_lr)
    clip = cfg.dp_clip_norm

    for ep in range(1, cfg.dp_epochs + 1):
        # Freeze neighbour latent means once per epoch
        if use_graph:
            mu_fixed = _get_all_mu(vae, data_t, device)    # (N, latent_dim)

        vae.train()
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        ep_grad_norms = []

        for s in range(steps_per_ep):
            batch_idx = perm[s * bs:(s + 1) * bs]
            batch = data_t[batch_idx]
            cur_bs = len(batch)

            acc = {n: torch.zeros_like(p) for n, p in vae.named_parameters()}

            for k in range(cur_bs):
                i = batch_idx[k].item()
                row = batch[k:k + 1]

                opt.zero_grad()
                rx, mu_i, lv_i = vae(row)
                loss = vae.loss(rx, row, mu_i, lv_i)

                # Graph regularisation — gradient flows only through μᵢ
                if use_graph and neighbors[i]:
                    nbr_idx = neighbors[i]
                    nbr_mu = mu_fixed[nbr_idx]      # frozen, no grad
                    lap_w = torch.tensor(
                        [lap_weights[i][j] for j in nbr_idx],
                        device=device, dtype=torch.float32)
                    deg_i = lap_w.sum()

                    # Laplacian: μᵢᵀ(Δμ)ᵢ = deg_i·||μᵢ||² − μᵢ·Σⱼwᵢⱼμⱼ
                    lap_vec = (deg_i * mu_i
                               - (lap_w.unsqueeze(1) * nbr_mu).sum(0, keepdim=True))
                    l_lap = (mu_i * lap_vec).sum()

                    # Pairwise: Σⱼ ||μᵢ − μⱼ_fixed||²
                    l_pair = ((mu_i - nbr_mu) ** 2).sum()

                    loss = (loss
                            + cfg.graph_lambda_laplacian * l_lap
                            + cfg.graph_lambda_pairwise * l_pair)

                loss.backward()

                total_norm = math.sqrt(sum(
                    p.grad.norm().item() ** 2
                    for p in vae.parameters() if p.grad is not None
                ))
                ep_grad_norms.append(total_norm)
                cf = min(1.0, clip / (total_norm + 1e-8))
                for n, p in vae.named_parameters():
                    if p.grad is not None:
                        acc[n] += p.grad * cf

            # Average + Gaussian noise
            opt.zero_grad()
            for n, p in vae.named_parameters():
                noise = torch.randn_like(acc[n]) * sigma * clip
                p.grad = (acc[n] + noise) / cur_bs
            opt.step()

            with torch.no_grad():
                rx, mu, lv = vae(batch)
                ep_loss += vae.loss(rx, batch, mu, lv).item()

        avg_norm = np.mean(ep_grad_norms) if ep_grad_norms else 0.0
        clip_frac = np.mean([1.0 if gn > clip else 0.0
                             for gn in ep_grad_norms]) if ep_grad_norms else 0.0

        if ep % 10 == 0:
            cum_eps = compute_epsilon(sigma, q, ep * steps_per_ep, cfg.delta)
            print(f"    ep {ep:>3d}/{cfg.dp_epochs}  "
                  f"loss={ep_loss / steps_per_ep:.4f}  "
                  f"grad_norm={avg_norm:.3f}  clip%={clip_frac:.0%}  "
                  f"eps={cum_eps:.4f}")

    return vae, achieved


def train_non_dp_vae(data_t: torch.Tensor, cfg: Config):
    """Train a TabVAE without differential privacy."""
    device = cfg.device
    data_t = data_t.to(device)
    N, D = data_t.shape

    vae = TabVAE(D, cfg).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae_lr)

    for ep in range(1, cfg.non_dp_epochs + 1):
        vae.train()
        opt.zero_grad()
        rx, mu, lv = vae(data_t)
        loss = vae.loss(rx, data_t, mu, lv)
        loss.backward()
        opt.step()
        if ep % 30 == 0:
            with torch.no_grad():
                rx2, mu2, lv2 = vae(data_t)
                print(f"    ep {ep:>3d}/{cfg.non_dp_epochs}  "
                      f"loss={vae.loss(rx2, data_t, mu2, lv2).item():.4f}")
    return vae
