"""
experiment.py - RSSM geometric fidelity sweep

Trains a small RSSM on Pendulum-v1 and tracks how well the learned latent
space preserves the true state space topology (S¹ × ℝ) across training.

Sweep: latent dims × training epochs × policies
Checkpoints TDA metrics every TOPO_CHECKPOINT epochs.

Outputs:
  wasserstein_over_epochs.png   - main result
  loss_vs_wasserstein.png       - do they actually correlate?
  fidelity_heatmap.png          - final W₁ per (latent_dim, policy)
  true_vs_latent_diagrams.png   - persistence diagrams side by side
  geometry_comparison.png       - PCA of true vs latent (visual sanity check)
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collect import (
    POLICIES,
    ReplayBuffer,
    fill_buffer,
    get_latents,
    sample_true_state_space,
)
from persim import plot_diagrams
from sklearn.decomposition import PCA
from tda_utils import (
    compute_dgm,
    geometric_fidelity,
    h0_total_persistence,
    loop_strength,
    normalize,
    topological_fingerprint,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── config ────────────────────────────────────────────────────────────────────
# Set conservatively so it runs on CPU in ~20-30 min.
# Bump TRAIN_EPOCHS + N_TRUE_SAMPLES if you have more headroom.

OBS_DIM = 3
ACT_DIM = 1
HIDDEN_DIM = 64
LATENT_DIMS = [8, 16, 32]
BATCH_SIZE = 256
TRAIN_EPOCHS = 60
TOPO_CHECKPOINT = 10
N_TRUE_SAMPLES = 300
N_LATENT_STEPS = 300


# ── model ─────────────────────────────────────────────────────────────────────


def build_models(latent_dim):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(OBS_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim * 2),
            )

        def forward(self, x):
            return torch.chunk(self.net(x), 2, dim=-1)

    class RSSM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRUCell(latent_dim + ACT_DIM, HIDDEN_DIM)
            self.prior = nn.Linear(HIDDEN_DIM, latent_dim * 2)
            self.post = nn.Linear(HIDDEN_DIM + latent_dim, latent_dim * 2)

        def forward(self, z, a, h):
            h = self.rnn(torch.cat([z, a], dim=-1), h)
            pm, plv = torch.chunk(self.prior(h), 2, dim=-1)
            return h, pm, plv

        def posterior(self, h, enc_z):
            return torch.chunk(self.post(torch.cat([h, enc_z], dim=-1)), 2, dim=-1)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(HIDDEN_DIM + latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, OBS_DIM),
            )

        def forward(self, h, z):
            return self.net(torch.cat([h, z], dim=-1))

    enc = Encoder().to(device)
    rssm = RSSM().to(device)
    dec = Decoder().to(device)
    return enc, rssm, dec


def sample_z(mean, logvar):
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


def kl_div(m1, lv1, m2, lv2):
    return 0.5 * torch.sum(lv2 - lv1 + (lv1.exp() + (m1 - m2) ** 2) / lv2.exp() - 1)


# ── training ──────────────────────────────────────────────────────────────────


def train_and_track(enc, rssm, dec, buf, latent_dim, policy_fn, policy_name):
    params = list(enc.parameters()) + list(rssm.parameters()) + list(dec.parameters())
    opt = optim.Adam(params, lr=1e-3)
    history = {}

    print(f"    Computing true state space diagram ({policy_name})…")
    true_states = sample_true_state_space(policy_fn, n_samples=N_TRUE_SAMPLES)
    dgm_true = compute_dgm(true_states)
    ls_true = loop_strength(dgm_true)
    h0_true = h0_total_persistence(dgm_true)
    print(f"    True H1 loop strength: {ls_true:.4f} | H0 total persistence: {h0_true:.4f}")

    dgm_latent = None

    for epoch in range(TRAIN_EPOCHS):
        if len(buf) < BATCH_SIZE:
            continue

        obs_b, act_b, nobs_b = buf.sample(BATCH_SIZE)
        obs_t = torch.tensor(obs_b).to(device)
        act_t = torch.tensor(act_b).to(device)
        nobs_t = torch.tensor(nobs_b).to(device)

        h = torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(device)
        z = torch.zeros(BATCH_SIZE, latent_dim).to(device)

        enc_mean, enc_lv = enc(nobs_t)
        enc_z = sample_z(enc_mean, enc_lv)

        h, pm, plv = rssm(z, act_t, h)
        qm, qlv = rssm.posterior(h, enc_z)
        z_post = sample_z(qm, qlv)

        recon = dec(h, z_post)
        recon_loss = ((recon - nobs_t) ** 2).mean()
        kl_loss = kl_div(qm, qlv, pm, plv) / BATCH_SIZE
        loss = recon_loss + 0.1 * kl_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % TOPO_CHECKPOINT == 0:
            lats = get_latents(
                enc,
                rssm,
                dec,
                latent_dim,
                HIDDEN_DIM,
                policy_fn,
                n_steps=N_LATENT_STEPS,
                device=device,
            )
            dgm_latent = compute_dgm(lats)
            w_dist = geometric_fidelity(dgm_true, dgm_latent)
            fp = topological_fingerprint(dgm_latent)

            history[epoch] = dict(
                loss=float(loss.item()),
                wasserstein=w_dist,
                loop_strength=fp["h1_loop_strength"],
                h0_total_persist=fp["h0_total_persistence"],
                ls_true=ls_true,
                h0_true=h0_true,
            )
            print(
                f"    epoch {epoch:3d} | loss={loss.item():.3f} | "
                f"W1={w_dist:.4f} | LS_lat={fp['h1_loop_strength']:.4f} | "
                f"H0_tp={fp['h0_total_persistence']:.4f}"
            )

    return history, dgm_true, dgm_latent


# ── sweep ─────────────────────────────────────────────────────────────────────

results = {}

for ld in LATENT_DIMS:
    print(f"\n{'=' * 60}")
    print(f"  LATENT DIM = {ld}")
    print(f"{'=' * 60}")

    buf = ReplayBuffer()
    print("  Filling replay buffer…")
    fill_buffer(buf)

    for pname, pfn in POLICIES.items():
        print(f"\n  ── policy: {pname} ──")
        enc, rssm_model, dec = build_models(ld)

        hist, dgm_true, dgm_latent_final = train_and_track(
            enc,
            rssm_model,
            dec,
            buf,
            ld,
            policy_fn=pfn,
            policy_name=pname,
        )

        results[(ld, pname)] = dict(
            history=hist,
            dgm_true=dgm_true,
            dgm_latent=dgm_latent_final,
            enc=enc,
            rssm=rssm_model,
            dec=dec,
        )


# ── plots ─────────────────────────────────────────────────────────────────────

policies = list(POLICIES.keys())

# A) W1 over epochs
fig, axes = plt.subplots(1, len(policies), figsize=(6 * len(policies), 5), sharey=False)
for ax, pname in zip(axes, policies):
    for ld in LATENT_DIMS:
        hist = results[(ld, pname)]["history"]
        xs = sorted(hist.keys())
        ys = [hist[e]["wasserstein"] for e in xs]
        ax.plot(xs, ys, marker="o", label=f"latent={ld}")
    ax.set_title(f"Geometric fidelity - {pname}\n(lower W₁ = more faithful)")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("W₁ distance H1 (true vs latent)")
    ax.legend()
plt.suptitle("Does training improve geometric fidelity?", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("wasserstein_over_epochs.png", dpi=150, bbox_inches="tight")
plt.show()

# B) Loss vs W1 - do they actually correlate?
fig, axes = plt.subplots(1, len(policies), figsize=(6 * len(policies), 5))
for ax, pname in zip(axes, policies):
    for ld in LATENT_DIMS:
        hist = results[(ld, pname)]["history"]
        losses = [hist[e]["loss"] for e in sorted(hist)]
        wdists = [hist[e]["wasserstein"] for e in sorted(hist)]
        ax.scatter(losses, wdists, label=f"latent={ld}", alpha=0.7, s=40)
    ax.set_xlabel("Reconstruction loss")
    ax.set_ylabel("W₁ distance H1")
    ax.set_title(f"Loss vs geometric fidelity - {pname}")
    ax.legend()
plt.suptitle("Lower loss → better topology?", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("loss_vs_wasserstein.png", dpi=150, bbox_inches="tight")
plt.show()

# C) Final W1 heatmap
final_wmat = np.array(
    [
        [
            results[(ld, p)]["history"][max(results[(ld, p)]["history"])]["wasserstein"]
            for p in policies
        ]
        for ld in LATENT_DIMS
    ]
)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(final_wmat, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(policies)))
ax.set_xticklabels(policies)
ax.set_yticks(range(len(LATENT_DIMS)))
ax.set_yticklabels([f"ld={d}" for d in LATENT_DIMS])
for r in range(len(LATENT_DIMS)):
    for c in range(len(policies)):
        ax.text(c, r, f"{final_wmat[r, c]:.3f}", ha="center", va="center", fontsize=10)
plt.colorbar(im, ax=ax, label="W₁ (lower = better)")
ax.set_title("Final geometric fidelity")
plt.tight_layout()
plt.savefig("fidelity_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# D) Persistence diagrams: true vs latent
fig, axes = plt.subplots(
    len(LATENT_DIMS),
    len(policies) * 2,
    figsize=(5 * len(policies) * 2, 4 * len(LATENT_DIMS)),
)
for ri, ld in enumerate(LATENT_DIMS):
    for ci, pname in enumerate(policies):
        res = results[(ld, pname)]
        ax_t = axes[ri][ci * 2]
        plot_diagrams(res["dgm_true"], ax=ax_t, show=False)
        ax_t.set_title(f"TRUE ld={ld}/{pname}", fontsize=8)
        ax_l = axes[ri][ci * 2 + 1]
        plot_diagrams(res["dgm_latent"], ax=ax_l, show=False)
        w = results[(ld, pname)]["history"][max(results[(ld, pname)]["history"])]["wasserstein"]
        ax_l.set_title(f"LATENT ld={ld}/{pname}  W1={w:.3f}", fontsize=8)
plt.suptitle("True vs learned persistence diagrams", y=1.01, fontsize=12)
plt.tight_layout()
plt.savefig("true_vs_latent_diagrams.png", dpi=150, bbox_inches="tight")
plt.show()

# E) PCA of true vs latent geometry (visual sanity check)
fig, axes = plt.subplots(
    len(LATENT_DIMS),
    len(policies) * 2,
    figsize=(5 * len(policies) * 2, 4 * len(LATENT_DIMS)),
)
for ri, ld in enumerate(LATENT_DIMS):
    for ci, (pname, pfn) in enumerate(POLICIES.items()):
        res = results[(ld, pname)]

        true_states = sample_true_state_space(pfn)
        ax_t = axes[ri][ci * 2]
        ax_t.scatter(
            true_states[:, 0],
            true_states[:, 1],
            s=4,
            alpha=0.6,
            c=np.arange(len(true_states)),
            cmap="viridis",
        )
        ax_t.set_title(f"TRUE (θ,θ̇) ld={ld}/{pname}", fontsize=8)
        ax_t.set_xlabel("θ")
        ax_t.set_ylabel("θ̇")

        lats = get_latents(res["enc"], res["rssm"], res["dec"], ld, HIDDEN_DIM, pfn, device=device)
        proj = PCA(n_components=2).fit_transform(normalize(lats))
        ax_l = axes[ri][ci * 2 + 1]
        ax_l.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.6, c=np.arange(len(proj)), cmap="viridis")
        ax_l.set_title(f"LATENT PCA ld={ld}/{pname}", fontsize=8)
        ax_l.set_xlabel("PC1")
        ax_l.set_ylabel("PC2")

plt.suptitle("True state geometry vs latent geometry (PCA)", y=1.01, fontsize=12)
plt.tight_layout()
plt.savefig("geometry_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# ── summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print(f"{'GEOMETRIC FIDELITY SUMMARY':^80}")
print("=" * 80)
print(
    f"{'policy':<12} {'latent':>7} {'init W1':>9} {'final W1':>9} "
    f"{'delta W1':>9} {'LS_true':>8} {'LS_lat':>8} {'H0_tp_lat':>10}"
)
print("-" * 80)
for ld in LATENT_DIMS:
    for pname in policies:
        hist = results[(ld, pname)]["history"]
        epochs = sorted(hist.keys())
        init_w = hist[epochs[0]]["wasserstein"]
        fin_w = hist[epochs[-1]]["wasserstein"]
        ls_t = hist[epochs[-1]]["ls_true"]
        ls_l = hist[epochs[-1]]["loop_strength"]
        h0_l = hist[epochs[-1]]["h0_total_persist"]
        print(
            f"{pname:<12} {ld:>7} {init_w:>9.4f} {fin_w:>9.4f} "
            f"{fin_w - init_w:>9.4f} {ls_t:>8.4f} {ls_l:>8.4f} {h0_l:>10.4f}"
        )
print("=" * 80)
