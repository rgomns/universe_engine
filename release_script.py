"""
Release script: real reversible universe → emergent QM

Features:
- Phase sweep over (coupling_theta, eps_nn)
- Classical observables → CSV + JSONL
- Heatmap GIFs for phase diagram
- Full QM pipeline for a chosen sweet-spot:
  - observables.jsonl
  - mapping_qm.jsonl
  - qm_reconstruction.json
  - console summary of locality, level statistics, PRs
"""

import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import csv

# ============================================================
# Global configuration
# ============================================================

@dataclass
class UniverseConfig:
    n_agents: int = 32
    bits_per_agent: int = 32
    coupling_theta: float = 0.0
    measure_interval: int = 40
    n_steps: int = 20000
    log_dir: str = "logs_phase"
    seed: int = 1234
    eps_nn: float = 0.2      # nearest-neighbor coupling strength
    eps_nnn: float = 0.05    # next-nearest-neighbor coupling strength

    @property
    def n_bits(self) -> int:
        return self.n_agents * self.bits_per_agent

    @property
    def state_dim(self) -> int:
        return self.n_bits


# ============================================================
# Utility
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    s = np.linalg.norm(v)
    if s == 0:
        return v
    return v / s


def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# ============================================================
# Universe state and dynamics
# ============================================================

@dataclass
class UniverseState:
    x: np.ndarray  # real state vector


@dataclass
class RuleLayer:
    W: np.ndarray  # orthogonal matrix


def make_random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(M)
    return Q


def make_block_local_orthogonal(dim: int, cfg: UniverseConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Construct a block-local biased matrix, then orthogonalize it globally.

    - Strong mixing within each agent (intra-agent blocks).
    - Weaker mixing between nearest neighbors (eps_nn).
    - Even weaker mixing between next-nearest neighbors (eps_nnn).
    """
    n_agents = cfg.n_agents
    bpa = cfg.bits_per_agent
    assert dim == n_agents * bpa

    W_pre = np.zeros((dim, dim), dtype=float)

    def agent_slice(a: int) -> slice:
        return slice(a * bpa, (a + 1) * bpa)

    # Intra-agent strong random orthogonals
    for a in range(n_agents):
        sl = agent_slice(a)
        O_a = make_random_orthogonal(bpa, rng)
        W_pre[sl, sl] += O_a

    # Nearest-neighbor couplings
    for a in range(n_agents):
        sl_a = agent_slice(a)
        b = (a + 1) % n_agents
        sl_b = agent_slice(b)
        C_ab = rng.normal(size=(bpa, bpa))
        C_ba = rng.normal(size=(bpa, bpa))
        W_pre[sl_a, sl_b] += cfg.eps_nn * C_ab
        W_pre[sl_b, sl_a] += cfg.eps_nn * C_ba

    # Next-nearest-neighbor couplings (weaker)
    for a in range(n_agents):
        sl_a = agent_slice(a)
        b = (a + 2) % n_agents
        sl_b = agent_slice(b)
        C_ab = rng.normal(size=(bpa, bpa))
        C_ba = rng.normal(size=(bpa, bpa))
        W_pre[sl_a, sl_b] += cfg.eps_nnn * C_ab
        W_pre[sl_b, sl_a] += cfg.eps_nnn * C_ba

    Q, _ = np.linalg.qr(W_pre)
    return Q


def build_rule_layer(dim: int, cfg: UniverseConfig, rng: np.random.Generator) -> RuleLayer:
    """
    Locality-biased rule layer:
    - Start from block-local orthogonal Q_local.
    - Blend with identity using coupling_theta.
    - Re-orthogonalize to get final W.
    """
    Q_local = make_block_local_orthogonal(dim, cfg, rng)
    W_pre = math.cos(cfg.coupling_theta) * np.eye(dim) + math.sin(cfg.coupling_theta) * Q_local
    Q2, _ = np.linalg.qr(W_pre)
    return RuleLayer(W=Q2)


def step_universe(state: UniverseState, rules: List[RuleLayer]) -> UniverseState:
    x = state.x
    for layer in rules:
        x = layer.W @ x
    return UniverseState(x=x)


# ============================================================
# Classical observables
# ============================================================

def compute_bit_marginals(x: np.ndarray) -> np.ndarray:
    p = x**2
    p = p / np.sum(p)
    return p


def compute_bit_entropy(x: np.ndarray) -> float:
    p = compute_bit_marginals(x)
    return shannon_entropy(p)


def compute_agent_entropy(x: np.ndarray, cfg: UniverseConfig) -> float:
    n_agents = cfg.n_agents
    bpa = cfg.bits_per_agent
    energy_per_agent = []
    for a in range(n_agents):
        sl = slice(a * bpa, (a + 1) * bpa)
        energy_per_agent.append(np.sum(x[sl] ** 2))
    energy_per_agent = np.array(energy_per_agent)
    energy_per_agent /= np.sum(energy_per_agent)
    return shannon_entropy(energy_per_agent)


def compute_coherence(x: np.ndarray, cfg: UniverseConfig) -> float:
    xc = x - np.mean(x)
    return float(np.linalg.norm(xc) / np.linalg.norm(x))


def compute_rule_stats(rules: List[RuleLayer]) -> Tuple[int, float]:
    if not rules:
        return 0, 0.0
    W_all = np.stack([r.W for r in rules], axis=0)
    rule_count = W_all.shape[0]
    rule_strength_mean = float(np.mean(np.abs(W_all)))
    return rule_count, rule_strength_mean


# ============================================================
# Real → complex mapping and QM observables
# ============================================================

def real_to_complex(x: np.ndarray) -> np.ndarray:
    dim = x.shape[0]
    if dim % 2 == 1:
        x = x[:-1]
        dim -= 1
    N = dim // 2
    q = x[:N]
    p = x[N:]
    psi = q + 1j * p
    norm = np.linalg.norm(psi)
    if norm == 0:
        return psi
    return psi / norm


def complex_density_matrix(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, np.conjugate(psi))


def qm_observables_from_real_state(x: np.ndarray) -> Dict:
    psi = real_to_complex(x)
    rho = complex_density_matrix(psi)
    evals = np.linalg.eigvalsh(rho)
    evals = np.real(evals)
    evals = evals[evals > 1e-12]
    if evals.size == 0:
        S_vN = 0.0
    else:
        S_vN = float(-np.sum(evals * np.log2(evals)))
    return {
        "psi_real": psi.real.tolist(),
        "psi_imag": psi.imag.tolist(),
        "rho_diag": np.real(np.diag(rho)).tolist(),
        "von_neumann_entropy": S_vN,
    }


# ============================================================
# Single run + summary metrics (for phase sweep)
# ============================================================

def run_universe_and_summarize(cfg: UniverseConfig) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.seed)

    x0 = rng.normal(size=(cfg.state_dim,))
    x0 = normalize(x0)
    state = UniverseState(x=x0)

    n_layers = 4
    rules = [build_rule_layer(cfg.state_dim, cfg, rng) for _ in range(n_layers)]
    rule_count, rule_strength_mean = compute_rule_stats(rules)

    bit_entropies = []
    agent_entropies = []
    coherences = []

    for t in range(cfg.n_steps):
        state = step_universe(state, rules)
        if t % cfg.measure_interval == 0:
            x = state.x.copy()
            bit_H = compute_bit_entropy(x)
            agent_H = compute_agent_entropy(x, cfg)
            coh = compute_coherence(x, cfg)
            bit_entropies.append(bit_H)
            agent_entropies.append(agent_H)
            coherences.append(coh)

    bit_entropies = np.array(bit_entropies)
    agent_entropies = np.array(agent_entropies)
    coherences = np.array(coherences)

    entropy_osc_score = float(np.std(bit_entropies))

    return {
        "coherence_mean": float(np.mean(coherences)),
        "agent_entropy_mean": float(np.mean(agent_entropies)),
        "rules_mean": rule_strength_mean,
        "entropy_osc_score": entropy_osc_score,
        "rule_count": rule_count,
    }


# ============================================================
# Phase sweep (classical metrics)
# ============================================================

def phase_sweep() -> Tuple[List[Dict], np.ndarray, List[float]]:
    """
    Sweep over (coupling_theta, eps_nn), compute classical metrics,
    and save JSONL + CSV.
    """
    coupling_thetas = np.linspace(0.0, 0.08, 9)  # 0.0 ... 0.08
    eps_nns = [0.0, 0.05, 0.1, 0.2]
    measure_interval = 40
    n_steps = 20000

    results = []
    log_dir = "logs_phase"
    os.makedirs(log_dir, exist_ok=True)

    jsonl_path = os.path.join(log_dir, "phase_diagram_results.jsonl")
    csv_path = os.path.join(log_dir, "phase_diagram_summary.csv")

    with open(jsonl_path, "w") as jf, open(csv_path, "w", newline="") as cf:
        fieldnames = [
            "coupling_theta",
            "eps_nn",
            "measure_interval",
            "coherence_mean",
            "agent_entropy_mean",
            "rules_mean",
            "entropy_osc_score",
            "rule_count",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for eps_nn in eps_nns:
            for ct in coupling_thetas:
                cfg = UniverseConfig(
                    coupling_theta=float(ct),
                    measure_interval=measure_interval,
                    n_steps=n_steps,
                    log_dir=log_dir,
                    seed=2025,
                    eps_nn=float(eps_nn),
                    eps_nnn=0.05,
                )
                print(f"[SWEEP] coupling_theta={ct:.4f}, eps_nn={eps_nn:.3f}")
                summary = run_universe_and_summarize(cfg)
                rec = {
                    "coupling_theta": float(ct),
                    "eps_nn": float(eps_nn),
                    "measure_interval": measure_interval,
                    **summary,
                }
                results.append(rec)
                jf.write(json.dumps(rec) + "\n")
                writer.writerow(rec)

    print(f"Phase sweep complete. JSONL: {jsonl_path}, CSV: {csv_path}")
    return results, coupling_thetas, eps_nns


# ============================================================
# Visualization: heatmap GIFs
# ============================================================

def build_grid(results: List[Dict], coupling_thetas: np.ndarray, eps_nns: List[float]) -> Dict[Tuple[float, float], Dict]:
    table = {}
    for r in results:
        key = (r["coupling_theta"], r["eps_nn"])
        table[key] = r
    return table


def make_heatmap_gifs(results: List[Dict], coupling_thetas: np.ndarray, eps_nns: List[float], out_prefix: str = "phase_diagram") -> None:
    """
    Create static heatmap GIFs for key observables over (coupling_theta, eps_nn).
    """
    table = build_grid(results, coupling_thetas, eps_nns)

    cts = sorted(set(coupling_thetas))
    ens = sorted(set(eps_nns))

    def build_Z(field: str) -> np.ndarray:
        Z = np.zeros((len(ens), len(cts)))
        for i, eps_nn in enumerate(ens):
            for j, ct in enumerate(cts):
                r = table[(float(ct), float(eps_nn))]
                Z[i, j] = r[field]
        return Z

    fields = [
        ("coherence_mean", "Coherence"),
        ("agent_entropy_mean", "Agent Entropy"),
        ("rules_mean", "Rule Density"),
        ("entropy_osc_score", "Entropy Oscillation Score"),
    ]

    for field, title in fields:
        Z = build_Z(field)
        fig, ax = plt.subplots(figsize=(6, 5))
        writer = PillowWriter(fps=1)
        gif_name = f"{out_prefix}_{field}.gif"
        with writer.saving(fig, gif_name, dpi=100):
            ax.clear()
            im = ax.imshow(
                Z,
                origin="lower",
                cmap="viridis",
                extent=[min(cts), max(cts), min(ens), max(ens)],
                aspect="auto",
            )
            plt.colorbar(im, ax=ax, label=title)
            ax.set_xlabel("coupling_theta")
            ax.set_ylabel("eps_nn (locality strength)")
            ax.set_title(title)
            plt.tight_layout()
            writer.grab_frame()
        plt.close(fig)
        print(f"Saved GIF: {gif_name}")


# ============================================================
# QM reconstruction pipeline (single chosen point)
# ============================================================

def run_universe_observables_log(cfg: UniverseConfig, log_name: str) -> None:
    """
    Run universe and log classical observables over time to observables.jsonl.
    """
    os.makedirs(cfg.log_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    x0 = rng.normal(size=(cfg.state_dim,))
    x0 = normalize(x0)
    state = UniverseState(x=x0)

    n_layers = 4
    rules = [build_rule_layer(cfg.state_dim, cfg, rng) for _ in range(n_layers)]
    rule_count, rule_strength_mean = compute_rule_stats(rules)

    history_for_attractor: List[np.ndarray] = []
    log_path = os.path.join(cfg.log_dir, log_name)
    with open(log_path, "w") as f:
        for t in range(cfg.n_steps):
            state = step_universe(state, rules)
            if t % cfg.measure_interval == 0:
                x = state.x.copy()
                bit_H = compute_bit_entropy(x)
                agent_H = compute_agent_entropy(x, cfg)
                coh = compute_coherence(x, cfg)
                history_for_attractor.append(x)
                H = np.stack(history_for_attractor, axis=0)
                Hq = np.round(H, 3)
                uniq = np.unique(Hq, axis=0)
                attractor_div = float(len(uniq) / len(history_for_attractor))
                rec = {
                    "t": t,
                    "coupling_theta": cfg.coupling_theta,
                    "measure_interval": cfg.measure_interval,
                    "coherence": coh,
                    "agent_entropy": agent_H,
                    "bit_entropy": bit_H,
                    "rule_count": rule_count,
                    "rule_strength_mean": rule_strength_mean,
                    "attractor_diversity": attractor_div,
                }
                f.write(json.dumps(rec) + "\n")
    print(f"Observables log written to {log_path}")


def run_mapping_log(cfg: UniverseConfig, mapping_log_name: str) -> None:
    """
    Run universe and log real→complex mapped states and QM observables to mapping_qm.jsonl.
    """
    os.makedirs(cfg.log_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 999)

    x0 = rng.normal(size=(cfg.state_dim,))
    x0 = normalize(x0)
    state = UniverseState(x=x0)

    n_layers = 4
    rules = [build_rule_layer(cfg.state_dim, cfg, rng) for _ in range(n_layers)]

    log_path = os.path.join(cfg.log_dir, mapping_log_name)
    with open(log_path, "w") as f:
        for t in range(cfg.n_steps):
            state = step_universe(state, rules)
            if t % cfg.measure_interval == 0:
                x = state.x.copy()
                qm_obs = qm_observables_from_real_state(x)
                rec = {
                    "t": t,
                    "coupling_theta": cfg.coupling_theta,
                    "measure_interval": cfg.measure_interval,
                    "state_dim": cfg.state_dim,
                    **qm_obs,
                }
                f.write(json.dumps(rec) + "\n")
    print(f"Mapping log written to {log_path}")


def load_mapping_log(path: str) -> Tuple[np.ndarray, np.ndarray]:
    times = []
    psi_list = []
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            psi_real = np.array(rec["psi_real"], dtype=float)
            psi_imag = np.array(rec["psi_imag"], dtype=float)
            psi = psi_real + 1j * psi_imag
            psi = normalize(psi)
            psi_list.append(psi)
            times.append(rec["t"])
    times = np.array(times, dtype=int)
    psi_t = np.stack(psi_list, axis=0)
    return times, psi_t


def compute_fidelity_and_phase(times: np.ndarray, psi_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    psi0 = psi_t[0]
    overlaps = psi_t @ np.conjugate(psi0)
    fidelity = np.abs(overlaps) ** 2
    phase = np.angle(overlaps)
    phase_unwrapped = np.unwrap(phase)
    return fidelity, phase_unwrapped


def estimate_effective_unitary(psi_t: np.ndarray) -> np.ndarray:
    T, N = psi_t.shape
    A = psi_t[:-1].T
    B = psi_t[1:].T
    M = B @ np.linalg.pinv(A)
    U, s, Vh = np.linalg.svd(M)
    U_eff = U @ Vh
    return U_eff


def effective_hamiltonian_from_unitary(U: np.ndarray, dt: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(U)
    eigvals = eigvals / np.abs(eigvals)
    phases = np.angle(eigvals)
    H_diag = np.diag(-phases / dt)
    V = eigvecs
    Vinv = np.linalg.inv(V)
    H = V @ H_diag @ Vinv
    H = 0.5 * (H + H.conjugate().T)
    return H


def qm_reconstruction_pipeline(mapping_log_path: str, output_path: str, measure_interval: int) -> None:
    times, psi_t = load_mapping_log(mapping_log_path)
    fidelity, phase_unwrapped = compute_fidelity_and_phase(times, psi_t)

    U_eff = estimate_effective_unitary(psi_t)
    dt = float(measure_interval)
    H_eff = effective_hamiltonian_from_unitary(U_eff, dt)

    evals, _ = np.linalg.eigh(H_eff)
    evals = evals.real

    out = {
        "times": times.tolist(),
        "fidelity_vs_initial": fidelity.tolist(),
        "phase_vs_initial_unwrapped": phase_unwrapped.tolist(),
        "H_eff_real": H_eff.real.tolist(),
        "H_eff_imag": H_eff.imag.tolist(),
        "H_eigenvalues": evals.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"QM reconstruction written to {output_path}")


# ============================================================
# Hamiltonian analysis (locality, spectrum, eigenvectors)
# ============================================================

def reshape_H_to_agent_blocks(H: np.ndarray, cfg: UniverseConfig) -> np.ndarray:
    n_agents = cfg.n_agents
    bpa = cfg.bits_per_agent
    cpa = bpa // 2
    N_complex = n_agents * cpa
    H = H[:N_complex, :N_complex]
    H_blocks = H.reshape(n_agents, cpa, n_agents, cpa)
    return H_blocks


def compute_agent_coupling_matrix(H: np.ndarray, cfg: UniverseConfig) -> np.ndarray:
    H_blocks = reshape_H_to_agent_blocks(H, cfg)
    n_agents = cfg.n_agents
    C = np.zeros((n_agents, n_agents), dtype=float)
    for i in range(n_agents):
        for j in range(n_agents):
            block = H_blocks[i, :, j, :]
            C[i, j] = float(np.linalg.norm(block))
    return C


def compute_coupling_vs_distance(C_agents: np.ndarray) -> Dict[str, List[float]]:
    n_agents = C_agents.shape[0]
    max_dist = n_agents - 1
    sums = np.zeros(max_dist + 1, dtype=float)
    counts = np.zeros(max_dist + 1, dtype=int)

    for i in range(n_agents):
        for j in range(n_agents):
            d = abs(j - i)
            d = min(d, n_agents - d)
            sums[d] += C_agents[i, j]
            counts[d] += 1

    avg = []
    for d in range(max_dist + 1):
        if counts[d] > 0:
            avg.append(float(sums[d] / counts[d]))
        else:
            avg.append(0.0)

    return {"distance": list(range(max_dist + 1)), "avg_coupling": avg}


def level_statistics(evals: np.ndarray) -> Dict[str, object]:
    evals = np.sort(evals)
    spacings = np.diff(evals)
    spacings = spacings[spacings > 0]
    if spacings.size == 0:
        return {
            "mean_spacing": 0.0,
            "var_spacing": 0.0,
            "min_spacing": 0.0,
            "max_spacing": 0.0,
            "spacings": [],
        }
    return {
        "mean_spacing": float(np.mean(spacings)),
        "var_spacing": float(np.var(spacings)),
        "min_spacing": float(np.min(spacings)),
        "max_spacing": float(np.max(spacings)),
        "spacings": spacings.tolist(),
    }


def analyze_eigenvectors(H: np.ndarray) -> Dict[str, object]:
    evals, evecs = np.linalg.eigh(H)
    N_c = evecs.shape[0]
    PRs = []
    for k in range(N_c):
        v = evecs[:, k]
        p = np.abs(v) ** 2
        PR = 1.0 / float(np.sum(p**2))
        PRs.append(PR)
    return {"participation_ratios": PRs}


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    # 1) Phase sweep (classical metrics) + CSV + JSONL + GIFs
    results, coupling_thetas, eps_nns = phase_sweep()
    make_heatmap_gifs(results, coupling_thetas, eps_nns, out_prefix="phase_diagram")

    print("\nSample phase sweep results:")
    for r in results[:5]:
        print(r)

    # 2) Full QM pipeline for a chosen sweet-spot configuration
    sweet_cfg = UniverseConfig(
        coupling_theta=0.026666666666666665,  # adjust if desired
        measure_interval=40,
        log_dir="logs_best",
        n_steps=20000,
        seed=2025,
        eps_nn=0.2,
        eps_nnn=0.05,
    )

    # 2a) Observables log
    run_universe_observables_log(sweet_cfg, log_name="observables.jsonl")

    # 2b) Mapping log
    mapping_log_name = "mapping_qm.jsonl"
    run_mapping_log(sweet_cfg, mapping_log_name=mapping_log_name)

    # 2c) QM reconstruction
    mapping_log_path = os.path.join(sweet_cfg.log_dir, mapping_log_name)
    reconstruction_output_path = os.path.join(sweet_cfg.log_dir, "qm_reconstruction.json")
    qm_reconstruction_pipeline(
        mapping_log_path=mapping_log_path,
        output_path=reconstruction_output_path,
        measure_interval=sweet_cfg.measure_interval,
    )

    # 2d) Hamiltonian analysis
    with open(reconstruction_output_path, "r") as f:
        rec = json.load(f)

    H_real = np.array(rec["H_eff_real"])
    H_imag = np.array(rec["H_eff_imag"])
    H_eff = H_real + 1j * H_imag

    C_agents = compute_agent_coupling_matrix(H_eff, sweet_cfg)
    coupling_vs_dist = compute_coupling_vs_distance(C_agents)
    print("\n[QM] Coupling vs distance:", coupling_vs_dist)

    evals = np.array(rec["H_eigenvalues"])
    ls = level_statistics(evals)
    print("\n[QM] Level statistics:", ls)

    eig_struct = analyze_eigenvectors(H_eff)
    print("\n[QM] Example participation ratios (first 10):", eig_struct["participation_ratios"][:10])

    print("\nFull package complete:")
    print("- Phase sweep: logs_phase/phase_diagram_results.jsonl, phase_diagram_summary.csv")
    print("- GIFs: phase_diagram_*.gif")
    print("- Sweet-spot observables: logs_best/observables.jsonl")
    print("- Sweet-spot mapping: logs_best/mapping_qm.jsonl")
    print("- Sweet-spot QM reconstruction: logs_best/qm_reconstruction.json")
