import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np

# ============================================================
# Configuration
# ============================================================

@dataclass
class UniverseConfig:
    n_agents: int = 32          # number of agents
    bits_per_agent: int = 32    # bits per agent
    coupling_theta: float = 0.0
    measure_interval: int = 10
    n_steps: int = 20000
    log_dir: str = "logs"
    seed: int = 1234

    @property
    def n_bits(self) -> int:
        return self.n_agents * self.bits_per_agent

    @property
    def state_dim(self) -> int:
        # real Hilbert space dimension
        return self.n_bits

# ============================================================
# Utility: entropy, etc.
# ============================================================

def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))

def normalize(v: np.ndarray) -> np.ndarray:
    s = np.linalg.norm(v)
    if s == 0:
        return v
    return v / s

# ============================================================
# Universe state and dynamics
# ============================================================

@dataclass
class UniverseState:
    x: np.ndarray  # real state vector, shape (state_dim,)

@dataclass
class RuleLayer:
    # simple linear reversible layer with local couplings
    W: np.ndarray  # shape (state_dim, state_dim)

def make_random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(M)
    return Q

def build_rule_layer(dim: int, coupling_theta: float, rng: np.random.Generator) -> RuleLayer:
    # base orthogonal
    Q = make_random_orthogonal(dim, rng)
    # mix with identity via coupling_theta
    W = math.cos(coupling_theta) * np.eye(dim) + math.sin(coupling_theta) * Q
    # re-orthogonalize to keep reversibility tight
    Q2, _ = np.linalg.qr(W)
    return RuleLayer(W=Q2)

def step_universe(state: UniverseState, rules: List[RuleLayer]) -> UniverseState:
    x = state.x
    for layer in rules:
        x = layer.W @ x
    return UniverseState(x=x)

# ============================================================
# Observables
# ============================================================

def compute_bit_marginals(x: np.ndarray) -> np.ndarray:
    # interpret x as real amplitudes over bits; here we just take squared value per bit
    p = x**2
    p = p / np.sum(p)
    return p

def compute_bit_entropy(x: np.ndarray) -> float:
    p = compute_bit_marginals(x)
    return shannon_entropy(p)

def compute_agent_entropy(x: np.ndarray, cfg: UniverseConfig) -> float:
    # group bits per agent, compute marginal per agent, then entropy over agents
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
    # simple proxy: L2 norm of mean-centered state
    xc = x - np.mean(x)
    return float(np.linalg.norm(xc) / np.linalg.norm(x))

def compute_entanglement_by_distance(x: np.ndarray, cfg: UniverseConfig) -> Tuple[float, List[float]]:
    # very simple "entanglement-like" measure:
    # treat each agent as a block, compute pairwise correlation vs distance
    n_agents = cfg.n_agents
    bpa = cfg.bits_per_agent
    agent_vecs = []
    for a in range(n_agents):
        sl = slice(a * bpa, (a + 1) * bpa)
        agent_vecs.append(normalize(x[sl]))
    agent_vecs = np.stack(agent_vecs, axis=0)  # (n_agents, bpa)

    max_dist = n_agents - 1
    corr_by_dist = [[] for _ in range(max_dist)]
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            d = j - i
            c = float(np.dot(agent_vecs[i], agent_vecs[j]))
            corr_by_dist[d - 1].append(abs(c))

    ent_by_dist = []
    for d in range(max_dist):
        if len(corr_by_dist[d]) == 0:
            ent_by_dist.append(0.0)
        else:
            ent_by_dist.append(float(np.mean(corr_by_dist[d])))

    ent_global = float(np.mean(ent_by_dist))
    return ent_global, ent_by_dist

def compute_rule_stats(rules: List[RuleLayer]) -> Tuple[int, int, float]:
    # crude proxies: count layers, diversity ~ distinct rows, mean |W_ij|
    if not rules:
        return 0, 0, 0.0
    W_all = np.stack([r.W for r in rules], axis=0)  # (L, D, D)
    L, D, _ = W_all.shape
    rule_count = L
    # diversity: count unique rows (approx via rounding)
    rows = W_all.reshape(L * D, D)
    rows_rounded = np.round(rows, 3)
    unique_rows = np.unique(rows_rounded, axis=0)
    rule_diversity = unique_rows.shape[0]
    rule_strength_mean = float(np.mean(np.abs(W_all)))
    return rule_count, rule_diversity, rule_strength_mean

def compute_attractor_diversity(history: List[np.ndarray]) -> float:
    # measure how many distinct states we visit (up to coarse binning)
    if not history:
        return 0.0
    H = np.stack(history, axis=0)
    Hq = np.round(H, 3)
    uniq = np.unique(Hq, axis=0)
    return float(len(uniq) / len(history))

# ============================================================
# Real → complex mapping (fixed complex structure)
# ============================================================

def real_to_complex(x: np.ndarray) -> np.ndarray:
    """
    Fixed complex structure: split R^{2N} into (q, p) and map to C^N via q + i p.
    If dim is odd, we drop the last component.
    """
    dim = x.shape[0]
    if dim % 2 == 1:
        x = x[:-1]
        dim -= 1
    N = dim // 2
    q = x[:N]
    p = x[N:]
    psi = q + 1j * p
    # normalize to unit vector (QM state)
    norm = np.linalg.norm(psi)
    if norm == 0:
        return psi
    return psi / norm

def complex_density_matrix(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, np.conjugate(psi))

def qm_observables_from_real_state(x: np.ndarray) -> Dict:
    psi = real_to_complex(x)
    rho = complex_density_matrix(psi)
    # von Neumann entropy for pure state is ~0, but we compute anyway
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
# Run a single universe and log
# ============================================================

def run_universe(cfg: UniverseConfig, log_name: str) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    # initial state: random real vector, normalized
    x0 = rng.normal(size=(cfg.state_dim,))
    x0 = normalize(x0)
    state = UniverseState(x=x0)

    # build rules (you can tune number of layers)
    n_layers = 4
    rules = [build_rule_layer(cfg.state_dim, cfg.coupling_theta, rng) for _ in range(n_layers)]

    rule_count, rule_diversity, rule_strength_mean = compute_rule_stats(rules)

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
                ent_global, ent_by_dist = compute_entanglement_by_distance(x, cfg)

                history_for_attractor.append(x)

                # attractor diversity over window so far
                attractor_div = compute_attractor_diversity(history_for_attractor)

                rec = {
                    "t": t,
                    "coupling_theta": cfg.coupling_theta,
                    "measure_interval": cfg.measure_interval,
                    "coherence": coh,
                    "agent_entropy": agent_H,
                    "bit_entropy": bit_H,
                    "rule_count": rule_count,
                    "rule_diversity": rule_diversity,
                    "rule_strength_mean": rule_strength_mean,
                    "entanglement_global": ent_global,
                    "entanglement_by_distance": ent_by_dist,
                    "attractor_diversity": attractor_div,
                }
                f.write(json.dumps(rec) + "\n")

    print(f"run complete, log written to {log_path}")

# ============================================================
# Run mapping log for the best point
# ============================================================

def run_mapping_for_best_point(cfg: UniverseConfig, mapping_log_name: str) -> None:
    """
    Re-run a single trajectory with the chosen (coupling_theta, measure_interval),
    and log the real→complex mapping and QM-style observables at each measurement.
    """
    os.makedirs(cfg.log_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 999)

    x0 = rng.normal(size=(cfg.state_dim,))
    x0 = normalize(x0)
    state = UniverseState(x=x0)

    n_layers = 4
    rules = [build_rule_layer(cfg.state_dim, cfg.coupling_theta, rng) for _ in range(n_layers)]

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

    print(f"mapping run complete, log written to {log_path}")

# ============================================================
# Sweep helper (if you want to regenerate the phase diagram)
# ============================================================

def sweep_and_log():
    log_dir = "logs_sweep"
    os.makedirs(log_dir, exist_ok=True)

    coupling_thetas = [0.0,
                       0.013333333333333332,
                       0.026666666666666665,
                       0.039999999999999994,
                       0.05333333333333333]
    measure_intervals = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for ct in coupling_thetas:
        for mi in measure_intervals:
            cfg = UniverseConfig(
                coupling_theta=ct,
                measure_interval=mi,
                log_dir=log_dir,
                n_steps=20000,
                seed=1234,
            )
            log_name = f"ct_{ct:.6f}__mi_{mi}.jsonl"
            run_universe(cfg, log_name)

# ============================================================
# Main: run best point + mapping
# ============================================================

if __name__ == "__main__":
    # Best point from your logs:
    # coupling_theta = 0.026666666666666665, measure_interval = 40
    best_cfg = UniverseConfig(
        coupling_theta=0.026666666666666665,
        measure_interval=40,
        log_dir="logs_best",
        n_steps=20000,
        seed=2025,
    )

    # 1) Run the universe and log classical-style observables
    run_universe(best_cfg, log_name="best_point_observables.json")

    # 2) Run mapping log: real → complex QM representation
    run_mapping_for_best_point(best_cfg, mapping_log_name="best_point_mapping_qm.json")

    # If you want to regenerate the whole sweep, uncomment:
    # sweep_and_log()
