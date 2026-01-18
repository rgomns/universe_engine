import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ============================================================
# Configuration
# ============================================================

@dataclass
class UniverseConfig:
    n_agents: int = 32
    bits_per_agent: int = 32
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

def make_block_local_orthogonal(dim: int, cfg: UniverseConfig, rng: np.random.Generator,
                                eps_nn: float = 0.2, eps_nnn: float = 0.05) -> np.ndarray:
    """
    Construct a block-local biased matrix, then orthogonalize it globally.

    - Strong mixing within each agent (intra-agent blocks).
    - Weaker mixing between nearest neighbors (nn).
    - Even weaker mixing between next-nearest neighbors (nnn).
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
        W_pre[sl_a, sl_b] += eps_nn * C_ab
        W_pre[sl_b, sl_a] += eps_nn * C_ba

    # Next-nearest-neighbor couplings (weaker)
    for a in range(n_agents):
        sl_a = agent_slice(a)
        b = (a + 2) % n_agents
        sl_b = agent_slice(b)
        C_ab = rng.normal(size=(bpa, bpa))
        C_ba = rng.normal(size=(bpa, bpa))
        W_pre[sl_a, sl_b] += eps_nnn * C_ab
        W_pre[sl_b, sl_a] += eps_nnn * C_ba

    # Global orthogonalization via QR
    Q, _ = np.linalg.qr(W_pre)
    return Q

def build_rule_layer(dim: int, coupling_theta: float, cfg: UniverseConfig,
                     rng: np.random.Generator) -> RuleLayer:
    """
    Locality-biased rule layer:
    - Start from block-local orthogonal Q_local.
    - Blend with identity using coupling_theta.
    - Re-orthogonalize to get final W.
    """
    Q_local = make_block_local_orthogonal(dim, cfg, rng)
    W_pre = math.cos(coupling_theta) * np.eye(dim) + math.sin(coupling_theta) * Q_local
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

def compute_entanglement_by_distance(x: np.ndarray, cfg: UniverseConfig) -> Tuple[float, List[float]]:
    n_agents = cfg.n_agents
    bpa = cfg.bits_per_agent
    agent_vecs = []
    for a in range(n_agents):
        sl = slice(a * bpa, (a + 1) * bpa)
        agent_vecs.append(normalize(x[sl]))
    agent_vecs = np.stack(agent_vecs, axis=0)

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
    if not rules:
        return 0, 0, 0.0
    W_all = np.stack([r.W for r in rules], axis=0)
    L, D, _ = W_all.shape
    rule_count = L
    rows = W_all.reshape(L * D, D)
    rows_rounded = np.round(rows, 3)
    unique_rows = np.unique(rows_rounded, axis=0)
    rule_diversity = unique_rows.shape[0]
    rule_strength_mean = float(np.mean(np.abs(W_all)))
    return rule_count, rule_diversity, rule_strength_mean

def compute_attractor_diversity(history: List[np.ndarray]) -> float:
    if not history:
        return 0.0
    H = np.stack(history, axis=0)
    Hq = np.round(H, 3)
    uniq = np.unique(Hq, axis=0)
    return float(len(uniq) / len(history))

# ============================================================
# Real → complex mapping
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
# Run universe and mapping logs
# ============================================================

def run_universe(cfg: UniverseConfig, log_name: str) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    x0 = rng.normal(size=(cfg.state_dim,))
    x0 = normalize(x0)
    state = UniverseState(x=x0)

    n_layers = 4
    rules = [build_rule_layer(cfg.state_dim, cfg.coupling_theta, cfg, rng) for _ in range(n_layers)]
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

def run_mapping_for_best_point(cfg: UniverseConfig, mapping_log_name: str) -> None:
    os.makedirs(cfg.log_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed + 999)

    x0 = rng.normal(size=(cfg.state_dim,))
    x0 = normalize(x0)
    state = UniverseState(x=x0)

    n_layers = 4
    rules = [build_rule_layer(cfg.state_dim, cfg.coupling_theta, cfg, rng) for _ in range(n_layers)]

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
# QM reconstruction pipeline
# ============================================================

def load_mapping_log(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mapping log and return:
      times: shape (T,)
      psi_t: shape (T, N) complex
    """
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

def compute_fidelity_and_phase(times: np.ndarray, psi_t: np.ndarray):
    """
    Correct vectorized fidelity + phase computation.
    psi_t: shape (T, N)
    """
    psi0 = psi_t[0]  # (N,)
    overlaps = psi_t @ np.conjugate(psi0)  # shape (T,)
    fidelity = np.abs(overlaps) ** 2
    phase = np.angle(overlaps)
    phase_unwrapped = np.unwrap(phase)
    return fidelity, phase_unwrapped

def estimate_effective_unitary(psi_t: np.ndarray) -> np.ndarray:
    """
    Estimate a single-step effective unitary U_eff that maps psi_k -> psi_{k+1}
    in least-squares sense.
    """
    T, N = psi_t.shape
    A = psi_t[:-1].T   # shape (N, T-1)
    B = psi_t[1:].T    # shape (N, T-1)
    M = B @ np.linalg.pinv(A)
    U, s, Vh = np.linalg.svd(M)
    U_eff = U @ Vh
    return U_eff

def effective_hamiltonian_from_unitary(U: np.ndarray, dt: float) -> np.ndarray:
    """
    Given unitary U ≈ exp(-i H dt), reconstruct H via matrix logarithm.
    """
    eigvals, eigvecs = np.linalg.eig(U)
    eigvals = eigvals / np.abs(eigvals)
    phases = np.angle(eigvals)
    H_diag = np.diag(-phases / dt)
    V = eigvecs
    Vinv = np.linalg.inv(V)
    H = V @ H_diag @ Vinv
    H = 0.5 * (H + H.conjugate().T)
    return H

def qm_reconstruction_pipeline(
    mapping_log_path: str,
    output_path: str,
    measure_interval: int,
) -> None:
    times, psi_t = load_mapping_log(mapping_log_path)
    fidelity, phase_unwrapped = compute_fidelity_and_phase(times, psi_t)

    U_eff = estimate_effective_unitary(psi_t)
    dt = float(measure_interval)
    H_eff = effective_hamiltonian_from_unitary(U_eff, dt)

    evals, evecs = np.linalg.eigh(H_eff)
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
# Hamiltonian analysis: locality, spectrum, eigenvectors
# ============================================================

def reshape_H_to_agent_blocks(H: np.ndarray, cfg: UniverseConfig) -> np.ndarray:
    """
    Reshape H (N_c x N_c complex) into (n_agents, cpa, n_agents, cpa),
    where N_c = complex dimension = (n_agents * bits_per_agent) // 2,
    and cpa = bits_per_agent // 2.
    """
    n_agents = cfg.n_agents
    bpa = cfg.bits_per_agent
    cpa = bpa // 2  # complex dims per agent
    N_complex = n_agents * cpa  # should be 512

    H = H[:N_complex, :N_complex]
    H_blocks = H.reshape(n_agents, cpa, n_agents, cpa)
    return H_blocks

def compute_agent_coupling_matrix(H: np.ndarray, cfg: UniverseConfig) -> np.ndarray:
    """
    Compute an (n_agents x n_agents) matrix of coupling strengths between agents,
    using Frobenius norm of each agent-agent block.
    """
    H_blocks = reshape_H_to_agent_blocks(H, cfg)
    n_agents = cfg.n_agents

    C = np.zeros((n_agents, n_agents), dtype=float)
    for i in range(n_agents):
        for j in range(n_agents):
            block = H_blocks[i, :, j, :]
            C[i, j] = float(np.linalg.norm(block))
    return C

def compute_coupling_vs_distance(C_agents: np.ndarray) -> Dict[str, List[float]]:
    """
    Average coupling strength as a function of agent distance on a ring.
    """
    n_agents = C_agents.shape[0]
    max_dist = n_agents - 1
    sums = np.zeros(max_dist + 1, dtype=float)
    counts = np.zeros(max_dist + 1, dtype=int)

    for i in range(n_agents):
        for j in range(n_agents):
            d = abs(j - i)
            d = min(d, n_agents - d)  # ring distance
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
    """
    Simple level-spacing statistics for sorted eigenvalues.
    """
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

def analyze_eigenvectors(H: np.ndarray, cfg: UniverseConfig) -> Dict[str, object]:
    """
    Analyze eigenvectors: participation ratios, per-agent weights, etc.
    """
    evals, evecs = np.linalg.eigh(H)
    evecs = evecs  # shape (N_c, N_c)
    N_c = evecs.shape[0]

    # Participation ratio: 1 / sum |psi_i|^4
    PRs = []
    for k in range(N_c):
        v = evecs[:, k]
        p = np.abs(v) ** 2
        PR = 1.0 / float(np.sum(p**2))
        PRs.append(PR)

    return {
        "participation_ratios": PRs,
    }

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Example sweet-spot configuration
    best_cfg = UniverseConfig(
        coupling_theta=0.026666666666666665,
        measure_interval=40,
        log_dir="logs_best",
        n_steps=20000,
        seed=2025,
    )

    # 1) Classical observables at best point
    run_universe(best_cfg, log_name="best_point_observables.jsonl")

    # 2) Mapping log: real → complex QM representation
    mapping_log_name = "best_point_mapping_qm.jsonl"
    run_mapping_for_best_point(best_cfg, mapping_log_name=mapping_log_name)

    # 3) Full QM reconstruction pipeline
    mapping_log_path = os.path.join(best_cfg.log_dir, mapping_log_name)
    reconstruction_output_path = os.path.join(best_cfg.log_dir, "best_point_qm_reconstruction.json")
    qm_reconstruction_pipeline(
        mapping_log_path=mapping_log_path,
        output_path=reconstruction_output_path,
        measure_interval=best_cfg.measure_interval,
    )

    # 4) Load reconstructed Hamiltonian for analysis
    with open(reconstruction_output_path, "r") as f:
        rec = json.load(f)

    H_real = np.array(rec["H_eff_real"])
    H_imag = np.array(rec["H_eff_imag"])
    H_eff = H_real + 1j * H_imag

    # 5) Coupling vs distance
    C_agents = compute_agent_coupling_matrix(H_eff, best_cfg)
    coupling_vs_dist = compute_coupling_vs_distance(C_agents)
    print("Coupling vs distance:", coupling_vs_dist)

    # 6) Level statistics
    evals = np.array(rec["H_eigenvalues"])
    ls = level_statistics(evals)
    print("Level statistics:", ls)

    # 7) Eigenvector structure
    eig_struct = analyze_eigenvectors(H_eff, best_cfg)
    print("Example participation ratios (first 10):", eig_struct["participation_ratios"][:10])
