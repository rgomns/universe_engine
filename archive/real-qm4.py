import numpy as np
from collections import Counter
import json
import os
from math import log2

# =========================================================
# REAL-VALUED QUANTUM-LIKE UNIVERSE (32 BITS)
# =========================================================

class RealQMUniverse:
    """
    Real-valued quantum-like universe with (q, p) ∈ R^{2N}.
    """

    def __init__(
        self,
        n_bits=32,
        pop_size=400,
        seed=42,
        local_theta=0.04,
        coupling_theta=0.07,
        measure_interval=25,
        measure_noise=0.01,
        emergence_rate=0.05,
        initial_rule_strength=0.40,
        hebb_decay=0.995,
        hebb_reinforce=0.02,
        hebb_survival=0.08,
        hebb_success_thresh=0.92,
        max_rule_strength=0.70,
    ):
        self.n_bits = n_bits
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)

        self.population = self.rng.normal(
            loc=0.0, scale=1.0, size=(pop_size, 2 * n_bits)
        ).astype(np.float32)
        self._renormalize()

        self.local_theta = local_theta
        self.coupling_theta = coupling_theta

        self.shifts = np.array([1, 5, 17], dtype=np.int32)
        idx = np.arange(self.n_bits, dtype=np.int32)
        self.coupling_pairs = {
            s: (idx, (idx + s) % self.n_bits) for s in self.shifts
        }

        self.emergence_rate = emergence_rate
        self.initial_rule_strength = initial_rule_strength
        self.hebb_decay = hebb_decay
        self.hebb_reinforce = hebb_reinforce
        self.hebb_survival = hebb_survival
        self.hebb_success_thresh = hebb_success_thresh
        self.max_rule_strength = max_rule_strength

        self.measure_interval = measure_interval
        self.measure_noise = measure_noise

        self.couplings = []

        self.cycle = 0
        self.entropy_history = []
        self.agent_entropy_history = []
        self.coherence_history = []

    # ---------------- core utilities ----------------

    def _split_qp(self):
        q = self.population[:, :self.n_bits]
        p = self.population[:, self.n_bits:]
        return q, p

    def _renormalize(self):
        norms = np.linalg.norm(self.population, axis=1, keepdims=True) + 1e-12
        self.population /= norms

    # ---------------- metrics ----------------

    def get_entropy(self):
        q, p = self._split_qp()
        probs = q**2 + p**2
        p_bits = probs.mean(axis=0)
        Z = p_bits.sum() + 1e-12
        p_bits /= Z
        entropy = -np.sum(p_bits * np.log2(p_bits + 1e-12))
        entropy /= np.log2(self.n_bits + 1e-12)
        return float(entropy)

    def get_agent_entropy(self):
        q, p = self._split_qp()
        probs = q**2 + p**2
        Z = probs.sum(axis=1, keepdims=True) + 1e-12
        probs_norm = probs / Z
        ent = -np.sum(probs_norm * np.log2(probs_norm + 1e-12), axis=1)
        ent /= np.log2(self.n_bits + 1e-12)
        return float(ent.mean()), float(ent.std())

    def get_coherence(self):
        q, p = self._split_qp()
        psi = q + 1j * p
        norms = np.linalg.norm(psi, axis=1, keepdims=True) + 1e-12
        psi = psi / norms

        sample_size = min(50, self.pop_size)
        idx = self.rng.choice(self.pop_size, size=sample_size, replace=False)
        psi_s = psi[idx]

        overlaps = []
        for i in range(sample_size - 1):
            for j in range(i + 1, sample_size):
                ov = np.vdot(psi_s[i], psi_s[j])
                overlaps.append(abs(ov))

        overlaps = np.array(overlaps)
        return float(overlaps.mean()), float(overlaps.std())

    # ---------------- micro-dynamics ----------------

    def _local_qp_rotation(self, theta):
        q = self.population[:, :self.n_bits]
        p = self.population[:, self.n_bits:]
        c = np.cos(theta)
        s = np.sin(theta)
        q_new = c * q - s * p
        p_new = s * q + c * p
        self.population[:, :self.n_bits] = q_new
        self.population[:, self.n_bits:] = p_new

    def _coupling_rotation(self, theta, idx, jdx):
        q = self.population[:, :self.n_bits]
        p = self.population[:, self.n_bits:]
        c = np.cos(theta)
        s = np.sin(theta)

        q_i = q[:, idx]
        q_j = q[:, jdx]
        p_i = p[:, idx]
        p_j = p[:, jdx]

        q_i_new = c * q_i + s * q_j
        q_j_new = -s * q_i + c * q_j
        p_i_new = c * p_i + s * p_j
        p_j_new = -s * p_i + c * p_j

        q[:, idx] = q_i_new
        q[:, jdx] = q_j_new
        p[:, idx] = p_i_new
        p[:, jdx] = p_j_new

    def real_unitary_micro_update(self):
        self._local_qp_rotation(self.local_theta)
        for s in self.shifts:
            idx, jdx = self.coupling_pairs[int(s)]
            self._coupling_rotation(self.coupling_theta, idx, jdx)

    # ---------------- constraints + hebbian ----------------

    def apply_constraints(self):
        if not self.couplings:
            return
        q = self.population[:, :self.n_bits]
        for scope, forbidden, strength in self.couplings:
            vals = np.sign(q[:, scope].sum(axis=1))
            violators = vals == forbidden
            if not np.any(violators):
                continue
            spin_sum = q[violators][:, scope].sum(axis=1)
            direction = np.sign(spin_sum)
            q[violators][:, scope] -= strength * 0.05 * direction[:, None]

    def hebbian_update(self):
        if not self.couplings:
            return
        q = self.population[:, :self.n_bits]
        next_rules = []
        for c in self.couplings:
            scope, forbidden, strength = c
            strength *= self.hebb_decay
            vals = np.sign(q[:, scope].sum(axis=1))
            success = np.mean(vals != forbidden)
            if success > self.hebb_success_thresh:
                strength = min(
                    strength + self.hebb_reinforce,
                    self.max_rule_strength
                )
            if strength > self.hebb_survival:
                c[2] = strength
                next_rules.append(c)
        self.couplings = next_rules

    # ---------------- measurement ----------------

    def measure(self):
        q, p = self._split_qp()
        probs = q**2 + p**2
        probs = np.clip(probs, 0.0, 1.0)
        rand = self.rng.random(size=probs.shape)
        collapse = rand < probs
        q_collapsed = np.where(collapse, np.sign(q), q)
        p_collapsed = np.where(collapse, 0.0, p)
        q_collapsed += self.measure_noise * self.rng.normal(size=q.shape)
        p_collapsed += self.measure_noise * self.rng.normal(size=p.shape)
        self.population[:, :self.n_bits] = q_collapsed
        self.population[:, self.n_bits:] = p_collapsed

    # ---------------- rule emergence + manual rules ----------------

    def maybe_emerge_rule(self):
        if self.rng.random() >= self.emergence_rate:
            return
        size = 3 if self.rng.random() < 0.3 else 2
        scope = self.rng.choice(self.n_bits, size=size, replace=False)
        q = self.population[:, :self.n_bits]
        vals = np.sign(q[:, scope].sum(axis=1))
        mean_spin = np.mean(vals)
        forbidden = int(np.sign(mean_spin))
        if forbidden == 0:
            forbidden = self.rng.choice([-1, 1])
        self.couplings.append(
            [scope, forbidden, self.initial_rule_strength]
        )

    def add_rule(self, scope, forbidden, strength=None):
        if strength is None:
            strength = self.initial_rule_strength
        scope = np.array(scope, dtype=int)
        self.couplings.append([scope, int(forbidden), float(strength)])

    # ---------------- step ----------------

    def step(self):
        self.cycle += 1
        self.real_unitary_micro_update()
        self.maybe_emerge_rule()
        self.apply_constraints()
        self.hebbian_update()
        if self.cycle % self.measure_interval == 0:
            self.measure()
        self._renormalize()

        entropy = self.get_entropy()
        agent_ent_mean, _ = self.get_agent_entropy()
        coh_mean, _ = self.get_coherence()

        self.entropy_history.append(entropy)
        self.agent_entropy_history.append(agent_ent_mean)
        self.coherence_history.append(coh_mean)

        return entropy


# =========================================================
# HELPERS
# =========================================================

def mutual_information_bits_from_spins(spins, A_indices, B_indices):
    def encode(rows, idxs):
        return [tuple(int(s) for s in row[idxs]) for row in rows]

    A_pats = encode(spins, A_indices)
    B_pats = encode(spins, B_indices)
    AB_pats = list(zip(A_pats, B_pats))

    def probs(pats):
        c = Counter(pats)
        total = sum(c.values())
        return {k: v / total for k, v in c.items()}

    PA = probs(A_pats)
    PB = probs(B_pats)
    PAB = probs(AB_pats)

    I = 0.0
    for (a, b), pab in PAB.items():
        pa = PA[a]
        pb = PB[b]
        I += pab * log2((pab + 1e-12) / (pa * pb + 1e-12))
    return I

def compute_entanglement_profile(spins):
    """
    spins: (pop_size, n_bits) with entries in {-1, +1}
    Returns:
      global_entanglement: float
      entanglement_by_distance: list of length max_distance
    """
    pop_size, n_bits = spins.shape
    # precompute all 2-bit subsets
    bit_pairs = [(i, j) for i in range(n_bits) for j in range(i+1, n_bits)]
    # group pairs by distance
    dist_groups = {}
    for i, j in bit_pairs:
        d = min(abs(j - i), n_bits - abs(j - i))  # ring distance
        dist_groups.setdefault(d, []).append((i, j))

    ent_by_dist = {}
    total_I = 0.0
    total_pairs = 0

    # to avoid insane cost, we sample a subset of pairs per distance
    max_pairs_per_dist = 40

    for d, pairs in dist_groups.items():
        if len(pairs) > max_pairs_per_dist:
            rng = np.random.default_rng(123 + d)
            pairs = list(pairs)
            idx = rng.choice(len(pairs), size=max_pairs_per_dist, replace=False)
            pairs = [pairs[k] for k in idx]

        I_vals = []
        for (i1, j1) in pairs:
            for (i2, j2) in pairs:
                if (i1, j1) == (i2, j2):
                    continue
                A = [i1, j1]
                B = [i2, j2]
                I = mutual_information_bits_from_spins(spins, A, B)
                I_vals.append(I)
        if I_vals:
            mean_I = float(np.mean(I_vals))
            ent_by_dist[d] = mean_I
            total_I += sum(I_vals)
            total_pairs += len(I_vals)

    global_ent = float(total_I / total_pairs) if total_pairs > 0 else 0.0

    max_d = max(ent_by_dist.keys()) if ent_by_dist else 0
    ent_list = [ent_by_dist.get(d, 0.0) for d in range(1, max_d + 1)]

    return global_ent, ent_list

def compute_attractor_diversity(spins_samples, prob_threshold=1e-3):
    """
    spins_samples: list of spin configurations (tuples) collected over time.
    """
    c = Counter(spins_samples)
    total = sum(c.values())
    if total == 0:
        return 0.0
    kept = [cfg for cfg, cnt in c.items() if cnt / total >= prob_threshold]
    D = len(kept) / total
    return float(D)

def summarize_rule_ecology(universe):
    scopes = []
    strengths = []
    for scope, forbidden, strength in universe.couplings:
        scopes.append(tuple(sorted(scope.tolist())))
        strengths.append(strength)
    rule_count = len(scopes)
    rule_diversity = len(set(scopes))
    rule_strength_mean = float(np.mean(strengths)) if strengths else 0.0
    return rule_count, rule_diversity, rule_strength_mean


# =========================================================
# SWEEP CONFIG
# =========================================================

N_BITS = 32
POP_SIZE = 400
SEED = 123

LOCAL_THETA = 0.04
EMERGENCE_RATE = 0.05
MEASURE_NOISE = 0.01

COUPLING_THETA_VALUES = np.linspace(0.0, 0.12, 10)
MEASURE_INTERVAL_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

STEPS_EQUIL = 1500
STEPS_SAMPLE = 500

LOG_DIR = "phase_sweep_logs"
os.makedirs(LOG_DIR, exist_ok=True)

SUMMARY_FILE = os.path.join(LOG_DIR, "summary_results.jsonl")


# =========================================================
# MAIN SWEEP
# =========================================================

def run_single_point(coupling_theta, measure_interval):
    u = RealQMUniverse(
        n_bits=N_BITS,
        pop_size=POP_SIZE,
        seed=SEED,
        local_theta=LOCAL_THETA,
        coupling_theta=coupling_theta,
        measure_interval=measure_interval,
        measure_noise=MEASURE_NOISE,
        emergence_rate=EMERGENCE_RATE,
    )

    # Equilibrate
    for _ in range(STEPS_EQUIL):
        u.step()

    # Sample spins over time for attractor + entanglement
    spins_samples = []
    for _ in range(STEPS_SAMPLE):
        u.step()
        q, _ = u._split_qp()
        spins = np.sign(q)
        spins[spins == 0] = 1
        # store one representative config per step (e.g. first agent)
        spins_samples.append(tuple(int(s) for s in spins[0]))

    # Use last snapshot for entanglement profile
    q, _ = u._split_qp()
    spins = np.sign(q)
    spins[spins == 0] = 1

    global_ent, ent_by_dist = compute_entanglement_profile(spins)
    attractor_div = compute_attractor_diversity(spins_samples)

    rule_count, rule_diversity, rule_strength_mean = summarize_rule_ecology(u)

    coh_mean, _ = u.get_coherence()
    agent_ent_mean, _ = u.get_agent_entropy()
    bit_entropy = u.get_entropy()

    result = {
        "coupling_theta": float(coupling_theta),
        "measure_interval": int(measure_interval),
        "coherence": float(coh_mean),
        "agent_entropy": float(agent_ent_mean),
        "bit_entropy": float(bit_entropy),
        "rule_count": int(rule_count),
        "rule_diversity": int(rule_diversity),
        "rule_strength_mean": float(rule_strength_mean),
        "entanglement_global": float(global_ent),
        "entanglement_by_distance": ent_by_dist,
        "attractor_diversity": float(attractor_div),
    }

    return result


def main():
    # open summary log in append mode
    with open(SUMMARY_FILE, "a", encoding="utf-8") as f_sum:
        total_points = len(COUPLING_THETA_VALUES) * len(MEASURE_INTERVAL_VALUES)
        idx = 0
        for ct in COUPLING_THETA_VALUES:
            for mi in MEASURE_INTERVAL_VALUES:
                idx += 1
                print(f"[{idx}/{total_points}] ct={ct:.4f}, meas={mi}")
                res = run_single_point(ct, mi)

                # write one JSON line per parameter point
                f_sum.write(json.dumps(res) + "\n")
                f_sum.flush()

                # also write a small per-point log if you want
                point_log = os.path.join(
                    LOG_DIR,
                    f"ct_{ct:.4f}_meas_{mi}.json"
                )
                with open(point_log, "w", encoding="utf-8") as f_pt:
                    json.dump(res, f_pt, indent=2)


if __name__ == "__main__":
    main()
