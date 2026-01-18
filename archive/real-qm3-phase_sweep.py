import numpy as np
import time
import json
import os
import threading
import multiprocessing as mp


class RealQMUniverse:
    """
    Real-valued quantum-like universe with explicit (q, p) structure.

    - State: x = (q, p) ∈ R^{2N}, with N = n_bits
    - Complex wavefunction: ψ = q + i p
    - Complex structure: J(q, p) = (-p, q)
    - Micro-dynamics: orthogonal, J-compatible (real representation of unitary)
    - Constraints: emergent rules acting on q (spin-like) + Hebbian adaptation
    - Measurement: Born rule from q^2 + p^2
    """

    def __init__(
        self,
        n_bits=256,
        pop_size=2500,
        seed=42,
        local_theta=0.12,
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

        # State: population of agents, each with (q, p) ∈ R^{2N}
        # Layout: [q_0 ... q_{N-1}, p_0 ... p_{N-1}]
        self.population = self.rng.normal(
            loc=0.0, scale=1.0, size=(pop_size, 2 * n_bits)
        ).astype(np.float32)
        self._renormalize()

        # Orthogonal micro-dynamics parameters
        self.local_theta = local_theta
        self.coupling_theta = coupling_theta

        # Precompute coupling index maps for speed
        self.shifts = np.array([1, 5, 17, 41], dtype=np.int32)
        idx = np.arange(self.n_bits, dtype=np.int32)
        self.coupling_pairs = {
            s: (idx, (idx + s) % self.n_bits) for s in self.shifts
        }

        # Rule parameters
        self.emergence_rate = emergence_rate
        self.initial_rule_strength = initial_rule_strength
        self.hebb_decay = hebb_decay
        self.hebb_reinforce = hebb_reinforce
        self.hebb_survival = hebb_survival
        self.hebb_success_thresh = hebb_success_thresh
        self.max_rule_strength = max_rule_strength

        # Measurement parameters
        self.measure_interval = measure_interval
        self.measure_noise = measure_noise

        # Constraints: list of [scope_indices, forbidden_spin, strength]
        self.couplings = []

        # Diagnostics
        self.cycle = 0
        self.record_low = 1.0
        self.entropy_history = []
        self.magnet_history = []
        self.agent_entropy_history = []
        self.coherence_history = []

        self.start_time = time.time()
        self.running = True

    # ---------------------------------------------------------
    # CORE UTILITIES
    # ---------------------------------------------------------

    def _split_qp(self):
        q = self.population[:, :self.n_bits]
        p = self.population[:, self.n_bits:]
        return q, p

    def _renormalize(self):
        norms = np.linalg.norm(self.population, axis=1, keepdims=True) + 1e-12
        self.population /= norms

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    def get_entropy(self):
        """
        Ensemble bit-entropy from Born probabilities over bits.
        """
        q, p = self._split_qp()
        probs = q**2 + p**2
        p_bits = probs.mean(axis=0)
        Z = p_bits.sum() + 1e-12
        p_bits /= Z
        entropy = -np.sum(p_bits * np.log2(p_bits + 1e-12))
        entropy /= np.log2(self.n_bits + 1e-12)
        return float(entropy)

    def get_agent_entropy(self):
        """
        Per-agent entropy over bits, then averaged over agents.
        """
        q, p = self._split_qp()
        probs = q**2 + p**2  # (pop_size, n_bits)
        Z = probs.sum(axis=1, keepdims=True) + 1e-12
        probs_norm = probs / Z
        ent = -np.sum(probs_norm * np.log2(probs_norm + 1e-12), axis=1)
        ent /= np.log2(self.n_bits + 1e-12)
        return float(ent.mean()), float(ent.std())

    def get_coherence(self):
        """
        Lightweight coherence proxy:
        average |<psi_i, psi_j>| over random agent pairs.
        """
        q, p = self._split_qp()
        psi = q + 1j * p

        # normalize
        norms = np.linalg.norm(psi, axis=1, keepdims=True) + 1e-12
        psi = psi / norms

        # sample random pairs
        sample_size = min(50, self.pop_size)
        idx = self.rng.choice(self.pop_size, size=sample_size, replace=False)
        psi_s = psi[idx]

        # compute pairwise overlaps
        overlaps = []
        for i in range(sample_size - 1):
            for j in range(i + 1, sample_size):
                ov = np.vdot(psi_s[i], psi_s[j])
                overlaps.append(abs(ov))

        overlaps = np.array(overlaps)
        return float(overlaps.mean()), float(overlaps.std())


    def get_magnetization(self):
        q, _ = self._split_qp()
        return float(np.sum(np.sign(q)))

    def get_rule_stats(self):
        if not self.couplings:
            return 0, 0.0, 0.0
        strengths = np.array([c[2] for c in self.couplings], dtype=np.float32)
        return len(strengths), float(strengths.mean()), float(strengths.max())

    # ---------------------------------------------------------
    # SAVE / LOAD
    # ---------------------------------------------------------

    def save_state(self, filename="real_qm_universe.json"):
        data = {
            "cycle": self.cycle,
            "population": self.population.tolist(),
            "couplings": [
                [scope.tolist(), int(forbidden), float(strength)]
                for scope, forbidden, strength in self.couplings
            ],
            "record_low": self.record_low,
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"[SAVE] cycle={self.cycle}, rules={len(self.couplings)}")

    def load_state(self, filename="real_qm_universe.json"):
        if not os.path.exists(filename):
            print("[SYSTEM] No save found.")
            return
        with open(filename, "r") as f:
            data = json.load(f)
        self.cycle = data["cycle"]
        self.population = np.array(data["population"], dtype=np.float32)
        self.couplings = [
            [np.array(scope, dtype=int), forbidden, strength]
            for scope, forbidden, strength in data["couplings"]
        ]
        self.record_low = data["record_low"]
        self._renormalize()
        print(f"[LOAD] cycle={self.cycle}, rules={len(self.couplings)}")

    # ---------------------------------------------------------
    # REAL ORTHOGONAL MICRO-DYNAMICS (J-COMPATIBLE)
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # REAL CONSTRAINT POTENTIALS (ON q ONLY)
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # HEBBIAN RULE SELECTION
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # MEASUREMENT (REAL BORN RULE)
    # ---------------------------------------------------------

    def measure(self):
        q = self.population[:, :self.n_bits]
        p = self.population[:, self.n_bits:]
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

    # ---------------------------------------------------------
    # RULE EMERGENCE
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------

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
        magnet = self.get_magnetization()
        agent_ent_mean, agent_ent_std = self.get_agent_entropy()
        coh_mean, coh_std = self.get_coherence()

        self.entropy_history.append(entropy)
        self.magnet_history.append(magnet)
        self.agent_entropy_history.append(agent_ent_mean)
        self.coherence_history.append(coh_mean)

        return entropy

    # ---------------------------------------------------------
    # TESTS
    # ---------------------------------------------------------

    def test_complex_norm(self):
        q, p = self._split_qp()
        psi = q + 1j * p
        norms = np.linalg.norm(psi, axis=1)
        return float(np.std(norms)), float(np.mean(norms))

    def test_born_rule(self):
        q, p = self._split_qp()
        probs = q**2 + p**2
        p_mean = probs.mean(axis=0)
        return float(abs(p_mean.sum() - 1.0))

    def debug_print_tests(self):
        std_norm, mean_norm = self.test_complex_norm()
        born_err = self.test_born_rule()
        print(
            f"[TEST] cycle={self.cycle} | "
            f"norm_mean={mean_norm:.6f}, norm_std={std_norm:.6e}, "
            f"born_err={born_err:.6e}"
        )


# ---------------------------------------------------------
# OSCILLATION DIAGNOSTICS
# ---------------------------------------------------------

def autocorrelation(x, max_lag):
    """
    Simple normalized autocorrelation for a 1D signal x.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = np.dot(x, x)
    if var < 1e-18:
        return np.zeros(max_lag + 1)
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size // 2:]
    ac = ac[:max_lag + 1] / var
    return ac


# ---------------------------------------------------------
# PHASE DIAGRAM: SINGLE POINT
# ---------------------------------------------------------

def run_phase_point(
    local_theta,
    coupling_theta,
    measure_interval,
    emergence_rate,
    n_bits=128,
    pop_size=400,
    steps=1000,
    seed=42,
):
    u = RealQMUniverse(
        n_bits=n_bits,
        pop_size=pop_size,
        seed=seed,
        local_theta=local_theta,
        coupling_theta=coupling_theta,
        measure_interval=measure_interval,
        emergence_rate=emergence_rate,
    )

    entropies = []
    rule_counts = []
    avg_strengths = []
    max_strengths = []
    agent_ent_means = []
    coherence_means = []

    for _ in range(steps):
        ent = u.step()
        entropies.append(ent)
        rc, avg_s, max_s = u.get_rule_stats()
        rule_counts.append(rc)
        avg_strengths.append(avg_s)
        max_strengths.append(max_s)
        agent_ent_means.append(u.agent_entropy_history[-1])
        coherence_means.append(u.coherence_history[-1])

    entropies = np.array(entropies)
    rule_counts = np.array(rule_counts)
    avg_strengths = np.array(avg_strengths)
    max_strengths = np.array(max_strengths)
    agent_ent_means = np.array(agent_ent_means)
    coherence_means = np.array(coherence_means)

    max_lag = 100
    ac = autocorrelation(entropies, max_lag=max_lag)
    osc_score = float(np.max(np.abs(ac[1:])))

    result = {
        "local_theta": local_theta,
        "coupling_theta": coupling_theta,
        "measure_interval": measure_interval,
        "emergence_rate": emergence_rate,
        "entropy_mean": float(entropies.mean()),
        "entropy_std": float(entropies.std()),
        "entropy_min": float(entropies.min()),
        "entropy_last": float(entropies[-1]),
        "rules_mean": float(rule_counts.mean()),
        "rules_max": float(rule_counts.max()),
        "rules_last": int(rule_counts[-1]),
        "avg_strength_mean": float(avg_strengths.mean()),
        "avg_strength_max": float(avg_strengths.max()),
        "max_strength_mean": float(max_strengths.mean()),
        "max_strength_max": float(max_strengths.max()),
        "agent_entropy_mean": float(agent_ent_means.mean()),
        "agent_entropy_std": float(agent_ent_means.std()),
        "coherence_mean": float(coherence_means.mean()),
        "coherence_std": float(coherence_means.std()),
        "entropy_osc_score": osc_score,
    }
    return result


# ---------------------------------------------------------
# PARALLEL PHASE SWEEP
# ---------------------------------------------------------

def run_phase_point_wrapper(args):
    return run_phase_point(*args)


def sweep_phase_diagram_parallel():
    local_thetas = [0.04, 0.08, 0.12]
    coupling_thetas = [0.03, 0.07, 0.12]
    measure_intervals = [10, 25, 50]
    emergence_rates = [0.02, 0.05, 0.10]

    n_bits = 128
    pop_size = 400
    steps = 1000
    seed = 123

    jobs = []
    for lt in local_thetas:
        for ct in coupling_thetas:
            for mi in measure_intervals:
                for er in emergence_rates:
                    jobs.append((
                        lt, ct, mi, er,
                        n_bits, pop_size, steps, seed
                    ))

    output_file = "phase_diagram_results_stream.jsonl"
    print("Streaming results to", output_file)
    print("lt   ct    meas  emerg | "
          "S_mean  S_std   S_min   S_last | "
          "rules_mean rules_max rules_last | "
          "agentS_mean coh_mean osc_score")

    start = time.time()

    with open(output_file, "w") as f:
        num_workers = 2  # safe on Windows
        with mp.Pool(num_workers) as pool:

            for res in pool.imap_unordered(run_phase_point_wrapper, jobs):
                print(
                    f"{res['local_theta']:4.2f} {res['coupling_theta']:4.2f} "
                    f"{res['measure_interval']:4d} {res['emergence_rate']:5.2f} | "
                    f"{res['entropy_mean']:.4f} {res['entropy_std']:.4f} "
                    f"{res['entropy_min']:.4f} {res['entropy_last']:.4f} | "
                    f"{res['rules_mean']:.2f} {res['rules_max']:.0f} {res['rules_last']:3d} | "
                    f"{res['agent_entropy_mean']:.4f} {res['coherence_mean']:.4f} "
                    f"{res['entropy_osc_score']:.4f}"
                )
                f.write(json.dumps(res) + "\n")
                f.flush()

    elapsed = (time.time() - start) / 60.0
    print(f"\n[SWEEP DONE] elapsed {elapsed:.1f} minutes")
    print("Results saved to", output_file)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    # Option 1: run the parallel phase sweep (recommended now)
    sweep_phase_diagram_parallel()

    # Option 2: interactive single-universe mode (uncomment to use)
    """
    u = RealQMUniverse()
    u.load_state()

    def input_thread():
        while u.running:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            if cmd == "s":
                u.save_state()
            elif cmd == "q":
                u.save_state()
                u.running = False
            elif cmd == "t":
                u.debug_print_tests()

    threading.Thread(target=input_thread, daemon=True).start()

    print(f"{'Cycle':<8} | {'Entropy':<8} | {'Best':<8} | {'Rules'}")
    best = u.record_low

    while u.running:
        ent = u.step()
        if ent < best:
            best = ent
            u.record_low = best
            print(f"{u.cycle:<8} | {ent:.4f}* | {best:.4f} | {len(u.couplings)}")
        if u.cycle % 1000 == 0:
            rules, avg_s, max_s = u.get_rule_stats()
            elapsed = (time.time() - u.start_time) / 60.0
            print(
                f"{u.cycle:<8} | {ent:.4f}  | {best:.4f} | "
                f"{rules} (avgS={avg_s:.3f}, maxS={max_s:.3f}, T+{elapsed:.1f}m)"
            )
            u.debug_print_tests()
    """
