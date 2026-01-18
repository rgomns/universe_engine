import numpy as np
import time
import json
import os
import threading


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

        self.start_time = time.time()
        self.running = True

    # ---------------------------------------------------------
    # CORE UTILITIES
    # ---------------------------------------------------------

    def _split_qp(self):
        """Return (q, p) views of the population."""
        q = self.population[:, :self.n_bits]
        p = self.population[:, self.n_bits:]
        return q, p

    def _renormalize(self):
        """Normalize each agent's (q, p) to unit norm."""
        norms = np.linalg.norm(self.population, axis=1, keepdims=True) + 1e-12
        self.population /= norms

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    def get_entropy(self):
        """
        Ensemble bit-entropy from Born probabilities.

        p_i = ⟨q_i^2 + p_i^2⟩_agents
        Normalize so Σ_i p_i = 1, then:
            S = - Σ_i p_i log2 p_i / log2(N)
        so S ∈ [0, 1].
        """
        q, p = self._split_qp()
        probs = q**2 + p**2  # shape: (pop_size, n_bits)
        p_bits = probs.mean(axis=0)
        Z = p_bits.sum() + 1e-12
        p_bits /= Z
        entropy = -np.sum(p_bits * np.log2(p_bits + 1e-12))
        entropy /= np.log2(self.n_bits + 1e-12)
        return float(entropy)

    def get_magnetization(self):
        """
        Magnetization from q-signs (spin-like).
        """
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
        """
        Apply local (q_i, p_i) rotation:
            (q_i, p_i) -> (q_i cosθ - p_i sinθ, q_i sinθ + p_i cosθ)
        This is exactly multiplication by e^{iθ} in complex form.
        """
        q = self.population[:, :self.n_bits]
        p = self.population[:, self.n_bits:]
        c = np.cos(theta)
        s = np.sin(theta)

        q_new = c * q - s * p
        p_new = s * q + c * p

        self.population[:, :self.n_bits] = q_new
        self.population[:, self.n_bits:] = p_new

    def _coupling_rotation(self, theta, idx, jdx):
        """
        Couple bits via index pairs (idx, jdx): rotate (q_i, q_j) and (p_i, p_j)
        with the same 2x2 orthogonal matrix. This is J-compatible.
        """
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
        """
        Real-valued reversible evolution (SO(2N)), J-compatible.
        This is the real representation of a complex unitary.
        """
        self._local_qp_rotation(self.local_theta)
        for s in self.shifts:
            idx, jdx = self.coupling_pairs[int(s)]
            self._coupling_rotation(self.coupling_theta, idx, jdx)
        # no renormalization here: this is purely orthogonal

    # ---------------------------------------------------------
    # REAL CONSTRAINT POTENTIALS (ON q ONLY)
    # ---------------------------------------------------------

    def apply_constraints(self):
        """
        Constraints act on q (spin-like). They are not unitary;
        they implement an effective potential / projection.
        """
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
        """
        Decay, reinforce, and prune rules based on how often they
        successfully avoid forbidden spin configurations.
        """
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
        """
        Real Born-rule measurement on bits.

        For each agent and bit:
            p_i = q_i^2 + p_i^2
            collapse with probability p_i to a definite spin in q,
            set p_i -> 0 on collapse,
            add small noise.
        """
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
        """
        With some probability, create a new constraint rule on a random scope.
        """
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

        # 1. Purely unitary (orthogonal, J-compatible)
        self.real_unitary_micro_update()

        # 2. Emergent constraints (non-unitary)
        self.maybe_emerge_rule()
        self.apply_constraints()

        # 3. Hebbian survival / adaptation
        self.hebbian_update()

        # 4. Measurement (non-unitary)
        if self.cycle % self.measure_interval == 0:
            self.measure()

        # 5. Renormalize once after all non-unitary operations
        self._renormalize()

        # 6. Diagnostics
        entropy = self.get_entropy()
        magnet = self.get_magnetization()
        self.entropy_history.append(entropy)
        self.magnet_history.append(magnet)

        return entropy

    # ---------------------------------------------------------
    # REAL ↔ COMPLEX QM TESTS
    # ---------------------------------------------------------

    def test_complex_norm(self):
        """
        Test: reconstruct ψ = q + i p and check norm preservation.
        Should be ~1 for each agent if micro-dynamics are unitary.
        """
        q, p = self._split_qp()
        psi = q + 1j * p
        norms = np.linalg.norm(psi, axis=1)
        return float(np.std(norms)), float(np.mean(norms))

    def test_born_rule(self):
        """
        Test: sum of Born probabilities over bits should be ~1.
        """
        q, p = self._split_qp()
        probs = q**2 + p**2
        p_mean = probs.mean(axis=0)
        return float(abs(p_mean.sum() - 1.0))

    def test_real_inner_product_preservation(self, sample_size=100):
        """
        Test: real inner products <x_i, x_j> are preserved by the micro-update
        for a sample of agents. This is the correct invariant for an orthogonal map.
        """
        idx = self.rng.choice(self.pop_size, size=min(sample_size, self.pop_size), replace=False)
        x_before = self.population[idx].copy()

        pop_backup = self.population.copy()
        self.real_unitary_micro_update()
        x_after = self.population[idx].copy()
        self.population = pop_backup

        G_before = x_before @ x_before.T
        G_after = x_after @ x_after.T

        diff = np.linalg.norm(G_before - G_after)
        base = np.linalg.norm(G_before) + 1e-12
        return float(diff / base)

    def debug_print_tests(self):
        std_norm, mean_norm = self.test_complex_norm()
        born_err = self.test_born_rule()
        real_unit_rel_err = self.test_real_inner_product_preservation(sample_size=50)
        print(
            f"[TEST] cycle={self.cycle} | "
            f"norm_mean={mean_norm:.6f}, norm_std={std_norm:.6e}, "
            f"born_err={born_err:.6e}, real_unit_rel_err={real_unit_rel_err:.6e}"
        )


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

if __name__ == "__main__":
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
