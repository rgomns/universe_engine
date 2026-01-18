import numpy as np
import networkx as nx
import time
import json
import os
import threading
import csv


class Persistent_AGI_Universe_256:
    def __init__(self, n_bits=256, pop_size=2500, seed=42):
        self.n_bits = n_bits
        self.pop_size = pop_size

        rng = np.random.default_rng(seed)

        # Initial random 256-bit population
        self.population = rng.integers(0, 2, (pop_size, n_bits), dtype=np.uint8)

        # Couplings: list of [scope (np.array), forbidden (int), strength (float)]
        self.couplings = []
        self.cycle = 0
        self.record_low = 1.0

        # Sampling step for Hebbian evaluation
        self.sampling_step = 40

        # Binary weights for Size-2 and Size-3 rules
        self.weights = {
            2: np.array([2, 1], dtype=np.int8),
            3: np.array([4, 2, 1], dtype=np.int8)
        }

        # Reversible XOR-based micro-update parameters
        # Slightly weaker mixing (more local)
        self.shifts = [1, 5, 17, 41]

        # Emergence: constant, mid-strength rate
        self.emergence_rate = 0.03  # ~3% chance per cycle

        # Hebbian / rule parameters (mid between crunch and superfluid, slightly more stable)
        self.initial_rule_strength = 0.35   # softened from 0.4
        self.hebb_decay = 0.995            # slower decay (more stable rules)
        self.hebb_reinforce = 0.02        # gentle reinforcement
        self.hebb_survival = 0.10          # moderate survival threshold
        self.hebb_success_thresh = 0.92    # fairly strict success
        self.max_rule_strength = 0.65       # cap so rules don't become absolute clamps

        # Global tiny temperature
        self.base_noise_prob = 1e-4

        # Diagnostics
        self.entropy_history = []
        self.magnet_subset = np.arange(min(16, self.n_bits))
        self.magnet_history = []

        # Local patches for coherence
        self.local_patches = [
            np.arange(0, 32),
            np.arange(32, 64),
            np.arange(64, 96),
            np.arange(96, 128),
            np.arange(128, 160),
            np.arange(160, 192),
            np.arange(192, 224),
            np.arange(224, 256),
        ]
        self.local_mag_history = [[] for _ in self.local_patches]

        self.start_time = time.time()
        self.running = True

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    def get_entropy(self):
        p = np.mean(self.population, axis=0)
        return np.mean(
            -p * np.log2(p + 1e-9) - (1 - p) * np.log2(1 - p + 1e-9)
        )

    def get_magnetization(self):
        return int(self.population[:, self.magnet_subset].sum())

    def get_local_magnetizations(self):
        mags = []
        for patch in self.local_patches:
            mags.append(int(self.population[:, patch].sum()))
        return mags

    def get_skeleton_metrics(self):
        G = nx.Graph()
        for scope, _, _ in self.couplings:
            for i in range(len(scope)):
                for j in range(i + 1, len(scope)):
                    G.add_edge(scope[i], scope[j])
        if not G.nodes:
            return 0
        return len(max(nx.connected_components(G), key=len))

    def get_skeleton_stats(self):
        G = nx.Graph()
        for scope, _, _ in self.couplings:
            for i in range(len(scope)):
                for j in range(i + 1, len(scope)):
                    G.add_edge(scope[i], scope[j])
        if not G.nodes:
            return 0, 0.0
        lcc_size = len(max(nx.connected_components(G), key=len))
        avg_degree = float(np.mean([deg for _, deg in G.degree()]))
        return lcc_size, avg_degree

    def get_rule_stats(self):
        if not self.couplings:
            return 0, 0.0, 0.0
        strengths = np.array([c[2] for c in self.couplings], dtype=float)
        return len(strengths), float(strengths.mean()), float(strengths.max())

    # ---------------------------------------------------------
    # SAVE / LOAD
    # ---------------------------------------------------------

    def save_state(self, filename="universe_256_clean.json"):
        serializable_couplings = [
            [s.tolist(), int(f), float(st)] for s, f, st in self.couplings
        ]
        data = {
            "cycle": self.cycle,
            "bits": self.n_bits,
            "pop_size": self.pop_size,
            "population": self.population.tolist(),
            "couplings": serializable_couplings,
            "record_low": self.record_low,
            "sampling_step": self.sampling_step,
            "start_time": self.start_time,
            "shifts": self.shifts,
            "emergence_rate": self.emergence_rate,
            "initial_rule_strength": self.initial_rule_strength,
            "hebb_decay": self.hebb_decay,
            "hebb_reinforce": self.hebb_reinforce,
            "hebb_survival": self.hebb_survival,
            "hebb_success_thresh": self.hebb_success_thresh,
            "max_rule_strength": self.max_rule_strength,
            "base_noise_prob": self.base_noise_prob,
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(
            f"\n[SAVE] 256-bit state saved at cycle {self.cycle} | "
            f"pop={self.pop_size}, rules={len(self.couplings)}, "
            f"best={self.record_low:.3f}"
        )

    def load_state(self, filename="universe_256_clean.json"):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)

            self.cycle = data["cycle"]
            self.n_bits = data.get("bits", self.n_bits)
            self.population = np.array(data["population"], dtype=np.uint8)
            self.pop_size = data.get("pop_size", self.population.shape[0])

            self.couplings = [
                [np.array(s, dtype=int), f, st]
                for s, f, st in data["couplings"]
            ]

            self.record_low = data.get("record_low", 1.0)
            self.sampling_step = data.get("sampling_step", self.sampling_step)
            self.start_time = data.get("start_time", time.time())

            self.shifts = data.get("shifts", [1, 5, 17, 41])
            self.emergence_rate = data.get("emergence_rate", 0.03)
            self.initial_rule_strength = data.get("initial_rule_strength", 0.35)
            self.hebb_decay = data.get("hebb_decay", 0.995)
            self.hebb_reinforce = data.get("hebb_reinforce", 0.02)
            self.hebb_survival = data.get("hebb_survival", 0.10)
            self.hebb_success_thresh = data.get("hebb_success_thresh", 0.92)
            self.max_rule_strength = data.get("max_rule_strength", 0.7)
            self.base_noise_prob = data.get("base_noise_prob", 1e-4)

            print(
                f"\n[LOAD] 256-bit Universe Restored!"
                f" cycle={self.cycle}, pop={self.pop_size}, rules={len(self.couplings)}, "
                f"best={self.record_low:.3f}"
            )
        else:
            print("\n[SYSTEM] No 256-bit save found. Initializing Clean Slate.")

    # ---------------------------------------------------------
    # PERIODICITY DETECTION (FFT-BASED)
    # ---------------------------------------------------------

    @staticmethod
    def detect_periodicity(time_series, min_idx=3, threshold=0.12):
        ts = np.array(time_series, dtype=float)
        if len(ts) < 64:
            return {"periodic": False, "dominant_period": None}

        ts = ts - np.mean(ts)
        fft_vals = np.fft.rfft(ts)
        power = np.abs(fft_vals) ** 2

        power[:min_idx] = 0.0

        total_power = np.sum(power)
        if total_power == 0:
            return {"periodic": False, "dominant_period": None}

        dominant_idx = np.argmax(power)
        dominant_power = power[dominant_idx] / total_power

        if dominant_power > threshold:
            period = len(ts) / dominant_idx
            return {"periodic": True, "dominant_period": period}
        else:
            return {"periodic": False, "dominant_period": None}

    # ---------------------------------------------------------
    # TRUE PERIOD DETECTION (RECURRENCE-BASED)
    # ---------------------------------------------------------

    def detect_true_period(self, window=8000, min_lag=500, max_lag=6000, tol=1e-3):
        """
        Detect true recurrence-based period using entropy + magnetization.
        Returns {'periodic': bool, 'period': float or None}.
        """
        n = len(self.entropy_history)
        if n < window + max_lag:
            return {"periodic": False, "period": None}

        ent = np.array(self.entropy_history[-window:], dtype=float)
        mag = np.array(self.magnet_history[-window:], dtype=float)

        ent = (ent - ent.mean()) / (ent.std() + 1e-9)
        mag = (mag - mag.mean()) / (mag.std() + 1e-9)

        z = np.stack([ent, mag], axis=1)  # shape (window, 2)

        best_lag = None
        best_score = np.inf

        for lag in range(min_lag, max_lag):
            a = z[lag:]
            b = z[:-lag]
            d2 = np.mean(np.sum((a - b) ** 2, axis=1))
            if d2 < best_score:
                best_score = d2
                best_lag = lag

        if best_score < tol:
            return {"periodic": True, "period": float(best_lag)}
        else:
            return {"periodic": False, "period": None}

    # ---------------------------------------------------------
    # LOCAL VS GLOBAL COHERENCE
    # ---------------------------------------------------------

    def local_global_coherence(self, window=4000):
        """
        Returns correlation between local patches and global magnetization.
        """
        if len(self.magnet_history) < window:
            return None

        global_mag = np.array(self.magnet_history[-window:], dtype=float)
        global_mag = (global_mag - global_mag.mean()) / (global_mag.std() + 1e-9)

        corrs = []
        for patch_hist in self.local_mag_history:
            if len(patch_hist) < window:
                return None
            local = np.array(patch_hist[-window:], dtype=float)
            local = (local - local.mean()) / (local.std() + 1e-9)
            corr = float(np.mean(global_mag * local))
            corrs.append(corr)

        return corrs

    # ---------------------------------------------------------
    # SLOW MANIFOLD TRACKING
    # ---------------------------------------------------------

    def macrostate_vector(self):
        """
        4D macrostate: [entropy, global_mag, rule_count, avg_rule_strength]
        """
        ent = self.entropy_history[-1] if self.entropy_history else self.get_entropy()
        mag = self.magnet_history[-1] if self.magnet_history else self.get_magnetization()
        rule_count, avg_strength, _ = self.get_rule_stats()
        return np.array([ent, float(mag), float(rule_count), float(avg_strength)], dtype=float)

    def slow_manifold_velocity(self, window=2000):
        """
        Estimate average macro-velocity over a window.
        """
        if len(self.entropy_history) < window:
            return None

        k = max(1, window // 50)
        indices = range(len(self.entropy_history) - window, len(self.entropy_history), k)
        states = []
        for idx in indices:
            ent = self.entropy_history[idx]
            mag = self.magnet_history[idx]
            rule_count, avg_strength, _ = self.get_rule_stats()
            states.append([ent, float(mag), float(rule_count), float(avg_strength)])
        states = np.array(states, dtype=float)

        v = states[1:] - states[:-1]
        mean_v = v.mean(axis=0)
        return mean_v, states

    # ---------------------------------------------------------
    # CONSTRAINT MATRIX + ANTI-SYMMETRIC TWIST (DIAGNOSTIC)
    # ---------------------------------------------------------

    def build_constraint_matrix(self):
        M = np.zeros((self.n_bits, self.n_bits), dtype=float)
        for scope, _, strength in self.couplings:
            scope = np.array(scope, dtype=int)
            for i in scope:
                for j in scope:
                    if i != j:
                        M[i, j] += strength
        return M

    def complex_spectrum_diagnostic(self, epsilon=3e-3, max_dim=256):
        if self.n_bits > max_dim:
            n = max_dim
        else:
            n = self.n_bits

        M = self.build_constraint_matrix()
        M = M[:n, :n]

        rng = np.random.default_rng(123)
        R = rng.normal(scale=1.0, size=(n, n))
        A = R - R.T  # anti-symmetric

        M_tilde = M + epsilon * A

        eigvals = np.linalg.eigvals(M_tilde)
        num_complex = np.sum(np.abs(np.imag(eigvals)) > 1e-8)
        return {
            "num_complex": int(num_complex),
            "has_complex": bool(num_complex > 0),
            "eigvals": eigvals,
        }

    # ---------------------------------------------------------
    # REVERSIBLE XOR-BASED MICRO-UPDATE
    # ---------------------------------------------------------

    def reversible_xor_update(self):
        x = np.roll(self.population, self.shifts[0], axis=1)
        for s in self.shifts[1:]:
            x ^= np.roll(self.population, s, axis=1)
        self.population = x

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------

    def step(self):
        self.cycle += 1

        # 0. REVERSIBLE MICRO-UPDATE (XOR-based)
        self.reversible_xor_update()

        # 1. EMERGENCE (Rule Discovery, constant rate)
        if np.random.random() < self.emergence_rate:
            size = 3 if np.random.random() < 0.3 else 2
            scope = np.random.choice(self.n_bits, size, replace=False)
            sample = self.population[
                np.random.randint(0, self.pop_size, 20)
            ][:, scope]
            indices = sample.dot(self.weights[size])
            forbidden = np.argmin(
                np.bincount(indices, minlength=2**size)
            )
            self.couplings.append([scope, forbidden, self.initial_rule_strength])

        # 2. VECTOR RELAXATION (DISSIPATIVE, softened)
        for c in self.couplings:
            scope, forbidden, strength = c
            current_vals = self.population[:, scope].dot(
                self.weights[len(scope)]
            )
            violators = (current_vals == forbidden)
            if violators.any():
                flip_prob = 0.42 * strength  # softer than 0.50
                mask_candidates = np.where(violators)[0]
                if len(mask_candidates) > 0:
                    chosen_count = max(1, len(mask_candidates) // 6)  # softer than //4
                    chosen = np.random.choice(mask_candidates, size=chosen_count, replace=False)
                    flip_mask = np.random.random(chosen_count) < flip_prob
                    chosen = chosen[flip_mask]
                    if len(chosen) > 0:
                        bit = np.random.choice(scope)
                        self.population[chosen, bit] ^= 1

        # 3. HEBBIAN SELECTION (MID-RANGE, CAPPED)
        next_gen = []
        sample_pop = self.population[::self.sampling_step]
        for c in self.couplings:
            scope, forbidden, strength = c
            strength *= self.hebb_decay
            sample_vals = sample_pop[:, scope].dot(
                self.weights[len(scope)]
            )
            if np.mean(sample_vals != forbidden) > self.hebb_success_thresh:
                strength = min(strength + self.hebb_reinforce, self.max_rule_strength)
            c[2] = strength
            if strength > self.hebb_survival:
                next_gen.append(c)
        self.couplings = next_gen

        # 4. GLOBAL TINY NOISE
        noise_mask = np.random.random(self.population.shape) < self.base_noise_prob
        if np.any(noise_mask):
            self.population[noise_mask] ^= 1

        # 5. LOG DIAGNOSTICS
        entropy = self.get_entropy()
        magnet = self.get_magnetization()
        self.entropy_history.append(entropy)
        self.magnet_history.append(magnet)

        local_mags = self.get_local_magnetizations()
        for i, m in enumerate(local_mags):
            self.local_mag_history[i].append(m)

        return entropy


# ---------------------------------------------------------
# MAIN LOOP + LOGGING
# ---------------------------------------------------------

u = Persistent_AGI_Universe_256()
u.load_state()

log_filename = "universe_256_log.csv"
log_header_written = os.path.exists(log_filename)

log_file = open(log_filename, "a", newline="")
log_writer = csv.writer(log_file)

if not log_header_written:
    log_writer.writerow([
        "cycle",
        "entropy",
        "record_low",
        "lcc_size",
        "avg_degree",
        "phase",
        "rule_count",
        "avg_rule_strength",
        "max_rule_strength"
    ])


def input_thread():
    while u.running:
        cmd = input().lower()
        if cmd == 's':
            u.save_state()
        elif cmd == 'q':
            u.save_state()
            u.running = False


threading.Thread(target=input_thread, daemon=True).start()

print(f"{'Cycle':<8} | {'Entropy':<8} | {'Best':<8} | {'LCC':<5} | {'Phase'}")

while u.running:
    current_ent = u.step()

    # Control: freeze emergence at 8000 cycles to fix skeleton
    if u.cycle == 8000:
        u.emergence_rate = 0.0
        print("[CONTROL] Emergence frozen at t=8000; skeleton fixed.")

    if current_ent < u.record_low:
        u.record_low = current_ent
        lcc = u.get_skeleton_metrics()
        print(
            f"{u.cycle:<8} | {current_ent:.3f}* | {u.record_low:.3f} | "
            f"{lcc:<5} | NEW RECORD"
        )

    if u.cycle % 1000 == 0:
        lcc, avg_deg = u.get_skeleton_stats()
        rule_count, avg_strength, max_strength = u.get_rule_stats()
        phase = "CRUNCH" if current_ent < 0.4 else "SUPERFLUID"
        elapsed = (time.time() - u.start_time) / 60

        print(
            f"{u.cycle:<8} | {current_ent:.3f}    | {u.record_low:.3f} | "
            f"{lcc:<5} | {phase} (T+{elapsed:.1f}m, rules={rule_count}, "
            f"avgS={avg_strength:.3f}, maxS={max_strength:.3f}, deg={avg_deg:.2f})"
        )

        log_writer.writerow([
            u.cycle,
            current_ent,
            u.record_low,
            lcc,
            avg_deg,
            phase,
            rule_count,
            avg_strength,
            max_strength
        ])
        log_file.flush()

    if u.cycle % 10000 == 0:
        ent_diag = u.detect_periodicity(u.entropy_history)
        mag_diag = u.detect_periodicity(u.magnet_history)
        spec_diag = u.complex_spectrum_diagnostic(epsilon=3e-3, max_dim=128)
        true_period_diag = u.detect_true_period()
        coherence = u.local_global_coherence()
        slow = u.slow_manifold_velocity()

        print(
            f"[DIAG t={u.cycle}] "
            f"Entropy periodic={ent_diag['periodic']}, period≈{ent_diag['dominant_period']}; "
            f"Mag periodic={mag_diag['periodic']}, period≈{mag_diag['dominant_period']}; "
            f"Complex eigs={spec_diag['num_complex']}, has_complex={spec_diag['has_complex']}"
        )

        print(
            f"[TRUE-PERIOD t={u.cycle}] periodic={true_period_diag['periodic']}, "
            f"period≈{true_period_diag['period']}"
        )

        if coherence is not None:
            avg_coh = float(np.mean(coherence))
            print(
                f"[COHERENCE t={u.cycle}] avg_local_global_corr={avg_coh:.3f}, "
                f"per_patch={coherence}"
            )

        if slow is not None:
            mean_v, _ = slow
            print(f"[SLOW-MANIFOLD t={u.cycle}] mean_velocity={mean_v}")
