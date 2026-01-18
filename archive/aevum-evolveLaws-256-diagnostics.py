import numpy as np
import networkx as nx
import time
import json
import os
import threading
import csv

from aevum_diagnostics import run_diagnostics


class Persistent_AGI_Universe_256:
    def __init__(self, n_bits=256, pop_size=2500):
        self.n_bits = n_bits
        self.pop_size = pop_size

        # Initializing a completely random 256-bit matrix
        self.population = np.random.randint(0, 2, (pop_size, n_bits), dtype=np.int8)
        self.couplings = []  # [scope, forbidden, strength]
        self.cycle = 0
        self.record_low = 1.0

        # Baseline noise
        self.noise_level = 0.00

        # Sampling step for Hebbian evaluation
        self.sampling_step = 40

        # Binary weights for Size-2 and Size-3 rules
        self.weights = {
            2: np.array([2, 1], dtype=np.int8),
            3: np.array([4, 2, 1], dtype=np.int8)
        }

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

    def get_skeleton_metrics(self):
        """
        Return size of largest connected component only.
        Used for fast per-record logging.
        """
        G = nx.Graph()
        for scope, _, _ in self.couplings:
            for i in range(len(scope)):
                for j in range(i + 1, len(scope)):
                    G.add_edge(scope[i], scope[j])
        if not G.nodes:
            return 0
        return len(max(nx.connected_components(G), key=len))

    def get_skeleton_stats(self):
        """
        Return (lcc_size, avg_degree) for skeleton.
        Used for periodic detailed logging and CSV.
        """
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
        """
        Return (rule_count, avg_strength, max_strength).
        """
        if not self.couplings:
            return 0, 0.0, 0.0
        strengths = np.array([c[2] for c in self.couplings], dtype=float)
        return len(strengths), float(strengths.mean()), float(strengths.max())

    # ---------------------------------------------------------
    # DIAGNOSTICS ADAPTERS
    # ---------------------------------------------------------

    def get_record_snapshot(self):
        """
        Produce a minimal Aevum-compatible record for diagnostics.
        """
        entropy = self.get_entropy()
        lcc, avg_deg = self.get_skeleton_stats()
        rule_count, avg_strength, max_strength = self.get_rule_stats()

        return {
            "cycle": self.cycle,
            "entropy": entropy,
            "num_constraints": rule_count,
            "avg_degree": avg_deg,
            "pattern_entropy": {3: 0.0, 4: 0.0},
        }

    def get_lineage_snapshot(self):
        """
        Convert couplings into Aevum-style lineage entries.
        """
        lineage = []
        for scope, forbidden, strength in self.couplings:
            lineage.append({
                "scope": list(scope),
                "final_strength": float(strength),
                "lifespan": 999999,
            })
        return lineage

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
            "noise_level": self.noise_level,
            "sampling_step": self.sampling_step,
            "start_time": self.start_time,
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
            self.population = np.array(data["population"], dtype=np.int8)
            self.pop_size = data.get("pop_size", self.population.shape[0])

            self.couplings = [
                [np.array(s, dtype=int), f, st]
                for s, f, st in data["couplings"]
            ]

            self.record_low = data.get("record_low", 1.0)
            self.noise_level = data.get("noise_level", self.noise_level)
            self.sampling_step = data.get("sampling_step", self.sampling_step)
            self.start_time = data.get("start_time", time.time())

            print(
                f"\n[LOAD] 256-bit Universe Restored!"
                f" cycle={self.cycle}, pop={self.pop_size}, rules={len(self.couplings)}, "
                f"best={self.record_low:.3f}"
            )
        else:
            print("\n[SYSTEM] No 256-bit save found. Initializing Clean Slate.")

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------

    def step(self):
        self.cycle += 1
        entropy = self.get_entropy()

        # 1. EMERGENCE (Rule Discovery)
        if np.random.random() < (0.03 * (entropy + 0.5)):
            size = 3 if np.random.random() < 0.3 else 2
            scope = np.random.choice(self.n_bits, size, replace=False)
            sample = self.population[
                np.random.randint(0, self.pop_size, 20)
            ][:, scope]
            indices = sample.dot(self.weights[size])
            forbidden = np.argmin(
                np.bincount(indices, minlength=2**size)
            )
            self.couplings.append([scope, forbidden, 0.6])

        # 2. VECTOR RELAXATION
        for c in self.couplings:
            scope, forbidden, strength = c
            current_vals = self.population[:, scope].dot(
                self.weights[len(scope)]
            )
            violators = (current_vals == forbidden)
            if violators.any():
                mask = (
                    (np.random.random(self.pop_size) < strength)
                    & violators
                )
                if mask.any():
                    self.population[mask, np.random.choice(scope)] ^= 1

        # 3. HEBBIAN SELECTION
        next_gen = []
        sample_pop = self.population[::self.sampling_step]
        for c in self.couplings:
            scope, forbidden, strength = c
            # Decay
            c[2] *= 0.95
            # Reinforcement
            sample_vals = sample_pop[:, scope].dot(
                self.weights[len(scope)]
            )
            if np.mean(sample_vals != forbidden) > 0.95:
                c[2] += 0.06
            # Survival threshold
            if c[2] > 0.15:
                next_gen.append(c)
        self.couplings = next_gen

        # 4. EXTERNAL DRIVE & NOISE
        self.population[:, 0:4] = (self.cycle // 20) % 2
        noise_mask = np.random.random(self.population.shape) < self.noise_level
        self.population[noise_mask] ^= 1


# ---------------------------------------------------------
# MAIN LOOP + LOGGING
# ---------------------------------------------------------

u = Persistent_AGI_Universe_256()
u.load_state()

# CSV LOGGING SETUP
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
        elif cmd == 'n':
            try:
                new_n = float(input("New Noise (e.g. 0.008): "))
                u.noise_level = new_n
                print(f"[NOISE] Updated noise_level to {u.noise_level}")
            except Exception:
                print("[NOISE] Invalid noise value.")
        elif cmd == 'q':
            u.save_state()
            u.running = False


threading.Thread(target=input_thread, daemon=True).start()

print(f"{'Cycle':<8} | {'Entropy':<8} | {'Best':<8} | {'LCC':<5} | {'Phase'}")

while u.running:
    u.step()
    current_ent = u.get_entropy()

    # New record detection
    if current_ent < u.record_low:
        u.record_low = current_ent
        lcc = u.get_skeleton_metrics()
        print(
            f"{u.cycle:<8} | {current_ent:.3f}* | {u.record_low:.3f} | "
            f"{lcc:<5} | NEW RECORD"
        )

    # Periodic detailed logging + CSV + diagnostics
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

        # ---------------------------------------------------------
        # DIAGNOSTICS
        # ---------------------------------------------------------
        record = u.get_record_snapshot()
        lineage = u.get_lineage_snapshot()

        diagnostics = run_diagnostics([record], lineage, n_bits=u.n_bits)

        diag_path = f"diagnostics_cycle_{u.cycle}.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)

        print(f"[DIAGNOSTICS] Saved diagnostics at cycle {u.cycle} → {diag_path}")
