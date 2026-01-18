import numpy as np
import networkx as nx
import time
import json
import os
import threading
import csv


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
        self.noise_level = 0.01

        # Sampling step for Hebbian evaluation
        self.sampling_step = 40

        # Binary weights for Size-2 and Size-3 rules
        self.weights = {
            2: np.array([2, 1], dtype=np.int8),
            3: np.array([4, 2, 1], dtype=np.int8)
        }

        self.start_time = time.time()
        self.running = True

        # --- Quantum analogy knobs ---------------------------------
        # "Temperature" for soft collapse to low-energy states
        self.collapse_temperature = 0.5

        # Fraction of population to replace during measurement
        self.collapse_fraction = 0.8

        # Minimum strength for a rule to contribute significantly to energy
        self.energy_strength_threshold = 0.05

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
    # ENERGY / POINTER STATES (QUANTUM ANALOGY)
    # ---------------------------------------------------------

    def compute_energies(self):
        """
        Interpret constraints as an energy functional.

        - Each violated constraint contributes positive energy.
        - Weight by constraint strength.
        - This defines a landscape where low energy = pointer states.
        """
        if not self.couplings:
            # No constraints => flat energy landscape
            return np.zeros(self.pop_size, dtype=float)

        energies = np.zeros(self.pop_size, dtype=float)

        for scope, forbidden, strength in self.couplings:
            if strength < self.energy_strength_threshold:
                # Very weak rules barely contribute to effective "Hamiltonian"
                continue

            vals = self.population[:, scope].dot(self.weights[len(scope)])
            violators = (vals == forbidden)

            # Add energy penalty to violators
            # You can tune the factor 1.0 if you want sharper landscapes
            energies[violators] += strength

        return energies

    def get_pointer_states(self, top_k=10):
        """
        Pointer states ~ lowest-energy attractors under current constraint set.
        Returns:
            unique_states: array of shape (M, n_bits) with unique low-energy states
            counts: how often each appears in the population
            energies: energy of each unique state
        """
        energies = self.compute_energies()
        # Sort individuals by energy
        idx_sorted = np.argsort(energies)
        low_indices = idx_sorted[:max(top_k, 1)]

        low_states = self.population[low_indices]
        low_energies = energies[low_indices]

        # Deduplicate low-energy states
        # Represent states as bytes for hashing
        state_bytes = [s.tobytes() for s in low_states]
        unique_map = {}
        for sb, e in zip(state_bytes, low_energies):
            if sb not in unique_map:
                unique_map[sb] = {
                    "state": None,
                    "count": 0,
                    "energy": e
                }
            unique_map[sb]["count"] += 1
            unique_map[sb]["energy"] = min(unique_map[sb]["energy"], e)

        unique_states = []
        counts = []
        energies_out = []
        for sb, info in unique_map.items():
            s = np.frombuffer(sb, dtype=np.int8)
            unique_states.append(s)
            counts.append(info["count"])
            energies_out.append(info["energy"])

        unique_states = np.array(unique_states, dtype=np.int8)
        counts = np.array(counts, dtype=int)
        energies_out = np.array(energies_out, dtype=float)

        # Sort by energy, then by count
        order = np.lexsort((-counts, energies_out))
        return unique_states[order], counts[order], energies_out[order]

    def measure(self):
        """
        "Measurement" operator (quantum analogy):

        - Sample current low-energy pointer states from the population.
        - Build a Boltzmann-like distribution over them (Born-rule analog).
        - Resample a large fraction of the population from this distribution.
        - This is an explicit collapse of the wavefunction (population distribution)
          into pointer states defined by the constraint landscape.
        """
        if self.pop_size == 0:
            return

        entropy_before = self.get_entropy()
        energies = self.compute_energies()

        pointer_states, counts, ptr_energies = self.get_pointer_states(top_k=20)

        if pointer_states.size == 0:
            print("[MEAS] No pointer states (no constraints). Skipping measurement.")
            return

        # Convert energies to probabilities ~ exp(-E/T)
        T = max(self.collapse_temperature, 1e-6)
        weights = np.exp(-ptr_energies / T)
        probs = weights / (weights.sum() + 1e-12)

        # How many individuals to "collapse" / overwrite
        num_collapse = int(self.pop_size * self.collapse_fraction)
        collapse_indices = np.random.choice(self.pop_size, size=num_collapse, replace=False)
        pointer_indices = np.random.choice(len(pointer_states), size=num_collapse, p=probs)

        # Perform collapse: reinitialize chosen individuals to sampled pointer states
        self.population[collapse_indices] = pointer_states[pointer_indices]

        entropy_after = self.get_entropy()

        # Some logging to see the "collapse" effect
        dominant_state = pointer_states[0]
        dominant_energy = ptr_energies[0]
        dominant_prob = probs[0]

        print(
            f"[MEAS] cycle={self.cycle} | "
            f"entropy_before={entropy_before:.3f} -> entropy_after={entropy_after:.3f} | "
            f"dominant_energy={dominant_energy:.3f}, dominant_prob={dominant_prob:.3f}, "
            f"num_pointer_states={len(pointer_states)}"
        )

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
            # Save quantum-analogy knobs too
            "collapse_temperature": self.collapse_temperature,
            "collapse_fraction": self.collapse_fraction,
            "energy_strength_threshold": self.energy_strength_threshold,
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

            # Restore quantum-analogy knobs if present
            self.collapse_temperature = data.get("collapse_temperature", self.collapse_temperature)
            self.collapse_fraction = data.get("collapse_fraction", self.collapse_fraction)
            self.energy_strength_threshold = data.get("energy_strength_threshold", self.energy_strength_threshold)

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

        # 2. VECTOR RELAXATION (constraint-enforced dynamics)
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

        # 3. HEBBIAN SELECTION (your "measurement" process)
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

        # NOTE: explicit "measurement" (collapse) is now done externally
        # by calling self.measure(), either periodically or via user input.


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
        "max_rule_strength",
        "avg_energy"  # new column: average energy of the wavefunction
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
        elif cmd == 'm':
            # Manual measurement: explicit collapse event
            u.measure()
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

    # Periodic detailed logging + CSV
    if u.cycle % 1000 == 0:
        lcc, avg_deg = u.get_skeleton_stats()
        rule_count, avg_strength, max_strength = u.get_rule_stats()
        phase = "CRUNCH" if current_ent < 0.4 else "SUPERFLUID"

        energies = u.compute_energies()
        avg_energy = float(energies.mean()) if len(energies) > 0 else 0.0

        elapsed = (time.time() - u.start_time) / 60

        print(
            f"{u.cycle:<8} | {current_ent:.3f}    | {u.record_low:.3f} | "
            f"{lcc:<5} | {phase} (T+{elapsed:.1f}m, rules={rule_count}, "
            f"avgS={avg_strength:.3f}, maxS={max_strength:.3f}, "
            f"deg={avg_deg:.2f}, avgE={avg_energy:.3f})"
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
            max_strength,
            avg_energy
        ])
        log_file.flush()
