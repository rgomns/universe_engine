import numpy as np
import networkx as nx
import time
import json
import os
import threading

class Persistent_AGI_Universe_256:
    def __init__(self, n_bits=256, pop_size=2500):
        self.n_bits = n_bits
        self.pop_size = pop_size

        # Initializing a completely random 256-bit matrix
        self.population = np.random.randint(0, 2, (pop_size, n_bits), dtype=np.int8)
        self.couplings = []  # [scope, forbidden, strength]
        self.cycle = 0
        self.record_low = 1.0

        # Baseline noise; can be adjusted live or made entropy-dependent inside step()
        self.noise_level = 0.01

        # Slightly denser sampling for 256 bits (affects Hebbian estimates)
        self.sampling_step = 40 
        
        # Binary weights for Size-2 and Size-3 rules
        self.weights = {
            2: np.array([2, 1], dtype=np.int8), 
            3: np.array([4, 2, 1], dtype=np.int8)
        }

        self.start_time = time.time()
        self.running = True

    def get_entropy(self):
        p = np.mean(self.population, axis=0)
        # Shannon Entropy calculation across all 256 bits
        return np.mean(-p * np.log2(p + 1e-9) - (1 - p) * np.log2(1 - p + 1e-9))

    def get_skeleton_metrics(self):
        G = nx.Graph()
        for scope, _, _ in self.couplings:
            for i in range(len(scope)):
                for j in range(i + 1, len(scope)):
                    G.add_edge(scope[i], scope[j])
        if not G.nodes:
            return 0
        # Returns the size of the Largest Connected Component (LCC)
        return len(max(nx.connected_components(G), key=len))

    def save_state(self, filename="universe_256_clean.json"):
        serializable_couplings = [[s.tolist(), int(f), float(st)] for s, f, st in self.couplings]
        data = {
            "cycle": self.cycle,
            "bits": self.n_bits,
            "population": self.population.tolist(),
            "couplings": serializable_couplings,
            "record_low": self.record_low,
            "noise_level": self.noise_level
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"\n[SAVE] 256-bit state saved at cycle {self.cycle}")

    def load_state(self, filename="universe_256_clean.json"):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            self.cycle = data["cycle"]
            self.population = np.array(data["population"], dtype=np.int8)
            self.couplings = [[np.array(s), f, st] for s, f, st in data["couplings"]]
            self.record_low = data["record_low"]
            self.noise_level = data.get("noise_level", self.noise_level)
            print(f"\n[LOAD] 256-bit Universe Restored! Record: {self.record_low:.3f}")
        else:
            print("\n[SYSTEM] No 256-bit save found. Initializing Clean Slate.")

    def step(self):
        self.cycle += 1
        entropy = self.get_entropy()

        # Optional: entropy-dependent noise (uncomment if you want adaptive noise)
        # self.noise_level = max(0.004, 0.02 * entropy)

        # 1. EMERGENCE (Rule Discovery)
        # Probability of discovery scales with current entropy (slowed for 256 bits)
        if np.random.random() < (0.03 * (entropy + 0.5)):
            size = 3 if np.random.random() < 0.3 else 2
            # Choosing from 256 available bits
            scope = np.random.choice(self.n_bits, size, replace=False)
            sample = self.population[np.random.randint(0, self.pop_size, 20)][:, scope]
            indices = sample.dot(self.weights[size])
            forbidden = np.argmin(np.bincount(indices, minlength=2**size))
            self.couplings.append([scope, forbidden, 0.6])

        # 2. VECTOR RELAXATION (Physics Engine)
        for c in self.couplings:
            scope, forbidden, strength = c
            current_vals = self.population[:, scope].dot(self.weights[len(scope)])
            violators = (current_vals == forbidden)
            if violators.any():
                mask = (np.random.random(self.pop_size) < strength) & violators
                if mask.any():
                    # Apply a bit-flip to one of the indices in the rule's scope
                    self.population[mask, np.random.choice(scope)] ^= 1

        # 3. DYNAMIC HEBBIAN CHECK (Rule Selection/Death)
        next_gen = []
        sample_pop = self.population[::self.sampling_step]
        for c in self.couplings:
            scope, forbidden, strength = c
            c[2] *= 0.95  # Entropy-driven decay
            sample_vals = sample_pop[:, scope].dot(self.weights[len(scope)])
            # If the rule is obeyed by 95% of the sample, it hardens
            if np.mean(sample_vals != forbidden) > 0.95:
                c[2] += 0.06
            if c[2] > 0.15:  # Survival threshold
                next_gen.append(c)
        self.couplings = next_gen

        # 4. EXTERNAL DRIVE & NOISE
        # Periodic pulse on the first 4 bits to drive system activity
        self.population[:, 0:4] = (self.cycle // 20) % 2
        noise_mask = np.random.random(self.population.shape) < self.noise_level
        self.population[noise_mask] ^= 1


# --- LIVE CONTROLLER THREAD ---
u = Persistent_AGI_Universe_256()
u.load_state()

def input_thread():
    while u.running:
        cmd = input().lower()
        if cmd == 's':
            u.save_state()
        elif cmd == 'n':
            try:
                new_n = float(input("New Noise (e.g. 0.008): "))
                u.noise_level = new_n
            except:
                pass
        elif cmd == 'q':
            u.save_state()
            u.running = False

threading.Thread(target=input_thread, daemon=True).start()

# --- MAIN LOOP ---
print(f"{'Cycle':<8} | {'Entropy':<8} | {'Best':<8} | {'LCC':<5} | {'Phase'}")
while u.running:
    u.step()
    current_ent = u.get_entropy()
    
    if current_ent < u.record_low:
        u.record_low = current_ent
        lcc = u.get_skeleton_metrics()
        print(f"{u.cycle:<8} | {current_ent:.3f}* | {u.record_low:.3f} | {lcc:<5} | NEW RECORD")

    if u.cycle % 1000 == 0:  # Periodic updates for 256-bit monitoring
        lcc = u.get_skeleton_metrics()
        phase = "CRUNCH" if current_ent < 0.4 else "SUPERFLUID"
        elapsed = (time.time() - u.start_time) / 60
        print(f"{u.cycle:<8} | {current_ent:.3f}    | {u.record_low:.3f} | {lcc:<5} | {phase} (T+{elapsed:.1f}m)")
