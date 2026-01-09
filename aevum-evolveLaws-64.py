import numpy as np
import networkx as nx
import time
import json
import os
import threading

class Persistent_AGI_Universe:
    def __init__(self, n_bits=64, pop_size=600):
        self.n_bits = n_bits
        self.pop_size = pop_size
        self.population = np.random.randint(0, 2, (pop_size, n_bits), dtype=np.int8)
        self.couplings = [] # [scope, forbidden, strength]
        self.cycle = 0
        self.record_low = 1.0
        self.noise_level = 0.00 # Live adjustable
        self.sampling_step = 50 # Live adjustable
        self.weights = {2: np.array([2, 1], dtype=np.int8), 
                        3: np.array([4, 2, 1], dtype=np.int8)}
        self.start_time = time.time()
        self.running = True

    def get_entropy(self):
        p = np.mean(self.population, axis=0)
        return np.mean(-p * np.log2(p + 1e-9) - (1-p) * np.log2(1-p + 1e-9))

    def get_skeleton_metrics(self):
        G = nx.Graph()
        for scope, _, _ in self.couplings:
            for i in range(len(scope)):
                for j in range(i + 1, len(scope)):
                    G.add_edge(scope[i], scope[j])
        if not G.nodes: return 0
        return len(max(nx.connected_components(G), key=len))

    def save_state(self, filename="universe_save.json"):
        # Convert couplings (lists/arrays) to serializable format
        serializable_couplings = [[s.tolist(), int(f), float(st)] for s, f, st in self.couplings]
        data = {
            "cycle": self.cycle,
            "population": self.population.tolist(),
            "couplings": serializable_couplings,
            "record_low": self.record_low,
            "noise_level": self.noise_level
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"\n[SAVE] Universe state locked at cycle {self.cycle}")

    def load_state(self, filename="universe_save.json"):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            self.cycle = data["cycle"]
            self.population = np.array(data["population"], dtype=np.int8)
            self.couplings = [[np.array(s), f, st] for s, f, st in data["couplings"]]
            self.record_low = data["record_low"]
            self.noise_level = data.get("noise_level", 0.02)
            print(f"\n[LOAD] Universe Restored! Record: {self.record_low:.3f}")
        else:
            print("\n[SYSTEM] No save found. Starting fresh universe.")

    def step(self):
        self.cycle += 1
        entropy = self.get_entropy()
        
        # 1. EMERGENCE
        if np.random.random() < (0.08 * (entropy + 0.5)):
            size = 3 if np.random.random() < 0.3 else 2
            scope = np.random.choice(self.n_bits, size, replace=False)
            sample = self.population[np.random.randint(0, self.pop_size, 20)][:, scope]
            indices = sample.dot(self.weights[size])
            forbidden = np.argmin(np.bincount(indices, minlength=2**size))
            self.couplings.append([scope, forbidden, 0.6])

        # 2. ULTRA-FAST VECTOR RELAXATION
        for c in self.couplings:
            scope, forbidden, strength = c
            current_vals = self.population[:, scope].dot(self.weights[len(scope)])
            violators = (current_vals == forbidden)
            if violators.any():
                mask = (np.random.random(self.pop_size) < strength) & violators
                if mask.any():
                    self.population[mask, np.random.choice(scope)] ^= 1

        # 3. DYNAMIC HEBBIAN CHECK
        next_gen = []
        sample_pop = self.population[::self.sampling_step] 
        for c in self.couplings:
            scope, forbidden, strength = c
            c[2] *= 0.95 # Decay
            sample_vals = sample_pop[:, scope].dot(self.weights[len(scope)])
            if np.mean(sample_vals != forbidden) > 0.95:
                c[2] += 0.06 # Hardening
            if c[2] > 0.15:
                next_gen.append(c)
        self.couplings = next_gen

        # 4. NOISE & DRIVE
        self.population[:, 0:4] = (self.cycle // 20) % 2
        noise_mask = np.random.random(self.population.shape) < self.noise_level
        self.population[noise_mask] ^= 1

# --- LIVE CONTROLLER THREAD ---
u = Persistent_AGI_Universe()
u.load_state()

def input_thread():
    while u.running:
        cmd = input().lower()
        if cmd == 's': u.save_state()
        elif cmd == 'n':
            try:
                new_n = float(input("New Noise (e.g. 0.05): "))
                u.noise_level = new_n
            except: pass
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

    if u.cycle % 5000 == 0:
        lcc = u.get_skeleton_metrics()
        phase = "CRUNCH" if current_ent < 0.2 else "SUPERFLUID"
        elapsed = (time.time() - u.start_time) / 60
        print(f"{u.cycle:<8} | {current_ent:.3f}    | {u.record_low:.3f} | {lcc:<5} | {phase} (T+{elapsed:.1f}m)")