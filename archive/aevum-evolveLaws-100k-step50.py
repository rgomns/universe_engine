import numpy as np
import networkx as nx
import time

class AGI_Endurance_Universe:
    def __init__(self, n_bits=64, pop_size=600):
        self.n_bits = n_bits
        self.pop_size = pop_size
        self.population = np.random.randint(0, 2, (pop_size, n_bits), dtype=np.int8)
        self.couplings = [] # [scope, forbidden, strength]
        self.cycle = 0
        self.record_low = 1.0
        self.weights = {2: np.array([2, 1], dtype=np.int8), 
                        3: np.array([4, 2, 1], dtype=np.int8)}
        self.start_time = time.time()

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

    def step(self, sampling_step=50):
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

        # 3. DYNAMIC HEBBIAN CHECK (The speed toggle)
        next_gen = []
        # We sample only every 'sampling_step' person to save CPU
        sample_pop = self.population[::sampling_step] 
        
        for c in self.couplings:
            scope, forbidden, strength = c
            c[2] *= 0.95 # Structural Decay
            
            sample_vals = sample_pop[:, scope].dot(self.weights[len(scope)])
            if np.mean(sample_vals != forbidden) > 0.95:
                c[2] += 0.06 # Hardening
            
            if c[2] > 0.15:
                next_gen.append(c)
        self.couplings = next_gen

        # 4. ENVIRONMENTAL DRIVE & NOISE
        self.population[:, 0:4] = (self.cycle // 20) % 2
        noise_mask = np.random.random(self.population.shape) < 0.02
        self.population[noise_mask] ^= 1

# --- 100,000 CYCLE RUN WITH WATCHDOG ---
universe = AGI_Endurance_Universe()
print(f"Starting 100k Run... Sampling Step: 50")
print(f"{'Cycle':<8} | {'Entropy':<8} | {'Best':<8} | {'LCC':<5} | {'Phase'}")
print("-" * 55)

for i in range(1, 100001):
    # Pass 50 for speed, 30 for precision
    universe.step(sampling_step=50)
    
    current_ent = universe.get_entropy()
    
    # WATCHDOG: Print immediately if a new record is hit
    if current_ent < universe.record_low:
        universe.record_low = current_ent
        lcc = universe.get_skeleton_metrics()
        print(f"{i:<8} | {current_ent:.3f}* | {universe.record_low:.3f} | {lcc:<5} | NEW RECORD")

    # Regular Logging every 5000 cycles to keep the terminal clean
    if i % 5000 == 0:
        lcc = universe.get_skeleton_metrics()
        phase = "CRUNCH" if current_ent < 0.2 else "SUPERFLUID"
        elapsed = (time.time() - universe.start_time) / 60
        print(f"{i:<8} | {current_ent:.3f}    | {universe.record_low:.3f} | {lcc:<5} | {phase} (T+{elapsed:.1f}m)")

print("100,000 Cycle Journey Complete.")