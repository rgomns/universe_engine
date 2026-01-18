import numpy as np
import networkx as nx

class SpontaneousUniverse:
    def __init__(self, n_bits=64, pop_size=600):
        self.n_bits = n_bits
        self.pop_size = pop_size
        # Using int8 to save memory and increase speed
        self.population = np.random.randint(0, 2, (pop_size, n_bits), dtype=np.int8)
        self.couplings = [] # [scope_array, forbidden_int, strength_float]
        self.cycle = 0
        self.record_low = 1.0
        
        # Pre-calculated bit weights for 2 and 3 bit constraints
        self.weights = {2: np.array([2, 1], dtype=np.int8), 
                        3: np.array([4, 2, 1], dtype=np.int8)}

    def get_entropy(self):
        p = np.mean(self.population, axis=0)
        # Vectorized Shannon entropy
        return np.mean(-p * np.log2(p + 1e-9) - (1-p) * np.log2(1-p + 1e-9))

    def get_skeleton_metrics(self):
        G = nx.Graph()
        for scope, _, _ in self.couplings:
            for i in range(len(scope)):
                for j in range(i + 1, len(scope)):
                    G.add_edge(scope[i], scope[j])
        if not G.nodes: return 0
        # Returns size of largest connected component
        return len(max(nx.connected_components(G), key=len))

    def step(self, noise_override=None):
        self.cycle += 1
        entropy = self.get_entropy()
        
        # 1. EMERGENCE (Mimicry)
        # Birth rate scaled by entropy to prevent stagnation
        if np.random.random() < (0.08 * (entropy + 0.5)):
            size = 3 if np.random.random() < 0.3 else 2
            scope = np.random.choice(self.n_bits, size, replace=False)
            
            # Mimicry: Observe a small sample of the population
            sample = self.population[np.random.randint(0, self.pop_size, 20)][:, scope]
            indices = sample.dot(self.weights[size])
            forbidden = np.argmin(np.bincount(indices, minlength=2**size))
            self.couplings.append([scope, forbidden, 0.6])

        # 2. VECTORIZED CASCADING RELAXATION
        # This is the "Engine" - updates the whole population in one go
        for c in self.couplings:
            scope, forbidden, strength = c
            current_vals = self.population[:, scope].dot(self.weights[len(scope)])
            violators = (current_vals == forbidden)
            
            if violators.any():
                # Probability filter based on constraint strength
                mask = (np.random.random(self.pop_size) < strength) & violators
                if mask.any():
                    # Flip a random bit within the violating pattern
                    flip_bit = np.random.choice(scope)
                    self.population[mask, flip_bit] ^= 1

        # 3. HEBBIAN HARDENING & DECAY
        next_gen = []
        sample_pop = self.population[::30] # Sample for efficiency
        for c in self.couplings:
            scope, forbidden, strength = c
            c[2] *= 0.95 # Global Fatigue
            
            # Hebbian Check: If the rule is obeyed, it hardens
            sample_vals = sample_pop[:, scope].dot(self.weights[len(scope)])
            if np.mean(sample_vals != forbidden) > 0.95:
                c[2] += 0.06 
            
            if c[2] > 0.15: # Bond "Snaps" if strength is too low
                next_gen.append(c)
        self.couplings = next_gen

        # 4. ENVIRONMENT & NOISE
        # Environmental Drive (Bits 0-4 flip every 20 cycles)
        self.population[:, 0:4] = (self.cycle // 20) % 2
        
        # Apply Noise
        floor = noise_override if noise_override is not None else 0.02
        noise_mask = np.random.random(self.population.shape) < floor
        self.population[noise_mask] ^= 1

# --- EXECUTION BLOCK ---
universe = SpontaneousUniverse()
print(f"{'Cycle':<8} | {'Entropy':<8} | {'LCC':<8} | {'Phase':<13} | {'Best'}")
print("-" * 60)

# Run for 30,000 cycles
for i in range(1, 30001):
    universe.step()
    
    current_ent = universe.get_entropy()
    if current_ent < universe.record_low:
        universe.record_low = current_ent
        
    if i % 500 == 0:
        lcc = universe.get_skeleton_metrics()
        
        if current_ent > 0.7: phase = "SUPERCRITICAL"
        elif current_ent > 0.2: phase = "SUPERFLUID"
        else: phase = "CRUNCH"
        
        print(f"{universe.cycle:<8} | {current_ent:.3f}    | {lcc:<8} | {phase:<13} | {universe.record_low:.3f}")