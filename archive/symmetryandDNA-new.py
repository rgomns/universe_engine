import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# ===============================
# CONFIGURATION (Synced with EVO_BASE)
# ===============================
N_BITS = 48
INITIAL_SAMPLE_SIZE = 12000 
MIN_OMEGA_BEFORE_CRUNCH = 80 
MAX_CYCLES = 7000 

# Selection Pressure Constants
GLOBAL_DECAY = 0.08        # Constant drain on all laws
CONSTRAINT_REWARD = 0.05    # Strength gain if law kills bitstrings
CONSTRAINT_DECAY = 0.01     # Strength loss if law is "lazy"
MIN_STRENGTH = 0.02
P_NEW_CONSTRAINT = 0.12     

# ===============================
# ANALYSIS TOOLS
# ===============================

class EmergenceTracker:
    def get_metrics(self, Omega, constraints):
        if not Omega: return 0, 0, 0, []
        
        unique_patterns = set(Omega)
        div = len(unique_patterns) / len(Omega)
        top_species = Counter(Omega).most_common(1)

        # Entropy calculation
        sample = random.sample(Omega, min(len(Omega), 400))
        h = 0
        for i in range(N_BITS):
            p1 = sum(1 for x in sample if x[i] == '1') / len(sample)
            if 0 < p1 < 1: h -= (p1 * math.log2(p1) + (1-p1) * math.log2(1-p1))
        h /= N_BITS

        # Skeleton (LCC) logic
        adj = defaultdict(list)
        for c in constraints:
            scope = c["scope"]
            for i in scope:
                for j in scope:
                    if i != j: adj[i].append(j)
        
        visited = [False] * N_BITS; lcc = 0
        for i in range(N_BITS):
            if not visited[i] and i in adj:
                curr_size = 0; q = [i]; visited[i] = True
                while q:
                    curr = q.pop(0); curr_size += 1
                    for n in adj[curr]:
                        if not visited[n]: visited[n] = True; q.append(n)
                lcc = max(lcc, curr_size)
                
        return h, lcc, div, top_species

def calculate_mi_matrix(Omega):
    """Computes bits of shared information between every bit pair."""
    mi_matrix = np.zeros((N_BITS, N_BITS))
    sample = random.sample(Omega, min(len(Omega), 1000))
    n = len(sample)
    
    for i in range(N_BITS):
        p_i1 = sum(1 for x in sample if x[i] == '1') / n
        p_i0 = 1 - p_i1
        for j in range(i + 1, N_BITS):
            p_j1 = sum(1 for x in sample if x[j] == '1') / n
            p_j0 = 1 - p_j1
            pairs = [x[i] + x[j] for x in sample]
            counts = Counter(pairs)
            mi = 0
            for ci, cj in [('0','0'), ('0','1'), ('1','0'), ('1','1')]:
                p_ij = counts[ci + cj] / n
                pi = p_i1 if ci == '1' else p_i0
                pj = p_j1 if cj == '1' else p_j0
                if p_ij > 0 and pi > 0 and pj > 0:
                    mi += p_ij * math.log2(p_ij / (pi * pj))
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    return mi_matrix

def plot_universe_state(cycle, Omega, Constraints):
    """Generates the Heatmap and Attractor visualization."""
    if not Omega: return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Mutual Information Heatmap
    mi = calculate_mi_matrix(Omega)
    sns.heatmap(mi, cmap='magma', ax=ax1, cbar_kws={'label': 'Bits Shared'})
    ax1.set_title(f"Cycle {cycle}: Logical DNA (MI Matrix)")
    
    # 2. Phase Space Projection
    points = []
    for x in random.sample(Omega, min(len(Omega), 800)):
        # Project 48-bits into 2D via regional density
        points.append((sum(int(b) for b in x[:24]), sum(int(b) for b in x[24:])))
    
    px, py = zip(*points) if points else ([], [])
    ax2.scatter(px, py, alpha=0.6, s=15, color='#00FFCC')
    ax2.set_title(f"Cycle {cycle}: Attractor State Space")
    ax2.set_facecolor('#111111')
    ax2.grid(color='#333333', linestyle='--')
    
    plt.tight_layout()
    plt.show()

# ===============================
# MAIN EXECUTION ENGINE
# ===============================

def run_universe():
    def rebirth(): return [''.join(random.choice("01") for _ in range(N_BITS)) for _ in range(INITIAL_SAMPLE_SIZE)]

    Omega = rebirth()
    Constraints = []
    tracker = EmergenceTracker()
    cycle = 0

    print("ENGINE START: Selection Pressure Enabled.")

    while cycle < MAX_CYCLES:
        # 1. APPLY LAWS & SELECTION PRESSURE
        if Constraints:
            weights = [max(c["strength"], 1e-6) for c in Constraints]
            c = random.choices(Constraints, weights=weights, k=1)[0]
            
            pre_size = len(Omega)
            scope = c["scope"]
            Omega = [x for x in Omega if ''.join(x[i] for i in scope) in c["allowed"]]
            removed = pre_size - len(Omega)

            # Reinforcement Logic from EVO_BASE
            if removed > 0:
                c["strength"] += CONSTRAINT_REWARD 
            else:
                c["strength"] *= (1 - CONSTRAINT_DECAY)

        # 2. CRUNCH EVENT / REBIRTH
        if len(Omega) <= MIN_OMEGA_BEFORE_CRUNCH:
            h, lcc, div, top_species = tracker.get_metrics(Omega, Constraints)
            
            if cycle % 10 == 0:
                print(f"CYCLE {cycle:4} | Pop: {len(Omega):3} | Entropy: {h:.3f} | Skeleton: {lcc:2}")
            
            # Visualization Triggers
            if cycle % 500 == 0 or (len(Omega) < 20 and cycle > 100):
                plot_universe_state(cycle, Omega, Constraints)

            # Global Decay & Law Extinction
            Constraints = [c for c in Constraints if (c["strength"] * (1-GLOBAL_DECAY)) >= MIN_STRENGTH]
            for c in Constraints: c["strength"] *= (1-GLOBAL_DECAY)

            Omega = rebirth()
            cycle += 1

        # 3. GENERATE NEW LAWS (2-Bit and 3-Bit Logic)
        if len(Constraints) < 100 and random.random() < P_NEW_CONSTRAINT:
            is_3bit = random.random() < 0.3
            if is_3bit:
                scope = tuple(sorted(random.sample(range(N_BITS), 3)))
                patterns = [f"{i:03b}" for i in range(8)]
            else:
                i = random.randrange(N_BITS)
                scope = (i, (i + 1) % N_BITS)
                patterns = ["00", "01", "10", "11"]
            
            forbidden = random.choice(patterns)
            allowed = set(p for p in patterns if p != forbidden)
            
            Constraints.append({"scope": scope, "allowed": allowed, "strength": 0.5})

if __name__ == "__main__":
    run_universe()