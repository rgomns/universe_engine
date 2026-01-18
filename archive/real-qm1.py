import numpy as np
import itertools
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------
# PARAMETERIZED UNIVERSE (CORE DYNAMICS)
# ---------------------------------------------------------

class Persistent_AGI_Universe_256:
    def __init__(
        self,
        n_bits=256,
        pop_size=2500,
        sampling_step=40,
        # sensitivity parameters
        initial_strength=0.6,
        survival_threshold=0.15,
        satisfaction_threshold=0.95,
        reinforcement_increment=0.06,
        decay_factor=0.95,
    ):
        self.n_bits = n_bits
        self.pop_size = pop_size

        self.population = np.random.randint(0, 2, (pop_size, n_bits), dtype=np.int8)
        # Each coupling: [scope (np.array), forbidden (int), strength (float)]
        self.couplings = []
        self.cycle = 0

        self.sampling_step = sampling_step

        # Weights for size-2 and size-3 rules
        self.weights = {
            2: np.array([2, 1], dtype=np.int8),
            3: np.array([4, 2, 1], dtype=np.int8),
        }

        # Sensitivity / meta-law parameters
        self.initial_strength = initial_strength
        self.survival_threshold = survival_threshold
        self.satisfaction_threshold = satisfaction_threshold
        self.reinforcement_increment = reinforcement_increment
        self.decay_factor = decay_factor

    # ---------------- METRICS ----------------

    def get_entropy(self):
        p = np.mean(self.population, axis=0)
        return np.mean(
            -p * np.log2(p + 1e-9) - (1 - p) * np.log2(1 - p + 1e-9)
        )

    def _build_adjacency(self):
        adj = [set() for _ in range(self.n_bits)]
        for scope, _, _ in self.couplings:
            size = len(scope)
            for i in range(size):
                si = int(scope[i])
                for j in range(i + 1, size):
                    sj = int(scope[j])
                    adj[si].add(sj)
                    adj[sj].add(si)
        return adj

    def get_skeleton_stats(self):
        """
        Return (lcc_size, avg_degree).
        """
        if not self.couplings:
            return 0, 0.0

        adj = self._build_adjacency()

        visited = set()
        best = 0

        for node in range(self.n_bits):
            if node in visited or not adj[node]:
                continue
            stack = [node]
            size = 0
            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                size += 1
                for nbr in adj[v]:
                    if nbr not in visited:
                        stack.append(nbr)
            if size > best:
                best = size

        degrees = [len(neigh) for neigh in adj if neigh]
        avg_deg = float(np.mean(degrees)) if degrees else 0.0

        return best, avg_deg

    def get_rule_stats(self):
        if not self.couplings:
            return 0, 0.0, 0.0
        strengths = np.fromiter((c[2] for c in self.couplings), dtype=float)
        return len(strengths), float(strengths.mean()), float(strengths.max())

    # ---------------- STEP ----------------

    def step(self):
        self.cycle += 1
        entropy = self.get_entropy()

        # 1. EMERGENCE (Rule Discovery)
        if np.random.random() < (0.03 * (entropy + 0.5)):
            size = 3 if np.random.random() < 0.3 else 2
            scope = np.random.choice(self.n_bits, size, replace=False).astype(np.int16)

            sample = self.population[
                np.random.randint(0, self.pop_size, 20)
            ][:, scope]

            indices = sample @ self.weights[size]
            forbidden = int(np.argmin(np.bincount(indices, minlength=2**size)))
            self.couplings.append([scope, forbidden, float(self.initial_strength)])

        # 2. VECTOR RELAXATION
        if self.couplings:
            randmask = np.random.random(self.pop_size)

            for c in self.couplings:
                scope, forbidden, strength = c
                weight = self.weights[len(scope)]
                current_vals = self.population[:, scope] @ weight
                violators = (current_vals == forbidden)
                if violators.any():
                    mask = (randmask < strength) & violators
                    if mask.any():
                        flip_index = np.random.choice(scope)
                        self.population[mask, flip_index] ^= 1

        # 3. HEBBIAN SELECTION
        next_gen = []
        sample_pop = self.population[::self.sampling_step]

        for c in self.couplings:
            scope, forbidden, strength = c

            # Decay
            strength *= self.decay_factor

            # Reinforcement
            weight = self.weights[len(scope)]
            sample_vals = sample_pop[:, scope] @ weight
            if np.mean(sample_vals != forbidden) > self.satisfaction_threshold:
                strength += self.reinforcement_increment

            c[2] = strength

            # Survival threshold
            if strength > self.survival_threshold:
                next_gen.append(c)

        self.couplings = next_gen

        return entropy


# ---------------------------------------------------------
# OBJECTIVE EVALUATION (MULTI-OBJECTIVE)
# ---------------------------------------------------------

def evaluate_params(params, cycles=5000, rule_norm_scale=500.0,
                    w_entropy=1.0, w_lcc=1.0, w_rules=0.5, n_bits=256):
    """
    Run a universe with given params, track metrics, and compute a scalar score.

    Returns:
      score (higher is better), metrics_dict
    """
    u = Persistent_AGI_Universe_256(
        n_bits=n_bits,
        pop_size=2500,
        sampling_step=40,
        initial_strength=params["initial_strength"],
        survival_threshold=params["survival_threshold"],
        satisfaction_threshold=params["satisfaction_threshold"],
        reinforcement_increment=params["reinforcement_increment"],
        decay_factor=params["decay_factor"],
    )

    min_entropy = float("inf")
    max_lcc = 0
    max_rule_count = 0

    for _ in range(cycles):
        ent = u.step()
        if ent < min_entropy:
            min_entropy = ent

        # Sample skeleton & rules occasionally (not every step to save time)
        if u.cycle % 200 == 0:
            lcc, _ = u.get_skeleton_stats()
            rc, _, _ = u.get_rule_stats()
            if lcc > max_lcc:
                max_lcc = lcc
            if rc > max_rule_count:
                max_rule_count = rc

    # Multi-objective scalarization
    # Normalize:
    # - entropy: want low → use (1 - min_entropy), assume entropy in [0,1]
    # - lcc: fraction of bits
    # - rule count: normalized by rule_norm_scale
    max_lcc_frac = max_lcc / float(n_bits) if n_bits > 0 else 0.0
    rule_norm = min(max_rule_count / rule_norm_scale, 1.0)

    score = (
        w_entropy * (1.0 - min_entropy) +
        w_lcc * max_lcc_frac +
        w_rules * rule_norm
    )

    metrics = {
        "min_entropy": min_entropy,
        "max_lcc": max_lcc,
        "max_lcc_frac": max_lcc_frac,
        "max_rule_count": max_rule_count,
        "score": score,
    }

    return score, metrics


# ---------------------------------------------------------
# PARALLEL HYPERPARAMETER SEARCH
# ---------------------------------------------------------

def generate_param_grid():
    """
    Define the search space. You can tune these ranges / grids.
    """
    param_space = {
        "initial_strength": [0.4, 0.6, 0.8],
        "survival_threshold": [0.10, 0.15, 0.20],
        "satisfaction_threshold": [0.90, 0.95, 0.98],
        "reinforcement_increment": [0.02, 0.04, 0.06],
        "decay_factor": [0.92, 0.95, 0.98],
    }

    grid = [
        dict(zip(param_space.keys(), values))
        for values in itertools.product(*param_space.values())
    ]
    return grid


def run_single_param_set(args):
    """
    Wrapper for parallel execution.
    args: (params, cycles, kwargs_for_evaluate)
    """
    params, cycles, eval_kwargs = args
    score, metrics = evaluate_params(params, cycles=cycles, **eval_kwargs)
    return params, score, metrics


def parallel_search(cycles=5000, max_workers=8, **eval_kwargs):
    """
    Parallel grid search over hyperparameters.
    """
    param_grid = generate_param_grid()

    best_params = None
    best_score = -float("inf")
    best_metrics = None

    tasks = [(p, cycles, eval_kwargs) for p in param_grid]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_param_set, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            params = futures[fut]
            try:
                p, score, metrics = fut.result()
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

            print(f"Tested {p} → score={score:.4f}, "
                  f"minH={metrics['min_entropy']:.3f}, "
                  f"LCCfrac={metrics['max_lcc_frac']:.3f}, "
                  f"rules={metrics['max_rule_count']}")

            if score > best_score:
                best_score = score
                best_params = deepcopy(p)
                best_metrics = deepcopy(metrics)

    return best_params, best_score, best_metrics


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    # You can tweak weights here to favor certain behaviors
    eval_kwargs = {
        "rule_norm_scale": 500.0,  # normalize rule counts
        "w_entropy": 1.0,          # weight for low entropy
        "w_lcc": 1.0,              # weight for large skeleton
        "w_rules": 0.5,            # weight for rule diversity
        "n_bits": 256,
    }

    best_params, best_score, best_metrics = parallel_search(
        cycles=4000,   # fewer cycles for faster search; increase for more accuracy
        max_workers=8, # adjust to your CPU
        **eval_kwargs
    )

    print("\nBEST PARAMS FOUND:")
    print(best_params)
    print(f"Best score: {best_score:.4f}")
    print("Metrics:", best_metrics)
