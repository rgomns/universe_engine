import os
import json
import random
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt

# ===============================
# GLOBAL CONFIG (shared defaults)
# ===============================

N_BITS = 48
INITIAL_SAMPLE = 12000

# Hard caps
MAX_STEPS = 100000         # total constraint-application steps per universe
MAX_CYCLES = 5000          # max cycles per universe (you can tune this)

MIN_OMEGA_BEFORE_CRUNCH = 80

CONSTRAINT_REWARD = 0.05
CONSTRAINT_DECAY = 0.01
GLOBAL_DECAY_DEFAULT = 0.05
MIN_STRENGTH = 0.02

MAX_CONSTRAINTS = 100
CONSTRAINTS_PER_STEP = 5

ENABLE_3BIT_DEFAULT = True
P_NEW_CONSTRAINT_DEFAULT = 0.08

PATTERN_WINDOW_SIZES = [3, 4]
TOP_PATTERNS_PER_CYCLE = 10

KL_SMOOTHING = 1e-12
PHASE_ENTROPY_DELTA = 0.2
PHASE_KL_THRESHOLD = 0.5

SNAPSHOT_INTERVAL = 100

FUNDAMENTAL_LIFESPAN = 2000

NUM_ATTRACTORS = 3
ATTRACTOR_KMEANS_ITERS = 20

random.seed(42)

_next_constraint_id = 0


def new_constraint_id():
    global _next_constraint_id
    cid = _next_constraint_id
    _next_constraint_id += 1
    return cid


# ===============================
# BASIC UTILITIES
# ===============================

def rand_bitstring():
    return ''.join(random.choice("01") for _ in range(N_BITS))


def random_universe():
    return set(rand_bitstring() for _ in range(INITIAL_SAMPLE))


def entropy(Omega):
    if not Omega:
        return 0.0
    H = 0.0
    for i in range(N_BITS):
        c = Counter(x[i] for x in Omega)
        p = [v / len(Omega) for v in c.values()]
        H += -sum(pi * math.log2(pi) for pi in p if pi > 0)
    return H / N_BITS


def kl_divergence(counter_p, counter_q):
    keys = set(counter_p.keys()) | set(counter_q.keys())
    total_p = sum(counter_p.values())
    total_q = sum(counter_q.values())
    if total_p == 0 or total_q == 0:
        return 0.0

    kl = 0.0
    for k in keys:
        p = counter_p.get(k, 0) / total_p
        q = counter_q.get(k, 0) / total_q
        p += KL_SMOOTHING
        q += KL_SMOOTHING
        kl += p * math.log2(p / q)
    return kl


# ===============================
# CONSTRAINTS
# ===============================

def random_constraint(current_cycle, enable_3bit):
    if enable_3bit and random.random() < 0.3:
        scope = tuple(sorted(random.sample(range(N_BITS), 3)))
        patterns = [f"{i:03b}" for i in range(8)]
    else:
        i = random.randrange(N_BITS)
        scope = (i, (i + 1) % N_BITS)
        patterns = ["00", "01", "10", "11"]

    forbidden = random.choice(patterns)
    allowed = set(p for p in patterns if p != forbidden)

    return {
        "id": new_constraint_id(),
        "scope": scope,
        "allowed": allowed,
        "strength": 0.5,
        "age": 0,
        "birth_cycle": current_cycle,
        "applied_count": 0,
        "removed_total": 0,
    }


def apply_constraint(Omega, c):
    new = set()
    removed = 0
    scope = c["scope"]
    for x in Omega:
        bits = ''.join(x[i] for i in scope)
        if bits in c["allowed"]:
            new.add(x)
        else:
            removed += 1
    return new, removed


# ===============================
# GRAPH METRICS
# ===============================

def constraint_graph_metrics(Constraints):
    adj = defaultdict(set)
    for c in Constraints:
        scope = c["scope"]
        for i in scope:
            for j in scope:
                if i != j:
                    adj[i].add(j)

    visited = set()
    components = 0

    for node in range(N_BITS):
        if node not in visited:
            components += 1
            stack = [node]
            while stack:
                v = stack.pop()
                if v not in visited:
                    visited.add(v)
                    stack.extend(adj[v])

    degrees = [len(adj[i]) for i in range(N_BITS)]
    avg_degree = sum(degrees) / N_BITS

    visited = set()
    largest_cc_size = 0
    for node in range(N_BITS):
        if node not in visited:
            size = 0
            stack = [node]
            while stack:
                v = stack.pop()
                if v not in visited:
                    visited.add(v)
                    size += 1
                    stack.extend(adj[v])
            largest_cc_size = max(largest_cc_size, size)

    deg_counts = Counter(degrees)
    total_nodes = sum(deg_counts.values())
    if total_nodes == 0:
        deg_entropy = 0.0
    else:
        probs = [c / total_nodes for c in deg_counts.values()]
        deg_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    return components, avg_degree, largest_cc_size, deg_entropy


# ===============================
# PATTERN SCANNING
# ===============================

def scan_local_patterns(Omega, window_sizes):
    pattern_counts = {w: Counter() for w in window_sizes}
    if not Omega:
        return pattern_counts

    for x in Omega:
        for w in window_sizes:
            for i in range(N_BITS - w + 1):
                pat = x[i:i + w]
                pattern_counts[w][pat] += 1

    return pattern_counts


def pattern_entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in counter.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ===============================
# MUTUAL INFORMATION
# ===============================

def compute_mutual_information_matrix(Omega):
    """
    Compute pairwise mutual information I(i,j) for bits i,j in [0, N_BITS).
    Returns:
        mi_matrix: N_BITS x N_BITS list of floats
        top_pairs: list of (i, j, MI) sorted by MI desc
    """
    if not Omega:
        mi_matrix = [[0.0 for _ in range(N_BITS)] for _ in range(N_BITS)]
        return mi_matrix, []

    n = len(Omega)

    # Marginals: P(bit=1) per position
    bit_counts = [0] * N_BITS
    for x in Omega:
        for i, b in enumerate(x):
            if b == '1':
                bit_counts[i] += 1

    H_single = [0.0] * N_BITS
    for i in range(N_BITS):
        p1 = bit_counts[i] / n
        p0 = 1 - p1
        H = 0.0
        for p in (p0, p1):
            if p > 0:
                H -= p * math.log2(p)
        H_single[i] = H

    # Joint distributions
    joint_counts = {}
    for i in range(N_BITS):
        for j in range(i + 1, N_BITS):
            joint_counts[(i, j)] = [0, 0, 0, 0]  # 00,01,10,11

    for x in Omega:
        bits = list(x)
        for i in range(N_BITS):
            bi = 1 if bits[i] == '1' else 0
            for j in range(i + 1, N_BITS):
                bj = 1 if bits[j] == '1' else 0
                idx = (bi << 1) | bj
                joint_counts[(i, j)][idx] += 1

    mi_matrix = [[0.0 for _ in range(N_BITS)] for _ in range(N_BITS)]
    top_pairs = []

    for (i, j), counts in joint_counts.items():
        total = sum(counts)
        if total == 0:
            continue
        H_joint = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                H_joint -= p * math.log2(p)
        Iij = H_single[i] + H_single[j] - H_joint
        mi_matrix[i][j] = Iij
        mi_matrix[j][i] = Iij
        top_pairs.append((i, j, Iij))

    top_pairs.sort(key=lambda t: t[2], reverse=True)
    return mi_matrix, top_pairs


def plot_mutual_information_heatmap(mi_matrix, plots_dir, label_prefix):
    plt.figure(figsize=(8, 6))
    plt.imshow(mi_matrix, cmap="viridis", origin="lower")
    plt.colorbar(label="Mutual Information")
    plt.xlabel("Bit index")
    plt.ylabel("Bit index")
    plt.title(f"{label_prefix} Mutual Information Heatmap")
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"{label_prefix.lower()}_mutual_information_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    return os.path.relpath(out_path, os.path.dirname(plots_dir))


# ===============================
# UNIVERSE RUNNER
# ===============================

def run_universe(
    evolving=True,
    label="EVO",
    p_new_constraint=P_NEW_CONSTRAINT_DEFAULT,
    global_decay=GLOBAL_DECAY_DEFAULT,
    enable_3bit=ENABLE_3BIT_DEFAULT,
    snapshot_interval=SNAPSHOT_INTERVAL
):
    Omega = random_universe()
    Constraints = []

    cycle = 0
    step = 0

    records = []
    global_pattern_stats = {w: Counter() for w in PATTERN_WINDOW_SIZES}
    constraint_lineage = []
    prev_cycle_patterns = {w: Counter() for w in PATTERN_WINDOW_SIZES}
    prev_entropy = None

    cycle_total_removed = 0
    cycle_total_applied = 0

    while step < MAX_STEPS:
        step += 1

        # Cycle boundary: reinitialize universe if too small
        if len(Omega) <= MIN_OMEGA_BEFORE_CRUNCH:
            H = entropy(Omega)

            total_strength = sum(c["strength"] for c in Constraints)
            strengths = [c["strength"] for c in Constraints]
            mean_s = sum(strengths) / len(strengths) if strengths else 0.0
            variance_strength = (
                sum((s - mean_s) ** 2 for s in strengths) / len(strengths)
                if strengths else 0.0
            )

            num_3bit = sum(1 for c in Constraints if len(c["scope"]) == 3)
            components, avg_degree, largest_cc, deg_entropy = constraint_graph_metrics(Constraints)

            cycle_patterns = scan_local_patterns(Omega, PATTERN_WINDOW_SIZES)
            for w in PATTERN_WINDOW_SIZES:
                global_pattern_stats[w].update(cycle_patterns[w])

            per_cycle_pattern_entropy = {
                w: pattern_entropy(cycle_patterns[w]) for w in PATTERN_WINDOW_SIZES
            }

            per_cycle_kl = {}
            for w in PATTERN_WINDOW_SIZES:
                if sum(prev_cycle_patterns[w].values()) == 0:
                    per_cycle_kl[w] = 0.0
                else:
                    per_cycle_kl[w] = kl_divergence(cycle_patterns[w], prev_cycle_patterns[w])

            per_cycle_top_patterns = {
                w: cycle_patterns[w].most_common(TOP_PATTERNS_PER_CYCLE)
                for w in PATTERN_WINDOW_SIZES
            }

            entropy_delta = 0.0 if prev_entropy is None else (H - prev_entropy)
            phase_transition = False
            phase_reasons = []

            if prev_entropy is not None and abs(entropy_delta) > PHASE_ENTROPY_DELTA:
                phase_transition = True
                phase_reasons.append(f"entropy_delta={entropy_delta:.3f}")
            for w in PATTERN_WINDOW_SIZES:
                if per_cycle_kl[w] > PHASE_KL_THRESHOLD:
                    phase_transition = True
                    phase_reasons.append(f"KL_w{w}={per_cycle_kl[w]:.3f}")

            record = {
                "label": label,
                "cycle": cycle,
                "omega": len(Omega),
                "entropy": H,
                "entropy_delta": entropy_delta,
                "total_strength": total_strength,
                "variance_strength": variance_strength,
                "num_constraints": len(Constraints),
                "num_3bit": num_3bit,
                "components": components,
                "avg_degree": avg_degree,
                "largest_cc": largest_cc,
                "deg_entropy": deg_entropy,
                "pattern_entropy": per_cycle_pattern_entropy,
                "kl_divergence": per_cycle_kl,
                "top_patterns": per_cycle_top_patterns,
                "total_removed": cycle_total_removed,
                "total_applied": cycle_total_applied,
                "removal_efficiency": (
                    cycle_total_removed / cycle_total_applied
                    if cycle_total_applied > 0 else 0.0
                ),
                "phase_transition": phase_transition,
                "phase_reasons": phase_reasons,
            }
            records.append(record)

            if cycle % snapshot_interval == 0:
                oldest_age = max((c["age"] for c in Constraints), default=0)
                strength_values = [c["strength"] for c in Constraints]
                p95_strength = (
                    sorted(strength_values)[int(0.95 * len(strength_values))]
                    if strength_values else 0.0
                )

                print(f"\n=== {label} Knowledge Snapshot @ Cycle {cycle} ===")
                print(f"  Ω size: {len(Omega)}")
                print(f"  Entropy: {H:.4f} (Δ={entropy_delta:.4f})")
                print(f"  Constraints: {len(Constraints)}")
                print(f"  Avg strength: {mean_s:.4f}")
                print(f"  Strength variance: {variance_strength:.4f}")
                print(f"  95th percentile strength: {p95_strength:.4f}")
                print(f"  Oldest constraint age: {oldest_age}")
                print(f"  3-bit constraints: {num_3bit}")
                print(f"  Graph components: {components}")
                print(f"  Largest CC size: {largest_cc}")
                print(f"  Degree entropy: {deg_entropy:.4f}")
                print(f"  Avg degree: {avg_degree:.3f}")
                for w in PATTERN_WINDOW_SIZES:
                    print(f"  Pattern entropy (w={w}): {per_cycle_pattern_entropy[w]:.4f}")
                    print(f"  KL divergence (w={w}): {per_cycle_kl[w]:.4f}")
                    print(f"  Top 3 patterns (w={w}): {per_cycle_top_patterns[w][:3]}")
                if phase_transition:
                    print(f"  Phase transition detected: {', '.join(phase_reasons)}")

            if evolving:
                surviving_constraints = []
                for c in Constraints:
                    old_strength = c["strength"]
                    c["strength"] *= (1 - global_decay)
                    if c["strength"] >= MIN_STRENGTH:
                        surviving_constraints.append(c)
                    else:
                        constraint_lineage.append({
                            "id": c["id"],
                            "birth_cycle": c["birth_cycle"],
                            "death_cycle": cycle,
                            "lifespan": cycle - c["birth_cycle"],
                            "final_strength": old_strength,
                            "scope": c["scope"],
                            "applied_count": c["applied_count"],
                            "removed_total": c["removed_total"],
                        })
                Constraints = surviving_constraints
            else:
                for c in Constraints:
                    constraint_lineage.append({
                        "id": c["id"],
                        "birth_cycle": c["birth_cycle"],
                        "death_cycle": cycle,
                        "lifespan": cycle - c["birth_cycle"],
                        "final_strength": c["strength"],
                        "scope": c["scope"],
                        "applied_count": c["applied_count"],
                        "removed_total": c["removed_total"],
                    })
                Constraints = []

            prev_cycle_patterns = cycle_patterns
            prev_entropy = H
            Omega = random_universe()
            cycle += 1
            cycle_total_removed = 0
            cycle_total_applied = 0

            # Max cycles guard
            if cycle >= MAX_CYCLES:
                break

        # introduce new constraint
        if evolving and len(Constraints) < MAX_CONSTRAINTS and random.random() < p_new_constraint:
            Constraints.append(random_constraint(cycle, enable_3bit))

        # apply constraints
        if Constraints:
            weights = [max(c["strength"], 1e-6) for c in Constraints]
            chosen = random.choices(
                Constraints,
                weights,
                k=min(CONSTRAINTS_PER_STEP, len(Constraints))
            )

            for c in chosen:
                Omega, removed = apply_constraint(Omega, c)
                cycle_total_applied += 1
                cycle_total_removed += removed
                c["applied_count"] += 1
                c["removed_total"] += removed
                if removed > 0:
                    c["strength"] += CONSTRAINT_REWARD
                else:
                    c["strength"] *= (1 - CONSTRAINT_DECAY)

        for c in Constraints:
            c["age"] += 1

    final_cycle = cycle
    for c in Constraints:
        constraint_lineage.append({
            "id": c["id"],
            "birth_cycle": c["birth_cycle"],
            "death_cycle": final_cycle,
            "lifespan": final_cycle - c["birth_cycle"],
            "final_strength": c["strength"],
            "scope": c["scope"],
            "applied_count": c["applied_count"],
            "removed_total": c["removed_total"],
        })

    return records, global_pattern_stats, constraint_lineage, Omega


# ===============================
# ANALYSIS / SUMMARY HELPERS
# ===============================

def extract_fundamental_constraints(lineage, min_lifespan=FUNDAMENTAL_LIFESPAN):
    return [c for c in lineage if c["lifespan"] >= min_lifespan]


def cluster_fundamentals_by_scope_and_impact(fundamentals):
    clusters = {}
    for c in fundamentals:
        scope = tuple(c["scope"])
        key = scope
        if key not in clusters:
            clusters[key] = {
                "scope": list(scope),
                "constraints": [],
                "avg_lifespan": 0.0,
                "avg_final_strength": 0.0,
                "avg_removal_efficiency": 0.0,
            }
        clusters[key]["constraints"].append(c)

    for scope, cl in clusters.items():
        lifespans = [x["lifespan"] for x in cl["constraints"]]
        strengths = [x["final_strength"] for x in cl["constraints"]]
        effs = []
        for x in cl["constraints"]:
            if x["applied_count"] > 0:
                effs.append(x["removed_total"] / x["applied_count"])

        cl["avg_lifespan"] = sum(lifespans) / len(lifespans)
        cl["avg_final_strength"] = sum(strengths) / len(strengths) if strengths else 0.0
        cl["avg_removal_efficiency"] = sum(effs) / len(effs) if effs else 0.0

    return list(clusters.values())


def detect_phase_transitions(records):
    return [r for r in records if r.get("phase_transition")]


def kmeans_simple(points, k, iters):
    if not points:
        return [], []

    dim = len(points[0])
    centroids = [list(p) for p in points[:k]]
    labels = [0] * len(points)

    for _ in range(iters):
        for i, p in enumerate(points):
            best = 0
            best_dist = float("inf")
            for j, c in enumerate(centroids):
                d = sum((p[d_] - c[d_]) ** 2 for d_ in range(dim))
                if d < best_dist:
                    best_dist = d
                    best = j
            labels[i] = best

        counts = [0] * k
        new_centroids = [[0.0] * dim for _ in range(k)]
        for lbl, p in zip(labels, points):
            counts[lbl] += 1
            for d_ in range(dim):
                new_centroids[lbl][d_] += p[d_]

        for j in range(k):
            if counts[j] > 0:
                for d_ in range(dim):
                    new_centroids[j][d_] /= counts[j]
            else:
                new_centroids[j] = centroids[j]

        centroids = new_centroids

    return centroids, labels


def assign_attractor_basins(records, k=NUM_ATTRACTORS, iters=ATTRACTOR_KMEANS_ITERS):
    if not records:
        return None

    points = []
    for r in records:
        pe3 = r["pattern_entropy"].get(3, 0.0)
        pe4 = r["pattern_entropy"].get(4, 0.0)
        points.append((r["entropy"], pe3, pe4))

    centroids, labels = kmeans_simple(points, k, iters)

    for r, lbl in zip(records, labels):
        r["attractor_id"] = int(lbl)

    return {"centroids": centroids, "k": k}


# ===============================
# PLOTTING
# ===============================

def plot_metrics(records, label_prefix, plots_dir):
    if not records:
        return {}

    os.makedirs(plots_dir, exist_ok=True)
    cycles = [r["cycle"] for r in records]
    omegas = [r["omega"] for r in records]
    entropies = [r["entropy"] for r in records]
    avg_degrees = [r["avg_degree"] for r in records]
    num_constraints = [r["num_constraints"] for r in records]

    paths = {}

    # Entropy
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, entropies, label="Entropy")
    plt.xlabel("Cycle")
    plt.ylabel("Entropy")
    plt.title(f"{label_prefix} Entropy over cycles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_entropy.png")
    plt.savefig(out)
    plt.close()
    paths["entropy"] = os.path.relpath(out, os.path.dirname(plots_dir))

    # Omega
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, omegas, label="|Ω|")
    plt.xlabel("Cycle")
    plt.ylabel("|Ω|")
    plt.title(f"{label_prefix} Omega size over cycles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_omega.png")
    plt.savefig(out)
    plt.close()
    paths["omega"] = os.path.relpath(out, os.path.dirname(plots_dir))

    # Avg degree
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, avg_degrees, label="Avg degree")
    plt.xlabel("Cycle")
    plt.ylabel("Average degree")
    plt.title(f"{label_prefix} Constraint graph degree over cycles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_avg_degree.png")
    plt.savefig(out)
    plt.close()
    paths["avg_degree"] = os.path.relpath(out, os.path.dirname(plots_dir))

    # Constraints
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, num_constraints, label="# constraints")
    plt.xlabel("Cycle")
    plt.ylabel("Number of constraints")
    plt.title(f"{label_prefix} constraints over cycles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_constraints.png")
    plt.savefig(out)
    plt.close()
    paths["constraints"] = os.path.relpath(out, os.path.dirname(plots_dir))

    return paths


def plot_phase_diagram(records, label_prefix, plots_dir):
    if not records:
        return {}

    os.makedirs(plots_dir, exist_ok=True)
    cycles = [r["cycle"] for r in records]
    entropies = [r["entropy"] for r in records]
    max_kls = [max(r["kl_divergence"].values()) for r in records]
    components = [r["components"] for r in records]
    phase_flags = [r["phase_transition"] for r in records]

    paths = {}

    # Phase diagram
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(cycles, entropies, c=max_kls, cmap="viridis", s=10)
    plt.colorbar(scatter, label="max KL divergence")
    plt.xlabel("Cycle")
    plt.ylabel("Entropy")
    plt.title(f"{label_prefix} Phase Diagram")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_phase_diagram.png")
    plt.savefig(out)
    plt.close()
    paths["phase_diagram"] = os.path.relpath(out, os.path.dirname(plots_dir))

    # Phase transitions
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, entropies, label="Entropy")
    pt_cycles = [c for c, f in zip(cycles, phase_flags) if f]
    pt_vals = [e for e, f in zip(entropies, phase_flags) if f]
    plt.scatter(pt_cycles, pt_vals, color="red", s=20, label="Phase transition")
    plt.xlabel("Cycle")
    plt.ylabel("Entropy")
    plt.title(f"{label_prefix} Phase Transitions")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_phase_transitions.png")
    plt.savefig(out)
    plt.close()
    paths["phase_transitions"] = os.path.relpath(out, os.path.dirname(plots_dir))

    # Components
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, components, label="# Components")
    plt.xlabel("Cycle")
    plt.ylabel("Components")
    plt.title(f"{label_prefix} Graph Components Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_components.png")
    plt.savefig(out)
    plt.close()
    paths["components"] = os.path.relpath(out, os.path.dirname(plots_dir))

    return paths


def plot_attractors(records, label_prefix, plots_dir):
    if not records or "attractor_id" not in records[0]:
        return {}

    os.makedirs(plots_dir, exist_ok=True)
    cycles = [r["cycle"] for r in records]
    entropies = [r["entropy"] for r in records]
    attractors = [r["attractor_id"] for r in records]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(cycles, entropies, c=attractors, cmap="tab10", s=8)
    plt.xlabel("Cycle")
    plt.ylabel("Entropy")
    plt.title(f"{label_prefix} Attractor Basins")
    plt.grid(True)
    plt.colorbar(scatter, label="Attractor ID")
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_attractors.png")
    plt.savefig(out)
    plt.close()
    return {"attractors": os.path.relpath(out, os.path.dirname(plots_dir))}


def plot_lineage(lineage, label_prefix, plots_dir):
    if not lineage:
        return {}

    os.makedirs(plots_dir, exist_ok=True)
    births = [c["birth_cycle"] for c in lineage]
    deaths = [c["death_cycle"] for c in lineage]
    lifespans = [c["lifespan"] for c in lineage]

    paths = {}

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(births, deaths, c=lifespans, cmap="plasma", s=10)
    plt.xlabel("Birth Cycle")
    plt.ylabel("Death Cycle")
    plt.title(f"{label_prefix} Constraint Lineage")
    plt.grid(True)
    plt.colorbar(scatter, label="Lifespan")
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_lineage.png")
    plt.savefig(out)
    plt.close()
    paths["lineage"] = os.path.relpath(out, os.path.dirname(plots_dir))

    plt.figure(figsize=(10, 6))
    plt.hist(lifespans, bins=50)
    plt.xlabel("Lifespan")
    plt.ylabel("Count")
    plt.title(f"{label_prefix} Lifespan Distribution")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(plots_dir, f"{label_prefix.lower()}_lifespan_hist.png")
    plt.savefig(out)
    plt.close()
    paths["lifespan_hist"] = os.path.relpath(out, os.path.dirname(plots_dir))

    return paths


# ===============================
# ENSEMBLE RUNNER
# ===============================

def run_single_universe(label, evolving, p_new_constraint, global_decay, enable_3bit):
    print(f"\n\n===== Running universe: {label} =====")
    base_dir = os.path.join("runs", label)
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # For reproducibility, seed per universe
    random.seed(42 + hash(label) % (10**6))

    records, patterns, lineage, final_Omega = run_universe(
        evolving=evolving,
        label=label,
        p_new_constraint=p_new_constraint,
        global_decay=global_decay,
        enable_3bit=enable_3bit
    )

    # Attractors
    attractors = assign_attractor_basins(records) if evolving else None

    # Phase transitions
    phase_transitions = detect_phase_transitions(records)

    # Fundamentals
    fundamentals = extract_fundamental_constraints(lineage) if evolving else []
    fundamental_clusters = cluster_fundamentals_by_scope_and_impact(fundamentals) if evolving else []

    # Mutual information from final Omega
    mi_matrix, top_pairs = compute_mutual_information_matrix(final_Omega)
    mi_heatmap_relpath = plot_mutual_information_heatmap(mi_matrix, plots_dir, label)

    # Plots
    metric_paths = plot_metrics(records, label, plots_dir)
    phase_paths = plot_phase_diagram(records, label, plots_dir)
    attractor_paths = plot_attractors(records, label, plots_dir) if evolving else {}
    lineage_paths = plot_lineage(lineage, label, plots_dir)

    # Full log
    log_data = {
        "label": label,
        "config": {
            "evolving": evolving,
            "P_NEW_CONSTRAINT": p_new_constraint,
            "GLOBAL_DECAY": global_decay,
            "ENABLE_3BIT": enable_3bit,
            "MAX_STEPS": MAX_STEPS,
            "MAX_CYCLES": MAX_CYCLES,
        },
        "records": records,
        "patterns": {str(w): dict(patterns[w]) for w in patterns},
        "lineage": lineage,
        "fundamentals": fundamentals,
        "fundamental_clusters": fundamental_clusters,
        "attractors": attractors,
        "phase_transitions": phase_transitions,
        "mutual_information_matrix": mi_matrix,
        "top_mi_pairs": top_pairs[:50],
    }

    log_path = os.path.join(base_dir, "universe_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    # Summary
    final_entropy = records[-1]["entropy"] if records else 0.0
    final_omega = records[-1]["omega"] if records else 0
    final_num_constraints = records[-1]["num_constraints"] if records else 0
    max_kl = 0.0
    for r in records:
        if r["kl_divergence"]:
            mk = max(r["kl_divergence"].values())
            if mk > max_kl:
                max_kl = mk

    # Compose plot paths summary (relative to base_dir)
    plot_paths = {}
    plot_paths.update(metric_paths)
    plot_paths.update(phase_paths)
    plot_paths.update(attractor_paths)
    plot_paths.update(lineage_paths)
    plot_paths["mutual_information_heatmap"] = mi_heatmap_relpath

    summary = {
        "label": label,
        "config": log_data["config"],
        "final_cycle": records[-1]["cycle"] if records else 0,
        "final_omega": final_omega,
        "final_entropy": final_entropy,
        "final_num_constraints": final_num_constraints,
        "num_fundamentals": len(fundamentals),
        "fundamental_clusters": fundamental_clusters,
        "attractor_centroids": attractors["centroids"] if attractors else None,
        "num_phase_transitions": len(phase_transitions),
        "max_kl_divergence": max_kl,
        "top_mi_pairs": top_pairs[:20],
        "plot_paths": plot_paths,
    }

    summary_path = os.path.join(base_dir, "universe_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Universe {label} complete.")
    print(f"  Log: {log_path}")
    print(f"  Summary: {summary_path}")


def run_ensemble():
    ensembles = [
        {
            "label": "EVO_BASE",
            "evolving": True,
            "P_NEW_CONSTRAINT": 0.08,
            "GLOBAL_DECAY": 0.05,
            "ENABLE_3BIT": True,
        },
        {
            "label": "EVO_CHAOS",
            "evolving": True,
            "P_NEW_CONSTRAINT": 0.20,
            "GLOBAL_DECAY": 0.01,
            "ENABLE_3BIT": True,
        },
        {
            "label": "EVO_FROZEN",
            "evolving": True,
            "P_NEW_CONSTRAINT": 0.02,
            "GLOBAL_DECAY": 0.20,
            "ENABLE_3BIT": True,
        },
        {
            "label": "CTRL_BASE",
            "evolving": False,
            "P_NEW_CONSTRAINT": 0.08,
            "GLOBAL_DECAY": 0.05,
            "ENABLE_3BIT": True,
        },
    ]

    os.makedirs("runs", exist_ok=True)

    # Parallel execution across universes
    with ProcessPoolExecutor() as executor:
        futures = []
        for cfg in ensembles:
            futures.append(
                executor.submit(
                    run_single_universe,
                    cfg["label"],
                    cfg["evolving"],
                    cfg["P_NEW_CONSTRAINT"],
                    cfg["GLOBAL_DECAY"],
                    cfg["ENABLE_3BIT"],
                )
            )
        for f in futures:
            # propagate exceptions if any
            f.result()


if __name__ == "__main__":
    run_ensemble()
