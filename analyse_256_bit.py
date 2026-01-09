import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


LOG_PATH = "universe_256_log.csv"


def load_log(path=LOG_PATH):
    data = defaultdict(list)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["cycle"].append(int(row["cycle"]))
            data["entropy"].append(float(row["entropy"]))
            data["record_low"].append(float(row["record_low"]))
            data["lcc_size"].append(int(row["lcc_size"]))
            data["avg_degree"].append(float(row["avg_degree"]))
            data["phase"].append(row["phase"])
            data["rule_count"].append(int(row["rule_count"]))
            data["avg_rule_strength"].append(float(row["avg_rule_strength"]))
            data["max_rule_strength"].append(float(row["max_rule_strength"]))
    return {k: np.array(v) for k, v in data.items()}


# ----------------------------------------------------------
# 1. Attractor basin size
# ----------------------------------------------------------

def analyze_attractor_behavior(log, tail_fraction=0.2):
    """
    Estimate whether the system has converged to a tight basin.
    Also approximate 'basin size' as the entropy variation near the end.
    """
    entropy = log["entropy"]
    record_low = log["record_low"]
    cycles = log["cycle"]

    n = len(cycles)
    start = int((1.0 - tail_fraction) * n)
    ent_tail = entropy[start:]
    rec_tail = record_low[start:]
    gap = ent_tail - rec_tail

    mean_gap = float(np.mean(gap))
    std_gap = float(np.std(gap))
    ent_std = float(np.std(ent_tail))
    rec_span = float(rec_tail.max() - rec_tail.min())

    print("=== Attractor analysis ===")
    print(f"Final segment cycles: {cycles[start]}–{cycles[-1]}")
    print(f"Mean gap (entropy - record_low): {mean_gap:.4f}")
    print(f"Std gap: {std_gap:.4f}")
    print(f"Std of entropy in final segment: {ent_std:.4f}")
    print(f"Span of record_low in final segment: {rec_span:.4f}")

    # Interpret this as an approximation of basin tightness
    if rec_span < 0.005 and mean_gap < 0.02 and ent_std < 0.02:
        print("Inference: Strong evidence of convergence to a tight attractor basin.")
        print(f"Approx. basin 'width' (entropy scale): ~{mean_gap + ent_std:.4f}")
    else:
        print("Inference: System still explores; attractor basin not tight or not unique.")
    print()


# ----------------------------------------------------------
# 2. Rule ecology stability & rule-cluster formation
# ----------------------------------------------------------

def analyze_rule_ecology(log):
    """
    Check whether rule ecology stabilizes, and look for 'cluster formation'
    via changes in average vs max strength and degree.
    """
    cycles = log["cycle"]
    rules = log["rule_count"]
    avgS = log["avg_rule_strength"]
    maxS = log["max_rule_strength"]
    avg_deg = log["avg_degree"]

    n = len(cycles)
    mid = n // 2

    def slope(x, y):
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m)

    slope_rules_early = slope(cycles[:mid], rules[:mid])
    slope_rules_late = slope(cycles[mid:], rules[mid:])
    slope_avgS_early = slope(cycles[:mid], avgS[:mid])
    slope_avgS_late = slope(cycles[mid:], avgS[mid:])
    slope_deg_early = slope(cycles[:mid], avg_deg[:mid])
    slope_deg_late = slope(cycles[mid:], avg_deg[mid:])

    print("=== Rule ecology analysis ===")
    print(f"Early rule_count slope: {slope_rules_early:.6f}")
    print(f"Late  rule_count slope: {slope_rules_late:.6f}")
    print(f"Early avg_strength slope: {slope_avgS_early:.6f}")
    print(f"Late  avg_strength slope: {slope_avgS_late:.6f}")
    print(f"Early avg_degree slope: {slope_deg_early:.6f}")
    print(f"Late  avg_degree slope: {slope_deg_late:.6f}")

    if abs(slope_rules_late) < abs(slope_rules_early) * 0.2 and \
       abs(slope_avgS_late) < abs(slope_avgS_early) * 0.2 and \
       abs(slope_deg_late) < abs(slope_deg_early) * 0.2:
        print("Inference: Rule ecology is approaching a stable configuration.")
    else:
        print("Inference: Rule ecology continues to evolve significantly over time.")

    # Rule-cluster formation heuristic:
    # If max_strength pulls away from avg_strength over time while avg_degree increases,
    # that suggests formation of strong clusters of rules.
    strength_gap = maxS - avgS
    gap_slope = slope(cycles, strength_gap)
    print(f"Slope of (max_strength - avg_strength): {gap_slope:.6f}")

    if gap_slope > 0 and slope_deg_late > 0:
        print("Inference: Evidence of rule-cluster formation (strong hubs emerging in the skeleton).")
    elif gap_slope < 0 and slope_deg_late > 0:
        print("Inference: Skeleton densifying but rule strengths are homogenizing.")
    else:
        print("Inference: No strong signature of clustered vs uniform rule structure yet.")
    print()


# ----------------------------------------------------------
# 3. Multiple phase transitions
# ----------------------------------------------------------

def analyze_phase_transitions(log, n_bins=25):
    """
    Detect multiple phase transitions by:
    - computing CRUNCH fraction vs entropy
    - looking for multiple large jumps
    """
    entropy = log["entropy"]
    phase = log["phase"]

    phase_num = np.array([1 if p == "CRUNCH" else 0 for p in phase])

    bins = np.linspace(entropy.min(), entropy.max(), n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    crunch_fraction = []

    for i in range(len(bins) - 1):
        mask = (entropy >= bins[i]) & (entropy < bins[i + 1])
        if mask.any():
            crunch_fraction.append(phase_num[mask].mean())
        else:
            crunch_fraction.append(np.nan)
    crunch_fraction = np.array(crunch_fraction)

    print("=== Phase transition analysis ===")
    print("Entropy bins and CRUNCH fraction:")
    for c, f in zip(bin_centers, crunch_fraction):
        if np.isnan(f):
            continue
        print(f"  H≈{c:.3f}: CRUNCH fraction={f:.2f}")

    valid = ~np.isnan(crunch_fraction)
    cf = crunch_fraction[valid]
    bc = bin_centers[valid]

    if len(cf) < 3:
        print("Not enough phase variation to detect transitions.")
        print()
        return

    diffs = np.diff(cf)
    abs_diffs = np.abs(diffs)

    # Find top 2–3 strongest jumps
    k = min(3, len(abs_diffs))
    top_indices = np.argsort(abs_diffs)[-k:][::-1]

    transitions = []
    for idx in top_indices:
        t_entropy = 0.5 * (bc[idx] + bc[idx + 1])
        transitions.append((t_entropy, diffs[idx]))

    print("Strongest phase-change candidates (entropy, signed jump):")
    for ent, d in transitions:
        print(f"  H≈{ent:.3f}, ΔCRUNCH≈{d:.2f}")

    print("Inference:")
    if len(transitions) == 0:
        print("  No clear phase transitions detected.")
    else:
        for ent, d in transitions:
            if 0.23 <= ent <= 0.27:
                print(f"  - One candidate near 0.25 (H≈{ent:.3f}).")
            else:
                print(f"  - Transition around H≈{ent:.3f} (not near 0.25).")
    print()

    # Optional: quick plot of CRUNCH fraction vs entropy
    plt.figure()
    plt.plot(bc, cf, marker="o")
    plt.xlabel("Entropy")
    plt.ylabel("CRUNCH fraction")
    plt.title("Phase vs entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# 4. Skeleton densification & evolution over time
# ----------------------------------------------------------

def analyze_skeleton_densification(log, n_bins=10):
    """
    Check how avg_degree behaves as entropy drops (densification vs simplification),
    and visualize skeleton evolution over time.
    """
    entropy = log["entropy"]
    avg_deg = log["avg_degree"]
    cycles = log["cycle"]

    # Trend vs entropy
    order = np.argsort(entropy)
    ent_sorted = entropy[order]
    deg_sorted = avg_deg[order]

    bins = np.linspace(ent_sorted.min(), ent_sorted.max(), n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_deg = []
    for i in range(len(bins) - 1):
        mask = (ent_sorted >= bins[i]) & (ent_sorted < bins[i + 1])
        if mask.any():
            mean_deg.append(deg_sorted[mask].mean())
        else:
            mean_deg.append(np.nan)
    mean_deg = np.array(mean_deg)

    print("=== Skeleton densification analysis ===")
    for c, d in zip(bin_centers, mean_deg):
        if np.isnan(d):
            continue
        print(f"  H≈{c:.3f}: avg_degree≈{d:.2f}")

    valid = ~np.isnan(mean_deg)
    if valid.sum() >= 3:
        A = np.vstack([bin_centers[valid], np.ones(valid.sum())]).T
        m, _ = np.linalg.lstsq(A, mean_deg[valid], rcond=None)[0]
        m = float(m)
        print(f"Approximate slope of avg_degree vs entropy: {m:.3f}")
        if m < 0:
            print("Inference: Skeleton densifies as entropy drops (degree increases when H decreases).")
        elif m > 0:
            print("Inference: Skeleton simplifies as entropy drops (degree decreases when H decreases).")
        else:
            print("Inference: No clear trend in skeleton density vs entropy.")
    else:
        print("Not enough data for a robust fit.")
    print()

    # Visualize evolution over time
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(cycles, avg_deg, label="avg_degree")
    plt.xlabel("Cycle")
    plt.ylabel("avg_degree")
    plt.title("Skeleton avg_degree over time")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(cycles, entropy, label="entropy", color="orange")
    plt.xlabel("Cycle")
    plt.ylabel("Entropy")
    plt.title("Entropy over time")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    log = load_log(LOG_PATH)

    analyze_attractor_behavior(log, tail_fraction=0.2)
    analyze_rule_ecology(log)
    analyze_phase_transitions(log, n_bins=25)
    analyze_skeleton_densification(log, n_bins=10)


if __name__ == "__main__":
    main()
