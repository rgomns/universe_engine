import os
import json
from datetime import datetime
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm  # pip install tqdm

from universe_engine import run_universe  # adjust import if needed

RUNS_DIR = "runs"
SWEEP_FILE = os.path.join(RUNS_DIR, "phase_sweep.json")


# ------------------------------------------------------------
#  PHASE CLASSIFIER (same logic, but factored)
# ------------------------------------------------------------

def classify_phase(summary):
    final_constraints = summary["final_num_constraints"]
    fundamentals = summary["num_fundamentals"]
    entropy = summary["final_entropy"]

    avg_strength = summary.get("avg_strength")
    oldest_age = summary.get("oldest_constraint_age")
    components = summary.get("components")
    largest_cc = summary.get("largest_cc")
    avg_degree = summary.get("avg_degree")

    # Supercritical: dense, strong, immortal-ish laws
    if (
        (fundamentals is not None and fundamentals >= 50)
        or (final_constraints >= 90 and avg_strength and avg_strength > 0.5)
        or (oldest_age and oldest_age > 3000)
        or (avg_degree and avg_degree > 5.5)
    ):
        return "supercritical"

    # Subcritical: weak, fragmented, low constraints
    if (
        final_constraints < 60
        or (fundamentals is not None and fundamentals < 5 and final_constraints < 80)
        or (components and components > 3)
        or (largest_cc and largest_cc < 40)
        or (avg_degree and avg_degree < 3.5)
        or (entropy > 0.90 and final_constraints < 60)
    ):
        return "subcritical"

    # Otherwise: critical
    return "critical"


# ------------------------------------------------------------
#  SINGLE POINT RUNNER (for use in pool)
# ------------------------------------------------------------

def run_single_point(args):
    p_new, decay, enable_3bit = args
    label = f"SWEEP_P{p_new:.3f}_D{decay:.3f}"

    records, patterns, lineage, final_omega = run_universe(
        evolving=True,
        label=label,
        p_new_constraint=p_new,
        global_decay=decay,
        enable_3bit=enable_3bit,
    )

    if records:
        last = records[-1]
        final_cycle = last.get("cycle", 0)
        final_entropy = last.get("entropy", 0.0)
        final_constraints = last.get("num_constraints", 0)
        num_phase_transitions = sum(1 for r in records if r.get("phase_transition"))

        avg_strength = last.get("avg_strength")
        oldest_age = last.get("oldest_constraint_age")
        components = last.get("components")
        largest_cc = last.get("largest_cc")
        avg_degree = last.get("avg_degree")
    else:
        final_cycle = 0
        final_entropy = 0.0
        final_constraints = 0
        num_phase_transitions = 0
        avg_strength = None
        oldest_age = None
        components = None
        largest_cc = None
        avg_degree = None

    FUNDAMENTAL_LIFESPAN = 2000
    fundamentals = [c for c in lineage if c.get("lifespan", 0) >= FUNDAMENTAL_LIFESPAN]
    num_fundamentals = len(fundamentals)

    summary = {
        "label": label,
        "config": {
            "P_NEW_CONSTRAINT": p_new,
            "GLOBAL_DECAY": decay,
            "ENABLE_3BIT": enable_3bit,
        },
        "final_cycle": final_cycle,
        "final_omega": list(final_omega) if isinstance(final_omega, set) else final_omega,
        "final_entropy": final_entropy,
        "final_num_constraints": final_constraints,
        "num_fundamentals": num_fundamentals,
        "num_phase_transitions": num_phase_transitions,
        "avg_strength": avg_strength,
        "oldest_constraint_age": oldest_age,
        "components": components,
        "largest_cc": largest_cc,
        "avg_degree": avg_degree,
    }   


    summary["phase"] = classify_phase(summary)

    # Save per-run summary
    run_dir = os.path.join(RUNS_DIR, label)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ------------------------------------------------------------
#  MAIN PARALLEL SWEEP
# ------------------------------------------------------------

def main():
    os.makedirs(RUNS_DIR, exist_ok=True)

    p_new_values = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.24]
    decay_values = [0.01, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20]

    points = [(p, d, True) for p, d in product(p_new_values, decay_values)]

    sweep_results = {
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "p_new_values": p_new_values,
            "decay_values": decay_values,
        },
        "runs": []
    }

    # Use all cores minus one by default
    n_workers = max(cpu_count() - 1, 1)

    with Pool(processes=n_workers) as pool:
        for summary in tqdm(pool.imap_unordered(run_single_point, points), total=len(points)):
            sweep_results["runs"].append(summary)
            # incremental save
            with open(SWEEP_FILE, "w") as f:
                json.dump(sweep_results, f, indent=2)

    # Final save (sorted by p, decay for readability)
    sweep_results["runs"].sort(
        key=lambda r: (r["config"]["P_NEW_CONSTRAINT"], r["config"]["GLOBAL_DECAY"])
    )
    with open(SWEEP_FILE, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\nParallel sweep complete. Saved to {SWEEP_FILE}")


if __name__ == "__main__":
    main()
