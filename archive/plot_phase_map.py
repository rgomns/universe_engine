import os
import json
import matplotlib.pyplot as plt

RUNS_DIR = "runs"
SWEEP_FILE = os.path.join(RUNS_DIR, "phase_sweep.json")

PHASE_COLORS = {
    "subcritical": "tab:blue",
    "critical": "tab:green",
    "supercritical": "tab:red",
}

PHASE_MARKERS = {
    "subcritical": "o",
    "critical": "s",
    "supercritical": "D",
}

def main():
    with open(SWEEP_FILE, "r") as f:
        sweep = json.load(f)

    runs = sweep["runs"]

    # 1) Phase diagram in parameter space
    plt.figure(figsize=(7, 6))
    for r in runs:
        cfg = r["config"]
        phase = r["phase"]
        x = cfg["P_NEW_CONSTRAINT"]
        y = cfg["GLOBAL_DECAY"]

        color = PHASE_COLORS.get(phase, "k")
        marker = PHASE_MARKERS.get(phase, "x")

        plt.scatter(x, y, color=color, marker=marker, s=80, alpha=0.8)

    # build legend manually
    for phase, color in PHASE_COLORS.items():
        plt.scatter([], [], color=color, marker=PHASE_MARKERS[phase], label=phase)

    plt.xlabel("P_NEW_CONSTRAINT")
    plt.ylabel("GLOBAL_DECAY")
    plt.title("Emergent law phases in parameter space")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Phase")
    plt.tight_layout()

    out_path = os.path.join(RUNS_DIR, "phase_diagram_full.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved full phase diagram to {out_path}")

    # 2) Optional: entropy vs. #constraints, colored by phase
    plt.figure(figsize=(7, 6))
    for r in runs:
        phase = r["phase"]
        x = r["final_entropy"]
        y = r["final_num_constraints"]

        color = PHASE_COLORS.get(phase, "k")
        marker = PHASE_MARKERS.get(phase, "x")

        plt.scatter(x, y, color=color, marker=marker, s=80, alpha=0.8)

    for phase, color in PHASE_COLORS.items():
        plt.scatter([], [], color=color, marker=PHASE_MARKERS[phase], label=phase)

    plt.xlabel("Final entropy")
    plt.ylabel("Final #constraints")
    plt.title("Emergent phases in behavior space")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Phase")
    plt.tight_layout()

    out_path2 = os.path.join(RUNS_DIR, "phase_diagram_behavior_full.png")
    plt.savefig(out_path2, dpi=200)
    plt.close()
    print(f"Saved behavioral phase diagram to {out_path2}")


if __name__ == "__main__":
    main()
