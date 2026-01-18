import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------
# LOAD JSONL RESULTS
# ---------------------------------------------------------

def load_results(filename):
    results = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


# ---------------------------------------------------------
# ORGANIZE RESULTS INTO A PARAMETER GRID
# ---------------------------------------------------------

def build_grid(results):
    # Extract unique parameter values
    lts = sorted(set(r["local_theta"] for r in results))
    cts = sorted(set(r["coupling_theta"] for r in results))
    mis = sorted(set(r["measure_interval"] for r in results))
    ers = sorted(set(r["emergence_rate"] for r in results))

    # Build lookup table
    table = {}
    for r in results:
        key = (r["local_theta"], r["coupling_theta"],
               r["measure_interval"], r["emergence_rate"])
        table[key] = r

    return lts, cts, mis, ers, table


# ---------------------------------------------------------
# HEATMAP UTILITY
# ---------------------------------------------------------

def plot_heatmap(x_vals, y_vals, Z, title, xlabel, ylabel):
    plt.figure(figsize=(6, 5))
    plt.imshow(Z, origin="lower", cmap="viridis",
               extent=[min(x_vals), max(x_vals),
                       min(y_vals), max(y_vals)],
               aspect="auto")
    plt.colorbar(label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# BUILD HEATMAPS FOR EACH MEASURE_INTERVAL SLICE
# ---------------------------------------------------------

from matplotlib.animation import PillowWriter

def visualize(results):
    lts, cts, mis, ers, table = build_grid(results)

    # Prepare animation writer
    writer = PillowWriter(fps=2)  # 2 frames per second
    fig = plt.figure(figsize=(6, 5))

    with writer.saving(fig, "phase_diagram.gif", dpi=100):

        for mi in mis:
            Z_coh = np.zeros((len(cts), len(ers)))
            Z_agent = np.zeros((len(cts), len(ers)))
            Z_rules = np.zeros((len(cts), len(ers)))
            Z_osc = np.zeros((len(cts), len(ers)))

            for i, ct in enumerate(cts):
                for j, er in enumerate(ers):
                    key = (lts[0], ct, mi, er)
                    r = table[key]
                    Z_coh[i, j] = r["coherence_mean"]
                    Z_agent[i, j] = r["agent_entropy_mean"]
                    Z_rules[i, j] = r["rules_mean"]
                    Z_osc[i, j] = r["entropy_osc_score"]

            # ---- FRAME 1: Coherence ----
            plt.clf()
            plt.imshow(Z_coh, origin="lower", cmap="viridis",
                       extent=[min(ers), max(ers),
                               min(cts), max(cts)],
                       aspect="auto")
            plt.colorbar(label="Coherence")
            plt.xlabel("emergence_rate")
            plt.ylabel("coupling_theta")
            plt.title(f"Coherence (measure_interval={mi})")
            plt.tight_layout()
            writer.grab_frame()

            # ---- FRAME 2: Agent Entropy ----
            plt.clf()
            plt.imshow(Z_agent, origin="lower", cmap="viridis",
                       extent=[min(ers), max(ers),
                               min(cts), max(cts)],
                       aspect="auto")
            plt.colorbar(label="Agent Entropy")
            plt.xlabel("emergence_rate")
            plt.ylabel("coupling_theta")
            plt.title(f"Agent Entropy (measure_interval={mi})")
            plt.tight_layout()
            writer.grab_frame()

            # ---- FRAME 3: Rule Density ----
            plt.clf()
            plt.imshow(Z_rules, origin="lower", cmap="viridis",
                       extent=[min(ers), max(ers),
                               min(cts), max(cts)],
                       aspect="auto")
            plt.colorbar(label="Rule Density")
            plt.xlabel("emergence_rate")
            plt.ylabel("coupling_theta")
            plt.title(f"Rule Density (measure_interval={mi})")
            plt.tight_layout()
            writer.grab_frame()

            # ---- FRAME 4: Oscillation Score ----
            plt.clf()
            plt.imshow(Z_osc, origin="lower", cmap="viridis",
                       extent=[min(ers), max(ers),
                               min(cts), max(cts)],
                       aspect="auto")
            plt.colorbar(label="Oscillation Score")
            plt.xlabel("emergence_rate")
            plt.ylabel("coupling_theta")
            plt.title(f"Oscillation Score (measure_interval={mi})")
            plt.tight_layout()
            writer.grab_frame()

    print("Saved GIF: phase_diagram.gif")



# ---------------------------------------------------------
# SCATTER PLOTS FOR CORRELATIONS
# ---------------------------------------------------------

def scatter_correlations(results):
    coh = np.array([r["coherence_mean"] for r in results])
    agent = np.array([r["agent_entropy_mean"] for r in results])
    rules = np.array([r["rules_mean"] for r in results])
    osc = np.array([r["entropy_osc_score"] for r in results])

    plt.figure(figsize=(6, 5))
    plt.scatter(coh, agent, c="blue", alpha=0.7)
    plt.xlabel("Coherence")
    plt.ylabel("Agent Entropy")
    plt.title("Coherence vs Agent Entropy")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(coh, rules, c="green", alpha=0.7)
    plt.xlabel("Coherence")
    plt.ylabel("Rule Density")
    plt.title("Coherence vs Rule Density")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(coh, osc, c="purple", alpha=0.7)
    plt.xlabel("Coherence")
    plt.ylabel("Oscillation Score")
    plt.title("Coherence vs Oscillation Score")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    filename = "phase_diagram_results_stream.json"
    results = load_results(filename)

    visualize(results)
    scatter_correlations(results)
