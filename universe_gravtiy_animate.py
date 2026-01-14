import json
import csv
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# LOAD STATE (for couplings + adjacency + layout)
# ----------------------------------------------------------

def load_state(state_file):
    with open(state_file, "r") as f:
        data = json.load(f)

    n_bits = len(data["population"][0])
    couplings = []
    for s, f_forb, st in data["couplings"]:
        scope = np.array(s, dtype=np.int64)
        couplings.append((scope, f_forb, st))

    return n_bits, couplings


def build_graph(n_bits, couplings, strength_threshold=0.15):
    G = nx.Graph()
    G.add_nodes_from(range(n_bits))

    for scope, _, strength in couplings:
        if strength < strength_threshold:
            continue
        s = len(scope)
        for i in range(s):
            for j in range(i + 1, s):
                a, b = int(scope[i]), int(scope[j])
                if G.has_edge(a, b):
                    G[a][b]["weight"] += strength
                else:
                    G.add_edge(a, b, weight=strength)

    return G


# ----------------------------------------------------------
# LOAD TIME-SERIES LOGS
# ----------------------------------------------------------

def load_degrees(degree_csv):
    df = pd.read_csv(degree_csv)
    cycles = df["cycle"].values
    degrees = df.drop(columns=["cycle"]).values
    return cycles, degrees


def load_mass(mass_csv):
    df = pd.read_csv(mass_csv)
    cycles = df["cycle"].values
    mass = df.drop(columns=["cycle"]).values
    return cycles, mass


# ----------------------------------------------------------
# ANIMATION
# ----------------------------------------------------------

def animate_universe(state_file, degree_csv, mass_csv, out_dir="frames", strength_threshold=0.15):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("[LOAD] Loading state...")
    n_bits, couplings = load_state(state_file)

    print("[GRAPH] Building skeleton graph...")
    G = build_graph(n_bits, couplings, strength_threshold=strength_threshold)

    print("[LAYOUT] Computing fixed layout...")
    pos = nx.spring_layout(G, seed=42, k=0.3)  # fixed layout for stable animation

    print("[LOGS] Loading degree & mass logs...")
    cycles_deg, degrees = load_degrees(degree_csv)
    cycles_mass, mass = load_mass(mass_csv)

    # Align cycles
    cycles = np.intersect1d(cycles_deg, cycles_mass)
    print(f"[INFO] Found {len(cycles)} cycles to animate.")

    # Build lookup tables
    deg_map = {c: degrees[i] for i, c in enumerate(cycles_deg)}
    mass_map = {c: mass[i] for i, c in enumerate(cycles_mass)}

    # ------------------------------------------------------
    # Generate frames
    # ------------------------------------------------------
    for idx, cycle in enumerate(cycles):
        print(f"[FRAME] Cycle {cycle} ({idx+1}/{len(cycles)})")

        node_deg = deg_map[cycle]
        node_mass = mass_map[cycle]

        # Normalize for visualization
        size = 200 + 20 * node_mass
        color = node_deg

        plt.figure(figsize=(8, 8))
        plt.title(f"Universe at Cycle {cycle}", fontsize=16)

        nx.draw_networkx_nodes(
            G, pos,
            node_size=size,
            node_color=color,
            cmap="viridis",
            linewidths=0.5,
            edgecolors="black"
        )
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.3,
            width=0.5
        )

        plt.axis("off")
        plt.tight_layout()

        frame_path = os.path.join(out_dir, f"frame_{cycle:06d}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close()

    print(f"[DONE] Frames saved to {out_dir}/")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

if __name__ == "__main__":
    run_id = "universe_run_20260109_223705"  # <-- CHANGE THIS
    state_file = f"{run_id}_state.json"
    degree_csv = f"{run_id}_degrees.csv"
    mass_csv = f"{run_id}_mass.csv"

    animate_universe(state_file, degree_csv, mass_csv)
