import json
import ast
import numpy as np
import pandas as pd
import networkx as nx
import os

# ==========================================================
# 1. LOADING STATE & GRAPH
# ==========================================================

def load_state(state_file):
    """
    Load a saved universe state and reconstruct:
    - n_bits
    - couplings
    - mass (array)
    """
    with open(state_file, "r") as f:
        data = json.load(f)
    n_bits = len(data["population"][0])
    couplings = []
    for s, f_forb, st in data["couplings"]:
        scope = np.array(s, dtype=np.int64)
        couplings.append((scope, f_forb, st))
    mass = np.array(data.get("mass", [0.0] * n_bits), dtype=np.float32)
    return n_bits, couplings, mass, data


def build_skeleton_graph(n_bits, couplings, strength_threshold=0.0):
    """
    Build skeleton graph from couplings.
    Nodes: bits
    Edges: between bits that co-occur in couplings with strength > threshold.
    Edge weight = sum of strengths.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_bits))
    for scope, _, strength in couplings:
        if strength <= strength_threshold:
            continue
        s = len(scope)
        if s < 2:
            continue
        for i in range(s):
            for j in range(i + 1, s):
                a, b = int(scope[i]), int(scope[j])
                if G.has_edge(a, b):
                    G[a][b]["weight"] += strength
                else:
                    G.add_edge(a, b, weight=strength)
    return G

# ==========================================================
# 2. PHASE SEGMENTATION (FROM METRICS CSV)
# ==========================================================

def load_phase_segments(metric_csv, min_segment_len=200):
    """
    From metrics.csv, extract contiguous segments of constant geom_phase
    with length >= min_segment_len (in cycles).
    Returns: list of (phase_name, start_cycle, end_cycle)
    """
    df = pd.read_csv(metric_csv)
    segments = []
    if df.empty:
        return segments

    current_phase = None
    start_cycle = None

    for _, row in df.iterrows():
        cycle = int(row["cycle"])
        phase = row["geom_phase"]
        if current_phase is None:
            current_phase = phase
            start_cycle = cycle
        elif phase != current_phase:
            if cycle - start_cycle >= min_segment_len:
                segments.append((current_phase, start_cycle, cycle - 1))
            current_phase = phase
            start_cycle = cycle

    # last segment
    last_cycle = int(df["cycle"].iloc[-1])
    if last_cycle - start_cycle >= min_segment_len:
        segments.append((current_phase, start_cycle, last_cycle))

    return segments

# ==========================================================
# 3. GRAVITY CSV HELPERS & WELL ANALYSIS
# ==========================================================

def parse_list_field(x):
    """
    gravity.csv fields are stored as strings like "[1, 2, 3]" or "[0.1, 0.2]".
    This parses them safely.
    """
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return []
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []


def detect_stable_wells_and_planets_phase(df_grav,
                                          mass_threshold=10.0,
                                          min_lifetime=200):
    """
    From a gravity.csv slice (already filtered to a phase or cycle window),
    detect:
    - stable wells: bits that appear in top_indices over many cycles with high mass
    - planetary systems: wells with significant orbit frequency

    Returns:
      summary dict with:
        - stable_wells
        - planetary_wells
        - multi_well_cycles
        - well_time (bit -> list of (cycle, mass, orbit_freq))
    """
    well_time = {}       # bit -> list of (cycle, mass, orbit_freq)
    multi_well_cycles = []

    for _, row in df_grav.iterrows():
        cycle = int(row["cycle"])
        top_idx = parse_list_field(row["top_indices"])
        top_masses = parse_list_field(row["top_masses"])
        orbit_freqs = parse_list_field(row["mass_orbit_freqs"])

        if not top_idx or not top_masses:
            continue

        heavy_mask = [m >= mass_threshold for m in top_masses]
        if sum(heavy_mask) >= 2:
            multi_well_cycles.append(cycle)

        for i, m, orb in zip(top_idx, top_masses, orbit_freqs):
            bit = int(i)
            if bit not in well_time:
                well_time[bit] = []
            well_time[bit].append((cycle, float(m), float(orb)))

    stable_wells = {}
    for bit, history in well_time.items():
        history = sorted(history, key=lambda x: x[0])
        cycles = [c for c, _, _ in history]
        if not cycles:
            continue
        lifetime = cycles[-1] - cycles[0]
        max_mass = max(m for _, m, _ in history)
        avg_orbit = float(np.mean([orb for _, _, orb in history]))
        if lifetime >= min_lifetime and max_mass >= mass_threshold:
            stable_wells[bit] = {
                "first_cycle": cycles[0],
                "last_cycle": cycles[-1],
                "lifetime": lifetime,
                "max_mass": max_mass,
                "avg_orbit_freq": avg_orbit,
            }

    planetary_wells = {bit: info for bit, info in stable_wells.items()
                       if info["avg_orbit_freq"] > 0.05}

    summary = {
        "stable_wells": stable_wells,
        "planetary_wells": planetary_wells,
        "multi_well_cycles": multi_well_cycles,
        "well_time": well_time,
    }
    return summary


def detect_mass_merging(well_time, min_overlap=500, corr_threshold=0.6):
    """
    Detect "merging-like" behavior based on mass dynamics:

    For two wells a, b, we say they exhibit merging-like behavior if:
      - they overlap in time >= min_overlap cycles
      - the mass change time series in the overlap are strongly anti-correlated
        (corr(diff(ma), diff(mb)) <= -corr_threshold)

    Returns: list of (a, b, overlap_start, overlap_end, corr)
    """
    wells = list(well_time.keys())
    merging_pairs = []

    for i in range(len(wells)):
        for j in range(i + 1, len(wells)):
            a, b = wells[i], wells[j]
            hist_a = sorted(well_time[a], key=lambda x: x[0])
            hist_b = sorted(well_time[b], key=lambda x: x[0])

            ca = [c for c, _, _ in hist_a]
            cb = [c for c, _, _ in hist_b]
            if not ca or not cb:
                continue

            start = max(ca[0], cb[0])
            end = min(ca[-1], cb[-1])
            if end - start < min_overlap:
                continue

            ma = {c: m for c, m, _ in hist_a if start <= c <= end}
            mb = {c: m for c, m, _ in hist_b if start <= c <= end}
            common_cycles = sorted(set(ma.keys()) & set(mb.keys()))
            if len(common_cycles) < min_overlap // 2:
                continue

            series_a = np.array([ma[c] for c in common_cycles])
            series_b = np.array([mb[c] for c in common_cycles])

            da = np.diff(series_a)
            db = np.diff(series_b)
            if len(da) < 5:
                continue

            corr = np.corrcoef(da, db)[0, 1]
            if corr <= -corr_threshold:
                merging_pairs.append((a, b, start, end, corr))

    return merging_pairs

# ==========================================================
# 4. FORCE LAW ANALYSIS WITH MULTIPLE FIELD DEFINITIONS
# ==========================================================

def compute_fields_at_wells(G, mass, wells, R_max=5):
    """
    Compute different field variants at each well:
      - F2: sum_j m_j / d(i,j)^2
      - F1: sum_j m_j / d(i,j)
      - F0: sum_j m_j for d(i,j) <= R_max

    We associate each well with a characteristic radius r = mean graph distance
    from that well to other nodes in its connected component.

    Returns:
      dict: well -> (r, F0, F1, F2)
    """
    if G.number_of_nodes() == 0:
        return {}

    all_lengths = dict(nx.all_pairs_shortest_path_length(G))
    results = {}

    for i in wells:
        if i not in all_lengths:
            continue
        lengths = all_lengths[i]
        dists = [d for j, d in lengths.items() if j != i]
        if not dists:
            continue
        r = float(np.mean(dists))

        F2 = 0.0
        F1 = 0.0
        F0 = 0.0
        for j, d in lengths.items():
            if j == i:
                continue
            dist = float(d)
            if dist <= 0:
                continue
            F2 += mass[j] / (dist ** 2)
            F1 += mass[j] / dist
            if dist <= R_max:
                F0 += mass[j]
        results[i] = (r, F0, F1, F2)

    return results


def fit_exponent(r_list, f_list, min_field=1e-6):
    """
    Fit F ~ (1/r)^alpha via log-log regression.
    Returns (alpha, alpha_std) for the given samples, or (None, None) if not enough data.
    """
    r_arr = np.array(r_list, dtype=float)
    f_arr = np.array(f_list, dtype=float)

    mask = (r_arr > 0) & (f_arr > min_field) & (~np.isnan(r_arr)) & (~np.isnan(f_arr))
    r_arr = r_arr[mask]
    f_arr = f_arr[mask]
    if len(r_arr) < 3:
        return None, None

    log_inv_r = np.log(1.0 / r_arr)
    log_f = np.log(f_arr)

    alpha, _ = np.polyfit(log_inv_r, log_f, 1)
    return float(alpha), 0.0  # single fit; use across samples for variance


def aggregate_force_exponents(G, mass, wells, R_max=5):
    """
    For a set of wells, compute F0/F1/F2 fields, and fit exponents per variant.
    Returns:
      dict with keys "F0", "F1", "F2" -> (alpha_mean, alpha_std, n_samples)
    """
    fields = compute_fields_at_wells(G, mass, wells, R_max=R_max)
    r_vals = []
    F0_vals = []
    F1_vals = []
    F2_vals = []

    for _, (r, F0, F1, F2) in fields.items():
        r_vals.append(r)
        F0_vals.append(F0)
        F1_vals.append(F1)
        F2_vals.append(F2)

    results = {}
    for name, flist in [("F0", F0_vals), ("F1", F1_vals), ("F2", F2_vals)]:
        if len(r_vals) < 3:
            results[name] = (None, None, 0)
            continue

        # We can fit exponent from all wells at once
        alpha, _ = fit_exponent(r_vals, flist)
        # For now, std=0 at this level; to get real std, resample or segment further
        results[name] = (alpha, 0.0, len(r_vals))

    return results

# ==========================================================
# 5. LENSING-LIKE GEODESIC BEHAVIOR
# ==========================================================

def compute_geodesic_stats(G, mass, high_mass_fraction=0.1,
                           n_pairs=300, rng=None):
    """
    Check if geodesics tend to pass through high-mass nodes more often
    than expected by chance (a crude gravitational lensing analogue).

    Returns: (avg_len, avg_hits, hit_rate)
    """
    if G.number_of_nodes() == 0:
        print("[LENS] Empty graph, skipping.")
        return None, None, None

    if rng is None:
        rng = np.random.default_rng()

    nodes = list(G.nodes())
    n_bits = len(nodes)
    node_arr = np.array(nodes, dtype=int)

    sorted_idx = np.argsort(mass)[::-1]
    k = max(1, int(len(sorted_idx) * high_mass_fraction))
    high_mass_nodes = set(sorted_idx[:k])

    paths_length = []
    paths_highmass_hits = []

    for _ in range(n_pairs):
        a, b = rng.choice(node_arr, size=2, replace=False)
        try:
            path = nx.shortest_path(G, int(a), int(b))
        except nx.NetworkXNoPath:
            continue
        length = len(path) - 1
        hits = sum(1 for n in path[1:-1] if n in high_mass_nodes)
        paths_length.append(length)
        paths_highmass_hits.append(hits)

    if not paths_length:
        print("[LENS] No valid paths found.")
        return None, None, None

    avg_len = float(np.mean(paths_length))
    avg_hits = float(np.mean(paths_highmass_hits))
    hit_rate = avg_hits / avg_len if avg_len > 0 else 0.0

    return avg_len, avg_hits, hit_rate

# ==========================================================
# 6. DISCRETE CURVATURE & EFFECTIVE DIMENSION
# ==========================================================

def compute_discrete_curvature_tensors(G):
    """
    Compute simple discrete curvature quantities:

    - Node curvature ~ clustering coefficient (0..1)
    - Edge curvature (Forman-like proxy):
        K(e) = 2 - (deg(u) + deg(v))

    Returns:
      node_curv: dict node -> curvature
      edge_curv: dict (u,v) -> curvature
    """
    if G.number_of_nodes() == 0:
        return {}, {}

    node_curv = nx.clustering(G)
    edge_curv = {}
    deg = dict(G.degree())
    for u, v in G.edges():
        edge_curv[(u, v)] = 2.0 - (deg[u] + deg[v])
    return node_curv, edge_curv


def estimate_graph_dimension(G, max_radius=4, n_samples=50, rng=None):
    """
    Rough estimate of effective graph dimension D from
    scaling of ball volume |B(r)| ~ r^D.

    Returns: (D_mean, D_std) or (None, None) if not enough data.
    """
    if G.number_of_nodes() == 0:
        return None, None

    if rng is None:
        rng = np.random.default_rng()

    nodes = list(G.nodes())
    dims = []

    for _ in range(n_samples):
        center = int(rng.choice(nodes))
        lengths = nx.single_source_shortest_path_length(G, center, cutoff=max_radius)

        radii = []
        volumes = []
        for r in range(1, max_radius + 1):
            vol = sum(1 for d in lengths.values() if d <= r)
            if vol > 1:
                radii.append(r)
                volumes.append(vol)
        if len(radii) >= 2:
            log_r = np.log(np.array(radii, dtype=float))
            log_v = np.log(np.array(volumes, dtype=float))
            D, _ = np.polyfit(log_r, log_v, 1)
            dims.append(D)

    if not dims:
        return None, None

    D_mean = float(np.mean(dims))
    D_std = float(np.std(dims))
    return D_mean, D_std

# ==========================================================
# 7. HIGH-LEVEL PHASE-AWARE TEST RUNNER
# ==========================================================

def run_gravity_tests_phase_aware(state_file,
                                  gravity_csv,
                                  metric_csv,
                                  strength_threshold=0.15,
                                  mass_threshold=10.0,
                                  min_lifetime=200):
    """
    High-level entry:
      - load snapshot and graph
      - segment metrics into phase intervals
      - for each phase segment:
          * filter gravity.csv rows by cycle ∈ segment
          * detect wells, planets, merging-like behavior
          * compute force exponents (F0/F1/F2)
          * compute lensing stats
          * compute curvature & dimension
      - print a compact summary
    """
    print("=== LOADING STATE & GRAPH ===")
    n_bits, couplings, mass, data = load_state(state_file)
    G = build_skeleton_graph(n_bits, couplings, strength_threshold=strength_threshold)
    print(f"[STATE] n_bits={n_bits}, nodes_in_graph={G.number_of_nodes()}, edges={G.number_of_edges()}")

    if not os.path.exists(metric_csv):
        print(f"[ERROR] metrics file not found: {metric_csv}")
        return
    df_metrics = pd.read_csv(metric_csv)

    if not os.path.exists(gravity_csv):
        print(f"[ERROR] gravity file not found: {gravity_csv}")
        return
    df_grav_all = pd.read_csv(gravity_csv)

    segments = load_phase_segments(metric_csv, min_segment_len=200)
    if not segments:
        print("[PHASE] No sufficiently long phase segments found; running on full data.")
        segments = [("ALL", int(df_metrics["cycle"].min()), int(df_metrics["cycle"].max()))]

    rng = np.random.default_rng()

    for phase_name, c_start, c_end in segments:
        print("\n==============================================")
        print(f"[PHASE] {phase_name} | cycles {c_start}–{c_end}")
        print("==============================================")

        # restrict gravity data to this cycle window
        df_grav = df_grav_all[
            (df_grav_all["cycle"] >= c_start) &
            (df_grav_all["cycle"] <= c_end)
        ].copy()
        if df_grav.empty:
            print("[PHASE] No gravity data in this window.")
            continue

        # 1) stable wells & planets
        summary = detect_stable_wells_and_planets_phase(
            df_grav,
            mass_threshold=mass_threshold,
            min_lifetime=min_lifetime
        )
        stable_wells = summary["stable_wells"]
        planetary_wells = summary["planetary_wells"]
        multi_well_cycles = summary["multi_well_cycles"]
        well_time = summary["well_time"]

        print(f"[WELLS] {len(stable_wells)} stable wells (lifetime >= {min_lifetime}, mass >= {mass_threshold}).")
        for bit, info in stable_wells.items():
            print(f"   - Well @ {bit}: lifetime={info['lifetime']}, max_mass={info['max_mass']:.2f}, avg_orbit={info['avg_orbit_freq']:.3f}")
        print(f"[PLANETS] {len(planetary_wells)} wells with significant orbiting activity (avg_orbit_freq>0.05).")
        print(f"[MULTI-WELL] cycles with >=2 heavy wells: {len(multi_well_cycles)}")

        # 2) merging-like behavior from mass dynamics
        merging_pairs = detect_mass_merging(well_time, min_overlap=500, corr_threshold=0.6)
        if merging_pairs:
            print("[MERGE] Pairs with strongly anti-correlated mass changes (potential competitive/merging dynamics):")
            for a, b, s, e, corr in merging_pairs[:10]:
                print(f"   - Wells {a},{b}: overlap {s}–{e}, corr(diff(ma),diff(mb))={corr:.3f}")
            if len(merging_pairs) > 10:
                print(f"       ... {len(merging_pairs)-10} more pairs omitted.")
        else:
            print("[MERGE] No strong mass-anticorrelation pairs detected in this phase.")

        # 3) force exponents from different field definitions
        well_bits = list(stable_wells.keys())
        if not well_bits:
            print("[FORCE] No stable wells in this phase for exponent fitting.")
        else:
            force_results = aggregate_force_exponents(G, mass, well_bits, R_max=5)
            for name, (alpha, alpha_std, n_samp) in force_results.items():
                if alpha is None:
                    print(f"[FORCE][{name}] Not enough samples to fit exponent.")
                else:
                    print(f"[FORCE][{name}] alpha ~ {alpha:.3f} (n={n_samp})  F_{name} ~ (1/r)^alpha")

        # 4) lensing-like stats
        avg_len, avg_hits, hit_rate = compute_geodesic_stats(
            G, mass,
            high_mass_fraction=0.1,
            n_pairs=500,
            rng=rng
        )
        if avg_len is not None:
            print(f"[LENS] avg geodesic length={avg_len:.3f}, avg high-mass hits={avg_hits:.3f}, hit_rate={hit_rate:.3f}")

        # 5) curvature & dimension
        node_curv, edge_curv = compute_discrete_curvature_tensors(G)
        if node_curv:
            vals = np.array(list(node_curv.values()), dtype=float)
            print(f"[CURV][nodes] mean={vals.mean():.3f}, std={vals.std():.3f}, min={vals.min():.3f}, max={vals.max():.3f}")
        else:
            print("[CURV][nodes] no data.")

        if edge_curv:
            e_vals = np.array(list(edge_curv.values()), dtype=float)
            print(f"[CURV][edges] mean={e_vals.mean():.3f}, std={e_vals.std():.3f}, min={e_vals.min():.3f}, max={e_vals.max():.3f}")
        else:
            print("[CURV][edges] no data.")

        D_mean, D_std = estimate_graph_dimension(G, max_radius=4, n_samples=50, rng=rng)
        if D_mean is not None:
            print(f"[DIM] effective graph dimension ~ {D_mean:.2f} ± {D_std:.2f}")
        else:
            print("[DIM] not enough data to estimate dimension.")

    print("\n[COMPLETE] Phase-aware gravity tests finished.")


# ==========================================================
# 8. SCRIPT ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    # Adjust this to match a specific run
    run_id = r"C:\universe_engine\universe_run_20260112_104006"

    state_file = f"{run_id}_state.json"
    gravity_csv = f"{run_id}_gravity.csv"
    metric_csv = f"{run_id}_metrics.csv"

    if not os.path.exists(state_file):
        print(f"[ERROR] state file not found: {state_file}")
    elif not os.path.exists(gravity_csv):
        print(f"[ERROR] gravity file not found: {gravity_csv}")
    elif not os.path.exists(metric_csv):
        print(f"[ERROR] metrics file not found: {metric_csv}")
    else:
        run_gravity_tests_phase_aware(
            state_file=state_file,
            gravity_csv=gravity_csv,
            metric_csv=metric_csv,
            strength_threshold=0.15,
            mass_threshold=10.0,
            min_lifetime=200,
        )
