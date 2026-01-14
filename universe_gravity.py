import numpy as np
import networkx as nx
import time
import json
import os
import threading
import csv

class Persistent_AGI_Universe:
    def __init__(self, n_bits=256, pop_size=2000, log_prefix="universe_run"):
        self.n_bits = n_bits
        self.pop_size = pop_size
        self.population = np.random.randint(0, 2, (pop_size, n_bits), dtype=np.int8)
        self.couplings = []  # [scope, forbidden, strength]
        self.cycle = 0
        self.record_low = 1.0
        self.noise_level = 0.00
        self.sampling_step = 50
        self.weights = {
            2: np.array([2, 1], dtype=np.int8),
            3: np.array([4, 2, 1], dtype=np.int8),
        }
        self.start_time = time.time()
        self.running = True

        # --- EMERGENT GEOMETRY / GRAVITY STATE ---
        self.adjacency = np.zeros((n_bits, n_bits), dtype=np.float32)
        self.mass = np.zeros(n_bits, dtype=np.float32)  # emergent mass per bit
        self.mass_scale = 3.0      # saturation scale for mass
        self.min_attraction = 0.1  # baseline attraction
        self.last_scopes = []      # for orbit-like pattern tracking
        self.scope_history_max = 500  # how many recent scopes to keep

        # --- LOGGING SETUP ---
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{log_prefix}_{timestamp}"
        self.state_file = f"{self.run_id}_state.json"
        self.metric_log_file = f"{self.run_id}_metrics.csv"
        self.degree_log_file = f"{self.run_id}_degrees.csv"
        self.mass_log_file = f"{self.run_id}_mass.csv"
        self.gravity_log_file = f"{self.run_id}_gravity.csv"  # wells, curvature, force, orbits

        self._init_metric_log()
        self._init_degree_log()
        self._init_mass_log()
        self._init_gravity_log()

    # ---------- LOGGING HELPERS ----------

    def _init_metric_log(self):
        with open(self.metric_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cycle",
                "entropy",
                "record_low",
                "lcc",
                "phase",
                "avg_degree",
                "max_degree",
                "avg_path_length",
                "clustering",
                "curvature",
                "geom_phase",
            ])

    def _init_degree_log(self):
        with open(self.degree_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["cycle"] + [f"deg_{i}" for i in range(self.n_bits)]
            writer.writerow(header)

    def _init_mass_log(self):
        with open(self.mass_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["cycle"] + [f"mass_{i}" for i in range(self.n_bits)]
            writer.writerow(header)

    def _init_gravity_log(self):
        """
        Logs high-level gravity/geometry relations per cycle for visibility:
        - top_k massive bits
        - local curvature around them
        - local avg distance
        - effective field strength
        - mass–curvature correlation
        - inverse-distance-like behavior (correlation between mass_i and 1/avg_dist_i)
        - orbit-like measure: how often each massive bit appears in new scopes
        """
        with open(self.gravity_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "cycle",
                "top_indices",           # list as string
                "top_masses",            # list as string
                "top_local_clustering",  # list as string
                "top_local_avg_dist",    # list as string
                "top_field_strength",    # list as string
                "mass_curvature_corr",   # scalar
                "mass_invdist_corr",     # scalar
                "mass_orbit_freqs",      # list as string
            ]
            writer.writerow(header)

    def log_metrics(self, cycle, entropy, lcc, phase,
                    avg_deg, max_deg, avg_path_len,
                    clustering, curvature, geom_phase):
        with open(self.metric_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                cycle,
                entropy,
                self.record_low,
                lcc,
                phase,
                avg_deg,
                max_deg,
                avg_path_len,
                clustering,
                curvature,
                geom_phase,
            ])

    def log_degrees(self, cycle, degrees):
        with open(self.degree_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = [cycle] + list(degrees)
            writer.writerow(row)

    def log_mass(self, cycle):
        with open(self.mass_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            row = [cycle] + list(self.mass)
            writer.writerow(row)

    def log_gravity(self, cycle, top_indices, top_masses,
                    top_local_clustering, top_local_avg_dist,
                    top_field_strength, mass_curv_corr,
                    mass_invdist_corr, mass_orbit_freqs):
        with open(self.gravity_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                cycle,
                str(list(map(int, top_indices))),
                str([float(x) for x in top_masses]),
                str([float(x) for x in top_local_clustering]),
                str([float(x) for x in top_local_avg_dist]),
                str([float(x) for x in top_field_strength]),
                float(mass_curv_corr) if mass_curv_corr is not None else "",
                float(mass_invdist_corr) if mass_invdist_corr is not None else "",
                str([float(x) for x in mass_orbit_freqs]),
            ])

    # ---------- GEOMETRY / GRAVITY UTILITIES ----------

    def update_emergent_geometry(self):
        """
        Build adjacency from couplings and apply mild decay.
        """
        self.adjacency.fill(0.0)
        for scope, _, strength in self.couplings:
            s = len(scope)
            if s < 2:
                continue
            for i in range(s):
                for j in range(i + 1, s):
                    a, b = scope[i], scope[j]
                    self.adjacency[a, b] += strength
                    self.adjacency[b, a] += strength

        # mild decay to avoid freezing history
        self.adjacency *= 0.99

    def update_mass_from_couplings(self):
        """
        Emergent mass per bit = sum of strengths of couplings involving that bit.
        """
        self.mass.fill(0.0)
        for scope, _, strength in self.couplings:
            for idx in scope:
                self.mass[idx] += strength

    def emergent_distance(self, i, j):
        """
        Simple emergent distance: larger adjacency => smaller distance.
        If adjacency=0, distance defaults to 1.
        """
        w = self.adjacency[i, j]
        return 1.0 / (1.0 + w)

    def effective_attraction(self):
        """
        Effective attraction per bit from mass with saturation.

        raw ~ min_attraction + mass
        saturation ~ 1 / (1 + mass / mass_scale)
        effective = raw * saturation

        - For small mass: grows with mass (gravity-like).
        - Around mass ~ mass_scale: peaks.
        - For very large mass: decays (saturation).
        """
        raw = self.min_attraction + self.mass
        saturation = 1.0 / (1.0 + (self.mass / self.mass_scale))
        eff = raw * saturation
        # Avoid zero everywhere
        if np.all(eff <= 0):
            eff = np.ones_like(eff)
        return eff

    def choose_scope_emergent_gravity(self, size):
        """
        Choose a scope using both:
        - emergent geometry (adjacency / distance)
        - emergent gravity from mass with saturation
        """
        eff = self.effective_attraction()

        # If no geometry yet, just use attraction-only bias
        if not self.couplings or np.all(self.adjacency == 0):
            probs = eff / eff.sum()
            return np.random.choice(self.n_bits, size, replace=False, p=probs)

        scope = []

        # First bit: choose by global attraction (mass-based)
        probs = eff / eff.sum()
        first = np.random.choice(self.n_bits, p=probs)
        scope.append(first)

        # Subsequent bits: combine locality and attraction
        while len(scope) < size:
            candidates = np.setdiff1d(np.arange(self.n_bits), np.array(scope))
            if len(candidates) == 0:
                break

            # distance to existing scope
            dists = []
            for c in candidates:
                d = np.mean([self.emergent_distance(c, s) for s in scope])
                dists.append(d)
            dists = np.array(dists)

            # locality: closer = higher weight
            locality_weight = 1.0 / (dists + 1e-9)

            # attraction for these candidates
            cand_attraction = eff[candidates]

            # combine: locality * attraction
            combined = locality_weight * cand_attraction
            if np.all(combined <= 0):
                combined = np.ones_like(combined)

            probs = combined / combined.sum()
            chosen = np.random.choice(candidates, p=probs)
            scope.append(chosen)

        return np.array(scope, dtype=np.int64)

    # ---------- METRICS ----------

    def get_entropy(self):
        p = np.mean(self.population, axis=0)
        return np.mean(-p * np.log2(p + 1e-9) - (1 - p) * np.log2(1 - p + 1e-9))

    def get_skeleton_graph(self):
        G = nx.Graph()
        for scope, _, _ in self.couplings:
            for i in range(len(scope)):
                for j in range(i + 1, len(scope)):
                    G.add_edge(scope[i], scope[j])
        return G

    def get_geometry_stats(self):
        """
        Return:
        - degrees vector
        - avg_degree
        - max_degree
        - avg_path_length on largest component
        - clustering on largest component
        - curvature proxy
        - geom_phase label
        """
        G = self.get_skeleton_graph()
        degrees = np.zeros(self.n_bits, dtype=np.int32)

        if not G.nodes:
            return degrees, 0.0, 0, np.nan, 0.0, 0.0, "GEOM_EMPTY"

        for i in range(self.n_bits):
            degrees[i] = G.degree[i] if i in G else 0

        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        H = G.subgraph(largest).copy()

        avg_deg = np.mean([d for _, d in H.degree()])
        max_deg = max([d for _, d in H.degree()])

        if H.number_of_nodes() > 1:
            try:
                avg_path_len = nx.average_shortest_path_length(H)
            except nx.NetworkXError:
                avg_path_len = np.nan
        else:
            avg_path_len = np.nan

        clustering = nx.average_clustering(H)

        # Curvature proxy: clustering / (scaled avg_path_length)
        N = H.number_of_nodes()
        if N > 1 and not np.isnan(avg_path_len):
            norm_factor = np.log(N + 1e-9)
            curvature = clustering / (avg_path_len / (norm_factor + 1e-9) + 1e-9)
        else:
            curvature = 0.0

        # Geometric phase
        lcc_ratio = len(largest) / self.n_bits
        if lcc_ratio < 0.25:
            geom_phase = "GEOM_DIFFUSE"
        elif lcc_ratio < 0.75:
            geom_phase = "GEOM_CLUSTERING"
        else:
            if clustering < 0.1:
                geom_phase = "GEOM_STRINGY"
            elif clustering < 0.3:
                geom_phase = "GEOM_MESH"
            else:
                geom_phase = "GEOM_RIGID"

        return degrees, avg_deg, max_deg, avg_path_len, clustering, curvature, geom_phase

    def get_lcc_size(self):
        if not self.couplings:
            return 0
        G = self.get_skeleton_graph()
        if not G.nodes:
            return 0
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        return len(largest)

    # ---------- GRAVITY ANALYSIS AROUND MASSIVE BITS ----------

    def gravity_analysis(self, degrees, top_k=5):
        """
        - measure curvature around massive bits
        - detect wells
        - compute local force field
        - test mass–curvature correlation
        - check inverse-distance-like behavior
        - orbit-like patterns from recent scopes
        """
        G = self.get_skeleton_graph()
        if not G.nodes:
            return [], [], [], [], None, None, []

        # sort bits by mass (heavy first)
        indices = np.arange(self.n_bits)
        sorted_idx = indices[np.argsort(self.mass)[::-1]]
        top_indices = sorted_idx[:top_k]

        # ego-graph based local stats for each top-mass bit
        local_clustering = []
        local_avg_dist = []
        local_field_strength = []

        # precompute all-pairs shortest paths on LCC for distance-based stats
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        H = G.subgraph(largest).copy()

        # map node -> shortest path lengths
        all_lengths = dict(nx.all_pairs_shortest_path_length(H))

        for i in top_indices:
            if i not in H:
                # isolated from LCC; treat as no local geometry
                local_clustering.append(0.0)
                local_avg_dist.append(np.nan)
                local_field_strength.append(0.0)
                continue

            # ego graph radius 1: neighbors & edges among neighbors
            ego = nx.ego_graph(H, i, radius=1)
            if ego.number_of_nodes() > 1:
                cl = nx.average_clustering(ego)
            else:
                cl = 0.0
            local_clustering.append(cl)

            # average distance from i to other nodes in LCC
            lengths = all_lengths.get(i, {})
            if len(lengths) > 1:
                dvals = [d for j, d in lengths.items() if j != i]
                avg_d = float(np.mean(dvals)) if dvals else np.nan
            else:
                avg_d = np.nan
            local_avg_dist.append(avg_d)

            # "field strength" at i: sum_j mass[j] / (dist(i,j)+eps)^2 over LCC
            field = 0.0
            for j, d in lengths.items():
                if j == i:
                    continue
                dist = float(d)
                field += self.mass[j] / ((dist + 1e-3) ** 2)
            local_field_strength.append(field)

        # mass–curvature correlation (global)
        # approximate per-bit curvature ~ degree * local clustering
        per_bit_curv = degrees * 0.0
        for node in H.nodes():
            # clustering coefficient at node
            per_bit_curv[node] = nx.clustering(H, node)

        mask = (self.mass > 0) & (per_bit_curv >= 0)
        if np.sum(mask) > 3:
            mvals = self.mass[mask]
            cvals = per_bit_curv[mask]
            mass_curv_corr = np.corrcoef(mvals, cvals)[0, 1]
        else:
            mass_curv_corr = None

        # mass vs inverse average distance (inverse-distance-like behavior)
        # for bits in LCC, compute avg shortest path distance to others
        invdist = np.zeros(self.n_bits, dtype=np.float32)
        valid = np.zeros(self.n_bits, dtype=bool)
        for i in H.nodes():
            lengths = all_lengths.get(i, {})
            if len(lengths) > 1:
                dvals = [d for j, d in lengths.items() if j != i]
                mean_d = float(np.mean(dvals))
                invdist[i] = 1.0 / (mean_d + 1e-9)
                valid[i] = True

        mask2 = valid & (self.mass > 0)
        if np.sum(mask2) > 3:
            mvals2 = self.mass[mask2]
            invvals = invdist[mask2]
            mass_invdist_corr = np.corrcoef(mvals2, invvals)[0, 1]
        else:
            mass_invdist_corr = None

        # orbit-like patterns: how often do top-mass bits appear in recent scopes?
        # approximate "orbit frequency" as count of appearances / len(history)
        orbit_counts = np.zeros(self.n_bits, dtype=np.int32)
        total_scopes = len(self.last_scopes)
        if total_scopes > 0:
            for sc in self.last_scopes:
                for b in sc:
                    orbit_counts[b] += 1
            orbit_freq = orbit_counts / float(total_scopes)
        else:
            orbit_freq = np.zeros(self.n_bits, dtype=np.float32)

        mass_orbit_freqs = [orbit_freq[i] for i in top_indices]

        return (top_indices,
                local_clustering,
                local_avg_dist,
                local_field_strength,
                mass_curv_corr,
                mass_invdist_corr,
                mass_orbit_freqs)

    # ---------- SAVE / LOAD ----------

    def save_state(self, filename=None):
        if filename is None:
            filename = self.state_file
        serializable_couplings = [[s.tolist(), int(f), float(st)] for s, f, st in self.couplings]
        data = {
            "cycle": self.cycle,
            "population": self.population.tolist(),
            "couplings": serializable_couplings,
            "record_low": self.record_low,
            "noise_level": self.noise_level,
            "adjacency": self.adjacency.tolist(),
            "mass": self.mass.tolist(),
            "run_id": self.run_id,
            "start_time": self.start_time,
            "mass_scale": self.mass_scale,
            "min_attraction": self.min_attraction,
        }
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"\n[SAVE] Universe state saved at cycle {self.cycle} -> {filename}")

    def load_state(self, filename=None):
        if filename is None:
            filename = self.state_file
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            self.cycle = data["cycle"]
            self.population = np.array(data["population"], dtype=np.int8)
            self.couplings = [[np.array(s), f_forb, st] for s, f_forb, st in data["couplings"]]
            self.record_low = data["record_low"]
            self.noise_level = data.get("noise_level", 0.0)
            self.adjacency = np.array(data.get("adjacency", np.zeros((self.n_bits, self.n_bits))), dtype=np.float32)
            self.mass = np.array(data.get("mass", np.zeros(self.n_bits)), dtype=np.float32)
            self.mass_scale = data.get("mass_scale", 3.0)
            self.min_attraction = data.get("min_attraction", 0.1)
            print(f"\n[LOAD] Universe restored from {filename}, record_low={self.record_low:.3f}")
        else:
            print(f"\n[SYSTEM] No save found at {filename}. Starting fresh universe.")

    # ---------- MAIN DYNAMICS ----------

    def step(self):
        self.cycle += 1
        entropy = self.get_entropy()

        # 1. EMERGENCE (GEOMETRY + GRAVITY-AWARE)
        if np.random.random() < (0.08 * (entropy + 0.5)):
            size = 3 if np.random.random() < 0.3 else 2
            scope = self.choose_scope_emergent_gravity(size)
            # track scope history for orbit-like analysis
            self.last_scopes.append(scope.copy())
            if len(self.last_scopes) > self.scope_history_max:
                self.last_scopes.pop(0)

            sample = self.population[np.random.randint(0, self.pop_size, 20)][:, scope]
            indices = sample.dot(self.weights[size])
            forbidden = np.argmin(np.bincount(indices, minlength=2**size))
            self.couplings.append([scope, forbidden, 0.6])

        # 2. VECTOR RELAXATION
        for c in self.couplings:
            scope, forbidden, strength = c
            current_vals = self.population[:, scope].dot(self.weights[len(scope)])
            violators = (current_vals == forbidden)
            if violators.any():
                mask = (np.random.random(self.pop_size) < strength) & violators
                if mask.any():
                    self.population[mask, np.random.choice(scope)] ^= 1

        # 3. HEBBIAN DECAY/HARDENING
        next_gen = []
        sample_pop = self.population[::self.sampling_step]
        for c in self.couplings:
            scope, forbidden, strength = c
            c[2] *= 0.95  # decay
            sample_vals = sample_pop[:, scope].dot(self.weights[len(scope)])
            if np.mean(sample_vals != forbidden) > 0.95:
                c[2] += 0.06  # hardening
            if c[2] > 0.15:
                next_gen.append(c)
        self.couplings = next_gen

        # 4. UPDATE GEOMETRY & MASS
        self.update_emergent_geometry()
        self.update_mass_from_couplings()

        # 5. DRIVE & NOISE
        self.population[:, 0:4] = (self.cycle // 20) % 2
        noise_mask = np.random.random(self.population.shape) < self.noise_level
        self.population[noise_mask] ^= 1

    # ---------- OBSERVABLES & LOGGING ----------

    def compute_and_log_observables(self, print_every=1000):
        current_ent = self.get_entropy()
        degrees, avg_deg, max_deg, avg_path_len, clustering, curvature, geom_phase = self.get_geometry_stats()
        lcc = self.get_lcc_size()
        phase = "CRUNCH" if current_ent < 0.2 else "SUPERFLUID"

        is_record = False
        if current_ent < self.record_low:
            self.record_low = current_ent
            is_record = True

        # core logs
        self.log_metrics(self.cycle, current_ent, lcc, phase,
                         avg_deg, max_deg, avg_path_len,
                         clustering, curvature, geom_phase)
        self.log_degrees(self.cycle, degrees)
        self.log_mass(self.cycle)

        # gravity & mass-curvature analysis
        (top_indices,
         top_local_clustering,
         top_local_avg_dist,
         top_field_strength,
         mass_curv_corr,
         mass_invdist_corr,
         mass_orbit_freqs) = self.gravity_analysis(degrees, top_k=5)

        if len(top_indices) > 0:
            top_masses = self.mass[top_indices]
        else:
            top_masses = []

        self.log_gravity(
            self.cycle,
            top_indices,
            top_masses,
            top_local_clustering,
            top_local_avg_dist,
            top_field_strength,
            mass_curv_corr,
            mass_invdist_corr,
            mass_orbit_freqs,
        )

        # Console visibility
        if is_record:
            print(f"{self.cycle:<8} | {current_ent:.3f}* | {self.record_low:.3f} | {lcc:<5} | {phase:<10} | {geom_phase}")
        elif self.cycle % print_every == 0:
            elapsed = (time.time() - self.start_time) / 60
            print(f"{self.cycle:<8} | {current_ent:.3f}  | {self.record_low:.3f} | {lcc:<5} | {phase:<10} | {geom_phase} (T+{elapsed:.1f}m)")
            # MASS SUMMARY
            if len(top_indices) > 0:
                heavy_str = ", ".join([f"{int(i)}:{self.mass[i]:.2f}" for i in top_indices])
                print(f"       | MASS TOP5 -> {heavy_str}")
            # GRAVITY SUMMARY
            if mass_curv_corr is not None and mass_invdist_corr is not None:
                print(f"       | corr(mass,local_curv)={mass_curv_corr:.3f}, corr(mass,1/dist)={mass_invdist_corr:.3f}")
            if len(top_indices) > 0:
                # show first heavy bit's local stats as "well"
                i0 = int(top_indices[0])
                cl0 = top_local_clustering[0] if len(top_local_clustering) > 0 else 0.0
                d0 = top_local_avg_dist[0] if len(top_local_avg_dist) > 0 else float('nan')
                f0 = top_field_strength[0] if len(top_field_strength) > 0 else 0.0
                orb0 = mass_orbit_freqs[0] if len(mass_orbit_freqs) > 0 else 0.0
                print(f"       | WELL @ {i0}: mass={self.mass[i0]:.2f}, clust={cl0:.3f}, avg_dist={d0:.3f}, field={f0:.2f}, orbit_freq={orb0:.3f}")

# ------------------------
# EXECUTION
# ------------------------

if __name__ == "__main__":
    u = Persistent_AGI_Universe()

    def input_thread():
        while u.running:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            if cmd == 'q':
                u.save_state()
                u.running = False
            elif cmd == 'n':
                try:
                    new_n = float(input("New noise level (e.g. 0.02): "))
                    u.noise_level = new_n
                    print(f"[UPDATE] noise_level = {u.noise_level}")
                except Exception:
                    print("[ERROR] Invalid noise value.")
            elif cmd == 's':
                u.save_state()
            elif cmd == 'm':
                try:
                    new_ms = float(input("New mass_scale (e.g. 3.0): "))
                    u.mass_scale = new_ms
                    print(f"[UPDATE] mass_scale = {u.mass_scale}")
                except Exception:
                    print("[ERROR] Invalid mass_scale.")
            elif cmd == 'a':
                try:
                    new_min = float(input("New min_attraction (e.g. 0.1): "))
                    u.min_attraction = new_min
                    print(f"[UPDATE] min_attraction = {u.min_attraction}")
                except Exception:
                    print("[ERROR] Invalid min_attraction.")
            else:
                print("Commands: [q]uit+save, [s]ave, [n]oise, [m]ass_scale, [a]ttraction")

    threading.Thread(target=input_thread, daemon=True).start()

    print(f"{'Cycle':<8} | {'Entropy':<8} | {'Best':<8} | {'LCC':<5} | {'Phase':<10} | {'GeomPhase'}")
    while u.running:
        u.step()
        u.compute_and_log_observables(print_every=1000)
