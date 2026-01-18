import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# ==========================================
# TSP INSTANCE + UTILITIES
# ==========================================

Point = Tuple[float, float]
Tour = List[int]


def euclidean_distance(p1: Point, p2: Point) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


def generate_euclidean_tsp(n_cities: int, seed: int = 42) -> List[Point]:
    rnd = random.Random(seed)
    return [(rnd.random(), rnd.random()) for _ in range(n_cities)]


def tour_length(tour: Tour, coords: List[Point]) -> float:
    if not tour:
        return 0.0
    total = 0.0
    n = len(tour)
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        total += euclidean_distance(coords[a], coords[b])
    return total


def random_tour(n_cities: int, rnd: random.Random) -> Tour:
    t = list(range(n_cities))
    rnd.shuffle(t)
    return t


# ==========================================
# BASELINE SOLVERS (NN + 2-OPT)
# ==========================================

def nearest_neighbor_tour(coords: List[Point], start: int = 0) -> Tour:
    n = len(coords)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: euclidean_distance(coords[cur], coords[j]))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


def two_opt(tour: Tour, coords: List[Point], max_iters: int = 2000) -> Tour:
    n = len(tour)
    if n < 4:
        return tour[:]
    best = tour[:]
    best_len = tour_length(best, coords)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if j == i + 1:
                    continue
                new_tour = best[:]
                new_tour[i:j] = reversed(new_tour[i:j])
                new_len = tour_length(new_tour, coords)
                if new_len + 1e-12 < best_len:
                    best = new_tour
                    best_len = new_len
                    improved = True
                    break
            if improved:
                break
    return best


def standard_heuristic_solver(coords: List[Point]) -> Tour:
    base = nearest_neighbor_tour(coords, start=0)
    improved = two_opt(base, coords)
    return improved


# ==========================================
# X-SPACE: CLUSTER GEOMETRY + PROJECTION
# ==========================================

@dataclass
class XState:
    """
    Cluster-geometry representation:
      - centers: indices of cities (cluster centers)
      - assignment: city -> cluster index
      - cluster_order: permutation of clusters
      - directions: per-cluster direction flag
    """
    centers: List[int]
    assignment: List[int]
    cluster_order: List[int]
    directions: List[int]


def compute_cluster_assignment(centers: List[int], coords: List[Point]) -> List[int]:
    n = len(coords)
    k = len(centers)
    assignment = [0] * n
    for city in range(n):
        best_c = 0
        best_d = float("inf")
        for ci, center_city in enumerate(centers):
            d = euclidean_distance(coords[city], coords[center_city])
            if d < best_d:
                best_d = d
                best_c = ci
        assignment[city] = best_c
    return assignment


def local_nn_order(cities: List[int], coords: List[Point]) -> List[int]:
    if not cities:
        return []
    remaining = set(cities)
    cur = cities[0]
    tour = [cur]
    remaining.remove(cur)
    while remaining:
        nxt = min(remaining, key=lambda j: euclidean_distance(coords[cur], coords[j]))
        tour.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return tour


def project_X_to_tour(x: XState, coords: List[Point]) -> Tour:
    """
    π : X -> Tour
    For each cluster in cluster_order:
      - take its cities
      - order them with local NN
      - flip direction if needed
    Concatenate.
    """
    n = len(coords)
    k = len(x.centers)
    clusters: List[List[int]] = [[] for _ in range(k)]
    for city in range(n):
        c = x.assignment[city]
        if 0 <= c < k:
            clusters[c].append(city)
        else:
            clusters[0].append(city)

    tour: List[int] = []
    for cluster_idx in x.cluster_order:
        if not (0 <= cluster_idx < k):
            continue
        cities = clusters[cluster_idx]
        if not cities:
            continue
        sub_tour = local_nn_order(cities, coords)
        if x.directions[cluster_idx] == 1:
            sub_tour.reverse()
        tour.extend(sub_tour)

    # repair permutation
    seen = set()
    repaired: List[int] = []
    for c in tour:
        if c not in seen:
            repaired.append(c)
            seen.add(c)
    for c in range(len(coords)):
        if c not in seen:
            repaired.append(c)
            seen.add(c)
    return repaired


# ==========================================
# SURROGATE ENERGY E[X]
# ==========================================

def energy_X(x: XState, coords: List[Point],
             w_intra: float = 1.0,
             w_inter: float = 1.0) -> float:
    """
    E[X] = w_intra * average city-center distance
         + w_inter * length of cluster-center tour (coarse path)
    """
    n = len(coords)
    k = len(x.centers)
    if k == 0:
        return float("inf")

    # Intra-cluster distortion
    intra_sum = 0.0
    for city in range(n):
        c_idx = x.assignment[city]
        if not (0 <= c_idx < k):
            c_idx = 0
        center_city = x.centers[c_idx]
        intra_sum += euclidean_distance(coords[city], coords[center_city])
    intra_term = intra_sum / n

    # Inter-cluster path: tour over centers in cluster_order
    center_coords = [coords[c] for c in x.centers]
    # reorder centers according to cluster_order
    ordered_centers = [center_coords[i] for i in x.cluster_order if 0 <= i < k]
    if len(ordered_centers) < 2:
        inter_term = 0.0
    else:
        inter_term = 0.0
        m = len(ordered_centers)
        for i in range(m):
            a = ordered_centers[i]
            b = ordered_centers[(i + 1) % m]
            inter_term += euclidean_distance(a, b)

    return w_intra * intra_term + w_inter * inter_term


# ==========================================
# X-SPACE UNIVERSE WITH ENERGY-BASED EVOLUTION
# ==========================================

@dataclass
class XUniverseConfig:
    population_size: int = 80
    n_generations: int = 220
    elite_fraction: float = 0.3
    mutation_rate: float = 0.5
    seed: int = 123
    k_min_base: int = 3
    k_max_base: int = 18
    w_intra: float = 1.0
    w_inter: float = 0.5


class XUniverseEnergy:
    def __init__(self, coords: List[Point], config: XUniverseConfig):
        self.coords = coords
        self.config = config
        self.rnd = random.Random(config.seed)
        self.n_cities = len(coords)
        self.k_min, self.k_max = self._compute_k_range()
        self.population: List[XState] = []
        self._init_population()

    def _compute_k_range(self) -> Tuple[int, int]:
        n = self.n_cities
        approx = max(2, int(math.sqrt(n)))
        k_min = max(2, min(self.config.k_min_base, approx))
        k_max = max(k_min + 1, min(self.config.k_max_base, approx * 2))
        return k_min, min(k_max, n)

    def _init_population(self):
        self.population = []
        for _ in range(self.config.population_size):
            k = self.rnd.randint(self.k_min, self.k_max)
            centers = self.rnd.sample(range(self.n_cities), k)
            assignment = compute_cluster_assignment(centers, self.coords)
            cluster_order = list(range(k))
            self.rnd.shuffle(cluster_order)
            directions = [self.rnd.randint(0, 1) for _ in range(k)]
            self.population.append(XState(centers, assignment, cluster_order, directions))

    def _energy(self, x: XState) -> float:
        return energy_X(
            x,
            self.coords,
            w_intra=self.config.w_intra,
            w_inter=self.config.w_inter,
        )

    def _mutate(self, x: XState) -> XState:
        k = len(x.centers)
        centers = x.centers[:]
        assignment = x.assignment[:]
        order = x.cluster_order[:]
        directions = x.directions[:]

        op = self.rnd.choice(["swap_order", "flip_dir", "nudge_center", "add_center", "remove_center"])

        if op == "swap_order" and k > 1:
            i, j = self.rnd.sample(range(k), 2)
            order[i], order[j] = order[j], order[i]

        elif op == "flip_dir":
            idx = self.rnd.randrange(k)
            directions[idx] ^= 1

        elif op == "nudge_center":
            idx = self.rnd.randrange(k)
            center_city = centers[idx]
            dists = [
                (city, euclidean_distance(self.coords[city], self.coords[center_city]))
                for city in range(self.n_cities)
            ]
            dists.sort(key=lambda t: t[1])
            neighborhood = [c for c, _ in dists[1 : min(8, self.n_cities)]]
            if neighborhood:
                centers[idx] = self.rnd.choice(neighborhood)
            assignment = compute_cluster_assignment(centers, self.coords)

        elif op == "add_center" and k < self.k_max and k < self.n_cities:
            candidates = [c for c in range(self.n_cities) if c not in centers]
            if candidates:
                centers.append(self.rnd.choice(candidates))
                assignment = compute_cluster_assignment(centers, self.coords)
                order = list(range(len(centers)))
                self.rnd.shuffle(order)
                directions = [self.rnd.randint(0, 1) for _ in range(len(centers))]

        elif op == "remove_center" and k > self.k_min:
            idx = self.rnd.randrange(k)
            centers.pop(idx)
            assignment = compute_cluster_assignment(centers, self.coords)
            order = list(range(len(centers)))
            self.rnd.shuffle(order)
            directions = [self.rnd.randint(0, 1) for _ in range(len(centers))]

        return XState(centers, assignment, order, directions)

    def evolve(self, verbose: bool = True) -> Tuple[float, Tour]:
        best_overall_len = float("inf")
        best_overall_tour: Tour = []

        for gen in range(self.config.n_generations):
            scored = [(self._energy(x), x) for x in self.population]
            scored.sort(key=lambda t: t[0])
            best_E, best_state = scored[0]
            best_tour = project_X_to_tour(best_state, self.coords)
            best_len = tour_length(best_tour, self.coords)

            if best_len < best_overall_len:
                best_overall_len = best_len
                best_overall_tour = best_tour[:]

            if verbose and (gen % max(1, self.config.n_generations // 10) == 0):
                print(
                    f"[GEN {gen:3d}] E_best={best_E:.4f} "
                    f"L_best={best_len:.4f} "
                    f"L_overall={best_overall_len:.4f} "
                    f"k={len(best_state.centers)} "
                    f"pop={len(self.population)}"
                )

            n_elite = max(1, int(len(scored) * self.config.elite_fraction))
            elites = [x for _, x in scored[:n_elite]]

            new_pop: List[XState] = []
            new_pop.extend(elites)

            while len(new_pop) < self.config.population_size:
                parent = self.rnd.choice(elites)
                child = parent
                if self.rnd.random() < self.config.mutation_rate:
                    child = self._mutate(parent)
                new_pop.append(child)

            self.population = new_pop

        return best_overall_len, best_overall_tour


# ==========================================
# EXPERIMENT DRIVER
# ==========================================

def run_experiment(n_cities: int = 40, seed: int = 42):
    print(f"=== X-space (energy) TSP experiment with {n_cities} cities ===")
    coords = generate_euclidean_tsp(n_cities, seed)

    rnd = random.Random(seed)
    rand_t = random_tour(n_cities, rnd)
    rand_len = tour_length(rand_t, coords)
    print(f"Random tour length:          {rand_len:.4f}")

    nn2_t = standard_heuristic_solver(coords)
    nn2_len = tour_length(nn2_t, coords)
    print(f"NN + 2-opt length:           {nn2_len:.4f}")

    cfg = XUniverseConfig(
        population_size=90,
        n_generations=200,
        elite_fraction=0.3,
        mutation_rate=0.5,
        seed=123,
        k_min_base=3,
        k_max_base=18,
        w_intra=1.0,
        w_inter=0.4,  # tune this if you like
    )
    universe = XUniverseEnergy(coords, cfg)
    best_len_X, best_tour_X = universe.evolve(verbose=True)
    print(f"\nX-space energy universe best:{best_len_X:.4f}")

    print("\n=== Comparison ===")
    print(f"Random:              {rand_len:.4f}")
    print(f"NN + 2-opt:          {nn2_len:.4f}")
    print(f"X-space energy:      {best_len_X:.4f}")


if __name__ == "__main__":
    run_experiment(n_cities=200, seed=42)
    # Try also:
    # run_experiment(n_cities=100, seed=42)
    # run_experiment(n_cities=200, seed=42)
