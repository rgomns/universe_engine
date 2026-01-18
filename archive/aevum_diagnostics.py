# ============================================================
# AEVUM DIAGNOSTICS (ROBUST, JSON-SAFE, OPTION B PCA)
# ============================================================

import numpy as np
from collections import defaultdict


# ------------------------------------------------------------
# JSON SAFETY
# ------------------------------------------------------------

def json_safe(x):
    """Recursively convert any object into JSON-serializable form."""
    import numpy as np

    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, complex):
        return f"{x.real}+{x.imag}j"
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    return x


# ------------------------------------------------------------
# SAFE HELPERS
# ------------------------------------------------------------

def _safe_array(x):
    """Convert to float array and replace NaN/inf with 0."""
    arr = np.array(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ------------------------------------------------------------
# AUTOCORRELATION + FFT
# ------------------------------------------------------------

def autocorrelation(x):
    x = _safe_array(x)
    if len(x) < 3:
        return np.zeros_like(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    denom = result[result.size // 2]
    if denom == 0:
        return np.zeros_like(x)
    return result[result.size // 2:] / denom


def fft_spectrum(x):
    x = _safe_array(x)
    if len(x) < 4:
        return [], []
    x = x - np.mean(x)
    try:
        fft_vals = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x))
        magnitudes = np.abs(fft_vals)
        magnitudes = _safe_array(magnitudes)
        return freqs, magnitudes
    except Exception:
        return [], []


def detect_periodicity(series, label, threshold=0.25):
    series = _safe_array(series)
    if len(series) < 10:
        return {"label": label, "periodic": False}

    ac = autocorrelation(series)
    freqs, mags = fft_spectrum(series)

    if len(mags) <= 1:
        return {"label": label, "periodic": False}

    dominant_idx = int(np.argmax(mags[1:]) + 1)
    dominant_freq = float(freqs[dominant_idx]) if len(freqs) > dominant_idx else 0.0
    dominant_mag = float(mags[dominant_idx]) if len(mags) > dominant_idx else 0.0

    periodic = dominant_mag > threshold * mags[0] if mags[0] != 0 else False

    return {
        "label": label,
        "periodic": periodic,
        "dominant_frequency": dominant_freq,
        "dominant_magnitude": dominant_mag,
    }


# ------------------------------------------------------------
# COMPLEX EIGENVALUES
# ------------------------------------------------------------

def build_constraint_matrix(lineage, n_bits):
    M = np.zeros((n_bits, n_bits))
    for c in lineage:
        scope = c.get("scope", [])
        strength = float(c.get("final_strength", 0.0))
        for i in scope:
            for j in scope:
                if i != j:
                    M[i, j] += strength
    return _safe_array(M)


def detect_complex_eigenvalues(lineage, n_bits=48):
    if not lineage:
        return {
            "has_complex_structure": False,
            "num_complex_eigenvalues": 0,
            "complex_eigenvalues": []
        }

    M = build_constraint_matrix(lineage, n_bits)

    try:
        eigenvals = np.linalg.eigvals(M)
        eigenvals = [complex(ev.real, ev.imag) for ev in eigenvals]
        complex_vals = [ev for ev in eigenvals if abs(ev.imag) > 1e-9]
        return {
            "has_complex_structure": len(complex_vals) > 0,
            "num_complex_eigenvalues": len(complex_vals),
            "complex_eigenvalues": complex_vals,
        }
    except Exception:
        return {
            "has_complex_structure": False,
            "num_complex_eigenvalues": 0,
            "complex_eigenvalues": []
        }


# ------------------------------------------------------------
# ROTATIONAL SYMMETRY
# ------------------------------------------------------------

def detect_cycles(lineage, n_bits=48):
    adj = defaultdict(set)
    for c in lineage:
        scope = c.get("scope", [])
        for i in scope:
            for j in scope:
                if i != j:
                    adj[i].add(j)

    visited = set()
    cycles = []

    def dfs(node, parent, path):
        visited.add(node)
        path.append(node)
        for neigh in adj[node]:
            if neigh == parent:
                continue
            if neigh in path:
                idx = path.index(neigh)
                cycles.append(tuple(sorted(path[idx:])))
            elif neigh not in visited:
                dfs(neigh, node, path.copy())

    for i in range(n_bits):
        if i not in visited:
            dfs(i, -1, [])

    unique_cycles = set(cycles)

    return {
        "num_cycles": len(unique_cycles),
        "cycles": [list(c) for c in unique_cycles],
        "has_rotational_symmetry": len(unique_cycles) > 0,
    }


# ------------------------------------------------------------
# HIDDEN DIMENSIONS (PCA, OPTION B)
# ------------------------------------------------------------

def detect_hidden_dimensions(records):
    if not records or len(records) < 2:
        return {
            "effective_dimensions": 1,
            "eigenvalues": [0.0],
            "explained_variance": [1.0],
            "higher_dimensional_behavior": False
        }

    data = []
    for r in records:
        e = float(r.get("entropy", 0.0))
        pe3 = float(r.get("pattern_entropy", {}).get(3, 0.0))
        pe4 = float(r.get("pattern_entropy", {}).get(4, 0.0))
        data.append([e, pe3, pe4])

    X = _safe_array(data)

    try:
        X = X - np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov = _safe_array(cov)

        eigvals, _ = np.linalg.eig(cov)
        eigvals = _safe_array(eigvals)

        total = float(np.sum(eigvals))
        if total <= 0:
            raise ValueError("Non-positive eigenvalue sum")

        explained = [float(ev / total) for ev in eigvals]
        dims = sum(ev > 0.05 for ev in explained)

        return {
            "eigenvalues": [float(ev) for ev in eigvals],
            "explained_variance": explained,
            "effective_dimensions": int(dims),
            "higher_dimensional_behavior": dims > 1,
        }

    except Exception:
        return {
            "effective_dimensions": 1,
            "eigenvalues": [0.0],
            "explained_variance": [1.0],
            "higher_dimensional_behavior": False
        }


# ------------------------------------------------------------
# LIMIT CYCLES
# ------------------------------------------------------------

def detect_limit_cycles(series, min_period=5):
    series = _safe_array(series)
    seen = {}
    for i, val in enumerate(series):
        rounded = round(val, 3)
        if rounded in seen:
            period = i - seen[rounded]
            if period >= min_period:
                return {"limit_cycle": True, "period": int(period)}
        else:
            seen[rounded] = i
    return {"limit_cycle": False}


# ------------------------------------------------------------
# MASTER ENTRY POINT
# ------------------------------------------------------------

def run_diagnostics(records, lineage, n_bits=48):
    ent = _safe_array([r.get("entropy", 0.0) for r in records])
    nc = _safe_array([r.get("num_constraints", 0.0) for r in records])
    deg = _safe_array([r.get("avg_degree", 0.0) for r in records])

    result = {
        "periodicity_entropy": detect_periodicity(ent, "entropy"),
        "periodicity_constraints": detect_periodicity(nc, "num_constraints"),
        "periodicity_degree": detect_periodicity(deg, "avg_degree"),

        "complex_eigenvalues": detect_complex_eigenvalues(lineage, n_bits),
        "rotational_symmetry": detect_cycles(lineage, n_bits),

        "hidden_dimensions": detect_hidden_dimensions(records),

        "limit_cycle_entropy": detect_limit_cycles(ent),
        "limit_cycle_constraints": detect_limit_cycles(nc),
    }

    return json_safe(result)
