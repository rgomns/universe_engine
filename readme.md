# 🧬 Real Reversible Universe Engine  
### **Emergent Quantum Mechanics from a Real, Local, Reversible Dynamical Law**

This repository contains a complete, reproducible experimental pipeline demonstrating how **complex quantum‑mechanical behavior** — including locality, chaos, Hermitian Hamiltonians, and extended eigenstates — can emerge from a **purely real, reversible, locally‑coupled dynamical system**.

The codebase provides:

- A **phase sweep** over key parameters  
- Classical observables (entropy, coherence, oscillation structure)  
- Heatmap **GIF phase diagrams**  
- A full **real → complex mapping**  
- **Effective Hamiltonian reconstruction**  
- Diagnostics: locality, level statistics, participation ratios  
- All logs in **CSV**, **JSONL**, and **GIF** formats  

This package is designed so a researcher can **run, inspect, and verify** every step of the emergence pipeline.

---

## 🔧 Features

### **1. Phase Sweep**
Sweeps over:
- `coupling_theta` — global mixing strength  
- `eps_nn` — nearest‑neighbor locality strength  

For each point, the engine computes:
- Mean coherence  
- Agent entropy  
- Rule density  
- Entropy oscillation score  
- Rule count  

Outputs:
- `logs_phase/phase_diagram_results.jsonl`  
- `logs_phase/phase_diagram_summary.csv`  
- GIFs:
  - `phase_diagram_coherence_mean.gif`
  - `phase_diagram_agent_entropy_mean.gif`
  - `phase_diagram_rules_mean.gif`
  - `phase_diagram_entropy_osc_score.gif`

These GIFs visualize the **phase structure** across the parameter grid.

---

### **2. Full QM Reconstruction Pipeline**
For a chosen “sweet‑spot” configuration, the script performs:

#### **a. Observables log**
- Classical observables over time  
- Attractor diversity  
- Saved to:  
  `logs_best/observables.jsonl`

#### **b. Real → Complex Mapping**
- Converts real state → complex wavefunction  
- Computes von Neumann entropy  
- Saved to:  
  `logs_best/mapping_qm.jsonl`

#### **c. Effective Hamiltonian Reconstruction**
- Estimates effective unitary  
- Extracts Hermitian generator  
- Computes eigenvalues and eigenvectors  
- Saved to:  
  `logs_best/qm_reconstruction.json`

#### **d. Hamiltonian Diagnostics**
Printed to console:
- Coupling vs distance (locality)
- Level statistics (chaos)
- Participation ratios (eigenvector structure)

---

## 📁 Repository Structure

```
├── release_script.py               # Main script (phase sweep + QM pipeline)
├── logs_phase/
│   ├── phase_diagram_results.jsonl
│   ├── phase_diagram_summary.csv
│   ├── phase_diagram_coherence_mean.gif
│   ├── phase_diagram_agent_entropy_mean.gif
│   ├── phase_diagram_rules_mean.gif
│   └── phase_diagram_entropy_osc_score.gif
├── logs_best/
│   ├── observables.jsonl
│   ├── mapping_qm.jsonl
│   └── qm_reconstruction.json
└── README.md
```

---

## ▶️ Running the Full Pipeline

### **Requirements**
- Python 3.9+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

### **Run everything**

```bash
python release_script.py
```

This will:

1. Run the full phase sweep  
2. Generate CSV + JSONL logs  
3. Produce GIF phase diagrams  
4. Run the sweet‑spot QM reconstruction  
5. Print Hamiltonian diagnostics  

---

## 📊 Interpreting the Results

### **Phase Diagram**
The GIFs show how the system transitions between:

- **Ordered phase** (low θ, low eps_nn)  
- **Quantum chaotic phase** (moderate θ, moderate eps_nn)  
- **Thermal / overmixed phase** (high θ or high eps_nn)  

### **QM Reconstruction**
The sweet‑spot point exhibits:

- **Local Hamiltonian** (coupling decays with distance)  
- **Wigner–Dyson level statistics** (quantum chaos)  
- **Extended eigenvectors** (high participation ratios)  
- **Stable complex structure**  

This demonstrates a **full emergent quantum phase** from a real reversible system.

---

## 🧪 Scientific Context

This project provides a constructive example of how:

- Complex Hilbert‑space structure  
- Hermitian Hamiltonians  
- Local interactions  
- Quantum chaos  
- Entanglement structure  

can emerge from a **real, reversible, deterministic** dynamical law.

It offers a platform for exploring foundational questions in:

- Quantum reconstruction  
- Emergence  
- Complexity science  
- Locality and chaos  
- Effective field behavior  

---

## 📬 Contact

If you are a researcher interested in collaborating, analyzing the model, or extending the reconstruction pipeline, feel free to reach out.




# Microscopic Law (Fundamental Dynamics)

At the deepest level, the model is defined by a **real**, **deterministic**, **time‑independent**, and **reversible** update rule acting on a high‑dimensional real state vector.

## Microscopic State

The ontic state of the universe at time $t$ is a real vector:
$$
x_t \in \mathbb{R}^N,
$$
where
$$
N = \text{n\_agents} \times \text{bits\_per\_agent}.
$$

No complex numbers, amplitudes, phases, or probabilistic elements appear at the fundamental level.

---

# Fundamental Update Rule

Time evolution consists of applying a fixed sequence of locality‑biased orthogonal transformations:
$$
x_{t+1} = W_L W_{L-1} \cdots W_2 W_1 \, x_t,
$$
where each $W_i \in \mathbb{R}^{N \times N}$ is an **orthogonal matrix** satisfying:
$$
W_i^T W_i = I.
$$

This guarantees:

- **Reversibility**  
  $$
  x_{t} = W_1^{-1} W_2^{-1} \cdots W_L^{-1} x_{t+1}
  $$

- **Norm preservation**  
  $$
  \|x_{t+1}\| = \|x_t\|
  $$

- **Deterministic dynamics**

- **No drift or dissipation**

---

# Construction of Each Rule Layer

Each orthogonal matrix $W_i$ is constructed from simple locality‑biased components:

### 1. Intra‑agent mixing

Each agent $a$ has a strongly mixing orthogonal block:
$$
O_a \in \mathbb{R}^{b \times b},
$$
where $b = \text{bits\_per\_agent}$.

### 2. Nearest‑neighbor couplings

Scaled by a locality parameter $\varepsilon_{\text{nn}}$.

### 3. Next‑nearest‑neighbor couplings

Scaled by $\varepsilon_{\text{nnn}}$.

### 4. Global mixing parameter

A single scalar $\theta$ blends identity with the local structure:
$$
W_{\text{pre}} = \cos\theta \, I + \sin\theta \, Q_{\text{local}}.
$$

### 5. Final orthogonalization

A QR decomposition produces the exact reversible update:
$$
W = \text{QR}(W_{\text{pre}}).
$$

This ensures the microscopic rule is **strictly orthogonal** and therefore **strictly reversible**.

---

# Summary of the Microscopic Law

In one sentence:

> **The fundamental dynamics are a fixed sequence of locality‑biased orthogonal transformations acting on a real state vector.**

This is the only rule at the microscopic level.

There is:

- no Hamiltonian  
- no complex structure  
- no Schrödinger equation  
- no probabilistic postulates  
- no learning or adaptation  
- no hidden variables  
- no collapse mechanism  

All of those appear **only at the emergent level**.

---

# Why This Counts as Emergence

From this simple real reversible rule, the model exhibits:

- a stable **complex Hilbert space**  
- an emergent **Hermitian Hamiltonian**  
- **local** effective interactions  
- **Wigner–Dyson** level statistics  
- **extended eigenvectors**  
- **quantum‑chaotic** behavior  
- a full **phase diagram** (ordered → quantum chaotic → thermal)

None of these structures are present in the microscopic rule.  
They arise only at the macroscopic, coarse‑grained level.

This is emergence in the strongest sense used in physics.
