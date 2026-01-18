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
.
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
