# Aevum: Emergent Robustness in Autonomous Bit-Environments

**Aevum** is a complexity science project designed to observe the spontaneous emergence of order within a 48-bit digital universe. Unlike traditional simulations that enforce specific outcomes, Aevum subjects a population of bitstrings to constant "Global Decay" and periodic "Big Crunches" (phase transitions), documenting how the system evolves its own internal "Laws" to maintain stability.

---

## 🌌 The Core Dynamics

The project explores a **Subtractive Darwinian** environment where order is a byproduct of survival. The universe is governed by three autonomous forces:

* **Autonomous Global Decay:** Bit-patterns that fail to satisfy emergent constraints are naturally pruned, favoring stable or symmetric configurations.
* **The "Natural" Big Crunch:** When internal logic becomes too restrictive or the population size ($\Omega$) drops below a critical threshold (default: 80), the system undergoes a collapse and re-initializes.
* **Ghost Laws:** Following a Crunch, the universe retains "fossilized" logic—remnant bit-constraints that have achieved high "strength" and longevity, accelerating the re-formation of order in subsequent cycles.

---

## 🔬 Key Research Findings

* **Anchor Bits:** The system autonomously identifies specific bit-indices as "high-utility nodes" essential for global stability, often visualized through degree centrality in the constraint graph.
* **Digital Homeostasis:** The engine tracks "Constraint Reward" and "Decay," allowing the system to reinforce rules that successfully prune entropy while shedding ineffective ones.
* **Phase Transitions:** By monitoring KL Divergence and Entropy Delta, the simulation detects sudden shifts in organizational structure, classified as "subcritical," "critical," or "supercritical".

---

## 🛠 Project Structure

The codebase is modularized to support both real-time simulation and long-term parameter analysis:

| File | Description |
| :--- | :--- |
| `universe_engine.py` | The core simulation engine. Manages entropy calculations, mutual information matrices, and constraint evolution. |
| `phase_sweep.py` | A parallel processing script that sweeps through parameter space (P-New vs. Decay) to identify phase boundaries. |
| `plot_phase_map.py` | Visualization tools to map the "Phase Diagram" of the universe in both parameter and behavior space. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* `matplotlib`, `tqdm`, `concurrent.futures`

### Running the Universe
To run the standard ensemble of four universe types (**EVO_BASE**, **EVO_CHAOS**, **EVO_FROZEN**, and **CTRL_BASE**):
```bash
python universe_engine.py