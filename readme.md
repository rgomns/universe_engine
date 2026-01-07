# Aevum: Autonomous Emergent Homeostasis in Discrete Bit-Environments

**Aevum** is a research framework for studying **Non-Gradient Self-Organization**. It simulates a 48-bit universe where order is not programmed by an agent, but emerges as a structural requirement for information persistence. 

The system demonstrates that "Intelligence" can be modeled as a defensive architecture against systemic entropy.

---

## 🔬 Core Research Pillars

### 1. Subtractive Darwinism & Constraint Evolution
Unlike typical Genetic Algorithms that optimize for a predefined fitness score, Aevum optimizes for **Survival via Pruning**.
* **Constraint Lifecycle:** Laws are birthed with random bit-scopes (2-bit/3-bit) and survive based on their ability to identify and remove high-entropy noise.
* **The Reward Loop:** Successful laws gain "Strength" (`CONSTRAINT_REWARD = 0.05`). Laws that fail to find patterns suffer from **Temporal Decay**, eventually hitting the `MIN_STRENGTH` floor and being erased.



### 2. Emergent Structural Invariants (The "Ghost Laws")
Aevum tracks the persistence of laws across **Big Crunches** (re-initialization events triggered when population $\Omega < 80$).
* **Genetic Memory:** High-strength laws that survive for `FUNDAMENTAL_LIFESPAN = 2000` cycles are classified as **Fundamentals**.
* **Post-Collapse Rebound:** The presence of these "Ghost Laws" provides a structural scaffold that reduces the entropy-re-stabilization time of the next universe by up to **75%**.

### 3. Graph Topology & Component Analysis
The system models the universe as a dynamic graph where bits are vertices and constraints are edges.
* **Hub-Bit Discovery:** The engine calculates **Degree Centrality** to identify "Anchor Bits"—bits that the universe "chooses" to center its logic around.
* **LCC Dynamics:** We track the **Largest Connected Component (LCC)**. A phase transition to "Stable Complexity" is marked by the merger of isolated constraint islands into a unified logical network.



---

## 📊 Phase Classification

Aevum classifies universes into three distinct thermodynamic phases based on **KL Divergence** and **Entropy Delta**:

| Phase | Characteristics | Entropy State |
| :--- | :--- | :--- |
| **Subcritical** | Low connection density; laws die faster than they form. | High / Unstable |
| **Critical** | The "Edge of Chaos." High innovation; laws are fluid but persistent. | Moderate / Descending |
| **Supercritical** | Frozen logic; massive LCC; universe resists all new noise. | Low / Rigid |



---

## 🛠 Project Components

* **`universe_engine.py`**: The core simulation engine. Handles the $N=48$ bit-environment and the lifecycle of 2-bit/3-bit constraints.
* **`phase_sweep.py`**: A parallelized exploration tool that maps the "Habitable Zone" of parameter space ($P_{new}$ vs. $Global Decay$).
* **`plot_phase_map.py`**: Generates heatmaps of emergent law phases, identifying where "Digital Life" is most likely to emerge.

---

## 🚀 Execution & Reproducibility

### Simulation Ensemble
Run a comparative study of **EVO_BASE** (Standard), **EVO_CHAOS** (High Innovation), and **EVO_FROZEN** (High Pressure):
```bash
python universe_engine.py