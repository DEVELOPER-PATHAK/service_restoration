# Supply Restoration Optimizer
### MANIT Bhopal — College Project

Binary PSO, Continuous PSO & Hybrid PSO-GA for electric supply restoration in smart distribution systems.

Addresses **challenges A, F, K, L, M** from the Goda et al. (2025) review paper.

---

## Setup & Run (3 steps)

### 1. Install Python (if not installed)
Download Python 3.9+ from https://www.python.org/downloads/

### 2. Install dependencies
Open terminal/command prompt **inside this folder** and run:

```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run ui/app.py
```

The browser opens automatically at http://localhost:8501

---

## Verify installation (optional)
```bash
python tests/test_algorithms.py
```
Expected output: `✅ All tests passed!`

---

## How to Use
1. Click **"Load IEEE 33-Bus"** in the sidebar
2. Pick a **fault bus** (e.g., Bus 26 — from the paper)
3. Tune **objective weights** for challenges A, F, K, L, M
4. Select **algorithm** (or "Compare All Three")
5. Hit **Run Optimization**
6. View convergence curves, metrics, switch tables, and post-restoration network

---

## Project Structure
```
restoration_project/
├── core/
│   ├── network.py       # Graph builder, radiality check
│   └── fitness.py       # Multi-objective cost function (A,F,K,L,M)
├── algorithms/
│   ├── bpso.py          # Binary PSO (sigmoid transfer)
│   ├── cpso.py          # Continuous PSO (threshold binarization)
│   └── hybrid_pso_ga.py # PSO + GA crossover/mutation
├── ui/
│   └── app.py           # Streamlit dashboard
├── data/
│   └── ieee33.json      # IEEE 33-bus test system
├── tests/
│   └── test_algorithms.py
└── requirements.txt
```

---

## Challenges Addressed
| Code | Challenge |
|------|-----------|
| A | Energy Not Supplied — maximize restored load |
| F | Load Priority — serve priority 1 loads first |
| K | Switch Sequence — optimal command ordering |
| L | Number of Switches — minimize operations |
| M | Switch Type — prefer automatic over manual |

---

## References
- Goda et al., *"Electric supply restoration in self-healed smart distribution systems: a review"*, Energy Informatics (2025) 8:114
- Kumar et al., *"Genetic algorithm for supply restoration in distribution system with priority customers"*, PMAPS 2006
- BPSO code adapted from: BPSO2.m (provided MATLAB reference)
