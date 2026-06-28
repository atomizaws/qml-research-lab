[![QML Research Lab CI](https://github.com/atomizaws/qml-research-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/atomizaws/qml-research-lab/actions/workflows/ci.yml)
# QML Research Lab ⚛️🧪

**Quantum Machine Learning (QML) Research Lab:**
This repository serves as a centralized platform for my experiments, simulations, and reproducible research in preparation for my future studies.

For the moment, the goal involves exploring hybrid quantum-classical algorithms, benchmarking simulation performance (CPU/GPU), and investigating noise resilience in Variational Quantum Circuits (VQC).

## 📂 Repository Structure

This repository follows a monorepo structure to maintain a consistent environment across different experiments.

```text
.
├── datasets/                # Shared datasets and static assets
├── notebooks/               # Experiment notebooks (Categorized by topic)
│   ├─── 01_basic_vqc_benchmark/   # Baseline VQC & CPU Performance tests
│   │   ├── 01_vqc_two_moons_cpu.ipynb
│   │   └── outputs/         # Experiment results (plots/logs)
│   ├─── 02_noise_mitigation/   # Noise Robustness & Error Mitigation
│   │   ├── 02_noise_impact_analysis.ipynb
│   │   └── outputs/
│   ├─── ...
│   │   
│   ...
│
├── src/                     # Shared Python modules and utilities
├── Dockerfile               # Reproducible environment definition
├── docker-compose.yml       # Container orchestration
└── requirements.txt         # Python dependencies

```

## 📂 Research Roadmap & Archives


| No. | Theme | Key Technologies | Status |
| :--- | :--- | :--- | :--- |
| **01** | [**Ideal VQC Simulation**](notebooks/01_basic_vqc_benchmark/) | PennyLane, PyTorch, VQC | ✅ Completed |
| **02** | [**Noise Robustness & Error Mitigation**](notebooks/02_noise_mitigation/) | Qiskit Aer, ZNE, Noise Models | ✅ Completed |
| **03** | [**Quantum Architecture Search & Expressivity**](notebooks/03_qas_expressivity_search/) | PennyLane, QAS (Rotoselect), Data Re-uploading | ✅ Completed |
| **04** | *Coming Soon...* | | 🚧 Planned |


## 🛠 Tech Stack & Environment

* **Language:** Python 3.10
* **Quantum Framework:** PennyLane (optimized with `lightning.qubit` & OpenMP)
* **ML Framework:** PyTorch / Scikit-learn
* **Infrastructure:** Docker & Docker Compose (CPU-optimized for Apple Silicon & x86)

## 🚀 Getting Started

### 1. Build the Lab Environment

The Docker setup compiles quantum simulators from source for maximum CPU performance. This step may take a few minutes.

```bash
docker-compose up -d --build

```

### 2. Access JupyterLab

Open your browser and navigate to:
[http://localhost:8888](https://www.google.com/search?q=http://localhost:8888)

### 3. Run Experiments

Navigate to the `notebooks/` directory within JupyterLab.

* **Start here:** `01_basic_vqc_benchmark/01_vqc_two_moons_cpu.ipynb`
* Generated plots and logs are automatically saved to the `outputs/` folder within each experiment directory.

## ⚙️ Performance Tuning

You can adjust resource allocation in `docker-compose.yml` to match your hardware:

* **`OMP_NUM_THREADS`**: Set this to your physical CPU core count (e.g., `8` for M1 Max, `4` for typical laptops) to maximize parallel simulation speed.
* **Memory Limits**: Adjust `deploy.resources.limits.memory` if needed.

## 📝 License

[MIT License](https://opensource.org/licenses/MIT) - Feel free to use and modify for your own research.

```

```
