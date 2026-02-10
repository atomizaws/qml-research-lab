# 01: Baseline VQC & CPU Performance Benchmark

## 🎯 Objective

To establish a baseline Quantum Machine Learning (QML) pipeline using a Variational Quantum Circuit (VQC) and to benchmark the training performance on a CPU-based simulator (**PennyLane Lightning**) without GPU acceleration.

## 🧪 Key Experiments

### 1. Non-linear Classification Task

We utilized the **"Two Moons"** dataset (Scikit-learn) to test the VQC's ability to classify non-linearly separable data.

* **Encoding:** Angle Embedding (converting classical data to quantum states).
* **Ansatz:** Strongly Entangling Layers (3 layers).
* **Measurement:** Pauli-Z expectation value.

### 2. CPU Optimization Benchmark

Instead of using GPUs, we focused on optimizing CPU performance for small-scale quantum simulations (2-4 qubits).

* **Backend:** `lightning.qubit` (C++ optimized state-vector simulator).
* **Parallelization:** Tuning `OMP_NUM_THREADS` to leverage multi-core architectures (e.g., Apple Silicon M1/M2/M3).

## 📊 Results

The model successfully learned the non-linear decision boundary with high efficiency, demonstrating that CPU-based simulation is sufficient and faster for low-qubit prototyping.

| Metric | Result | Note |
| --- | --- | --- |
| **Final Accuracy** | **~97.5%** | Successfully classified Two Moons data. |
| **Training Time** | **< 15 sec** | 20 Epochs (on M1 Max / 8 Threads). |
| **Convergence** | Stable | Loss dropped consistently (MSE < 0.75). |

## 📂 Files

* **`01_vqc_two_moons_cpu.ipynb`**: The foundational notebook containing the VQC implementation, training loop, and CPU benchmark setup.