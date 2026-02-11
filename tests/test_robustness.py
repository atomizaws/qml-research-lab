import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
from sklearn.datasets import make_moons
from qiskit_aer import noise
from qiskit_aer.noise import ReadoutError
import warnings

# --- Configuration for Month 2 Benchmark ---
# Match parameters with '02_noise_impact_analysis.ipynb'
N_QUBITS = 2
N_LAYERS = 3       # Depth of StronglyEntanglingLayers
EPOCHS = 25        # Sufficient for convergence on Two Moons
SAMPLES = 200      # Match notebook sample size
SHOTS = 1000       # Number of measurement shots
NOISE_LEVEL = 0.1  # Data noise for make_moons

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Suppress specific warnings for cleaner CI logs
warnings.filterwarnings("ignore", category=UserWarning, module="pennylane_qiskit")

# --- 1. Noise Model Factory ---
def get_noise_model(prob_gate=0.01, prob_readout=0.03):
    """
    Constructs a noise model simulating real hardware imperfections.
    Based on the analysis in 02_noise_impact_analysis.ipynb.
    """
    noise_model = noise.NoiseModel()
    
    # Gate error (Depolarizing error)
    error_1 = noise.depolarizing_error(prob_gate, 1)
    noise_model.add_all_qubit_quantum_error(error_1, ['rx', 'ry', 'rz'])
    error_2 = noise.depolarizing_error(prob_gate * 5, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    
    # Readout error (Dominant factor)
    probabilities = [[1 - prob_readout, prob_readout], [prob_readout, 1 - prob_readout]]
    readout_error = ReadoutError(probabilities)
    noise_model.add_all_qubit_readout_error(readout_error)
    
    return noise_model

# --- 2. Dataset Creation (Updated to Two Moons) ---
def create_dataset():
    """
    Generates the 'Two Moons' dataset using scikit-learn, 
    matching the logic in Month 2 notebooks.
    """
    X, y = make_moons(n_samples=SAMPLES, noise=NOISE_LEVEL, random_state=42)
    
    # Normalize inputs to be within [0, pi] range for AngleEmbedding
    # Shift and scale: range roughly [-1, 2] -> [0, pi]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min) * np.pi
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_tensor), 
        batch_size=10, 
        shuffle=True
    )

# --- 3. Quantum Circuit & Model ---
def circuit_logic(inputs, weights):
    # AngleEmbedding fits well with normalized Two Moons data
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]

def train_and_get_model():
    """
    Trains a fresh model using the Lightning backend (Plan A).
    """
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    # Adjoint differentiation is crucial for speed in CI
    qnode = qml.QNode(circuit_logic, dev, interface="torch", diff_method="adjoint")
    
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    
    model = nn.Sequential(qlayer, nn.Linear(N_QUBITS, 2))
    
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    dataloader = create_dataset()
    
    # Training loop
    model.train()
    print(f"Training for {EPOCHS} epochs...")
    for _ in range(EPOCHS):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_accuracy(model, backend_name, noise_model=None):
    """
    Performs inference and returns accuracy percentage.
    """
    if backend_name == "qiskit.aer":
        dev = qml.device("qiskit.aer", wires=N_QUBITS, noise_model=noise_model, shots=SHOTS)
    else:
        dev = qml.device(backend_name, wires=N_QUBITS)
        
    qnode = qml.QNode(circuit_logic, dev, interface="torch")
    
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
    test_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    test_model = nn.Sequential(test_layer, nn.Linear(N_QUBITS, 2))
    
    # Load trained weights
    with torch.no_grad():
        test_model[0].weights.copy_(model[0].weights)
        test_model[1].weight.copy_(model[1].weight)
        test_model[1].bias.copy_(model[1].bias)
    
    test_model.eval()
    dataloader = create_dataset()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = test_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

# --- 4. Test Cases ---

def test_training_and_ideal_accuracy():
    """
    Test 1: Verify that the model can learn the Two Moons dataset 
    in an ideal environment (lightning.qubit).
    """
    print("\n--- Starting Training (Two Moons) ---")
    model = train_and_get_model()
    
    print("--- Evaluating on Ideal Simulator ---")
    acc_ideal = evaluate_accuracy(model, "lightning.qubit")
    print(f"Ideal Accuracy: {acc_ideal:.2f}%")
    
    # Goal: Ideal simulation should easily classify Two Moons (> 90%)
    assert acc_ideal >= 90.0, f"Training failed! Ideal accuracy too low: {acc_ideal}%"

def test_noise_robustness():
    """
    Test 2: Verify performance under noise (Qiskit Aer).
    Goal: Check if the model maintains acceptable accuracy (> 75%)
    even with Readout Errors, matching Month 2 analysis.
    """
    model = train_and_get_model()
    
    # Noise Config: Readout error 3% (Typical hardware level)
    noise_model = get_noise_model(prob_readout=0.03)
    
    print("\n--- Evaluating on Qiskit Aer (Noisy) ---")
    acc_noisy = evaluate_accuracy(model, "qiskit.aer", noise_model)
    print(f"Noisy Accuracy: {acc_noisy:.2f}%")
    
    # Assertion: 
    # Based on Month 2 notebooks, we expect ~86%. 
    # Setting threshold to 75% to prevent flaky CI failures.
    assert acc_noisy >= 75.0, f"Model is too fragile! Accuracy dropped below 75%: {acc_noisy}%"
