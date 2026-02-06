import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
from qiskit_aer import noise, ReadoutError

# --- Constants (Lightweight for CI, but preserving trends) ---
N_QUBITS = 2
N_LAYERS = 3
EPOCHS = 15        # Number of training epochs
SAMPLES = 100      # Number of data samples
SHOTS = 1000       # Number of shots for quantum measurement

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Noise Model Factory ---
def get_noise_model(prob_gate=0.01, prob_readout=0.05):
    """
    Constructs a noise model for Qiskit Aer with specified error rates.
    """
    noise_model = noise.NoiseModel()
    
    # Gate error (Depolarizing error)
    error_1 = noise.depolarizing_error(prob_gate, 1)
    noise_model.add_all_qubit_quantum_error(error_1, ['rx', 'ry', 'rz'])
    error_2 = noise.depolarizing_error(prob_gate * 5, 2)
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    
    # Readout error (Dominant factor in this experiment)
    probabilities = [[1 - prob_readout, prob_readout], [prob_readout, 1 - prob_readout]]
    readout_error = ReadoutError(probabilities)
    noise_model.add_all_qubit_readout_error(readout_error)
    
    return noise_model

# --- 2. Dataset Creation ---
def create_dataset():
    """
    Creates a dataset with a distribution that is learnable but susceptible to noise.
    Using synthetic data similar to 'Two Moons' but simplified for Qubit rotation logic.
    """
    # Class 0: Clustered around (0.0, 0.0)
    X0 = torch.randn(SAMPLES // 2, 2) * 0.4 + 0.0
    y0 = torch.zeros(SAMPLES // 2).long()
    
    # Class 1: Clustered around (1.2, 1.2) 
    # (Separated but close enough to cause misclassification under noise)
    X1 = torch.randn(SAMPLES // 2, 2) * 0.4 + 1.2
    y1 = torch.ones(SAMPLES // 2).long()
    
    X = torch.cat([X0, X1])
    y = torch.cat([y0, y1])
    
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), 
        batch_size=10, 
        shuffle=True
    )

# --- 3. Quantum Circuit & Model ---
def circuit_logic(inputs, weights):
    # Using StronglyEntanglingLayers as established in the research notebook
    qml.templates.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]

def train_and_get_model():
    """
    Trains a model using the Lightning backend for speed and returns the trained model.
    """
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    # Use Adjoint differentiation for speed
    qnode = qml.QNode(circuit_logic, dev, interface="torch", diff_method="adjoint")
    
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
    
    model = nn.Sequential(qlayer, nn.Linear(N_QUBITS, 2))
    
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    dataloader = create_dataset()
    
    # Training loop
    model.train()
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
    Performs inference on the specified backend and returns the accuracy.
    """
    if backend_name == "qiskit.aer":
        # Device for noise simulation
        dev = qml.device("qiskit.aer", wires=N_QUBITS, noise_model=noise_model, shots=SHOTS)
    else:
        # Device for ideal simulation
        dev = qml.device(backend_name, wires=N_QUBITS)
        
    qnode = qml.QNode(circuit_logic, dev, interface="torch")
    
    # Transfer weights and rebuild model for inference
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
    test_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
    test_model = nn.Sequential(test_layer, nn.Linear(N_QUBITS, 2))
    
    # Load trained weights
    with torch.no_grad():
        # Copy weights for qlayer (index 0)
        test_model[0].weights.copy_(model[0].weights)
        # Copy weights and bias for Linear layer (index 1)
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

# --- 4. Test Cases (Executed by Pytest) ---

def test_pipeline_integrity():
    """
    Tests if the Training -> Inference pipeline works and achieves high accuracy 
    in an ideal environment (Lightning).
    """
    print("\n--- Starting Training ---")
    model = train_and_get_model()
    
    print("--- Evaluating on Ideal Simulator ---")
    acc_ideal = evaluate_accuracy(model, "lightning.qubit")
    print(f"Ideal Accuracy: {acc_ideal:.2f}%")
    
    # Criteria: Should achieve >90% in an ideal environment
    assert acc_ideal >= 90.0, f"Training failed! Accuracy too low: {acc_ideal}%"

def test_noise_impact():
    """
    Regression test to verify behavior under noisy environments.
    Checks if Qiskit Aer works and if the model degrades as expected under noise.
    """
    model = train_and_get_model()
    
    # Case A: Acceptable Noise (Readout error 5%)
    # Common level in real hardware. Should remain robust (>80%)
    noise_low = get_noise_model(prob_readout=0.05)
    acc_low = evaluate_accuracy(model, "qiskit.aer", noise_low)
    print(f"\nLow Noise Accuracy: {acc_low:.2f}%")
    
    # Case B: Critical Noise (Readout error 20%)
    # Should fail or degrade significantly
    noise_high = get_noise_model(prob_readout=0.20)
    acc_high = evaluate_accuracy(model, "qiskit.aer", noise_high)
    print(f"High Noise Accuracy: {acc_high:.2f}%")
    
    # Assertion for robustness
    assert acc_low >= 80.0, f"Model is too fragile! Failed at 5% noise: {acc_low}%"
    
    # Verify noise simulation is working:
    # Accuracy should drop under high noise compared to low noise, or fall below 70%
    assert acc_high < acc_low or acc_high < 70.0, \
        "Noise simulation might be broken! High noise didn't degrade performance correctly."

