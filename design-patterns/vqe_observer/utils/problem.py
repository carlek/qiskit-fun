import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from vqe_observer.utils.pauli import parse_pauli_observable

def build_2qubit_problem(obs_str: str):
    """
    Create a 2-qubit Sparse Pauli Operator and Circuit from a Pauli observable string
    Args:
        obs_str (str): The observable string to parse.
    Returns:
        H (SparsePauliOp): 
        ansatz (QuantumCircuit): TwoLocal (Ry/Rz, CX, reps=1) with |10> ref
    """
    pauli_list = parse_pauli_observable(obs_str)
    H = SparsePauliOp.from_list(pauli_list)

    if H.num_qubits != 2:
        raise ValueError(f"Observable={obs_str} has {H.num_qubits} qubits: Only 2 is supported.")

    ansatz = TwoLocal(
        num_qubits=H.num_qubits,
        rotation_blocks=["ry", "rz"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=1,
        insert_barriers=False,
    )

    # Optional HF-like reference: |10>
    reference = QuantumCircuit(H.num_qubits)
    reference.x(0)
    ansatz = reference.compose(ansatz)

    return H, ansatz

def initial_params(ansatz: QuantumCircuit, mode: str="zeros", seed: int=42) -> np.ndarray:
    """
    Construct initial parameter vector for the ansatz.
    mode: "zeros" | "random"
    """
    np.random.seed(seed)
    n = ansatz.num_parameters
    if mode == "random":
        return 2 * np.pi * np.random.rand(n)
    return np.zeros(n, dtype=float)
