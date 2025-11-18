import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

def verify_final_state(ansatz: QuantumCircuit, theta_opt: np.ndarray, H: SparsePauliOp) -> None:
    """Print exact <H> at optimum and overlap with |Φ+>."""
    psi_opt = Statevector.from_instruction(ansatz.assign_parameters(theta_opt))
    E_check = float(np.real(psi_opt.expectation_value(H)))
    print("Check <H> at optimum:", E_check)

    # |Φ+> = (|00> + |11>)/sqrt(2)
    phi_plus = (Statevector.from_label("00") + Statevector.from_label("11")).data / np.sqrt(2)
    overlap = np.vdot(phi_plus, psi_opt.data)
    print("|<Φ+|ψ_opt>|^2 =", float(np.abs(overlap) ** 2))
