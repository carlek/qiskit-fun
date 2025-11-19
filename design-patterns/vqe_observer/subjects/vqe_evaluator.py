import numpy as np
from typing import Sequence
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from vqe_observer.core.subject import Subject

class VQEEvaluator(Subject):
    """Statevector-based VQE evaluator (Subject) for local, exact expectations.

    This Subject computes the energy expectation
    `E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩`
    exactly using a statevector simulator, then notifies observers with
    (energy, params, eval_count) on each *official* evaluation.

    This class binds parameters into the ansatz, prepares a full statevector, 
    and computes the expectation in-memory.
    
    Args:
        H: Observable as a SparsePauliOp.
        ansatz: Parameterized circuit `U(θ)`.
        theta0: Initial parameter vector in the circuit's parameter order.

    Notes:
        - Observers receive a single notification per iteration
        - Optimizers use gradient_param_shift() to obtain values for iteration
    """
    def __init__(self, H: SparsePauliOp, ansatz: QuantumCircuit, theta0: Sequence[float]) -> None:
        super().__init__()
        self.H = H
        self.ansatz = ansatz
        self.params = np.asarray(theta0, dtype=float)
        self.eval_count = 0
        self.stop = False  # toggle for observers to stop evaluation

    # Low-level evaluation
    def _statevector_from(self, theta: Sequence[float]) -> Statevector:
        """Bind parameters and construct the exact statevector for the ansatz.

        Args:
            theta: Parameter vector matching ``ansatz.parameters`` order.

        Returns:
            Statevector: a vector prepared from the parameterized circuit.
        """
        circ = self.ansatz.assign_parameters(theta)
        return Statevector.from_instruction(circ)

    def _energy_raw(self, theta: Sequence[float]) -> float:
        """Helper function to retrieve energy ⟨H⟩ from state vector"""
        psi = self._statevector_from(theta)
        return float(np.real(psi.expectation_value(self.H)))

    def energy(self, theta: Sequence[float]) -> float:
        """Compute exact ⟨H⟩ via statevector and notify observers

        Args:
            theta: Parameter vector at which to evaluate the energy.

        Returns:
            float: The scalar energy expectation value
        """       
        E = self._energy_raw(theta)
        self.eval_count += 1
        self.notify(E, np.asarray(theta, dtype=float), self.eval_count)
        return E

    def gradient_param_shift(self, theta: Sequence[float], shift: float) -> np.ndarray:
        """Compute ∂E/∂θ via the parameter-shift rule.

        Uses: ∂E/∂θ_i = 0.5 * [E(θ + s e_i) - E(θ - s e_i)], s = shift

        """
        theta = np.asarray(theta, dtype=float)
        grad = np.zeros_like(theta, dtype=float)

        for i in range(theta.size):
            tp = theta.copy()
            tm = theta.copy()
            tp[i] += shift
            tm[i] -= shift

            Ep = self._energy_raw(tp)
            Em = self._energy_raw(tm)
            grad[i] = 0.5 * (Ep - Em)

        return grad

    def execute(self, max_iters: int = 100) -> None:
        """Run the main VQE loop: one official evaluation per iteration.
        Args:
            max_iters: Maximum number of iterations (notifications). Early
                termination occurs if an observer resets self.stop = True.
        """
        self.stop = False
        for _ in range(max_iters):
            if self.stop:
                break
            # Evaluate at current params and notify observers
            _ = self.energy(self.params)
