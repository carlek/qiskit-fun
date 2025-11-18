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
        - Observers receive a single notification per iteration via
          `notify=True`. Optimizers may perform additional *silent*
          probes with notify=False (e.g., parameter-shift gradients) that do
          not re-notify or increment the evaluation counter.
    """
    def __init__(self, H: SparsePauliOp, ansatz: QuantumCircuit, theta0: Sequence[float], max_iters: int) -> None:
        super().__init__()
        self.H = H
        self.ansatz = ansatz
        self.params = np.asarray(theta0, dtype=float)
        self.max_iters = max_iters
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

    def energy(self, theta: Sequence[float], notify: bool = True) -> float:
        """Compute exact ⟨H⟩ via statevector; optionally notify observers.

        Args:
            theta: Parameter vector at which to evaluate the energy.
            notify: If `True`, increments `eval_count` and calls
                `notify(energy, theta, eval_count)`; if `False`, performs a
                *silent* probe (no notify, no counter increment).

        Returns:
            float: The scalar energy expectation value.
        """       
        psi = self._statevector_from(theta)
        E = float(np.real(psi.expectation_value(self.H)))
        if notify:
            self.eval_count += 1
            self.notify(E, np.asarray(theta, dtype=float), self.eval_count)
        return E

    def execute(self) -> None:
        """Run the main VQE loop: one official evaluation per iteration.
        Args:
            max_iters: Maximum number of iterations (notifications). Early
                termination occurs if an observer resets self.stop = True.
        """
        self.stop = False
        for _ in range(self.max_iters):
            if self.stop:
                break
            # Evaluate at current params and notify observers
            _ = self.energy(self.params, notify=True)
