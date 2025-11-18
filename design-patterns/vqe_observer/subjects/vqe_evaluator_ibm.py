from typing import Sequence, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
from vqe_observer.core.subject import Subject

class IBMVQEEvaluator(Subject):
    """VQE Evaluator using IBM Quantum Runtime Estimator.

    This Subject computes `E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩` on an IBM backend using the
    Estimator primitive, then notifies observers with (energy, params, eval_count).

    Args:
        H: Observable as a SparsePauliOp.
        ansatz: Parameterized circuit U(θ).
        theta0: Initial parameter vector.
        backend: IBM backend name (e.g., "ibm_brisbane"). If None, a simulator or
            default choice may be selected by your account/instance.
        service: Optional pre-initialized QiskitRuntimeService (advanced use).
        instance: IBM Quantum instance string (e.g., "ibm-q/open/main"). Required
            if your account has access to multiple instances.
        shots: Number of shots per Estimator evaluation.
        optimization_level: Transpile optimization level (0-3).
        resilience_level: Runtime resilience setting (0-3) for error mitigation.
        seed: Optional transpiler/primitive seed for reproducibility.

    Notes:
        - You must have authenticated with IBM Quantum (saved account or env token).
        - Estimator accepts the *symbolic* ansatz and a list of parameter values.
    """

    def __init__(
        self,
        H: SparsePauliOp,
        ansatz: QuantumCircuit,
        theta0: Sequence[float],
        backend: str = None,
        service: Optional[QiskitRuntimeService] = None,
        instance: Optional[str] = None,
        shots: int = 512,
        optimization_level: int = 1,
        resilience_level: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.H = H
        self.ansatz = ansatz
        self.params = np.asarray(theta0, dtype=float)
        self.eval_count = 0
        self.stop = False

        # save backend services
        self._service = service
        self._backend = backend
        self._instance = instance
        self._shots = shots
        self._opt_level = optimization_level
        self._res_level = resilience_level
        self._seed = seed
        
        self._estimator: Estimator
        self._isa_ansatz: QuantumCircuit
        self._isa_obs: SparsePauliOp

    def _prepare_isa(self) -> None:
        """Transpile ansatz to backend ISA and setup observable"""
        pm = generate_preset_pass_manager(
            backend=self._backend,
            optimization_level=int(self._opt_level),
            seed_transpiler=self._seed,
        )
        isa_psi = pm.run(self.ansatz)
        isa_obs = self.H.apply_layout(isa_psi.layout)
        self._isa_ansatz = isa_psi
        self._isa_obs = isa_obs

    def energy(self, theta: Sequence[float], notify: bool = True) -> float:
        """Compute ⟨H⟩ on IBM hardware/simulator; optionally notify observers.

        Args:
            theta: Parameter vector for the ansatz in the circuit's parameter order.
            notify: If True, increments eval_count and notifies observers.

        Returns:
            float: Energy expectation value.

        Note:
            Assumes a live Estimator has been created by execute(...).
        """
        job = self._estimator.run(
            [ (self._isa_ansatz,
               self._isa_obs, 
               [np.asarray(theta, float).tolist()])
            ]
        )

        pub= job.result()[0]
        E = float(pub.data.evs[0])

        if notify:
            self.eval_count += 1
            self.notify(E, np.asarray(theta, dtype=float), self.eval_count)
        return E

    def execute(self, max_iters: int = 200) -> None:
        """Run the main VQE loop with a single long-running Estimator.

        An Estimator are created once at the start of this
        method, reused for all evaluations, and closed automatically on exit.

        Args:
            max_iters: Maximum number of iterations (notifications). Early
                termination occurs if an observer sets self.stop = True.
        """
    
        self._prepare_isa()
        
        self._estimator = Estimator(mode=self._backend)
        self._estimator.options.default_shots = int(self._shots) if self._shots else None
        self._estimator.options.resilience_level = int(self._res_level) if self._res_level else None
        self._estimator.options.seed_estimator = int(self._seed) if self._seed else None

        self.stop = False
        for _ in range(max_iters):
            if self.stop:
                break
            _ = self.energy(self.params, notify=True)

        # run is finished        
        self._estimator = None

    def dump_backend_info(self) -> None:
        """Best-effort print of backend/options while active."""
        try:
            if self._session is not None:
                print("Session backend:", self._backend)
        except Exception:
            pass
        