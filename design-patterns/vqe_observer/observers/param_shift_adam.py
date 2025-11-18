import numpy as np
from typing import Optional
from vqe_observer.core.observer import Observer
from vqe_observer.subjects.vqe_evaluator import VQEEvaluator

class ParamShiftAdam(Observer):
    """Adaptive Moment (Adam) optimizer using parameter-shift gradients.
    
    Each update() does exactly one optimization step:
      - compute paramater shift gradient without notifying other observers (notify=False)
      - update 1st & 2nd Adam moments (beta1 & beta2)
      - set new parameters in evaluator
      - optional early-stop on small energy deltas
    """

    def __init__(
        self,
        evaluator: VQEEvaluator,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-6,
        shift: float = np.pi / 2,
        limit: int = 8,
    ) -> None:
        """Initialize an Adam optimizer that uses parameter-shift gradients.

        Args:
            evaluator: The VQE subject providing energy(theta, notify=...) and
                holding the mutable parameter vector in evaluator.params.
            lr: Base learning rate (step size) for Adam updates.
            beta1: Exponential decay rate for first-moment (mean) estimates.
            beta2: Exponential decay rate for second-moment (variance) estimates.
            epsilon: Numerical stability constant added to the denominator.
            shift: Shift value (π/2) used in the parameter-shift rule.
            limit: Stop after limit iterations with energy change below epsilon.
        
        Note:
            The gradient for each parameter θ is estimated:
            `0.5·(E(θ + shift) - E(θ - shift))` via calls to
            evaluator.energy(..., notify=False) to avoid re-notifying observers.
        """
        self.evaluator = evaluator
        self.lr, self.b1, self.b2 = lr, beta1, beta2
        self.epsilon = epsilon
        self.shift = shift
        self.limit = limit

        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0
        self._last_energy: Optional[float] = None
        self._streak: int = 0

    def _gradient_param_shift(self, theta: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(theta, dtype=float)
        for i in range(theta.size):
            tp = theta.copy(); tp[i] += self.shift
            tm = theta.copy(); tm[i] -= self.shift
            Ep = self.evaluator.energy(tp, notify=False)
            Em = self.evaluator.energy(tm, notify=False)
            grad[i] = 0.5 * (Ep - Em)
        return grad

    def update(self, energy: float, theta: np.ndarray, eval_count: int) -> None:
        # Early stop based on energy improvement
        if self._last_energy is not None and abs(self._last_energy - energy) < self.epsilon:
            self._streak += 1
            if self._streak >= self.limit:
                self.evaluator.stop = True
        else:
            self._streak = 0
        self._last_energy = energy

        # Compute gradient (silent)
        g = self._gradient_param_shift(theta)

        # Lazy init moments
        if self.m is None:
            self.m = np.zeros_like(g)
            self.v = np.zeros_like(g)

        # Adam moments
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)

        # Bias-corrected moments
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)

        # Parameter update
        step = self.lr * mhat / (np.sqrt(vhat) + self.epsilon)
        self.evaluator.params = theta - step
