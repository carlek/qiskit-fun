from typing import Optional
import numpy as np
from vqe_observer.core.observer import Observer
from vqe_observer.subjects.vqe_evaluator import VQEEvaluator

class ParamShiftGD(Observer):
    """Plain (stochastic) gradient-descent step using parameter-shift gradients.

    Each update() does exactly one optimization GD step.
      - Observes (E, theta), fetching the gradient from evaluator.gradient_param_shift(...),
      - Optional gradient clipping and momentum are supported. 
      - set new parameters in evaluator.params
      - optional early-stop on small energy deltas
    """

    def __init__(
        self,
        evaluator: VQEEvaluator,
        lr: float = 0.05,
        epsilon: float = 1e-8,
        shift: float = np.pi / 2,
        limit: int = 8,
        grad_clip: Optional[float] = None,
        momentum: float = 0.0,
    ) -> None:
        """Initialize a parameter-shift gradient-descent optimizer.

        Args:
            evaluator: The VQE subject providing energy(theta, notify=...) and
                holding the mutable parameter vector in evaluator.params.
            lr: Learning rate (step size) for the GD update.
            epsilon: Numerical stability constant used in internal calculations.
            shift: Shift value (typically π/2) used in the parameter-shift rule.
            limit: Stop after this many consecutive iterations with energy change below epsilon tolerance.
            grad_clip: If set set norm of the gradient to this before applying momentum/updates.
            momentum: Exponential smoothing factor μ in [0, 1). 
                When > 0, a velocity buffer accumulates past gradients to reduce "zig-zagging".

        Note:
            The gradient for each parameter θ is estimated:
            0.5·(E(θ + shift) - E(θ - shift)) via calls to
            evaluator.energy(..., notify=False) to avoid re-notifying observers.
        """
        self.evaluator = evaluator
        self.lr = lr
        self.shift = shift
        self.epsilon = epsilon
        self.limit = limit
        self.grad_clip = grad_clip
        self.momentum = momentum
        self._last_energy: Optional[float] = None
        self._streak = 0
        self._vel: Optional[np.ndarray] = None

    def update(self, energy: float, theta: np.ndarray, eval_count: int) -> None:
        # stop if epsilon or limit is reached.
        if self._last_energy is not None and abs(self._last_energy - energy) < self.epsilon:
            self._streak += 1
            if self._streak >= self.limit:
                self.evaluator.stop = True
        else:
            self._streak = 0
        self._last_energy = energy

        # fetch gradient from evaluator
        g = self.evaluator.gradient_param_shift(theta, self.shift)

        # Optional gradient clipping
        if self.grad_clip is not None:
            norm = np.linalg.norm(g)
            if norm > self.grad_clip and norm > 0:
                g = g * (self.grad_clip / norm)

        # Momentum (optional)
        if self.momentum and (self._vel is not None):
            self._vel = self.momentum * self._vel + (1.0 - self.momentum) * g
            step = self.lr * self._vel
        elif self.momentum:
            self._vel = g.copy()
            step = self.lr * self._vel
        else:
            step = self.lr * g

        # Gradient descent step
        new_params = theta - step
        self.evaluator.params = new_params
