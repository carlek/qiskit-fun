from typing import Optional
import numpy as np
from vqe_observer.core.observer import Observer

class BestTracker(Observer):
    """Tracks lowest energy and associated parameters."""

    def __init__(self) -> None:
        self.best_energy: Optional[float] = None
        self.best_params: Optional[np.ndarray] = None
        self.best_eval: Optional[int] = None

    def update(self, energy: float, params: np.ndarray, eval_count: int) -> None:
        if self.best_energy is None or energy < self.best_energy:
            self.best_energy = energy
            self.best_params = params.copy()
            self.best_eval = eval_count

    def dump(self) -> None:
        if self.best_energy is None:
            print("No evaluations recorded.")
        else:
            print(f"Best @ eval {self.best_eval}: E = {self.best_energy:.8f}")
            print(f"theta* = {np.array2string(self.best_params, precision=6, floatmode='fixed')}")
