from typing import List, Tuple
import numpy as np
from vqe_observer.core.observer import Observer

class Logger(Observer):
    """Accumulates evaluations: printing is deferred to dump()."""

    def __init__(self) -> None:
        self.history: List[Tuple[int, float, np.ndarray]] = []

    def update(self, energy: float, params: np.ndarray, eval_count: int) -> None:
        self.history.append((eval_count, energy, params.copy()))

    def dump(self) -> None:
        for k, e, p in self.history:
            print(f"[{k:04d}] E = {e:.8f}  theta = {np.array2string(p, precision=6, floatmode='fixed')}")

