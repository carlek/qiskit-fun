from typing import List
import numpy as np
from vqe_observer.core.observer import Observer

class Visualizer(Observer):
    """Collects energies and plots convergence in dump()."""

    def __init__(self, method: str) -> None:
        self.energies: List[float] = []
        self.method: str = method

    def update(self, energy: float, params: np.ndarray, eval_count: int) -> None:
        self.energies.append(energy)

    def dump(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping plot.")
            return
        plt.plot(self.energies, marker='o')
        plt.title(f"VQE Energy: {self.method}")
        plt.xlabel("Evaluation")
        plt.ylabel("Energy")
        plt.grid(True)
        plt.show()
