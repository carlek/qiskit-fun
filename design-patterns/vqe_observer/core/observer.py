from abc import ABC, abstractmethod
import numpy as np

class Observer(ABC):
    """Observer: Interface for receiving events from a Subject."""

    @abstractmethod
    def update(self, energy: float, params: np.ndarray, eval_count: int) -> None:
        """New energy evaluation from given parameters"""
        raise NotImplementedError

    # Optional hooks (no-ops by default)
    def dump(self) -> None:
        """Report or plot"""
        return None

    def cleanup(self) -> None:
        """Release resources."""
        return None

    def name(self) -> str:
        """Observer name"""
        return type(self).__name__