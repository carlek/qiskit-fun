from abc import ABC
from typing import List
import numpy as np
from .observer import Observer

class Subject(ABC):
    """Subject: publishes events to all attached observers."""

    def __init__(self) -> None:
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def clear_observers(self) -> None:
        """Detach all observers."""
        self._observers.clear()

    def notify(self, energy: float, params: np.ndarray, eval_count: int) -> None:
        """Notify all observers."""
        for ob in self._observers:
            ob.update(energy, params, eval_count)
    
    def name(self):
        """Subject name"""
        return type(self).__name__
