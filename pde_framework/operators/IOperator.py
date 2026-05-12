"""Abstract interface for differential operators."""

from __future__ import annotations

from abc import ABC, abstractmethod


class IOperator(ABC):
    """Abstract operator interface.

    Implementations take a field and return a new field instance.
    """

    @abstractmethod
    def apply(self, field):
        """Apply the operator and return a new field."""
