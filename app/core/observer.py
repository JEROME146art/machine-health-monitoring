"""Observer pattern primitives used by the alert subsystem.

The :class:`Subject` keeps a list of observers and notifies them whenever a
state change occurs.  :class:`AlertService` acts as a concrete subject and
:class:`EventLogger` / :class:`AlertStore` as concrete observers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Observer(ABC):
    """Interface for objects that react to events published by a subject."""

    @abstractmethod
    def update(self, event: Any) -> None:
        """Handle a published event.

        Args:
            event: The event payload published by the subject.
        """
        raise NotImplementedError


class Subject(ABC):
    """Base class implementing the observer subscription mechanics.

    Attributes:
        _observers: Registered observers, kept in subscription order.
    """

    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def subscribe(self, observer: Observer) -> None:
        """Register an observer.

        Args:
            observer: The observer to notify on future events.
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe(self, observer: Observer) -> None:
        """Remove an observer if present.

        Args:
            observer: The observer to remove.
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event: Any) -> None:
        """Publish an event to every registered observer.

        A failing observer must not prevent the remaining observers from
        receiving the event, so exceptions are isolated per subscriber.

        Args:
            event: The event payload to publish.
        """
        for observer in list(self._observers):
            try:
                observer.update(event)
            except Exception:  # noqa: BLE001 - observer isolation is intentional
                # An observer failure is contained so the remaining
                # subscribers still receive the event.
                continue

    @property
    def observer_count(self) -> int:
        """Number of currently registered observers."""
        return len(self._observers)