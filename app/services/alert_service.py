"""Alert service implementing the Observer pattern.

:class:`AlertService` is the *subject*: it publishes alerts to any number of
subscribers.  Two concrete observers ship with the project:

* :class:`AlertStore`  - keeps the alerts in memory for the API to serve.
* :class:`EventLogger` - writes a line to the application log.

Adding a new consumer (e.g. an e-mail notifier or a webhook dispatcher) is
just a matter of subclassing :class:`~app.core.observer.Observer`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from app.core.exceptions import DuplicateSubscriberError
from app.core.observer import Observer, Subject
from app.models.readings import Alert, AlertSeverity

logger = logging.getLogger(__name__)

#: Maximum number of alerts retained in memory.
MAX_STORED_ALERTS: int = 200


class AlertStore(Observer):
    """Observer that retains alerts in a bounded in-memory list.

    Attributes:
        _alerts: The retained alerts, newest first.
    """

    def __init__(self, max_alerts: int = MAX_STORED_ALERTS) -> None:
        """Create the store.

        Args:
            max_alerts: Maximum number of alerts to retain.
        """
        self._alerts: list[Alert] = []
        self._max_alerts = max_alerts

    def update(self, event: Alert) -> None:
        """Store a published alert.

        Args:
            event: The alert to store.
        """
        if not isinstance(event, Alert):
            return
        self._alerts.insert(0, event)
        del self._alerts[self._max_alerts :]

    @property
    def alerts(self) -> list[Alert]:
        """A copy of the retained alerts, newest first."""
        return list(self._alerts)

    def clear(self) -> None:
        """Remove every retained alert."""
        self._alerts.clear()


class EventLogger(Observer):
    """Observer that writes alerts to the application log."""

    def update(self, event: Alert) -> None:
        """Log a published alert.

        Args:
            event: The alert to log.
        """
        if isinstance(event, Alert):
            logger.warning(
                "ALERT [%s] %s - %s", event.severity.value, event.machine_id, event.cause
            )


class AlertService(Subject):
    """Publishes alerts to its subscribers and keeps a queryable history.

    Attributes:
        _store: The built-in :class:`AlertStore` observer.
    """

    def __init__(self) -> None:
        """Create the service and register the built-in observers."""
        super().__init__()
        self._store = AlertStore()
        self.subscribe(self._store)
        self.subscribe(EventLogger())

    # ------------------------------------------------------------------
    # Subject / Observer
    # ------------------------------------------------------------------
    def subscribe(self, observer: Observer) -> None:
        """Register an observer, rejecting duplicate subscriptions.

        Args:
            observer: The observer to register.

        Raises:
            DuplicateSubscriberError: If the observer is already registered.
        """
        if observer in self._observers:
            raise DuplicateSubscriberError(
                "This observer is already subscribed to the alert service.",
                observer=type(observer).__name__,
            )
        self._observers.append(observer)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def publish(
        self,
        machine_id: str,
        severity: AlertSeverity,
        cause: str,
        alert_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> Alert:
        """Create an alert and notify every subscriber.

        Args:
            machine_id: The machine the alert relates to.
            severity: The alert severity.
            cause: Human readable description of what happened.
            alert_id: Optional explicit identifier.
            timestamp: Optional explicit timestamp.

        Returns:
            The published :class:`Alert`.
        """
        alert = Alert(
            id=alert_id or uuid.uuid4().hex[:12],
            machine_id=machine_id,
            severity=severity,
            timestamp=timestamp or datetime.utcnow(),
            cause=cause,
        )
        self.notify(alert)
        return alert

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def list_alerts(
        self,
        limit: int = 50,
        machine_id: str | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Return the retained alerts, newest first.

        Args:
            limit: Maximum number of alerts to return.
            machine_id: Optional filter by machine.
            severity: Optional filter by severity.

        Returns:
            The filtered alert list.
        """
        alerts = self._store.alerts
        if machine_id:
            alerts = [a for a in alerts if a.machine_id == machine_id]
        if severity:
            alerts = [a for a in alerts if a.severity is severity]
        return alerts[:limit]

    def acknowledge(self, alert_id: str, user: str = "operator") -> Optional[Alert]:
        """Mark an alert as acknowledged.

        Args:
            alert_id: The identifier of the alert to acknowledge.
            user: Who acknowledged it.

        Returns:
            The updated alert, or ``None`` if the id is unknown.
        """
        for alert in self._store._alerts:  # noqa: SLF001 - internal mutation
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = user
                return alert
        return None

    def clear(self) -> None:
        """Clear the stored alert history."""
        self._store.clear()

    @property
    def store(self) -> AlertStore:
        """The built-in alert store observer."""
        return self._store
