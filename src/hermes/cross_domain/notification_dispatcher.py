from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os


@dataclass
class NotificationEvent:
    event_type: str
    title: str
    message: str
    severity: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


EVENT_TYPES = {
    "snapshot_recorded": "快照记录",
    "degradation_detected": "退化检测",
    "correction_applied": "自动修复",
    "pattern_discovered": "新模式发现",
    "transfer_breakthrough": "迁移突破",
    "policy_updated": "策略更新",
    "experiment_completed": "实验完成",
    "milestone_achieved": "里程碑达成",
}


class ConsoleHandler:
    def send(self, event: NotificationEvent) -> bool:
        prefix = {
            "info": "[INFO]",
            "warning": "[WARN]",
            "critical": "[CRIT]",
            "success": "[OK]",
        }.get(event.severity, "[INFO]")

        print(f"{prefix} [{event.event_type}] {event.title}")
        print(f"  {event.message}")
        return True


class LogFileHandler:
    def __init__(self, log_path: str = "loop_output/notifications.log"):
        self._log_path = log_path
        directory = os.path.dirname(log_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def send(self, event: NotificationEvent) -> bool:
        try:
            with open(self._log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False


class WebhookHandler:
    def __init__(self, webhook_url: Optional[str] = None):
        self._webhook_url = webhook_url

    def send(self, event: NotificationEvent) -> bool:
        if not self._webhook_url:
            return False
        try:
            import urllib.request
            data = json.dumps(event.to_dict()).encode('utf-8')
            req = urllib.request.Request(
                self._webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False


class NotificationDispatcher:
    def __init__(self):
        self._handlers: List[Callable] = [
            ConsoleHandler().send,
            LogFileHandler().send,
        ]
        self._history: List[NotificationEvent] = []

    def add_handler(self, handler: Callable):
        self._handlers.append(handler)

    def add_webhook(self, url: str):
        self._handlers.append(WebhookHandler(url).send)

    def dispatch(self, event_type: str, title: str, message: str,
                 severity: str = "info", metadata: Optional[Dict[str, Any]] = None) -> NotificationEvent:
        event = NotificationEvent(
            event_type=event_type,
            title=title,
            message=message,
            severity=severity,
            metadata=metadata or {},
        )
        self._history.append(event)

        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                pass

        return event

    def get_history(self, limit: int = 20) -> List[NotificationEvent]:
        return self._history[-limit:]

    def get_history_dict(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._history[-limit:]]


_default_dispatcher = None


def get_dispatcher() -> NotificationDispatcher:
    global _default_dispatcher
    if _default_dispatcher is None:
        _default_dispatcher = NotificationDispatcher()
    return _default_dispatcher


def notify(event_type: str, title: str, message: str,
           severity: str = "info", metadata: Optional[Dict[str, Any]] = None) -> NotificationEvent:
    return get_dispatcher().dispatch(event_type, title, message, severity, metadata)