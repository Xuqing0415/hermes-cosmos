"""
Custom exceptions for Hermes
"""

from typing import Any, Optional


class HermesError(Exception):
    """Base exception for all Hermes errors"""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code or "HERMES_ERROR"
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(HermesError):
    """Configuration related errors"""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class ValidationError(HermesError):
    """Validation related errors"""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(message, code="VALIDATION_ERROR", details=error_details)


class AuthenticationError(HermesError):
    """Authentication related errors"""

    def __init__(self, message: str = "Authentication failed", details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="AUTHENTICATION_ERROR", details=details)


class AuthorizationError(HermesError):
    """Authorization related errors"""

    def __init__(self, message: str = "Access denied", details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="AUTHORIZATION_ERROR", details=details)


class RateLimitError(HermesError):
    """Rate limit exceeded errors"""

    def __init__(self, retry_after: Optional[int] = None, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        if retry_after:
            error_details["retry_after"] = retry_after
        super().__init__("Rate limit exceeded", code="RATE_LIMIT_EXCEEDED", details=error_details)


class ResourceNotFoundError(HermesError):
    """Resource not found errors"""

    def __init__(self, resource_type: str, resource_id: str, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        error_details.update({"resource_type": resource_type, "resource_id": resource_id})
        super().__init__(f"{resource_type} not found: {resource_id}", code="RESOURCE_NOT_FOUND", details=error_details)


class ResourceExhaustedError(HermesError):
    """Resource exhausted errors"""

    def __init__(
        self,
        resource_type: str,
        requested: int,
        available: int,
        details: Optional[dict[str, Any]] = None,
    ):
        error_details = details or {}
        error_details.update({
            "resource_type": resource_type,
            "requested": requested,
            "available": available,
        })
        super().__init__(
            f"Insufficient {resource_type}: requested {requested}, available {available}",
            code="RESOURCE_EXHAUSTED",
            details=error_details,
        )


class SchedulerError(HermesError):
    """Scheduler related errors"""

    def __init__(self, message: str, job_id: Optional[str] = None, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        if job_id:
            error_details["job_id"] = job_id
        super().__init__(message, code="SCHEDULER_ERROR", details=error_details)


class SchedulingFailedError(SchedulerError):
    """Job scheduling failed errors"""

    def __init__(self, job_id: str, reason: str, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        error_details["reason"] = reason
        super().__init__(f"Failed to schedule job {job_id}: {reason}", job_id=job_id, details=error_details)


class CheckpointError(HermesError):
    """Checkpoint related errors"""

    def __init__(self, message: str, checkpoint_id: Optional[str] = None, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        if checkpoint_id:
            error_details["checkpoint_id"] = checkpoint_id
        super().__init__(message, code="CHECKPOINT_ERROR", details=error_details)


class CheckpointFailedError(CheckpointError):
    """Checkpoint creation failed errors"""

    def __init__(self, job_id: str, reason: str, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        error_details.update({"job_id": job_id, "reason": reason})
        super().__init__(f"Checkpoint failed for job {job_id}: {reason}", details=error_details)


class CheckpointRestoreError(CheckpointError):
    """Checkpoint restore failed errors"""

    def __init__(self, checkpoint_id: str, reason: str, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        error_details["reason"] = reason
        super().__init__(f"Failed to restore checkpoint {checkpoint_id}: {reason}", checkpoint_id=checkpoint_id, details=error_details)


class NetworkError(HermesError):
    """Network related errors"""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="NETWORK_ERROR", details=details)


class StorageError(HermesError):
    """Storage related errors"""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="STORAGE_ERROR", details=details)


class DatabaseError(HermesError):
    """Database related errors"""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="DATABASE_ERROR", details=details)


class CacheError(HermesError):
    """Cache related errors"""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="CACHE_ERROR", details=details)


class FaultPredictionError(HermesError):
    """Fault prediction related errors"""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="FAULT_PREDICTION_ERROR", details=details)


class MigrationError(HermesError):
    """Job migration related errors"""

    def __init__(self, job_id: str, reason: str, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        error_details.update({"job_id": job_id, "reason": reason})
        super().__init__(f"Migration failed for job {job_id}: {reason}", code="MIGRATION_ERROR", details=error_details)


class TimeoutError(HermesError):
    """Timeout errors"""

    def __init__(self, operation: str, timeout_seconds: int, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        error_details.update({"operation": operation, "timeout_seconds": timeout_seconds})
        super().__init__(f"Operation '{operation}' timed out after {timeout_seconds}s", code="TIMEOUT_ERROR", details=error_details)


class InternalError(HermesError):
    """Internal server errors"""

    def __init__(self, message: str = "Internal server error", details: Optional[dict[str, Any]] = None):
        super().__init__(message, code="INTERNAL_ERROR", details=details)


class ServiceUnavailableError(HermesError):
    """Service unavailable errors"""

    def __init__(self, service: str, reason: Optional[str] = None, details: Optional[dict[str, Any]] = None):
        error_details = details or {}
        error_details["service"] = service
        if reason:
            error_details["reason"] = reason
        message = f"Service '{service}' is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(message, code="SERVICE_UNAVAILABLE", details=error_details)
