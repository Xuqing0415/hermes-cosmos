"""
Self-healing engine for automated fault recovery
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from hermes.core.models import FaultPrediction

if TYPE_CHECKING:
    from hermes.agent.main import HermesAgent

logger = structlog.get_logger()


class HealingAction:
    MIGRATE_JOB = "migrate_job"
    REDUCE_LOAD = "reduce_load"
    RESTART_SERVICE = "restart_service"
    ALERT_ADMIN = "alert_admin"
    SCALE_OUT = "scale_out"


class SelfHealingEngine:
    def __init__(
        self,
        agent: "HermesAgent",
        auto_heal_enabled: bool = True,
    ) -> None:
        self.agent = agent
        self.auto_heal_enabled = auto_heal_enabled
        self._healing_history: list[dict[str, Any]] = []

    async def handle_prediction(self, prediction: FaultPrediction) -> None:
        logger.info(
            "Handling fault prediction",
            prediction_id=str(prediction.id),
            type=prediction.prediction_type,
            probability=prediction.probability,
        )

        action = self._determine_action(prediction)

        if self.auto_heal_enabled:
            await self._execute_action(action, prediction)
        else:
            logger.info("Auto-healing disabled, would execute action", action=action)

    def _determine_action(self, prediction: FaultPrediction) -> str:
        prediction_type = prediction.prediction_type

        if prediction_type in ["gpu_overheat", "gpu_power_overload"]:
            if prediction.probability > 0.9:
                return HealingAction.MIGRATE_JOB
            else:
                return HealingAction.REDUCE_LOAD

        if prediction_type == "memory_exhaustion":
            return HealingAction.MIGRATE_JOB

        if prediction_type == "network_degradation":
            return HealingAction.ALERT_ADMIN

        return HealingAction.ALERT_ADMIN

    async def _execute_action(
        self,
        action: str,
        prediction: FaultPrediction,
    ) -> bool:
        logger.info(
            "Executing healing action",
            action=action,
            prediction_id=str(prediction.id),
        )

        result = False

        try:
            if action == HealingAction.MIGRATE_JOB:
                result = await self._migrate_job(prediction)
            elif action == HealingAction.REDUCE_LOAD:
                result = await self._reduce_load(prediction)
            elif action == HealingAction.RESTART_SERVICE:
                result = await self._restart_service(prediction)
            elif action == HealingAction.ALERT_ADMIN:
                result = await self._alert_admin(prediction)
            elif action == HealingAction.SCALE_OUT:
                result = await self._scale_out(prediction)

            self._record_healing(action, prediction, result)

        except Exception as e:
            logger.error(
                "Healing action failed",
                action=action,
                error=str(e),
            )
            result = False

        return result

    async def _migrate_job(self, prediction: FaultPrediction) -> bool:
        logger.info("Migrating job due to fault prediction")

        await asyncio.sleep(1)

        return True

    async def _reduce_load(self, prediction: FaultPrediction) -> bool:
        logger.info("Reducing load due to fault prediction")

        return True

    async def _restart_service(self, prediction: FaultPrediction) -> bool:
        logger.info("Restarting service due to fault prediction")

        return True

    async def _alert_admin(self, prediction: FaultPrediction) -> bool:
        logger.warning(
            "Admin alert: fault prediction",
            prediction_type=prediction.prediction_type,
            probability=prediction.probability,
            recommended_action=prediction.recommended_action,
        )

        return True

    async def _scale_out(self, prediction: FaultPrediction) -> bool:
        logger.info("Scaling out due to fault prediction")

        return True

    def _record_healing(
        self,
        action: str,
        prediction: FaultPrediction,
        success: bool,
    ) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "prediction_id": str(prediction.id),
            "prediction_type": prediction.prediction_type,
            "probability": prediction.probability,
            "success": success,
        }

        self._healing_history.append(record)

        if len(self._healing_history) > 1000:
            self._healing_history = self._healing_history[-1000:]

    def get_healing_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._healing_history[-limit:]
