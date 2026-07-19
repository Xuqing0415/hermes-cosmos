"""
Job queue management
"""

import asyncio
import json
from datetime import datetime
from typing import Optional
from uuid import UUID

import redis.asyncio as redis
import structlog

from hermes.core.models import Job, JobStatus, JobPriority

logger = structlog.get_logger()


class JobQueue:
    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None

    async def connect(self) -> None:
        logger.info("Connecting to Redis", url=self.redis_url)
        self._redis = redis.from_url(self.redis_url)
        await self._redis.ping()
        logger.info("Connected to Redis")

    async def disconnect(self) -> None:
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        logger.info("Disconnected from Redis")

    async def enqueue(self, job: Job) -> None:
        assert self._redis is not None

        priority_score = self._get_priority_score(job.priority)
        job_data = job.model_dump_json()

        await self._redis.zadd(
            "hermes:jobs:pending",
            {job_data: priority_score},
        )

        await self._redis.hset(
            f"hermes:jobs:{job.id}",
            mapping={"data": job_data, "status": job.status.value},
        )

        await self._redis.publish("hermes:events", json.dumps({
            "type": "job_enqueued",
            "job_id": str(job.id),
            "timestamp": datetime.utcnow().isoformat(),
        }))

        logger.info("Job enqueued", job_id=str(job.id), priority=job.priority.value)

    async def dequeue(self) -> Optional[Job]:
        assert self._redis is not None

        results = await self._redis.zpopmax("hermes:jobs:pending")

        if not results:
            return None

        job_data = results[0][0]
        job = Job.model_validate_json(job_data)

        await self._redis.hset(
            f"hermes:jobs:{job.id}",
            "status",
            JobStatus.QUEUED.value,
        )

        logger.info("Job dequeued", job_id=str(job.id))
        return job

    async def get_pending_jobs(self, limit: int = 100) -> list[Job]:
        assert self._redis is not None

        results = await self._redis.zrevrange(
            "hermes:jobs:pending",
            0,
            limit - 1,
            withscores=False,
        )

        jobs = []
        for job_data in results:
            job = Job.model_validate_json(job_data)
            jobs.append(job)

        return jobs

    async def get_job(self, job_id: UUID) -> Optional[Job]:
        assert self._redis is not None

        job_data = await self._redis.hget(f"hermes:jobs:{job_id}", "data")
        if not job_data:
            return None

        return Job.model_validate_json(job_data)

    async def update_status(self, job_id: UUID, status: JobStatus) -> None:
        assert self._redis is not None

        await self._redis.hset(
            f"hermes:jobs:{job_id}",
            "status",
            status.value,
        )

        await self._redis.publish("hermes:events", json.dumps({
            "type": "job_status_changed",
            "job_id": str(job_id),
            "status": status.value,
            "timestamp": datetime.utcnow().isoformat(),
        }))

        logger.info("Job status updated", job_id=str(job_id), status=status.value)

    async def mark_scheduled(self, job_id: UUID, placement: dict) -> None:
        assert self._redis is not None

        await self._redis.hset(
            f"hermes:jobs:{job_id}",
            mapping={
                "status": JobStatus.SCHEDULING.value,
                "placement": json.dumps(placement),
            },
        )

        await self._redis.zrem("hermes:jobs:pending", await self._redis.hget(f"hermes:jobs:{job_id}", "data"))

    async def get_queue_size(self) -> int:
        assert self._redis is not None
        return await self._redis.zcard("hermes:jobs:pending")

    def _get_priority_score(self, priority: JobPriority) -> float:
        scores = {
            JobPriority.CRITICAL: 100.0,
            JobPriority.HIGH: 75.0,
            JobPriority.NORMAL: 50.0,
            JobPriority.LOW: 25.0,
        }
        return scores.get(priority, 50.0)
