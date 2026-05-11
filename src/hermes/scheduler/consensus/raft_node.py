"""
Hermes Raft节点 - 基于PySyncObj实现分布式一致性
"""

import asyncio
from pysyncobj import SyncObj, SyncObjConf, replicated
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json

@dataclass
class JobState:
    """作业状态"""
    job_id: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED
    region: str
    gpu_count: int
    checkpoints: List[str]  # checkpoint IDs

@dataclass
class ClusterResource:
    """集群资源状态"""
    region: str
    total_gpus: int
    available_gpus: int
    pods_running: int

class RaftSchedulerState(SyncObj):
    """Raft调度器状态"""
    
    def __init__(self, selfNodeAddr: str, otherNodesAddrs: List[str]):
        cfg = SyncObjConf(
            appendEntriesTimeout=100,
            electionTimeout=300,
            replicaTimeout=5000,
            loggingLevel='INFO'
        )
        super(RaftSchedulerState, self).__init__(
            selfNodeAddr,
            otherNodesAddrs,
            cfg
        )
        self._jobs: Dict[str, JobState] = {}
        self._resources: Dict[str, ClusterResource] = {}
        self._leader_node: Optional[str] = None
    
    @replicated
    def add_job(self, job_id: str, status: str, region: str, gpu_count: int) -> bool:
        """添加作业"""
        if job_id in self._jobs:
            return False
        self._jobs[job_id] = JobState(
            job_id=job_id,
            status=status,
            region=region,
            gpu_count=gpu_count,
            checkpoints=[]
        )
        return True
    
    @replicated
    def update_job_status(self, job_id: str, status: str) -> bool:
        """更新作业状态"""
        if job_id not in self._jobs:
            return False
        self._jobs[job_id].status = status
        return True
    
    @replicated
    def add_checkpoint(self, job_id: str, checkpoint_id: str) -> bool:
        """添加Checkpoint"""
        if job_id not in self._jobs:
            return False
        self._jobs[job_id].checkpoints.append(checkpoint_id)
        return True
    
    @replicated
    def update_resources(self, region: str, available_gpus: int, pods_running: int) -> bool:
        """更新资源状态"""
        if region not in self._resources:
            self._resources[region] = ClusterResource(
                region=region,
                total_gpus=available_gpus + pods_running,
                available_gpus=available_gpus,
                pods_running=pods_running
            )
        else:
            self._resources[region].available_gpus = available_gpus
            self._resources[region].pods_running = pods_running
        return True
    
    @replicated
    def allocate_gpus(self, region: str, gpu_count: int) -> bool:
        """分配GPU资源"""
        if region not in self._resources:
            return False
        if self._resources[region].available_gpus < gpu_count:
            return False
        self._resources[region].available_gpus -= gpu_count
        self._resources[region].pods_running += 1
        return True
    
    @replicated
    def release_gpus(self, region: str, gpu_count: int) -> bool:
        """释放GPU资源"""
        if region not in self._resources:
            return False
        self._resources[region].available_gpus += gpu_count
        self._resources[region].pods_running -= 1
        return True
    
    def get_job(self, job_id: str) -> Optional[JobState]:
        """获取作业状态"""
        return self._jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, JobState]:
        """获取所有作业"""
        return self._jobs
    
    def get_resources(self) -> Dict[str, ClusterResource]:
        """获取资源状态"""
        return self._resources
    
    def is_leader(self) -> bool:
        """判断是否为Leader"""
        return self._getLeader() == self._selfAddr
    
    def get_leader(self) -> Optional[str]:
        """获取Leader地址"""
        return self._getLeader()

async def create_raft_cluster(node_addresses: List[str], node_index: int) -> RaftSchedulerState:
    """创建Raft集群节点"""
    self_addr = node_addresses[node_index]
    other_addrs = [addr for i, addr in enumerate(node_addresses) if i != node_index]
    
    raft_node = RaftSchedulerState(self_addr, other_addrs)
    
    # 等待集群稳定
    while not raft_node._isReady():
        await asyncio.sleep(0.1)
    
    return raft_node

# 示例用法
async def main():
    # 启动3节点集群
    nodes = [
        '127.0.0.1:10001',
        '127.0.0.1:10002',
        '127.0.0.1:10003'
    ]
    
    # 启动第一个节点
    node = await create_raft_cluster(nodes, 0)
    
    if node.is_leader():
        print("当前节点是Leader")
        
        # 添加测试作业
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            node.add_job,
            "test-job-001",
            "PENDING",
            "us-east",
            8
        )
        print(f"添加作业结果: {result}")
        
        # 更新资源
        await asyncio.get_event_loop().run_in_executor(
            None,
            node.update_resources,
            "us-east",
            200,
            10
        )
        
        # 获取状态
        jobs = await asyncio.get_event_loop().run_in_executor(
            None,
            node.get_all_jobs
        )
        print(f"作业列表: {jobs}")
        
        resources = await asyncio.get_event_loop().run_in_executor(
            None,
            node.get_resources
        )
        print(f"资源状态: {resources}")
    
    # 保持运行
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
