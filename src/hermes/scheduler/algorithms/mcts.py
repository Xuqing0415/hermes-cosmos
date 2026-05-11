"""
Hermes MCTS调度器 - 基于numpy加速的蒙特卡洛树搜索
"""

import numpy as np
import asyncio
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from uuid import uuid4

@dataclass
class ClusterState:
    """集群状态"""
    gpu_count: int
    available_gpus: int
    regions: Dict[str, int]  # region -> available_gpus
    carbon_intensity: Dict[str, float]  # region -> carbon intensity
    network_latency: Dict[Tuple[str, str], float]  # (from, to) -> latency ms

@dataclass
class JobRequest:
    """作业请求"""
    job_id: str
    gpu_count: int
    gpu_type: str = "nvidia-h100"
    memory_gb: int = 64
    region_preferences: List[str] = None
    carbon_aware: bool = True
    priority: int = 1  # 1=low, 2=normal, 3=high, 4=critical

@dataclass
class PlacementDecision:
    """放置决策"""
    region: str
    gpu_count: int
    score: float
    carbon_cost: float
    latency_ms: float

class MCTSNode:
    """MCTS节点"""
    
    def __init__(self, state: ClusterState, job_request: JobRequest):
        self.state = state
        self.job_request = job_request
        self.visits = 0
        self.value = 0.0
        self.children: Dict[str, 'MCTSNode'] = {}
        self.parent: Optional['MCTSNode'] = None
        self.action: Optional[str] = None  # 到达此节点的动作（选择的region）
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def ucb1(self, total_visits: int, exploration_weight: float = 1.414) -> float:
        """UCB1公式"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(total_visits) / self.visits)
        return exploitation + exploration
    
    def expand(self):
        """扩展节点"""
        # 获取可用region
        available_regions = [
            region for region, available in self.state.regions.items()
            if available >= self.job_request.gpu_count
        ]
        
        if self.job_request.region_preferences:
            # 优先选择偏好region
            available_regions = [
                r for r in self.job_request.region_preferences
                if r in available_regions
            ]
        
        for region in available_regions:
            # 创建新的状态（模拟分配）
            new_regions = self.state.regions.copy()
            new_regions[region] -= self.job_request.gpu_count
            
            new_state = ClusterState(
                gpu_count=self.state.gpu_count,
                available_gpus=self.state.available_gpus - self.job_request.gpu_count,
                regions=new_regions,
                carbon_intensity=self.state.carbon_intensity,
                network_latency=self.state.network_latency
            )
            
            child = MCTSNode(new_state, self.job_request)
            child.parent = self
            child.action = region
            self.children[region] = child

def evaluate(state: ClusterState, job_request: JobRequest) -> float:
    """评估函数 - 计算放置质量"""
    # 简单评估：优先选择碳强度低、延迟小的region
    score = 0.0
    
    # 碳感知权重
    if job_request.carbon_aware:
        avg_carbon = sum(state.carbon_intensity.values()) / len(state.carbon_intensity)
        score += (1 - avg_carbon / 100) * 0.4
    
    # 资源利用率权重
    utilization = (state.gpu_count - state.available_gpus) / state.gpu_count
    score += min(utilization, 0.85) * 0.3
    
    # 可用资源权重
    score += (state.available_gpus / state.gpu_count) * 0.3
    
    return score

def mcts_search(job_request: JobRequest, cluster_state: ClusterState, 
                iterations: int = 1000) -> PlacementDecision:
    """执行MCTS搜索"""
    root = MCTSNode(cluster_state, job_request)
    best_decision = None
    best_score = float('-inf')
    
    for _ in range(iterations):
        # 1. 选择
        node = root
        while not node.is_leaf:
            total_visits = sum(child.visits for child in node.children.values())
            node = max(node.children.values(), key=lambda c: c.ucb1(total_visits))
        
        # 2. 扩展
        if node.visits > 0:
            node.expand()
            if node.children:
                node = next(iter(node.children.values()))
        
        # 3. 模拟
        reward = evaluate(node.state, job_request)
        
        # 4. 回溯
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    # 选择访问次数最多的子节点
    if root.children:
        best_region = max(root.children.keys(), 
                         key=lambda r: root.children[r].visits)
        child = root.children[best_region]
        
        best_decision = PlacementDecision(
            region=best_region,
            gpu_count=job_request.gpu_count,
            score=child.value / child.visits,
            carbon_cost=cluster_state.carbon_intensity.get(best_region, 0),
            latency_ms=0
        )
    
    return best_decision

async def global_schedule(job_request: JobRequest, cluster_state: ClusterState) -> PlacementDecision:
    """异步调度函数"""
    loop = asyncio.get_running_loop()
    
    with ProcessPoolExecutor() as pool:
        decision = await loop.run_in_executor(
            pool,
            mcts_search,
            job_request,
            cluster_state,
            1000  # iterations
        )
    
    return decision

# 示例用法
async def main():
    # 创建集群状态
    cluster = ClusterState(
        gpu_count=1000,
        available_gpus=720,
        regions={
            "us-east": 200,
            "eu-west": 180,
            "asia-east": 150,
            "eu-central": 190
        },
        carbon_intensity={
            "us-east": 45,
            "eu-west": 25,
            "asia-east": 60,
            "eu-central": 30
        },
        network_latency={}
    )
    
    # 创建作业请求
    job = JobRequest(
        job_id=str(uuid4()),
        gpu_count=8,
        region_preferences=["us-east", "eu-west"],
        carbon_aware=True,
        priority=3
    )
    
    # 执行调度
    decision = await global_schedule(job, cluster)
    
    print(f"调度决策: {decision.region}")
    print(f"分数: {decision.score:.4f}")
    print(f"碳成本: {decision.carbon_cost}")

if __name__ == "__main__":
    asyncio.run(main())
