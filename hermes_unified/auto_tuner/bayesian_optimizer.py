"""
Bayesian Optimizer for Hyperparameter Tuning
贝叶斯超参数优化器
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from typing import List, Callable, Dict, Tuple
import random

class BayesianOptimizer:
    """贝叶斯优化器，使用高斯过程和EI获取函数"""
    
    def __init__(
        self,
        objective_func: Callable,
        search_space: List[Tuple],
        n_calls: int = 20,
        acq_func: str = 'EI'
    ):
        """
        Args:
            objective_func: 目标函数，输入超参数，输出损失值
            search_space: 搜索空间，格式为 [(name, type, low, high), ...]
                type: 'real' 或 'integer'
            n_calls: 总调用次数
            acq_func: 获取函数类型 ('EI', 'PI', 'UCB')
        """
        self.objective_func = objective_func
        self.search_space = search_space
        self.n_calls = n_calls
        self.acq_func = acq_func
        
        # 历史数据
        self.X = []
        self.y = []
        
        # 高斯过程模型
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
            random_state=42
        )
        
        print(f"🚀 BayesianOptimizer initialized with {n_calls} calls")
    
    def _normalize_params(self, params: List[float]) -> List[float]:
        """归一化参数到 [0, 1] 范围"""
        normalized = []
        for i, (name, p_type, low, high) in enumerate(self.search_space):
            if p_type == 'integer':
                normalized.append((params[i] - low) / (high - low))
            else:
                normalized.append((params[i] - low) / (high - low))
        return normalized
    
    def _denormalize_params(self, normalized: List[float]) -> List[float]:
        """反归一化参数"""
        params = []
        for i, (name, p_type, low, high) in enumerate(self.search_space):
            val = normalized[i] * (high - low) + low
            if p_type == 'integer':
                params.append(int(round(val)))
            else:
                params.append(val)
        return params
    
    def _acquisition_ei(self, x: np.ndarray, xi: float = 0.01) -> float:
        """期望改进获取函数"""
        x = np.atleast_2d(x)
        mu, sigma = self.gp.predict(x, return_std=True)
        
        if sigma == 0:
            return 0.0
        
        best_y = min(self.y)
        gamma = (best_y - mu - xi) / sigma
        
        ei = (best_y - mu - xi) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
        return -ei[0]  # 最小化问题，取负值
    
    def _acquisition_pi(self, x: np.ndarray, xi: float = 0.01) -> float:
        """概率改进获取函数"""
        x = np.atleast_2d(x)
        mu, sigma = self.gp.predict(x, return_std=True)
        
        if sigma == 0:
            return 0.0
        
        best_y = min(self.y)
        gamma = (best_y - mu - xi) / sigma
        
        pi = norm.cdf(gamma)
        return -pi[0]
    
    def _acquisition_ucb(self, x: np.ndarray, kappa: float = 2.576) -> float:
        """上置信边界获取函数"""
        x = np.atleast_2d(x)
        mu, sigma = self.gp.predict(x, return_std=True)
        
        ucb = mu + kappa * sigma
        return -ucb[0]
    
    def _optimize_acquisition(self) -> List[float]:
        """优化获取函数，找到下一个评估点"""
        best_x = None
        best_val = float('inf')
        
        # 随机采样多个点，选择获取函数值最小的
        for _ in range(100):
            x = [random.random() for _ in range(len(self.search_space))]
            
            if self.acq_func == 'EI':
                val = self._acquisition_ei(np.array(x))
            elif self.acq_func == 'PI':
                val = self._acquisition_pi(np.array(x))
            else:
                val = self._acquisition_ucb(np.array(x))
            
            if val < best_val:
                best_val = val
                best_x = x
        
        return best_x
    
    def suggest(self) -> Dict:
        """建议下一个要评估的超参数"""
        if len(self.X) < 3:
            # 初始阶段随机采样
            normalized = [random.random() for _ in range(len(self.search_space))]
        else:
            # 训练高斯过程并优化获取函数
            self.gp.fit(np.array(self.X), np.array(self.y))
            normalized = self._optimize_acquisition()
        
        params = self._denormalize_params(normalized)
        
        # 转换为字典格式
        result = {}
        for i, (name, p_type, low, high) in enumerate(self.search_space):
            result[name] = params[i]
        
        return result
    
    def tell(self, params: Dict, loss: float):
        """记录评估结果"""
        # 转换为列表并归一化
        normalized = []
        for name, p_type, low, high in self.search_space:
            normalized.append((params[name] - low) / (high - low))
        
        self.X.append(normalized)
        self.y.append(loss)
        
        print(f"📊 Trial {len(self.y)}: {params} -> loss={loss:.4f}")
    
    def run(self) -> Tuple[Dict, float]:
        """运行完整优化过程"""
        print("\n🔬 Starting Bayesian Optimization...")
        
        for i in range(self.n_calls):
            # 获取建议的超参数
            params = self.suggest()
            
            # 评估目标函数
            loss = self.objective_func(params)
            
            # 记录结果
            self.tell(params, loss)
        
        # 返回最佳结果
        best_idx = np.argmin(self.y)
        best_params = self._denormalize_params(self.X[best_idx])
        best_loss = self.y[best_idx]
        
        result = {}
        for i, (name, p_type, low, high) in enumerate(self.search_space):
            result[name] = best_params[i]
        
        print(f"\n🏆 Best params: {result}")
        print(f"🏆 Best loss: {best_loss:.4f}")
        
        return result, best_loss
    
    def get_history(self) -> Tuple[List, List]:
        """获取历史数据"""
        return self.X, self.y