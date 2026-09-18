"""
Attack Evolution Module

Uses genetic algorithm to automatically search for optimal attack parameters.
"""

import random
import copy
import numpy as np


class AttackEvolution:
    """使用遗传算法自动搜索最佳攻击参数"""
    
    def __init__(self, simulator_class, base_config, population_size=8, generations=5):
        self.simulator_class = simulator_class
        self.base_config = base_config
        self.pop_size = population_size
        self.generations = generations
        self.population = [self._random_individual() for _ in range(population_size)]
    
    def _random_individual(self):
        """生成随机个体（攻击配置）"""
        return {
            'attack_type': random.choice(['label_flip', 'gradient_scale', 'backdoor']),
            'intensity': random.uniform(1.5, 10.0),      # 梯度缩放倍数
            'mal_ratio': random.uniform(0.1, 0.5),       # 恶意客户端比例
            'start_round': random.randint(5, 30),        # 攻击开始轮次
            'trigger_pos': (random.randint(0, 31), random.randint(0, 31)) if random.random() > 0.5 else None  # 后门触发位置
        }
    
    def _fitness(self, individual):
        """计算适应度：攻击成功度 = 1 - 最终准确率"""
        config = copy.deepcopy(self.base_config)
        config['attack_config'] = individual
        
        try:
            sim = self.simulator_class(**config)
            results = sim.run_federated_training(
                num_rounds=30,
                clients_per_round=5,
                local_epochs=2
            )
            final_acc = results['accuracy_history'][-1] if results['accuracy_history'] else 1.0
            return 1 - final_acc  # 适应度越高表示攻击越成功
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            return 0.0
    
    def _crossover(self, p1, p2):
        """交叉操作"""
        child = {}
        for key in p1:
            if random.random() > 0.5:
                child[key] = p1[key]
            else:
                child[key] = p2[key]
        return child
    
    def _mutate(self, ind):
        """变异操作"""
        mutated = copy.deepcopy(ind)
        
        if random.random() < 0.3:
            mutated['intensity'] += random.uniform(-1.0, 1.0)
            mutated['intensity'] = max(1.0, min(20.0, mutated['intensity']))
        
        if random.random() < 0.3:
            mutated['mal_ratio'] += random.uniform(-0.1, 0.1)
            mutated['mal_ratio'] = max(0.0, min(0.5, mutated['mal_ratio']))
        
        if random.random() < 0.2:
            mutated['attack_type'] = random.choice(['label_flip', 'gradient_scale', 'backdoor'])
        
        if random.random() < 0.1:
            mutated['start_round'] = random.randint(5, 30)
        
        return mutated
    
    def evolve(self):
        """执行遗传算法进化"""
        print(f"🧬 Starting attack evolution with population size {self.pop_size}")
        
        for gen in range(self.generations):
            fitnesses = [self._fitness(ind) for ind in self.population]
            
            # 锦标赛选择
            new_population = []
            for _ in range(self.pop_size):
                i1 = random.choices(range(self.pop_size), weights=fitnesses, k=1)[0]
                i2 = random.choices(range(self.pop_size), weights=fitnesses, k=1)[0]
                child = self._crossover(self.population[i1], self.population[i2])
                child = self._mutate(child)
                new_population.append(child)
            
            self.population = new_population
            
            # 找到当前代最佳个体
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            best_individual = self.population[best_idx]
            
            print(f"\n🌟 Generation {gen + 1}/{self.generations}")
            print(f"   Best Fitness: {best_fitness:.4f} (higher = more effective attack)")
            print(f"   Best Attack Config:")
            print(f"     - Type: {best_individual['attack_type']}")
            print(f"     - Intensity: {best_individual['intensity']:.2f}")
            print(f"     - Malicious Ratio: {best_individual['mal_ratio']:.2f}")
            print(f"     - Start Round: {best_individual['start_round']}")
        
        # 返回最佳个体
        final_fitnesses = [self._fitness(ind) for ind in self.population]
        best_idx = np.argmax(final_fitnesses)
        
        print(f"\n🏆 Evolution Complete!")
        print(f"Best Attack Fitness: {final_fitnesses[best_idx]:.4f}")
        print(f"Best Attack Parameters: {self.population[best_idx]}")
        
        return self.population[best_idx]


def test_evolution():
    """测试攻击进化模块"""
    from .federated_simulator import FederatedSimulator
    
    base_config = {
        'num_clients': 20,
        'model_shape': (10, 784),
        'num_features': 784,
        'num_classes': 10
    }
    
    evolution = AttackEvolution(FederatedSimulator, base_config, population_size=4, generations=3)
    best_attack = evolution.evolve()
    
    print("\n🎉 Found optimal attack configuration:")
    print(best_attack)


if __name__ == "__main__":
    test_evolution()
