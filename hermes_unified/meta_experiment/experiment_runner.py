"""
Experiment Runner Module

Implements asynchronous experiment execution with multi-processing.
"""

import multiprocessing
from multiprocessing import Queue, Process
import time
import sqlite3
import os
from typing import Dict, List, Any, Optional
import numpy as np

from .experiment_config import ExperimentConfig


class ExperimentResult:
    """Represents results from a single experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.final_accuracy = 0.0
        self.final_loss = 0.0
        self.accuracy_history = []
        self.loss_history = []
        self.defense_effectiveness = 0.0
        self.convergence_round = -1
        self.runtime = 0.0
        self.success = False
        self.error_message = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage."""
        return {
            'config_hash': self.config.get_hash(),
            'attack_type': self.config.attack_type,
            'attack_intensity': self.config.attack_intensity,
            'malicious_ratio': self.config.malicious_ratio,
            'defense_type': self.config.defense_type,
            'num_clients': self.config.num_clients,
            'non_iid_alpha': self.config.non_iid_alpha,
            'num_rounds': self.config.num_rounds,
            'final_accuracy': self.final_accuracy,
            'final_loss': self.final_loss,
            'accuracy_history': str(self.accuracy_history),
            'loss_history': str(self.loss_history),
            'defense_effectiveness': self.defense_effectiveness,
            'convergence_round': self.convergence_round,
            'runtime': self.runtime,
            'success': self.success,
            'error_message': self.error_message
        }


class ResultDatabase:
    """Manages experiment results storage in SQLite."""
    
    def __init__(self, db_path: str = 'results.db'):
        """
        Initialize result database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_hash TEXT UNIQUE,
                attack_type TEXT,
                attack_intensity REAL,
                malicious_ratio REAL,
                defense_type TEXT,
                num_clients INTEGER,
                non_iid_alpha REAL,
                num_rounds INTEGER,
                final_accuracy REAL,
                final_loss REAL,
                accuracy_history TEXT,
                loss_history TEXT,
                defense_effectiveness REAL,
                convergence_round INTEGER,
                runtime REAL,
                success BOOLEAN,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_config_hash ON experiments(config_hash)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_defense_type ON experiments(defense_type)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_malicious_ratio ON experiments(malicious_ratio)
        ''')
        
        conn.commit()
        conn.close()
    
    def save_result(self, result: ExperimentResult):
        """Save experiment result to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data = result.to_dict()
        keys = list(data.keys())
        values = list(data.values())
        
        placeholders = ','.join(['?' for _ in keys])
        key_str = ','.join(keys)
        
        try:
            cursor.execute(f'''
                INSERT OR REPLACE INTO experiments ({key_str})
                VALUES ({placeholders})
            ''', values)
            conn.commit()
        except Exception as e:
            print(f"Error saving result: {e}")
        finally:
            conn.close()
    
    def get_explored_hashes(self) -> List[str]:
        """Get hashes of all explored configurations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT config_hash FROM experiments')
        hashes = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return hashes
    
    def get_all_results(self) -> List[Dict]:
        """Get all experiment results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM experiments')
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_results_by_defense(self, defense_type: str) -> List[Dict]:
        """Get results for a specific defense type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM experiments WHERE defense_type = ?', (defense_type,))
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_average_accuracy(self, filter_params: Dict = None) -> float:
        """Get average accuracy for filtered experiments."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT AVG(final_accuracy) FROM experiments WHERE success = 1'
        params = []
        
        if filter_params:
            conditions = []
            for key, value in filter_params.items():
                conditions.append(f"{key} = ?")
                params.append(value)
            
            if conditions:
                query += ' AND ' + ' AND '.join(conditions)
        
        cursor.execute(query, params)
        result = cursor.fetchone()[0]
        
        conn.close()
        return result if result else 0.0


def run_single_experiment(config: ExperimentConfig, queue: Queue):
    """Run a single experiment and put result in queue."""
    result = ExperimentResult(config)
    
    try:
        start_time = time.time()
        
        import sys
        import os
        file_path = os.path.abspath(__file__)
        parent_dir = os.path.dirname(file_path)
        grandparent_dir = os.path.dirname(parent_dir)
        great_grandparent_dir = os.path.dirname(grandparent_dir)
        sys.path.insert(0, great_grandparent_dir)
        
        from hermes_unified.federated.federated_simulator import FederatedSimulator
        
        attack_config = {
            'type': config.attack_type,
            'intensity': config.attack_intensity,
            'malicious_ratio': config.malicious_ratio,
            'start_round': 0
        }
        
        sim = FederatedSimulator(
            num_clients=config.num_clients,
            attack_config=attack_config,
            defense_type=config.defense_type
        )
        
        sim.run_federated_training(num_rounds=config.num_rounds)
        
        result.final_accuracy = sim.final_accuracy
        result.final_loss = sim.final_loss
        result.accuracy_history = sim.accuracy_history
        result.loss_history = sim.loss_history
        result.defense_effectiveness = sim.defense_effectiveness if hasattr(sim, 'defense_effectiveness') else 0.0
        result.convergence_round = sim.convergence_round if hasattr(sim, 'convergence_round') else -1
        result.runtime = time.time() - start_time
        result.success = True
        
    except Exception as e:
        result.success = False
        result.error_message = str(e)
        result.runtime = time.time() - start_time
    
    queue.put(result)


class ExperimentRunner:
    """Manages parallel experiment execution."""
    
    def __init__(self, num_workers: Optional[int] = None, db_path: str = 'results.db'):
        """
        Initialize experiment runner.
        
        Args:
            num_workers: Number of parallel workers (default: CPU count)
            db_path: Path to result database
        """
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.db = ResultDatabase(db_path)
        self.results = []
    
    def run_parallel(self, configs: List[ExperimentConfig], verbose: bool = True):
        """Run experiments in parallel."""
        num_configs = len(configs)
        completed = 0
        
        if verbose:
            print(f"Running {num_configs} experiments with {self.num_workers} workers...")
        
        while configs:
            batch_size = min(self.num_workers, len(configs))
            batch = configs[:batch_size]
            configs = configs[batch_size:]
            
            queue = Queue()
            processes = []
            
            for config in batch:
                p = Process(target=run_single_experiment, args=(config, queue))
                processes.append(p)
                p.start()
            
            for p in processes:
                p.join()
            
            while not queue.empty():
                result = queue.get()
                self.results.append(result)
                self.db.save_result(result)
                completed += 1
                
                if verbose:
                    status = "SUCCESS" if result.success else "FAILED"
                    print(f"[{completed}/{num_configs}] {result.config.get_hash()} - {status}")
    
    def run_sequential(self, configs: List[ExperimentConfig], verbose: bool = True):
        """Run experiments sequentially."""
        num_configs = len(configs)
        completed = 0
        
        if verbose:
            print(f"Running {num_configs} experiments sequentially...")
        
        for config in configs:
            result = ExperimentResult(config)
            
            try:
                start_time = time.time()
                
                from federated.federated_simulator import FederatedSimulator
                
                attack_config = {
                    'type': config.attack_type,
                    'intensity': config.attack_intensity,
                    'malicious_ratio': config.malicious_ratio,
                    'start_round': 0
                }
                
                sim = FederatedSimulator(
                    num_clients=config.num_clients,
                    attack_config=attack_config,
                    defense_type=config.defense_type
                )
                
                sim.run_federated_training(num_rounds=config.num_rounds)
                
                result.final_accuracy = sim.final_accuracy
                result.final_loss = sim.final_loss
                result.accuracy_history = sim.accuracy_history
                result.loss_history = sim.loss_history
                result.defense_effectiveness = sim.defense_effectiveness if hasattr(sim, 'defense_effectiveness') else 0.0
                result.convergence_round = sim.convergence_round if hasattr(sim, 'convergence_round') else -1
                result.runtime = time.time() - start_time
                result.success = True
                
            except Exception as e:
                result.success = False
                result.error_message = str(e)
                result.runtime = time.time() - start_time
            
            self.results.append(result)
            self.db.save_result(result)
            completed += 1
            
            if verbose:
                status = "SUCCESS" if result.success else "FAILED"
                print(f"[{completed}/{num_configs}] {result.config.get_hash()} - {status}")
    
    def get_results(self) -> List[ExperimentResult]:
        """Get all collected results."""
        return self.results


# Example usage
if __name__ == "__main__":
    from experiment_config import ParameterSpace
    
    ps = ParameterSpace()
    samples = ps.sample_random(5)
    
    runner = ExperimentRunner(num_workers=2, db_path='test_results.db')
    runner.run_parallel(samples)
    
    print("\nExperiment completed!")
    print(f"Total results: {len(runner.results)}")
    print(f"Successful: {sum(1 for r in runner.results if r.success)}")
    
    os.remove('test_results.db')
