"""
Auto Insights Module

Automatically generates insights and reports from experiment results.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
import os
import re
import random


class ExperimentAnalyzer:
    """Analyzes experiment results and generates insights."""
    
    def __init__(self, db_path: str = 'results.db'):
        """
        Initialize analyzer.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load data from database into DataFrame."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql('SELECT * FROM experiments WHERE success = 1', conn)
        conn.close()
        
        if not self.df.empty:
            self.df['accuracy_history'] = self.df['accuracy_history'].apply(self._parse_list)
            self.df['loss_history'] = self.df['loss_history'].apply(self._parse_list)
    
    def _parse_list(self, s: str) -> List[float]:
        """Parse string representation of list."""
        try:
            s = s.strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            if not s:
                return []
            return [float(x.strip()) for x in s.split(',')]
        except:
            return []
    
    def analyze_defense_effectiveness(self) -> Dict:
        """Analyze effectiveness of different defense mechanisms."""
        if self.df.empty:
            return {'error': 'No data available'}
        
        results = {}
        
        defense_groups = self.df.groupby('defense_type')
        
        for defense, group in defense_groups:
            avg_acc = group['final_accuracy'].mean()
            std_acc = group['final_accuracy'].std()
            max_acc = group['final_accuracy'].max()
            min_acc = group['final_accuracy'].min()
            
            results[defense] = {
                'avg_accuracy': avg_acc,
                'std_accuracy': std_acc,
                'max_accuracy': max_acc,
                'min_accuracy': min_acc,
                'count': len(group)
            }
        
        return results
    
    def analyze_malicious_ratio_impact(self) -> Dict:
        """Analyze how malicious ratio affects accuracy."""
        if self.df.empty:
            return {'error': 'No data available'}
        
        results = {}
        
        ratio_groups = self.df.groupby('malicious_ratio')
        
        for ratio, group in ratio_groups:
            avg_acc = group['final_accuracy'].mean()
            std_acc = group['final_accuracy'].std()
            
            results[ratio] = {
                'avg_accuracy': avg_acc,
                'std_accuracy': std_acc,
                'count': len(group)
            }
        
        return results
    
    def analyze_attack_types(self) -> Dict:
        """Analyze effectiveness of different attack types."""
        if self.df.empty:
            return {'error': 'No data available'}
        
        results = {}
        
        attack_groups = self.df.groupby('attack_type')
        
        for attack, group in attack_groups:
            avg_acc = group['final_accuracy'].mean()
            std_acc = group['final_accuracy'].std()
            
            results[attack] = {
                'avg_accuracy': avg_acc,
                'std_accuracy': std_acc,
                'count': len(group)
            }
        
        return results
    
    def correlation_analysis(self) -> Dict:
        """Perform correlation analysis between parameters and accuracy."""
        if self.df.empty:
            return {'error': 'No data available'}
        
        numeric_cols = ['attack_intensity', 'malicious_ratio', 'num_clients', 'non_iid_alpha', 'final_accuracy']
        subset = self.df[numeric_cols]
        
        corr_matrix = subset.corr()
        accuracy_corr = corr_matrix['final_accuracy'].drop('final_accuracy')
        
        return accuracy_corr.to_dict()
    
    def best_performing_configs(self, top_n: int = 10) -> List[Dict]:
        """Get top performing configurations."""
        if self.df.empty:
            return []
        
        sorted_df = self.df.sort_values('final_accuracy', ascending=False).head(top_n)
        
        results = []
        for _, row in sorted_df.iterrows():
            results.append({
                'config_hash': row['config_hash'],
                'attack_type': row['attack_type'],
                'attack_intensity': row['attack_intensity'],
                'malicious_ratio': row['malicious_ratio'],
                'defense_type': row['defense_type'],
                'num_clients': row['num_clients'],
                'non_iid_alpha': row['non_iid_alpha'],
                'final_accuracy': row['final_accuracy'],
                'runtime': row['runtime']
            })
        
        return results
    
    def generate_insights(self) -> List[str]:
        """Generate human-readable insights from analysis."""
        insights = []
        
        if self.df.empty:
            return ["No experiment data available."]
        
        # Defense effectiveness insight
        defense_effect = self.analyze_defense_effectiveness()
        if defense_effect:
            best_defense = max(defense_effect.keys(), key=lambda x: defense_effect[x]['avg_accuracy'])
            worst_defense = min(defense_effect.keys(), key=lambda x: defense_effect[x]['avg_accuracy'])
            
            best_acc = defense_effect[best_defense]['avg_accuracy']
            worst_acc = defense_effect[worst_defense]['avg_accuracy']
            
            insights.append(f"Best defense: {best_defense} (avg accuracy: {best_acc:.4f})")
            insights.append(f"Worst defense: {worst_defense} (avg accuracy: {worst_acc:.4f})")
            
            if best_defense != 'fedavg':
                insights.append(f"{best_defense} outperforms fedavg by {(best_acc - defense_effect.get('fedavg', {}).get('avg_accuracy', 0))*100:.1f}% on average")
        
        # Malicious ratio insight
        ratio_impact = self.analyze_malicious_ratio_impact()
        if ratio_impact:
            ratios = sorted(ratio_impact.keys())
            if len(ratios) >= 2:
                acc_drop = ratio_impact[ratios[0]]['avg_accuracy'] - ratio_impact[ratios[-1]]['avg_accuracy']
                insights.append(f"Increasing malicious ratio from {ratios[0]} to {ratios[-1]} reduces accuracy by {acc_drop*100:.1f} percentage points on average")
        
        # Correlation insight
        correlations = self.correlation_analysis()
        if correlations:
            strongest_positive = max(correlations.keys(), key=lambda x: correlations[x])
            strongest_negative = min(correlations.keys(), key=lambda x: correlations[x])
            
            insights.append(f"Strongest positive correlation with accuracy: {strongest_positive} (r={correlations[strongest_positive]:.2f})")
            insights.append(f"Strongest negative correlation with accuracy: {strongest_negative} (r={correlations[strongest_negative]:.2f})")
            
            if correlations.get('malicious_ratio', 0) < -0.5:
                insights.append("Malicious ratio has a strong negative impact on final accuracy")
        
        # Attack type insight
        attack_effect = self.analyze_attack_types()
        if attack_effect:
            most_damaging = min(attack_effect.keys(), key=lambda x: attack_effect[x]['avg_accuracy'])
            insights.append(f"Most damaging attack type: {most_damaging} (lowest avg accuracy)")
        
        # Sample size insight
        insights.append(f"Analysis based on {len(self.df)} successful experiments")
        
        return insights
    
    def generate_report(self, output_dir: str = 'reports') -> str:
        """Generate comprehensive markdown report."""
        os.makedirs(output_dir, exist_ok=True)
        
        report = "# Experiment Analysis Report\n\n"
        report += "## Overview\n\n"
        report += f"This report summarizes findings from {len(self.df)} experiments.\n\n"
        
        # Defense comparison
        report += "## Defense Effectiveness Comparison\n\n"
        report += "| Defense | Avg Accuracy | Std | Count |\n"
        report += "|---------|--------------|-----|-------|\n"
        
        defense_effect = self.analyze_defense_effectiveness()
        for defense, stats in sorted(defense_effect.items(), key=lambda x: -x[1]['avg_accuracy']):
            report += f"| {defense} | {stats['avg_accuracy']:.4f} | {stats['std_accuracy']:.4f} | {stats['count']} |\n"
        
        # Malicious ratio impact
        report += "\n## Malicious Ratio Impact\n\n"
        report += "| Ratio | Avg Accuracy | Std |\n"
        report += "|-------|--------------|-----|\n"
        
        ratio_impact = self.analyze_malicious_ratio_impact()
        for ratio, stats in sorted(ratio_impact.items()):
            report += f"| {ratio} | {stats['avg_accuracy']:.4f} | {stats['std_accuracy']:.4f} |\n"
        
        # Correlation analysis
        report += "\n## Correlation Analysis\n\n"
        report += "| Parameter | Correlation with Accuracy |\n"
        report += "|-----------|---------------------------|\n"
        
        correlations = self.correlation_analysis()
        for param, corr in sorted(correlations.items(), key=lambda x: -abs(x[1])):
            report += f"| {param} | {corr:.4f} |\n"
        
        # Key insights
        report += "\n## Key Insights\n\n"
        insights = self.generate_insights()
        for i, insight in enumerate(insights, 1):
            report += f"{i}. {insight}\n"
        
        # Best configurations
        report += "\n## Top Performing Configurations\n\n"
        report += "| Rank | Defense | Attack | Mal Ratio | Accuracy |\n"
        report += "|------|---------|--------|-----------|----------|\n"
        
        best_configs = self.best_performing_configs(5)
        for i, config in enumerate(best_configs, 1):
            report += f"| {i} | {config['defense_type']} | {config['attack_type']} | {config['malicious_ratio']} | {config['final_accuracy']:.4f} |\n"
        
        report_path = os.path.join(output_dir, 'experiment_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path


class MultiArmedBanditSampler:
    """Multi-armed bandit sampler for incremental experiment selection."""
    
    def __init__(self, parameter_space, db_path: str = 'results.db', exploration_rate: float = 0.2):
        """
        Initialize bandit sampler.
        
        Args:
            parameter_space: ParameterSpace instance
            db_path: Path to results database
            exploration_rate: Probability of exploring new configurations
        """
        self.parameter_space = parameter_space
        self.db_path = db_path
        self.exploration_rate = exploration_rate
        self.arm_rewards = {}
    
    def _calculate_reward(self, result: Dict) -> float:
        """Calculate reward for an experiment."""
        accuracy = result.get('final_accuracy', 0.0)
        malicious_ratio = result.get('malicious_ratio', 0.0)
        
        return accuracy * (1 + malicious_ratio)
    
    def _update_rewards(self):
        """Update reward estimates from database."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT config_hash, final_accuracy, malicious_ratio FROM experiments WHERE success = 1')
        results = cursor.fetchall()
        
        for config_hash, accuracy, mal_ratio in results:
            reward = accuracy * (1 + mal_ratio)
            self.arm_rewards[config_hash] = reward
        
        conn.close()
    
    def select_next_experiments(self, num_samples: int = 5) -> List:
        """Select next experiments using epsilon-greedy strategy."""
        self._update_rewards()
        
        explored_hashes = list(self.arm_rewards.keys())
        unexplored = self.parameter_space.sample_unexplored(explored_hashes, num_samples * 10)
        
        selected = []
        
        for _ in range(num_samples):
            if random.random() < self.exploration_rate or not self.arm_rewards:
                # Explore: select random unexplored
                if unexplored:
                    config = random.choice(unexplored)
                    unexplored.remove(config)
                    selected.append(config)
            else:
                # Exploit: select based on reward
                if unexplored:
                    best_config = None
                    best_score = float('-inf')
                    
                    for config in unexplored[:min(20, len(unexplored))]:
                        score = self._estimate_reward(config)
                        if score > best_score:
                            best_score = score
                            best_config = config
                    
                    if best_config:
                        unexplored.remove(best_config)
                        selected.append(best_config)
        
        return selected
    
    def _estimate_reward(self, config) -> float:
        """Estimate reward for a configuration based on similar past results."""
        reward = 0.5
        
        if self.arm_rewards:
            avg_reward = sum(self.arm_rewards.values()) / len(self.arm_rewards)
            reward = avg_reward
        
        return reward


# Example usage
if __name__ == "__main__":
    import random
    
    # Create analyzer
    analyzer = ExperimentAnalyzer('test_results.db')
    
    if not analyzer.df.empty:
        # Generate insights
        insights = analyzer.generate_insights()
        print("Generated Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        # Generate report
        report_path = analyzer.generate_report()
        print(f"\nReport saved to: {report_path}")
    else:
        print("No data available for analysis.")
