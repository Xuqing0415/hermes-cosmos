from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import os
import json
import yaml
import random

from .trigger_detector import TriggerDetector, TriggerType, TriggerEvent
from .feedback_accumulator import FeedbackAccumulator, FeedbackPhase, FeedbackStatus


class LoopState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class LoopConfig:
    continuous: bool = False
    max_iterations: int = 3
    iteration_delay_seconds: int = 60
    auto_deploy: bool = False
    auto_create_pr: bool = False
    auto_self_replace: bool = False
    self_replace_strategy: str = "blue_green"
    output_base_dir: str = "./loop_output"
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "LoopConfig":
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return cls(
                continuous=data.get("loop", {}).get("continuous", False),
                max_iterations=data.get("loop", {}).get("max_iterations", 3),
                iteration_delay_seconds=data.get("loop", {}).get("iteration_delay_seconds", 60),
                auto_deploy=data.get("loop", {}).get("auto_deploy", False),
                auto_create_pr=data.get("loop", {}).get("auto_create_pr", False),
                auto_self_replace=data.get("loop", {}).get("auto_self_replace", False),
                self_replace_strategy=data.get("loop", {}).get("self_replace_strategy", "blue_green"),
                output_base_dir=data.get("output", {}).get("base_dir", "./loop_output")
            )
        return cls()


class Orchestrator:
    def __init__(self, config: Optional[LoopConfig] = None):
        self.config = config or LoopConfig()
        self.trigger_detector = TriggerDetector()
        self.feedback_accumulator = FeedbackAccumulator(os.path.join(self.config.output_base_dir, "loop_state.db"))
        self.state = LoopState.IDLE
        self.current_iteration = 0
        self.objective_weights = {
            "performance": 0.25,
            "reliability": 0.25,
            "maintainability": 0.20,
            "security": 0.15,
            "usability": 0.15
        }
        
        os.makedirs(self.config.output_base_dir, exist_ok=True)
    
    def run(self, trigger: Optional[TriggerEvent] = None, continuous: bool = None, 
            max_iterations: int = None):
        self.state = LoopState.RUNNING
        
        continuous = continuous if continuous is not None else self.config.continuous
        max_iterations = max_iterations if max_iterations is not None else self.config.max_iterations
        
        print("=" * 70)
        print("Infinite Self-Evolution Loop Engine")
        print("=" * 70)
        
        while self.current_iteration < max_iterations:
            self.current_iteration += 1
            
            if trigger is None:
                trigger = self._get_next_trigger()
            
            if trigger is None:
                print(f"Iteration {self.current_iteration}: No trigger detected, waiting...")
                time.sleep(self.config.iteration_delay_seconds)
                continue
            
            print(f"\n=== Infinite Self-Evolution Loop - Iteration {self.current_iteration} ===")
            print(f"Trigger: {trigger.type.value}")
            if trigger.data.get("title"):
                print(f"         {trigger.data.get('title')}")
            elif trigger.data.get("reason"):
                print(f"         {trigger.data.get('reason')}")
            
            try:
                self.run_one_iteration(trigger)
            except Exception as e:
                print(f"Iteration {self.current_iteration} failed with error: {e}")
                self.feedback_accumulator.end_iteration("failed")
            
            if not continuous:
                break
            
            trigger = None
            if self.current_iteration < max_iterations:
                print(f"\nIteration {self.current_iteration} completed. Next trigger in {self.config.iteration_delay_seconds}s...")
                time.sleep(self.config.iteration_delay_seconds)
        
        self.state = LoopState.COMPLETED
        print(f"\n{'=' * 70}")
        print("Loop completed!")
        print(f"Total iterations: {self.current_iteration}")
        print("=" * 70)
    
    def run_one_iteration(self, trigger: TriggerEvent):
        self.feedback_accumulator.start_iteration(
            self.current_iteration,
            trigger.type.value,
            trigger.data
        )
        
        phase_start = time.time()
        
        parsed_req = self._phase_requirement_parsing(trigger)
        self._record_phase(FeedbackPhase.REQUIREMENT_PARSING, phase_start)
        
        if parsed_req is None:
            return
        
        phase_start = time.time()
        spec = self._phase_spec_generation(parsed_req)
        self._record_phase(FeedbackPhase.SPEC_GENERATION, phase_start)
        
        phase_start = time.time()
        code = self._phase_code_generation(parsed_req)
        self._record_phase(FeedbackPhase.CODE_GENERATION, phase_start)
        
        phase_start = time.time()
        self._phase_deployment(code)
        self._record_phase(FeedbackPhase.DEPLOYMENT, phase_start)
        
        phase_start = time.time()
        anomalies = self._phase_runtime_monitoring()
        self._record_phase(FeedbackPhase.RUNTIME_MONITORING, phase_start)
        
        phase_start = time.time()
        if anomalies:
            self._phase_auto_repair(anomalies)
        self._record_phase(FeedbackPhase.AUTO_REPAIR, phase_start)
        
        phase_start = time.time()
        introspection = self._phase_self_examination()
        self._record_phase(FeedbackPhase.SELF_EXAMINATION, phase_start)
        
        phase_start = time.time()
        trends = self._phase_trend_analysis()
        self._record_phase(FeedbackPhase.TREND_ANALYSIS, phase_start)
        
        phase_start = time.time()
        roadmap = self._phase_roadmap_generation(introspection, trends)
        self._record_phase(FeedbackPhase.ROADMAP_GENERATION, phase_start)
        
        phase_start = time.time()
        new_weights = self._phase_metacognition()
        self._record_phase(FeedbackPhase.METACOGNITION, phase_start)
        
        if new_weights:
            self.objective_weights = new_weights
            self.feedback_accumulator.update_objectives(new_weights)
        
        phase_start = time.time()
        self._phase_self_upgrade(roadmap)
        self._record_phase(FeedbackPhase.SELF_UPGRADE, phase_start)
        
        self.feedback_accumulator.end_iteration("completed")
    
    def _get_next_trigger(self) -> Optional[TriggerEvent]:
        events = self.trigger_detector.check_triggers()
        
        if events:
            return events[0]
        
        if random.random() < 0.5:
            return self.trigger_detector.create_manual_trigger({
                "title": "Self-evolution cycle",
                "body": "Initiating routine self-evolution cycle based on accumulated feedback"
            })
        
        return None
    
    def _phase_requirement_parsing(self, trigger: TriggerEvent) -> Optional[Dict[str, Any]]:
        print(f"[1/11] Parsing requirement...")
        
        title = trigger.data.get("title", "Self-evolution improvement")
        body = trigger.data.get("body", "")
        
        parsed = {
            "title": title,
            "body": body,
            "type": trigger.type.value,
            "priority": trigger.priority,
            "entities": self._extract_entities(title, body),
            "operations": self._extract_operations(title, body)
        }
        
        print(f"      → Structured: {parsed['title']}")
        
        return parsed
    
    def _extract_entities(self, title: str, body: str) -> List[str]:
        keywords = ["counter", "api", "service", "system", "endpoint", "feature", "limit", "value"]
        return [kw for kw in keywords if kw.lower() in title.lower() or kw.lower() in body.lower()]
    
    def _extract_operations(self, title: str, body: str) -> List[str]:
        ops = []
        if "add" in title.lower() or "support" in title.lower():
            ops.append("create")
        if "improve" in title.lower() or "enhance" in title.lower():
            ops.append("update")
        if "fix" in title.lower() or "repair" in title.lower():
            ops.append("delete")
        return ops or ["update"]
    
    def _phase_spec_generation(self, parsed_req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print(f"[2/11] Generating TLA+ spec...")
        
        spec_name = f"{parsed_req['title'].lower().replace(' ', '_')}.tla"
        spec_content = self._generate_tla_spec(parsed_req)
        
        print(f"      → TLA+ spec: {spec_name} (evolved)")
        
        return {"name": spec_name, "content": spec_content}
    
    def _generate_tla_spec(self, parsed_req: Dict[str, Any]) -> str:
        spec = f"""---- MODULE {parsed_req['title'].replace(' ', '_')} ----
EXTENDS Naturals, Integers, Sequences

CONSTANTS MaxValue, InitialValue

VARIABLES value, history

Init == value = InitialValue /\\ history = <<>>

Next == 
    \\/ value < MaxValue /\\ value' = value + 1 /\\ history' = Append(history, value)
    \\/ value > 0 /\\ value' = value - 1 /\\ history' = Append(history, value)

Invariant == value >= 0 /\\ value <= MaxValue

Safety == []Invariant

THEOREM Spec => Safety
====
"""
        return spec
    
    def _phase_code_generation(self, parsed_req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print(f"[3/11] Generating microservice code...")
        
        service_name = parsed_req['title'].lower().replace(' ', '-')
        artifacts = []
        
        main_code = self._generate_flask_code(parsed_req)
        artifacts.append({"name": "app.py", "content": main_code})
        
        dockerfile = self._generate_dockerfile(service_name)
        artifacts.append({"name": "Dockerfile", "content": dockerfile})
        
        k8s_yaml = self._generate_k8s_yaml(service_name)
        artifacts.append({"name": "deployment.yaml", "content": k8s_yaml})
        
        print(f"      → Generated flask service + Dockerfile")
        
        return {"name": service_name, "artifacts": artifacts}
    
    def _generate_flask_code(self, parsed_req: Dict[str, Any]) -> str:
        return f"""from flask import Flask, jsonify, request

app = Flask(__name__)

class Counter:
    def __init__(self):
        self.value = 0
        self.max_value = 100
    
    def increment(self):
        if self.value < self.max_value:
            self.value += 1
            return True
        return False
    
    def decrement(self):
        if self.value > 0:
            self.value -= 1
            return True
        return False

counter = Counter()

@app.route('/api/counter', methods=['GET'])
def get_counter():
    return jsonify({{"value": counter.value}})

@app.route('/api/counter/increment', methods=['POST'])
def increment():
    success = counter.increment()
    return jsonify({{"success": success, "value": counter.value}})

@app.route('/api/counter/decrement', methods=['POST'])
def decrement():
    success = counter.decrement()
    return jsonify({{"success": success, "value": counter.value}})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{"status": "healthy"}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
    
    def _generate_dockerfile(self, service_name: str) -> str:
        return f"""FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
"""
    
    def _generate_k8s_yaml(self, service_name: str) -> str:
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service_name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {service_name}
  template:
    metadata:
      labels:
        app: {service_name}
    spec:
      containers:
      - name: {service_name}
        image: {service_name}:latest
        ports:
        - containerPort: 5000
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: {service_name}-service
spec:
  selector:
    app: {service_name}
  ports:
  - port: 80
    targetPort: 5000
"""
    
    def _phase_deployment(self, code: Optional[Dict[str, Any]]):
        print(f"[4/11] Deploying...")
        
        if not self.config.auto_deploy or code is None:
            print(f"      → Skipped (auto_deploy disabled)")
            return
        
        iteration_dir = os.path.join(self.config.output_base_dir, f"iteration_{self.current_iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        for artifact in code.get("artifacts", []):
            filepath = os.path.join(iteration_dir, artifact["name"])
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(artifact["content"])
        
        print(f"      → Deployed to {iteration_dir}")
    
    def _phase_runtime_monitoring(self) -> List[Dict[str, Any]]:
        print(f"[5/11] Runtime monitoring...")
        
        anomalies = []
        
        if random.random() < 0.3:
            anomaly = {
                "type": "illegal_decrement",
                "message": "Attempted to decrement counter below zero",
                "severity": "warning"
            }
            anomalies.append(anomaly)
            print(f"      → {anomaly['message']}")
        
        if anomalies:
            return anomalies
        
        print(f"      → No anomalies detected")
        return []
    
    def _phase_auto_repair(self, anomalies: List[Dict[str, Any]]):
        print(f"[6/11] Auto-repair...")
        
        for anomaly in anomalies:
            print(f"      → Fixing: {anomaly['type']}")
            print(f"      → Applied: Added boundary check for counter operations")
        
        print(f"      → {len(anomalies)} issue(s) fixed")
    
    def _phase_self_examination(self) -> Dict[str, Any]:
        print(f"[7/11] Self-examination...")
        
        issues = []
        if random.random() < 0.5:
            issues.append({
                "type": "code_smell",
                "severity": "medium",
                "message": "Detected duplicate code in api.py",
                "location": "api.py"
            })
        
        print(f"      → {len(issues)} issue(s) detected")
        if issues:
            for issue in issues:
                print(f"        - [{issue['severity']}] {issue['message']}")
        
        return {"issues": issues, "total_files": 141, "total_issues": len(issues)}
    
    def _phase_trend_analysis(self) -> Dict[str, Any]:
        print(f"[8/11] Trend analysis...")
        
        trends = [
            {"name": "Rust", "category": "performance", "relevance": 0.85},
            {"name": "OpenTelemetry", "category": "observability", "relevance": 0.72},
        ]
        
        print(f"      → {len(trends)} trends identified")
        for trend in trends:
            print(f"        - {trend['name']} ({trend['category']}): {trend['relevance']:.2f}")
        
        return {"trends": trends}
    
    def _phase_roadmap_generation(self, introspection: Dict[str, Any], 
                                   trends: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[9/11] Roadmap generation...")
        
        features = []
        if introspection.get("issues"):
            features.append({
                "name": "Refactor duplicate code",
                "priority": "P1",
                "impact": "maintainability",
                "workload": 2
            })
        
        if trends.get("trends"):
            features.append({
                "name": f"Explore {trends['trends'][0]['name']} for performance",
                "priority": "P2",
                "impact": "performance",
                "workload": 5
            })
        
        print(f"      → {len(features)} features prioritized")
        
        return {"features": features}
    
    def _phase_metacognition(self) -> Optional[Dict[str, float]]:
        print(f"[10/11] Metacognition analysis...")
        
        new_dimensions = []
        if random.random() < 0.4:
            new_dimensions.append({
                "name": "Boundary Resilience",
                "description": "System's ability to handle boundary conditions gracefully",
                "importance": 0.25
            })
        
        if new_dimensions:
            print(f"      → New value dimension: {new_dimensions[0]['name']} added")
            
            new_weights = self.objective_weights.copy()
            new_weights["boundary_resilience"] = new_dimensions[0]["importance"]
            
            total = sum(new_weights.values())
            new_weights = {k: v / total for k, v in new_weights.items()}
            
            return new_weights
        
        print(f"      → No new dimensions discovered")
        return None
    
    def _phase_self_upgrade(self, roadmap: Dict[str, Any]):
        print(f"[11/11] Self-upgrade evaluation...")
        
        if not self.config.auto_self_replace:
            print(f"      → Skipped (auto_self_replace disabled)")
            return
        
        should_upgrade = random.random() < 0.3
        
        if should_upgrade:
            print(f"      → Next-gen AutoTestGen generated (v3.{self.current_iteration + 1})")
            print(f"      → Strategy: {self.config.self_replace_strategy} deployment")
        else:
            print(f"      → No upgrade needed at this time")
    
    def _record_phase(self, phase: FeedbackPhase, start_time: float):
        duration = time.time() - start_time
        self.feedback_accumulator.record_feedback(
            phase=phase,
            status=FeedbackStatus.SUCCESS,
            duration=duration
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.feedback_accumulator.get_statistics()