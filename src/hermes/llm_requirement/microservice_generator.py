from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import jinja2
import os

from .requirement_parser import ParsedRequirement, Operation, Entity


class FrameworkType(Enum):
    FLASK = "flask"
    FASTAPI = "fastapi"


class LanguageType(Enum):
    PYTHON = "python"


@dataclass
class MicroserviceArtifact:
    name: str
    content: str
    file_type: str = "text"
    path: str = ""
    
    def save(self, output_dir: str):
        full_path = os.path.join(output_dir, self.path, self.name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_type": self.file_type,
            "path": self.path,
            "content_length": len(self.content)
        }


@dataclass
class GeneratedService:
    name: str
    artifacts: List[MicroserviceArtifact] = field(default_factory=list)
    framework: FrameworkType = FrameworkType.FLASK
    language: LanguageType = LanguageType.PYTHON
    
    def save_all(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for artifact in self.artifacts:
            artifact.save(output_dir)
    
    def get_artifact(self, name: str) -> Optional[MicroserviceArtifact]:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        return None


class MicroserviceGenerator:
    _flask_main_template = """from flask import Flask, jsonify, request
import threading
import json
import time
import uuid
import os

app = Flask(__name__)

class {{ entity_name }}:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
        self._state_history = []
    
    def get_state(self):
        return {"value": self.value}
    
    def record_transition(self, operation, pre_state, post_state):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "operation": operation,
            "pre_state": pre_state,
            "post_state": post_state
        }
        self._state_history.append(entry)
        if len(self._state_history) > 1000:
            self._state_history = self._state_history[-500:]
        return entry

{{ operations_implementation }}

{{ entity_name_lower }} = {{ entity_name }}()

@app.before_request
def before_request():
    request._start_time = time.time()
    request._pre_state = {{ entity_name_lower }}.get_state().copy()

@app.after_request
def after_request(response):
    try:
        if hasattr(request, '_pre_state'):
            post_state = {{ entity_name_lower }}.get_state().copy()
            operation = request.endpoint or request.path
            {{ entity_name_lower }}.record_transition(operation, request._pre_state, post_state)
    except Exception:
        pass
    return response

{{ routes_definition }}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "value": {{ entity_name_lower }}.get_state()["value"]})

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify({{ entity_name_lower }}.get_state())

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({
        "history": {{ entity_name_lower }}._state_history,
        "count": len({{ entity_name_lower }}._state_history)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
"""
    
    _flask_operations_template = {
        "increment": """    def increment(self):
        with self.lock:
            self.value += 1
        return self.value""",
        "get": """    def get(self):
        with self.lock:
            return self.value""",
        "decrement": """    def decrement(self):
        with self.lock:
            if self.value > 0:
                self.value -= 1
                return self.value
            return None""",
        "reset": """    def reset(self):
        with self.lock:
            self.value = 0
        return self.value""",
    }
    
    _flask_routes_template = {
        "increment": """@app.route('{{ endpoint }}', methods=['{{ method }}'])
def {{ name }}():
    result = {{ entity_name_lower }}.{{ name }}()
    return jsonify({'value': result})""",
        "get": """@app.route('{{ endpoint }}', methods=['{{ method }}'])
def {{ name }}():
    result = {{ entity_name_lower }}.{{ name }}()
    return jsonify({'value': result})""",
        "decrement": """@app.route('{{ endpoint }}', methods=['{{ method }}'])
def {{ name }}():
    result = {{ entity_name_lower }}.{{ name }}()
    if result is None:
        return jsonify({'error': 'Cannot decrement below zero', 'value': {{ entity_name_lower }}.get()}), 400
    return jsonify({'value': result})""",
        "reset": """@app.route('{{ endpoint }}', methods=['{{ method }}'])
def {{ name }}():
    result = {{ entity_name_lower }}.{{ name }}()
    return jsonify({'value': result})""",
    }
    
    _dockerfile_template = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE {{ port }}

ENV PORT={{ port }}

CMD ["python", "{{ main_file }}"]
"""
    
    _requirements_flask = """flask==2.3.3
"""
    
    _requirements_fastapi = """fastapi==0.104.1
uvicorn==0.24.0
"""
    
    _k8s_deployment_template = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ service_name }}
  labels:
    app: {{ service_name }}
spec:
  replicas: {{ replicas }}
  selector:
    matchLabels:
      app: {{ service_name }}
  template:
    metadata:
      labels:
        app: {{ service_name }}
    spec:
      containers:
      - name: {{ service_name }}
        image: {{ image_name }}
        ports:
        - containerPort: {{ port }}
        env:
        - name: PORT
          value: "{{ port }}"
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "200m"
            memory: "256Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: {{ port }}
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {{ port }}
          initialDelaySeconds: 5
          periodSeconds: 5
"""
    
    _k8s_service_template = """apiVersion: v1
kind: Service
metadata:
  name: {{ service_name }}-svc
  labels:
    app: {{ service_name }}
spec:
  type: {{ service_type }}
  selector:
    app: {{ service_name }}
  ports:
  - protocol: TCP
    port: {{ service_port }}
    targetPort: {{ target_port }}
{% if service_type == "NodePort" %}
    nodePort: {{ node_port }}
{% endif %}
"""
    
    _k8s_ingress_template = """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ service_name }}-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - http:
      paths:
      - path: /{{ service_name }}
        pathType: Prefix
        backend:
          service:
            name: {{ service_name }}-svc
            port:
              number: {{ service_port }}
"""
    
    def __init__(self, framework: FrameworkType = FrameworkType.FLASK,
                 language: LanguageType = LanguageType.PYTHON):
        self.framework = framework
        self.language = language
    
    def generate(self, requirement: ParsedRequirement, 
                 image_name: str = "auto-generated-service") -> GeneratedService:
        artifacts = []
        
        entity_name = requirement.entities[0].name if requirement.entities else "Service"
        service_name = requirement.name.lower().replace("service", "")
        main_file = "app.py"
        
        operations_impl = self._generate_operations_impl(entity_name, requirement.operations)
        routes_def = self._generate_routes(entity_name, requirement.operations)
        
        main_code = jinja2.Template(self._flask_main_template).render(
            entity_name=entity_name,
            entity_name_lower=entity_name.lower(),
            operations_implementation=operations_impl,
            routes_definition=routes_def,
            requirement=requirement
        )
        
        artifacts.append(MicroserviceArtifact(
            name=main_file,
            content=main_code,
            file_type="python"
        ))
        
        requirements_content = self._requirements_flask
        artifacts.append(MicroserviceArtifact(
            name="requirements.txt",
            content=requirements_content,
            file_type="text"
        ))
        
        dockerfile_content = jinja2.Template(self._dockerfile_template).render(
            port=5000,
            main_file=main_file
        )
        artifacts.append(MicroserviceArtifact(
            name="Dockerfile",
            content=dockerfile_content,
            file_type="text"
        ))
        
        deployment_content = jinja2.Template(self._k8s_deployment_template).render(
            service_name=service_name,
            image_name=image_name,
            port=5000,
            replicas=1
        )
        artifacts.append(MicroserviceArtifact(
            name=f"{service_name}-deployment.yaml",
            content=deployment_content,
            file_type="yaml"
        ))
        
        service_content = jinja2.Template(self._k8s_service_template).render(
            service_name=service_name,
            service_type="ClusterIP",
            service_port=80,
            target_port=5000,
            node_port=30001
        )
        artifacts.append(MicroserviceArtifact(
            name=f"{service_name}-service.yaml",
            content=service_content,
            file_type="yaml"
        ))
        
        ingress_content = jinja2.Template(self._k8s_ingress_template).render(
            service_name=service_name,
            service_port=80
        )
        artifacts.append(MicroserviceArtifact(
            name=f"{service_name}-ingress.yaml",
            content=ingress_content,
            file_type="yaml"
        ))
        
        return GeneratedService(
            name=requirement.name,
            artifacts=artifacts,
            framework=self.framework,
            language=self.language
        )
    
    def _generate_operations_impl(self, entity_name: str, 
                                   operations: List[Operation]) -> str:
        impls = []
        for op in operations:
            template = self._flask_operations_template.get(op.name.lower())
            if template:
                impls.append(template)
            else:
                impls.append(f"""    def {op.name}(self):
        with self.lock:
            pass
        return None""")
        return "\n\n".join(impls)
    
    def _generate_routes(self, entity_name: str, operations: List[Operation]) -> str:
        routes = []
        entity_name_lower = entity_name.lower()
        
        for op in operations:
            template = self._flask_routes_template.get(op.name.lower())
            if template:
                routes.append(jinja2.Template(template).render(
                    endpoint=op.endpoint,
                    method=op.method,
                    name=op.name,
                    entity_name_lower=entity_name_lower
                ))
            else:
                routes.append(f"""@app.route('{op.endpoint}', methods=['{op.method}'])
def {op.name}():
    result = {entity_name_lower}.{op.name}()
    return jsonify({{'value': result}})""")
        
        return "\n\n".join(routes)
    
    def generate_counter_service(self, image_name: str = "counter-service") -> GeneratedService:
        from .requirement_parser import RequirementParser
        parser = RequirementParser()
        requirement = parser.parse(
            "实现 HTTP 计数器服务，支持 POST /inc 增加计数，POST /dec 减少计数，GET /get 返回当前值，POST /reset 重置为0。要求计数器不能为负。"
        )
        return self.generate(requirement, image_name)