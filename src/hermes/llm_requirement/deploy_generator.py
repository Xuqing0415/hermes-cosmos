from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import jinja2
import os

from .requirement_parser import ParsedRequirement


@dataclass
class DeployArtifact:
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


class DeployGenerator:
    _dockerfile_template = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE {{ port }}

CMD ["python", "app.py"]
"""
    
    _requirements_template = """flask==2.3.3
pytest==7.4.0
"""
    
    _docker_compose_template = """version: '3.8'

services:
  {{ service_name }}:
    build: .
    ports:
      - "{{ port }}:{{ port }}"
    environment:
      - FLASK_APP=app.py
      - FLASK_ENV=production
    restart: unless-stopped
"""
    
    _gitignore_template = """__pycache__/
*.pyc
*.pyo
.env
.venv/
venv/
*.log
"""
    
    def __init__(self):
        self._templates = {
            "Dockerfile": self._dockerfile_template,
            "requirements.txt": self._requirements_template,
            "docker-compose.yml": self._docker_compose_template,
            ".gitignore": self._gitignore_template,
        }
    
    def generate(self, requirement: ParsedRequirement, port: int = 5000) -> List[DeployArtifact]:
        artifacts = []
        
        service_name = requirement.name.lower()
        
        for filename, template_str in self._templates.items():
            template = jinja2.Template(template_str)
            content = template.render(
                service_name=service_name,
                port=port,
                requirement=requirement
            )
            artifacts.append(DeployArtifact(
                name=filename,
                content=content,
                file_type="text"
            ))
        
        return artifacts
    
    def generate_counter_deploy(self) -> List[DeployArtifact]:
        from .requirement_parser import RequirementParser
        parser = RequirementParser()
        requirement = parser.parse_counter_example()
        return self.generate(requirement)
    
    def save_all(self, artifacts: List[DeployArtifact], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for artifact in artifacts:
            artifact.save(output_dir)
    
    def generate_full_project(self, requirement: ParsedRequirement,
                              output_dir: str, port: int = 5000):
        deploy_artifacts = self.generate(requirement, port)
        self.save_all(deploy_artifacts, output_dir)
        
        code_content = self._generate_code(requirement)
        code_artifact = DeployArtifact(
            name="app.py",
            content=code_content,
            file_type="python"
        )
        code_artifact.save(output_dir)
        
        test_content = self._generate_tests(requirement)
        test_artifact = DeployArtifact(
            name="test_app.py",
            content=test_content,
            file_type="python"
        )
        test_artifact.save(output_dir)
        
        return deploy_artifacts + [code_artifact, test_artifact]
    
    def _generate_code(self, requirement: ParsedRequirement) -> str:
        entity_name = requirement.entities[0].name if requirement.entities else "Service"
        
        code_template = """from flask import Flask, jsonify
import threading

app = Flask(__name__)

class {{ entity_name }}:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
        return self.value
    
    def get(self):
        with self.lock:
            return self.value

{{ entity_name_lower }} = {{ entity_name }}()

{% for op in requirement.operations %}
@app.route('{{ op.endpoint }}', methods=['{{ op.method }}'])
def {{ op.name }}():
    {% if op.name == 'increment' %}
    result = {{ entity_name_lower }}.increment()
    return jsonify({'value': result})
    {% elif op.name == 'get' %}
    result = {{ entity_name_lower }}.get()
    return jsonify({'value': result})
    {% else %}
    return jsonify({'error': 'Not implemented'}), 501
    {% endif %}
{% endfor %}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
        
        template = jinja2.Template(code_template)
        return template.render(
            requirement=requirement, 
            entity_name=entity_name,
            entity_name_lower=entity_name.lower()
        )
    
    def _generate_tests(self, requirement: ParsedRequirement) -> str:
        entity_name = requirement.entities[0].name if requirement.entities else "Service"
        
        test_code_template = """import pytest
import threading

class {{ entity_name }}:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
        return self.value
    
    def get(self):
        with self.lock:
            return self.value

class Test{{ entity_name }}:
    def test_initial_value(self):
        obj = {{ entity_name }}()
        assert obj.get() == 0
    
    def test_increment(self):
        obj = {{ entity_name }}()
        obj.increment()
        assert obj.get() == 1
    
    def test_multiple_increments(self):
        obj = {{ entity_name }}()
        for i in range(10):
            obj.increment()
        assert obj.get() == 10
    
    def test_concurrent_increments(self):
        obj = {{ entity_name }}()
        threads = []
        
        def worker():
            for _ in range(100):
                obj.increment()
        
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert obj.get() == 1000

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
"""
        
        template = jinja2.Template(test_code_template)
        return template.render(requirement=requirement, entity_name=entity_name)