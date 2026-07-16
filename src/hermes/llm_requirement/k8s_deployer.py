from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os
import subprocess
import json
import time


class DeployerType(Enum):
    MOCK = "mock"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"


class DeploymentStatus(Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass
class DeploymentInfo:
    name: str
    type: str
    status: DeploymentStatus
    replicas: int = 0
    available_replicas: int = 0
    service_url: Optional[str] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "status": self.status.value,
            "replicas": self.replicas,
            "available_replicas": self.available_replicas,
            "service_url": self.service_url,
            "message": self.message
        }


@dataclass
class DeployResult:
    success: bool
    deployments: List[DeploymentInfo] = field(default_factory=list)
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "deployments": [d.to_dict() for d in self.deployments],
            "message": self.message
        }


class BaseDeployer:
    def __init__(self, deployer_type: DeployerType):
        self.deployer_type = deployer_type
    
    def deploy(self, manifest_files: List[str], namespace: str = "default") -> DeployResult:
        raise NotImplementedError
    
    def undeploy(self, name: str, namespace: str = "default") -> DeployResult:
        raise NotImplementedError
    
    def get_status(self, name: str, namespace: str = "default") -> DeploymentInfo:
        raise NotImplementedError
    
    def is_available(self) -> bool:
        raise NotImplementedError
    
    def get_deployer_type(self) -> DeployerType:
        return self.deployer_type


class MockDeployer(BaseDeployer):
    def __init__(self):
        super().__init__(DeployerType.MOCK)
        self._deployments: Dict[str, DeploymentInfo] = {}
        self._next_port = 30000
    
    def deploy(self, manifest_files: List[str], namespace: str = "default") -> DeployResult:
        deployments = []
        
        for manifest_file in manifest_files:
            filename = os.path.basename(manifest_file)
            
            if "deployment" in filename.lower():
                name = filename.replace("-deployment.yaml", "").replace(".yaml", "")
                self._next_port += 1
                service_url = f"http://localhost:{self._next_port}"
                
                info = DeploymentInfo(
                    name=name,
                    type="deployment",
                    status=DeploymentStatus.RUNNING,
                    replicas=1,
                    available_replicas=1,
                    service_url=service_url,
                    message=f"Mock deployment created for {name}"
                )
                self._deployments[name] = info
                deployments.append(info)
            
            elif "service" in filename.lower():
                name = filename.replace("-service.yaml", "").replace(".yaml", "")
                info = DeploymentInfo(
                    name=name + "-svc",
                    type="service",
                    status=DeploymentStatus.RUNNING,
                    message=f"Mock service created for {name}"
                )
                self._deployments[name + "-svc"] = info
                deployments.append(info)
        
        return DeployResult(
            success=True,
            deployments=deployments,
            message=f"Mock deployment completed: {len(deployments)} resources"
        )
    
    def undeploy(self, name: str, namespace: str = "default") -> DeployResult:
        removed = []
        
        if name in self._deployments:
            removed.append(self._deployments.pop(name))
        
        prefix_to_remove = [k for k in self._deployments if k.startswith(name)]
        for k in prefix_to_remove:
            removed.append(self._deployments.pop(k))
        
        return DeployResult(
            success=True,
            deployments=removed,
            message=f"Mock undeployed {len(removed)} resources"
        )
    
    def get_status(self, name: str, namespace: str = "default") -> DeploymentInfo:
        if name in self._deployments:
            return self._deployments[name]
        
        return DeploymentInfo(
            name=name,
            type="unknown",
            status=DeploymentStatus.DELETED,
            message=f"Deployment {name} not found"
        )
    
    def is_available(self) -> bool:
        return True


class KubernetesDeployer(BaseDeployer):
    def __init__(self, kubeconfig: Optional[str] = None):
        super().__init__(DeployerType.KUBERNETES)
        self.kubeconfig = kubeconfig
    
    def deploy(self, manifest_files: List[str], namespace: str = "default") -> DeployResult:
        deployments = []
        
        for manifest_file in manifest_files:
            if not os.path.exists(manifest_file):
                continue
            
            cmd = ["kubectl", "apply", "-f", manifest_file]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            cmd.extend(["-n", namespace])
            
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                
                if result.returncode == 0:
                    filename = os.path.basename(manifest_file)
                    if "deployment" in filename.lower():
                        name = filename.replace("-deployment.yaml", "").replace(".yaml", "")
                        info = DeploymentInfo(
                            name=name,
                            type="deployment",
                            status=DeploymentStatus.DEPLOYING,
                            message="Kubernetes deployment applied"
                        )
                        deployments.append(info)
                    elif "service" in filename.lower():
                        name = filename.replace("-service.yaml", "").replace(".yaml", "")
                        info = DeploymentInfo(
                            name=name + "-svc",
                            type="service",
                            status=DeploymentStatus.RUNNING,
                            message="Kubernetes service applied"
                        )
                        deployments.append(info)
                else:
                    deployments.append(DeploymentInfo(
                        name=os.path.basename(manifest_file),
                        type="unknown",
                        status=DeploymentStatus.FAILED,
                        message=result.stderr[:200]
                    ))
            
            except subprocess.TimeoutExpired:
                deployments.append(DeploymentInfo(
                    name=os.path.basename(manifest_file),
                    type="unknown",
                    status=DeploymentStatus.FAILED,
                    message="kubectl command timed out"
                ))
            except FileNotFoundError:
                return DeployResult(
                    success=False,
                    message="kubectl not found. Please install kubectl."
                )
        
        return DeployResult(
            success=all(d.status != DeploymentStatus.FAILED for d in deployments),
            deployments=deployments,
            message=f"Kubernetes deployment completed: {len(deployments)} resources"
        )
    
    def undeploy(self, name: str, namespace: str = "default") -> DeployResult:
        cmd = ["kubectl", "delete", "deployment,service", name, f"{name}-svc"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        cmd.extend(["-n", namespace, "--ignore-not-found=true"])
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                return DeployResult(
                    success=True,
                    message=f"Deleted {name} deployment and service"
                )
            else:
                return DeployResult(
                    success=False,
                    message=result.stderr[:200]
                )
        
        except subprocess.TimeoutExpired:
            return DeployResult(
                success=False,
                message="kubectl delete timed out"
            )
        except FileNotFoundError:
            return DeployResult(
                success=False,
                message="kubectl not found"
            )
    
    def get_status(self, name: str, namespace: str = "default") -> DeploymentInfo:
        cmd = ["kubectl", "get", "deployment", name, "-o", "json"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        cmd.extend(["-n", namespace])
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                spec_replicas = data.get("spec", {}).get("replicas", 0)
                available_replicas = data.get("status", {}).get("availableReplicas", 0)
                
                status = DeploymentStatus.RUNNING if available_replicas > 0 else DeploymentStatus.DEPLOYING
                
                return DeploymentInfo(
                    name=name,
                    type="deployment",
                    status=status,
                    replicas=spec_replicas,
                    available_replicas=available_replicas,
                    message="Status retrieved"
                )
            else:
                return DeploymentInfo(
                    name=name,
                    type="deployment",
                    status=DeploymentStatus.DELETED,
                    message="Deployment not found"
                )
        
        except subprocess.TimeoutExpired:
            return DeploymentInfo(
                name=name,
                type="deployment",
                status=DeploymentStatus.PENDING,
                message="Timeout getting status"
            )
        except FileNotFoundError:
            return DeploymentInfo(
                name=name,
                type="deployment",
                status=DeploymentStatus.FAILED,
                message="kubectl not found"
            )
        except json.JSONDecodeError:
            return DeploymentInfo(
                name=name,
                type="deployment",
                status=DeploymentStatus.PENDING,
                message="Failed to parse status"
            )
    
    def is_available(self) -> bool:
        try:
            cmd = ["kubectl", "version", "--client", "-o", "json"]
            if self.kubeconfig:
                cmd.extend(["--kubeconfig", self.kubeconfig])
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return False
            
            cmd_cluster = ["kubectl", "get", "nodes", "-o", "name", "--timeout=5s"]
            if self.kubeconfig:
                cmd_cluster.extend(["--kubeconfig", self.kubeconfig])
            
            result_cluster = subprocess.run(
                cmd_cluster, capture_output=True, text=True, timeout=10
            )
            return result_cluster.returncode == 0
        except FileNotFoundError:
            return False
    
    def wait_for_ready(self, name: str, namespace: str = "default", 
                       timeout: int = 60) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(name, namespace)
            if status.status == DeploymentStatus.RUNNING:
                return True
            time.sleep(2)
        
        return False


class DeployerManager:
    @staticmethod
    def get_deployer(kubeconfig: Optional[str] = None) -> BaseDeployer:
        k8s_deployer = KubernetesDeployer(kubeconfig)
        if k8s_deployer.is_available():
            return k8s_deployer
        
        return MockDeployer()
    
    @staticmethod
    def get_deployer_by_type(deployer_type: DeployerType, 
                             kubeconfig: Optional[str] = None) -> BaseDeployer:
        if deployer_type == DeployerType.KUBERNETES:
            return KubernetesDeployer(kubeconfig)
        elif deployer_type == DeployerType.DOCKER_COMPOSE:
            return MockDeployer()
        else:
            return MockDeployer()