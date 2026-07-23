#!/usr/bin/env python3
"""
 -  Hermes 


1.  Hermes
2. 
3. 
4. 
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone
from typing import Optional

try:
    import httpx
except ImportError:
    print(" httpx: pip install httpx")
    sys.exit(1)

# 
API_URL = os.environ.get("HERMES_API_URL", "http://localhost:50051")
OUTPUT_FILE = "hermes_job_result.json"

def submit_job(job_name: str, gpu_count: int, image: str) -> dict:
    """ Hermes"""
    print(f"\n : {job_name}")
    print(f"   GPU: {gpu_count}")
    print(f"   : {image}")
    
    start_time = time.time()
    
    response = httpx.post(
        f"{API_URL}/jobs",
        json={
            "name": job_name,
            "tenant_id": "ai-training-team",
            "user_id": "hermes-tester",
            "priority": "NORMAL",
            "requirements": {
                "gpu_count": gpu_count,
                "gpu_type": "nvidia-h100",
                "memory_gb": gpu_count * 64,
                "cpu_cores": gpu_count * 8,
                "storage_gb": 500,
                "network_bandwidth_gbps": 100,
                "max_duration_hours": 6,
                "checkpoint_interval_seconds": 300,
            },
            "constraints": {
                "regions": ["us-east"],
                "carbon_aware": True,
            },
            "image": image,
            "command": "python train_gpt.py",
            "environment": {
                "HERMES_CHECKPOINT_INTERVAL": "300",
                "OMP_NUM_THREADS": "8",
            },
            "checkpoint_enabled": True,
            "metadata": {
                "experiment": "nano-gpt-test",
                "model_size": "12-layer-768-hidden",
            },
        },
        timeout=30,
    )
    
    submit_time = time.time() - start_time
    
    if response.status_code != 201:
        print(f" : {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    job_id = result["job"]["id"]
    
    print(f" !")
    print(f"   ID: {job_id}")
    print(f"   : {submit_time:.2f}")
    
    return {
        "job_id": job_id,
        "submit_time": submit_time,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

def monitor_job(job_id: str, max_wait_minutes: int = 30) -> dict:
    """"""
    print(f"\n : {job_id}")
    
    start_time = time.time()
    scheduled_at = None
    running_at = None
    completed_at = None
    status_history = []
    
    timeout = max_wait_minutes * 60
    
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{API_URL}/jobs/{job_id}", timeout=10)
            
            if response.status_code != 200:
                print(f" : {response.status_code}")
                time.sleep(5)
                continue
            
            job = response.json()["job"]
            status = job["status"]
            
            # 
            status_history.append({
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            print(f"   [{datetime.now().strftime('%H:%M:%S')}] : {status}")
            
            # 
            if status == "SCHEDULING" and scheduled_at is None:
                scheduled_at = datetime.now(timezone.utc).isoformat()
                scheduling_delay = time.time() - start_time
                print(f"   ⏱ : {scheduling_delay:.2f}")
            
            if status == "RUNNING" and running_at is None:
                running_at = datetime.now(timezone.utc).isoformat()
                startup_delay = time.time() - start_time
                print(f"   ⏱ : {startup_delay:.2f}")
            
            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                completed_at = datetime.now(timezone.utc).isoformat()
                total_duration = time.time() - start_time
                print(f"\n !")
                print(f"   : {status}")
                print(f"   : {total_duration:.2f}")
                
                return {
                    "status": status,
                    "scheduled_at": scheduled_at,
                    "running_at": running_at,
                    "completed_at": completed_at,
                    "total_duration": total_duration,
                    "status_history": status_history,
                }
            
            time.sleep(5)
            
        except Exception as e:
            print(f" : {e}")
            time.sleep(5)
    
    print(f"⏰  ({max_wait_minutes})")
    return {
        "status": "TIMEOUT",
        "status_history": status_history,
    }

def simulate_failure(job_id: str, delay_seconds: int = 180):
    """"""
    print(f"\n  {delay_seconds}")
    time.sleep(delay_seconds)
    
    print(f" ...")
    print("   (PodGPU)")
    
    # 
    return {
        "failure_triggered_at": datetime.now(timezone.utc).isoformat(),
        "recovery_started_at": datetime.now(timezone.utc).isoformat(),
    }

def save_results(results: dict, filename: str = OUTPUT_FILE):
    """"""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n : {filename}")

def main():
    parser = argparse.ArgumentParser(description="Hermes ")
    parser.add_argument("--job-name", default="nano-gpt-test", help="")
    parser.add_argument("--gpu-count", type=int, default=8, help="GPU")
    parser.add_argument("--image", default="hermes-cosmos/training:latest", help="Docker")
    parser.add_argument("--max-wait", type=int, default=30, help="")
    parser.add_argument("--test-failure", action="store_true", help="")
    args = parser.parse_args()
    
    print("="*60)
    print(" Hermes ")
    print("="*60)
    print(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_URL}")
    print("="*60)
    
    # 1: 
    submit_result = submit_job(args.job_name, args.gpu_count, args.image)
    if not submit_result:
        print(" ")
        sys.exit(1)
    
    # 2: 
    failure_result = None
    if args.test_failure:
        failure_result = simulate_failure(submit_result["job_id"])
    
    # 3: 
    monitor_result = monitor_job(submit_result["job_id"], args.max_wait)
    
    # 
    results = {
        "test_info": {
            "job_name": args.job_name,
            "gpu_count": args.gpu_count,
            "image": args.image,
            "test_failure": args.test_failure,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        "submit_result": submit_result,
        "failure_result": failure_result,
        "monitor_result": monitor_result,
        "metrics": {
            "scheduling_delay": None,
            "startup_delay": None,
            "total_duration": monitor_result.get("total_duration"),
            "recovery_time": None,
        },
    }
    
    # 
    if submit_result and monitor_result.get("scheduled_at"):
        submit_dt = datetime.fromisoformat(submit_result["submitted_at"].replace("Z", "+00:00"))
        scheduled_dt = datetime.fromisoformat(monitor_result["scheduled_at"].replace("Z", "+00:00"))
        results["metrics"]["scheduling_delay"] = (scheduled_dt - submit_dt).total_seconds()
    
    if monitor_result.get("scheduled_at") and monitor_result.get("running_at"):
        scheduled_dt = datetime.fromisoformat(monitor_result["scheduled_at"].replace("Z", "+00:00"))
        running_dt = datetime.fromisoformat(monitor_result["running_at"].replace("Z", "+00:00"))
        results["metrics"]["startup_delay"] = (running_dt - scheduled_dt).total_seconds()
    
    if failure_result and monitor_result.get("completed_at"):
        recovery_dt = datetime.fromisoformat(failure_result["recovery_started_at"].replace("Z", "+00:00"))
        completed_dt = datetime.fromisoformat(monitor_result["completed_at"].replace("Z", "+00:00"))
        results["metrics"]["recovery_time"] = (completed_dt - recovery_dt).total_seconds()
    
    # 
    print("\n" + "="*60)
    print(" ")
    print("="*60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("="*60)
    
    # 
    save_results(results)
    
    # 
    print("\n :")
    print(f"   ID: {submit_result['job_id']}")
    print(f"   : {submit_result['submit_time']:.2f}")
    print(f"   : {results['metrics']['scheduling_delay']:.2f}")
    print(f"   : {results['metrics']['startup_delay']:.2f}")
    print(f"   : {results['metrics']['total_duration']:.2f}")
    if results["metrics"]["recovery_time"]:
        print(f"   : {results['metrics']['recovery_time']:.2f}")
    
    return results

if __name__ == "__main__":
    main()
