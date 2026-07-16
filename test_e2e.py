#!/usr/bin/env python3
"""
Hermes  - API
"""

import requests
import json
import time

# 
GATEWAY_URL = "http://localhost:8000"

def test_gateway_health():
    """API"""
    try:
        response = requests.get(f"{GATEWAY_URL}/api/v1/health")
        if response.status_code == 200:
            print(" API")
            return True
        else:
            print(f" API: {response.text}")
            return False
    except Exception as e:
        print(f" API: {e}")
        return False

def submit_real_job():
    """"""
    job_data = {
        "job_id": "real-test-001",
        "model_name": "tiny_model",
        "batch_size": 16,
        "epochs": 2,
        "gpu_type": "CPU",
        "num_gpus": 1,
        "checkpoint_interval": 1,
        "priority": "normal"
    }
    
    print(f"\n : {json.dumps(job_data, indent=2)}")
    
    try:
        response = requests.post(f"{GATEWAY_URL}/api/v1/jobs", json=job_data)
        print(f" : {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f" !")
            print(f"   job_id: {result['job_id']}")
            print(f"   status: {result['status']}")
            return result['job_id']
        else:
            print(f" : {response.text}")
            return None
    except Exception as e:
        print(f" : {e}")
        return None

def get_job_status(job_id):
    """"""
    try:
        response = requests.get(f"{GATEWAY_URL}/api/v1/jobs/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f" : {e}")
        return None

def simulate_failure(job_id):
    """"""
    print(f"\n : {job_id}")
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{GATEWAY_URL}/api/v1/jobs/{job_id}/fail")
        print(f" : {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            elapsed = (time.time() - start_time) * 1000
            
            print(f" !")
            print(f"   : {result['status']}")
            if 'details' in result:
                details = result['details']
                print(f"   : {details.get('previous_region', 'N/A')}")
                print(f"   : {details.get('new_region', 'N/A')}")
                print(f"   : {details.get('recovery_time_ms', 0)}ms")
            print(f"   : {elapsed:.2f}ms")
            
            return True
        else:
            print(f" : {response.text}")
            return False
    except Exception as e:
        print(f" : {e}")
        return False

def get_cluster_summary():
    """"""
    try:
        response = requests.get(f"{GATEWAY_URL}/api/v1/cluster")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f" : {e}")
        return None

def main():
    """"""
    print("==============================================")
    print(" Hermes ")
    print("==============================================")
    print()
    
    # 1. 
    print(" 1: ")
    if not test_gateway_health():
        print("\n API")
        return
    
    print()
    
    # 2. 
    print(" 2: ")
    job_id = submit_real_job()
    
    if not job_id:
        return
    
    print()
    
    # 3. 
    print(" 3: ")
    status = get_job_status(job_id)
    if status:
        print(f" : {status.get('status', 'UNKNOWN')}")
        print(f"   : {status.get('region', 'UNKNOWN')}")
    
    print()
    
    # 4. 
    print(" 4: ")
    cluster = get_cluster_summary()
    if cluster:
        print(f" :")
        print(f"   GPU: {cluster.get('total_gpus', 0)}")
        print(f"   GPU: {cluster.get('available_gpus', 0)}")
    
    print()
    
    # 5. 
    print(" 5: ")
    success = simulate_failure(job_id)
    
    print()
    print("==============================================")
    if success:
        print(" !")
        print(" API")
        print(" ")
        print(" ")
    else:
        print(" ")
    print("==============================================")

if __name__ == "__main__":
    main()
