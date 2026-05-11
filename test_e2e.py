#!/usr/bin/env python3
"""
Hermes 端到端测试脚本 - API网关版
"""

import requests
import json
import time

# 服务地址
GATEWAY_URL = "http://localhost:8000"

def test_gateway_health():
    """测试API网关健康检查"""
    try:
        response = requests.get(f"{GATEWAY_URL}/api/v1/health")
        if response.status_code == 200:
            print("✅ API网关健康检查通过")
            return True
        else:
            print(f"❌ API网关健康检查失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ API网关连接失败: {e}")
        return False

def submit_real_job():
    """提交真实的训练作业"""
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
    
    print(f"\n📤 提交作业: {json.dumps(job_data, indent=2)}")
    
    try:
        response = requests.post(f"{GATEWAY_URL}/api/v1/jobs", json=job_data)
        print(f"📥 响应: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 作业提交成功!")
            print(f"   job_id: {result['job_id']}")
            print(f"   status: {result['status']}")
            return result['job_id']
        else:
            print(f"❌ 作业提交失败: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 作业提交异常: {e}")
        return None

def get_job_status(job_id):
    """获取作业状态"""
    try:
        response = requests.get(f"{GATEWAY_URL}/api/v1/jobs/{job_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ 获取作业状态失败: {e}")
        return None

def simulate_failure(job_id):
    """模拟故障并观察恢复"""
    print(f"\n🔧 模拟作业故障: {job_id}")
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{GATEWAY_URL}/api/v1/jobs/{job_id}/fail")
        print(f"📥 响应: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            elapsed = (time.time() - start_time) * 1000
            
            print(f"✅ 故障恢复成功!")
            print(f"   状态: {result['status']}")
            if 'details' in result:
                details = result['details']
                print(f"   原区域: {details.get('previous_region', 'N/A')}")
                print(f"   新区域: {details.get('new_region', 'N/A')}")
                print(f"   恢复时间: {details.get('recovery_time_ms', 0)}ms")
            print(f"   实际耗时: {elapsed:.2f}ms")
            
            return True
        else:
            print(f"❌ 故障恢复失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 故障模拟异常: {e}")
        return False

def get_cluster_summary():
    """获取集群状态"""
    try:
        response = requests.get(f"{GATEWAY_URL}/api/v1/cluster")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"❌ 获取集群状态失败: {e}")
        return None

def main():
    """主测试流程"""
    print("==============================================")
    print("🚀 Hermes 端到端测试")
    print("==============================================")
    print()
    
    # 1. 健康检查
    print("📋 步骤1: 健康检查")
    if not test_gateway_health():
        print("\n❌ API网关未就绪，请先启动服务")
        return
    
    print()
    
    # 2. 提交真实作业
    print("📋 步骤2: 提交真实训练作业")
    job_id = submit_real_job()
    
    if not job_id:
        return
    
    print()
    
    # 3. 检查作业状态
    print("📋 步骤3: 检查作业状态")
    status = get_job_status(job_id)
    if status:
        print(f"✅ 作业状态: {status.get('status', 'UNKNOWN')}")
        print(f"   调度区域: {status.get('region', 'UNKNOWN')}")
    
    print()
    
    # 4. 获取集群状态
    print("📋 步骤4: 查看集群状态")
    cluster = get_cluster_summary()
    if cluster:
        print(f"✅ 集群状态:")
        print(f"   总GPU: {cluster.get('total_gpus', 0)}")
        print(f"   可用GPU: {cluster.get('available_gpus', 0)}")
    
    print()
    
    # 5. 模拟故障恢复
    print("📋 步骤5: 模拟故障恢复")
    success = simulate_failure(job_id)
    
    print()
    print("==============================================")
    if success:
        print("🎉 所有测试通过!")
        print("✅ API网关正常工作")
        print("✅ 调度器正常工作")
        print("✅ 故障恢复功能正常")
    else:
        print("❌ 测试失败，请检查日志")
    print("==============================================")

if __name__ == "__main__":
    main()
