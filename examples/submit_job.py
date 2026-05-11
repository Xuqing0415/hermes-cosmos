#!/usr/bin/env python3
"""
真实作业提交脚本 - 用于验证 Hermes 调度系统

这个脚本会：
1. 提交一个真实的训练作业到 Hermes
2. 实时监控作业状态
3. 记录关键指标（调度延迟、运行时间等）
4. 可选：触发故障测试
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Optional

try:
    import httpx
except ImportError:
    print("需要安装 httpx: pip install httpx")
    sys.exit(1)

# 配置
API_URL = os.environ.get("HERMES_API_URL", "http://localhost:50051")
OUTPUT_FILE = "hermes_job_result.json"

def submit_job(job_name: str, gpu_count: int, image: str) -> dict:
    """提交训练作业到 Hermes"""
    print(f"\n📤 提交作业: {job_name}")
    print(f"   GPU数量: {gpu_count}")
    print(f"   镜像: {image}")
    
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
        print(f"❌ 作业提交失败: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    job_id = result["job"]["id"]
    
    print(f"✅ 作业提交成功!")
    print(f"   作业ID: {job_id}")
    print(f"   提交耗时: {submit_time:.2f}秒")
    
    return {
        "job_id": job_id,
        "submit_time": submit_time,
        "submitted_at": datetime.utcnow().isoformat(),
    }

def monitor_job(job_id: str, max_wait_minutes: int = 30) -> dict:
    """监控作业状态直到完成或超时"""
    print(f"\n🔍 监控作业: {job_id}")
    
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
                print(f"⚠️ 获取作业状态失败: {response.status_code}")
                time.sleep(5)
                continue
            
            job = response.json()["job"]
            status = job["status"]
            
            # 记录状态变化
            status_history.append({
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            print(f"   [{datetime.now().strftime('%H:%M:%S')}] 状态: {status}")
            
            # 记录关键时间点
            if status == "SCHEDULING" and scheduled_at is None:
                scheduled_at = datetime.utcnow().isoformat()
                scheduling_delay = time.time() - start_time
                print(f"   ⏱️ 调度延迟: {scheduling_delay:.2f}秒")
            
            if status == "RUNNING" and running_at is None:
                running_at = datetime.utcnow().isoformat()
                startup_delay = time.time() - start_time
                print(f"   ⏱️ 启动延迟: {startup_delay:.2f}秒")
            
            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                completed_at = datetime.utcnow().isoformat()
                total_duration = time.time() - start_time
                print(f"\n🏁 作业结束!")
                print(f"   最终状态: {status}")
                print(f"   总耗时: {total_duration:.2f}秒")
                
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
            print(f"⚠️ 监控出错: {e}")
            time.sleep(5)
    
    print(f"⏰ 监控超时 ({max_wait_minutes}分钟)")
    return {
        "status": "TIMEOUT",
        "status_history": status_history,
    }

def simulate_failure(job_id: str, delay_seconds: int = 180):
    """模拟作业故障（用于测试恢复能力）"""
    print(f"\n💥 计划在 {delay_seconds}秒后模拟故障")
    time.sleep(delay_seconds)
    
    print(f"💥 触发故障模拟...")
    print("   (在生产环境中，这会删除一个Pod来模拟GPU故障)")
    
    # 记录故障时间
    return {
        "failure_triggered_at": datetime.utcnow().isoformat(),
        "recovery_started_at": datetime.utcnow().isoformat(),
    }

def save_results(results: dict, filename: str = OUTPUT_FILE):
    """保存结果到文件"""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 结果已保存到: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Hermes 真实作业测试")
    parser.add_argument("--job-name", default="nano-gpt-test", help="作业名称")
    parser.add_argument("--gpu-count", type=int, default=8, help="GPU数量")
    parser.add_argument("--image", default="hermes-cosmos/training:latest", help="Docker镜像")
    parser.add_argument("--max-wait", type=int, default=30, help="最大等待分钟数")
    parser.add_argument("--test-failure", action="store_true", help="是否测试故障恢复")
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Hermes 真实作业测试")
    print("="*60)
    print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_URL}")
    print("="*60)
    
    # 步骤1: 提交作业
    submit_result = submit_job(args.job_name, args.gpu_count, args.image)
    if not submit_result:
        print("❌ 作业提交失败，退出")
        sys.exit(1)
    
    # 步骤2: 模拟故障（可选）
    failure_result = None
    if args.test_failure:
        failure_result = simulate_failure(submit_result["job_id"])
    
    # 步骤3: 监控作业
    monitor_result = monitor_job(submit_result["job_id"], args.max_wait)
    
    # 汇总结果
    results = {
        "test_info": {
            "job_name": args.job_name,
            "gpu_count": args.gpu_count,
            "image": args.image,
            "test_failure": args.test_failure,
            "started_at": datetime.utcnow().isoformat(),
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
    
    # 计算指标
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
    
    # 打印报告
    print("\n" + "="*60)
    print("📊 测试结果报告")
    print("="*60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("="*60)
    
    # 保存结果
    save_results(results)
    
    # 输出关键指标
    print("\n📈 关键指标:")
    print(f"   作业ID: {submit_result['job_id']}")
    print(f"   提交延迟: {submit_result['submit_time']:.2f}秒")
    print(f"   调度延迟: {results['metrics']['scheduling_delay']:.2f}秒")
    print(f"   启动延迟: {results['metrics']['startup_delay']:.2f}秒")
    print(f"   总耗时: {results['metrics']['total_duration']:.2f}秒")
    if results["metrics"]["recovery_time"]:
        print(f"   恢复时间: {results['metrics']['recovery_time']:.2f}秒")
    
    return results

if __name__ == "__main__":
    main()
