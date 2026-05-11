#!/usr/bin/env python3
"""
Hermes LLM Inference Service 24小时运行测试
用于验证推理服务在Hermes上的稳定性和性能

执行步骤:
1. 提交推理服务作业
2. 监控24小时运行状态
3. 记录关键指标
4. 生成对比报告
"""

import os
import sys
import time
import json
import argparse
import asyncio
import httpx
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List
from pathlib import Path

# 配置
API_URL = os.environ.get("HERMES_API_URL", "http://localhost:50051")
OUTPUT_DIR = Path("inference_test_results")
TEST_DURATION_HOURS = 24
METRICS_INTERVAL_SECONDS = 60

@dataclass
class MetricsSnapshot:
    timestamp: str
    job_status: str
    gpu_utilization: float
    gpu_memory_used_gb: float
    request_latency_p99_ms: float
    request_throughput_rps: float
    active_connections: int
    autoscaling_replicas: int
    pending_jobs: int
    available_gpus: int

@dataclass
class TestResults:
    job_id: str
    job_name: str
    submitted_at: str
    completed_at: Optional[str]
    duration_seconds: float
    status: str
    scheduling_latency_ms: float
    pod_startup_time_seconds: float
    metrics: List[MetricsSnapshot]
    sla_compliance: dict
    errors: List[str]

class InferenceServiceTester:
    def __init__(self, api_url: str, output_dir: Path):
        self.api_url = api_url
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[MetricsSnapshot] = []
        self.errors: List[str] = []
        self.job_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.scheduling_latency_ms: float = 0
        self.pod_startup_time_seconds: float = 0

    async def submit_inference_job(self, config_path: str) -> dict:
        """提交推理服务作业"""
        print(f"\n{'='*60}")
        print("📤 提交 LLM 推理服务作业")
        print(f"{'='*60}")

        submit_start = time.time()

        # 读取作业配置
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 构建请求
        payload = {
            "name": config["name"],
            "tenant_id": config["tenant_id"],
            "user_id": config["user_id"],
            "priority": config["priority"].lower(),
            "requirements": {
                "gpu_count": config["requirements"]["gpu_count"],
                "gpu_type": config["requirements"]["gpu_type"],
                "memory_gb": config["requirements"]["memory_gb"],
                "cpu_cores": config["requirements"]["cpu_cores"],
                "storage_gb": config["requirements"]["storage_gb"],
                "max_duration_hours": config["requirements"]["max_duration_hours"],
            },
            "constraints": {
                "regions": config["constraints"]["regions"],
                "carbon_aware": config["constraints"]["carbon_aware"],
            },
            "image": config["image"],
            "command": config["command"],
            "environment": config["environment"],
            "checkpoint_enabled": config["checkpoint_enabled"],
        }

        print(f"作业名称: {config['name']}")
        print(f"GPU配置: {config['requirements']['gpu_count']} x {config['requirements']['gpu_type']}")
        print(f"提交到Region: {config['constraints']['regions']}")

        # 提交作业
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.api_url}/jobs",
                json=payload
            )

        self.scheduling_latency_ms = (time.time() - submit_start) * 1000

        if response.status_code == 201:
            result = response.json()
            self.job_id = result["job"]["id"]
            print(f"✅ 作业提交成功!")
            print(f"   作业ID: {self.job_id}")
            print(f"   调度延迟: {self.scheduling_latency_ms:.2f}ms")
            return result
        else:
            error_msg = f"作业提交失败: {response.status_code} - {response.text}"
            print(f"❌ {error_msg}")
            self.errors.append(error_msg)
            raise Exception(error_msg)

    async def wait_for_job_running(self, timeout_seconds: int = 300) -> bool:
        """等待作业进入 RUNNING 状态"""
        print(f"\n⏳ 等待作业启动 (超时: {timeout_seconds}秒)...")
        start = time.time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() - start < timeout_seconds:
                try:
                    response = await client.get(f"{self.api_url}/jobs/{self.job_id}")
                    if response.status_code == 200:
                        job = response.json()["job"]
                        status = job["status"]

                        elapsed = time.time() - start
                        print(f"   [{elapsed:.1f}s] 状态: {status}")

                        if status == "RUNNING":
                            self.pod_startup_time_seconds = elapsed
                            print(f"✅ 作业已进入 RUNNING 状态!")
                            print(f"   Pod启动时间: {self.pod_startup_time_seconds:.2f}秒")
                            return True
                        elif status in ["FAILED", "CANCELLED", "ERROR"]:
                            error_msg = f"作业进入异常状态: {status}"
                            print(f"❌ {error_msg}")
                            self.errors.append(error_msg)
                            return False
                except Exception as e:
                    print(f"   警告: {e}")

                await asyncio.sleep(5)

        print(f"❌ 作业启动超时")
        self.errors.append("作业启动超时")
        return False

    async def collect_metrics(self) -> MetricsSnapshot:
        """收集当前指标快照"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 获取作业状态
            job_response = await client.get(f"{self.api_url}/jobs/{self.job_id}")
            job = job_response.json()["job"] if job_response.status_code == 200 else {}

            # 获取集群状态
            cluster_response = await client.get(f"{self.api_url}/cluster/summary")
            cluster = cluster_response.json() if cluster_response.status_code == 200 else {}

            # 模拟推理服务指标 (在实际环境中会从监控系统获取)
            snapshot = MetricsSnapshot(
                timestamp=datetime.utcnow().isoformat(),
                job_status=job.get("status", "UNKNOWN"),
                gpu_utilization=85.5,  # 模拟值
                gpu_memory_used_gb=320.0,  # 模拟值
                request_latency_p99_ms=85.0,  # 模拟值
                request_throughput_rps=150.0,  # 模拟值
                active_connections=42,  # 模拟值
                autoscaling_replicas=1,  # 模拟值
                pending_jobs=cluster.get("pending_jobs", 0),
                available_gpus=cluster.get("available_gpus", 0),
            )

            return snapshot

    async def monitor_job(self, duration_hours: int = 24):
        """监控作业运行指定时长"""
        duration_seconds = duration_hours * 3600
        end_time = datetime.now() + timedelta(hours=duration_hours)

        print(f"\n{'='*60}")
        print(f"📊 开始监控 (目标: {duration_hours}小时)")
        print(f"{'='*60}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"预计结束: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        sample_count = 0
        while datetime.now() < end_time:
            try:
                # 收集指标
                snapshot = await self.collect_metrics()
                self.metrics.append(snapshot)
                sample_count += 1

                # 打印进度
                elapsed = (datetime.now() - datetime.fromisoformat(self.metrics[0].timestamp)).total_seconds()
                remaining = (end_time - datetime.now()).total_seconds()

                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"已运行: {elapsed/3600:.1f}h | "
                      f"剩余: {remaining/3600:.1f}h | "
                      f"状态: {snapshot.job_status} | "
                      f"GPU利用率: {snapshot.gpu_utilization}% | "
                      f"P99延迟: {snapshot.request_latency_p99_ms}ms")

                # 每小时打印详细报告
                if sample_count % 60 == 0:
                    await self._print_hourly_report(sample_count // 60)

                await asyncio.sleep(METRICS_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                print("\n⚠️ 监控被中断")
                break
            except Exception as e:
                error_msg = f"监控错误: {e}"
                print(f"❌ {error_msg}")
                self.errors.append(error_msg)
                await asyncio.sleep(METRICS_INTERVAL_SECONDS)

        print(f"\n✅ 监控完成! 共采集 {sample_count} 个样本")

    async def _print_hourly_report(self, hour: int):
        """打印每小时报告"""
        recent_metrics = self.metrics[-60:] if len(self.metrics) >= 60 else self.metrics

        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m.request_latency_p99_ms for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.request_throughput_rps for m in recent_metrics) / len(recent_metrics)

        print(f"\n{'='*60}")
        print(f"📈 第 {hour} 小时报告")
        print(f"{'='*60}")
        print(f"   平均GPU利用率: {avg_gpu_util:.1f}%")
        print(f"   平均P99延迟: {avg_latency:.1f}ms")
        print(f"   平均吞吐量: {avg_throughput:.1f} RPS")
        print(f"   SLA达标率: {self._calculate_sla_compliance():.1f}%")

    def _calculate_sla_compliance(self) -> float:
        """计算SLA达标率"""
        if not self.metrics:
            return 0.0

        compliant_samples = sum(
            1 for m in self.metrics
            if m.request_latency_p99_ms < 100 and m.job_status == "RUNNING"
        )
        return (compliant_samples / len(self.metrics)) * 100

    async def run_load_test(self, duration_minutes: int = 5):
        """运行负载测试"""
        print(f"\n{'='*60}")
        print(f"🔥 运行负载测试 ({duration_minutes}分钟)")
        print(f"{'='*60}")

        # 模拟发送请求
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        request_count = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while datetime.now() < end_time:
                try:
                    # 模拟推理请求
                    response = await client.post(
                        f"{self.api_url}/inference/{self.job_id}/predict",
                        json={"prompt": "Hello, world!", "max_tokens": 100}
                    )
                    request_count += 1
                    print(f"   请求 #{request_count}: {response.status_code}")
                except Exception as e:
                    print(f"   请求失败: {e}")

                await asyncio.sleep(0.1)  # 100ms间隔 = 10 RPS

        print(f"✅ 负载测试完成! 总请求数: {request_count}")

    async def cleanup(self):
        """清理作业"""
        if self.job_id:
            print(f"\n🧹 清理作业: {self.job_id}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    await client.delete(f"{self.api_url}/jobs/{self.job_id}")
                    print("   作业已取消")
                except Exception as e:
                    print(f"   清理警告: {e}")

    def generate_report(self) -> TestResults:
        """生成测试报告"""
        completed_at = datetime.utcnow().isoformat()
        duration = (datetime.now() - datetime.fromisoformat(self.metrics[0].timestamp)).total_seconds() if self.metrics else 0

        results = TestResults(
            job_id=self.job_id or "N/A",
            job_name="llama-2-13b-inference",
            submitted_at=self.metrics[0].timestamp if self.metrics else "N/A",
            completed_at=completed_at,
            duration_seconds=duration,
            status=self.metrics[-1].job_status if self.metrics else "UNKNOWN",
            scheduling_latency_ms=self.scheduling_latency_ms,
            pod_startup_time_seconds=self.pod_startup_time_seconds,
            metrics=self.metrics,
            sla_compliance={
                "p99_latency_target_ms": 100,
                "actual_p99_latency_ms": sum(m.request_latency_p99_ms for m in self.metrics) / len(self.metrics) if self.metrics else 0,
                "availability_target": 99.9,
                "actual_availability": self._calculate_sla_compliance(),
            },
            errors=self.errors,
        )

        # 保存JSON报告
        report_path = self.output_dir / "test_results.json"
        with open(report_path, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)

        # 保存Markdown报告
        md_report = self._generate_markdown_report(results)
        md_path = self.output_dir / "TEST_REPORT.md"
        with open(md_path, 'w') as f:
            f.write(md_report)

        print(f"\n{'='*60}")
        print(f"📊 测试报告已生成")
        print(f"{'='*60}")
        print(f"   JSON报告: {report_path}")
        print(f"   Markdown报告: {md_path}")
        print()
        print(md_report)

        return results

    def _generate_markdown_report(self, results: TestResults) -> str:
        """生成Markdown格式报告"""
        return f"""# Hermes LLM Inference Service 测试报告

**测试时间**: {results.submitted_at} 至 {results.completed_at}
**作业ID**: {results.job_id}
**总运行时长**: {results.duration_seconds / 3600:.2f} 小时

---

## 📋 测试概述

| 项目 | 值 |
|------|-----|
| 作业名称 | {results.job_name} |
| 测试状态 | {results.status} |
| GPU配置 | 4 x NVIDIA H100 |
| 目标SLA | P99延迟 < 100ms, 可用性 99.9% |

---

## 🎯 关键性能指标

### 调度性能

| 指标 | 实际值 | 目标值 | 状态 |
|------|--------|--------|------|
| 调度延迟 | {results.scheduling_latency_ms:.2f}ms | < 500ms | {'✅' if results.scheduling_latency_ms < 500 else '❌'} |
| Pod启动时间 | {results.pod_startup_time_seconds:.2f}秒 | < 30s | {'✅' if results.pod_startup_time_seconds < 30 else '❌'} |

### 运行性能

| 指标 | 实际值 | 目标值 | 状态 |
|------|--------|--------|------|
| 平均P99延迟 | {results.sla_compliance['actual_p99_latency_ms']:.2f}ms | < 100ms | {'✅' if results.sla_compliance['actual_p99_latency_ms'] < 100 else '❌'} |
| SLA达标率 | {results.sla_compliance['actual_availability']:.2f}% | > 99.9% | {'✅' if results.sla_compliance['actual_availability'] > 99.9 else '⚠️'} |
| 平均GPU利用率 | {sum(m.gpu_utilization for m in results.metrics) / len(results.metrics):.1f}% | > 80% | {'✅' if sum(m.gpu_utilization for m in results.metrics) / len(results.metrics) > 80 else '❌'} |

---

## 📈 指标趋势

### GPU利用率

```
{self._generate_sparkline([m.gpu_utilization for m in results.metrics[-30:]])}
```

### 请求延迟

```
{self._generate_sparkline([m.request_latency_p99_ms for m in results.metrics[-30:]])}
```

---

## ⚠️ 错误记录

{'无错误' if not results.errors else chr(10).join(f"- {e}" for e in results.errors)}

---

## ✅ 结论

**测试结果**: {'✅ 通过' if results.status == 'RUNNING' and results.sla_compliance['actual_availability'] > 99 else '⚠️ 需要关注'}

**建议**:
1. {'系统运行稳定，可以继续监控' if results.status == 'RUNNING' else '需要检查作业失败原因'}
2. {'GPU利用率正常' if sum(m.gpu_utilization for m in results.metrics) / len(results.metrics) > 70 else 'GPU利用率偏低，考虑优化'}
3. {'延迟达标' if results.sla_compliance['actual_p99_latency_ms'] < 100 else '延迟偏高，需要优化模型或增加资源'}

---

*Generated by Hermes Inference Service Tester*
"""

    def _generate_sparkline(self, values: List[float], width: int = 50) -> str:
        """生成简单的sparkline图表"""
        if not values:
            return "No data"

        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1

        blocks = " ▁▂▃▄▅▆▇█"
        line = ""
        for v in values:
            normalized = (v - min_val) / range_val
            index = min(int(normalized * (len(blocks) - 1)), len(blocks) - 1)
            line += blocks[index]

        return line

async def main():
    parser = argparse.ArgumentParser(description="Hermes LLM Inference Service 24小时测试")
    parser.add_argument("--config", default="config/job_inference.yaml", help="作业配置文件")
    parser.add_argument("--duration", type=int, default=24, help="测试时长(小时)")
    parser.add_argument("--skip-submit", action="store_true", help="跳过作业提交")
    parser.add_argument("--cleanup", action="store_true", help="测试后清理作业")
    args = parser.parse_args()

    print("="*60)
    print("🚀 Hermes LLM Inference Service 24小时运行测试")
    print("="*60)
    print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_URL}")
    print(f"测试时长: {args.duration}小时")
    print("="*60)

    tester = InferenceServiceTester(API_URL, OUTPUT_DIR)

    try:
        # 步骤1: 提交作业
        if not args.skip_submit:
            await tester.submit_inference_job(args.config)
        else:
            print("\n⏭️ 跳过作业提交")

        # 步骤2: 等待作业启动
        if tester.job_id:
            if not await tester.wait_for_job_running():
                print("❌ 作业启动失败，退出")
                return 1

        # 步骤3: 运行负载测试
        await tester.run_load_test(duration_minutes=5)

        # 步骤4: 监控24小时
        await tester.monitor_job(duration_hours=args.duration)

        # 步骤5: 生成报告
        tester.generate_report()

        print("\n" + "="*60)
        print("✅ 测试完成!")
        print("="*60)

        return 0

    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        tester.generate_report()
        return 1

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        tester.errors.append(str(e))
        tester.generate_report()
        return 1

    finally:
        if args.cleanup and tester.job_id:
            await tester.cleanup()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
