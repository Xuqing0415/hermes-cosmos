"""
演示：自动生成可演进的生产系统

流程：
1. 用户输入自然语言需求
2. 生成完整微服务代码（Flask API + Dockerfile + K8s YAML）
3. 部署到集群（Mock/Kubernetes）
4. 运行测试序列，监控运行时行为
5. 检测违规并自动修复
6. 重新部署验证
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hermes.llm_requirement import SelfDeployingEngine, SelfDeployResult


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(result: SelfDeployResult):
    print("\n" + "-" * 80)
    print("部署结果汇总")
    print("-" * 80)
    
    status_icon = {
        "completed": "✅",
        "detected_anomaly": "⚠️",
        "failed": "❌"
    }.get(result.status.value, "🔄")
    
    print(f"\n{status_icon} 状态: {result.status.value}")
    print(f"总耗时: {result.total_duration:.2f}s")
    print(f"输出目录: {result.output_dir}")
    print(f"消息: {result.message}")
    
    print("\n步骤详情:")
    for step in result.steps:
        status_color = "✅" if step.status == "completed" else "❌" if step.status == "failed" else "🔄"
        print(f"  {status_color} {step.step}: {step.status} ({step.duration:.2f}s)")
        if step.details:
            for key, value in step.details.items():
                print(f"      {key}: {value}")
    
    print("\n" + "-" * 80)
    print("生成的工件")
    print("-" * 80)
    
    if os.path.exists(result.output_dir):
        for item in os.listdir(result.output_dir):
            item_path = os.path.join(result.output_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                print(f"  - {item} ({size} bytes)")
            else:
                print(f"  - {item}/ (目录)")
    
    if result.fix_actions_taken:
        print("\n" + "-" * 80)
        print("执行的修复动作")
        print("-" * 80)
        for i, action in enumerate(result.fix_actions_taken, 1):
            print(f"  {i}. {action}")
    
    if result.violations:
        print("\n" + "-" * 80)
        print("检测到的违规")
        print("-" * 80)
        for violation in result.violations:
            print(f"  ⚠️ {violation.violation_type.value}: {violation.message}")


def main():
    print_header("AutoTestGen 自部署引擎演示")
    
    print("\n需求描述：")
    print('"实现 HTTP 计数器服务，支持 POST /inc 增加计数，POST /dec 减少计数，GET /get 返回当前值，POST /reset 重置为0。要求计数器不能为负。"')
    
    print("\n" + "-" * 80)
    print("测试序列: inc x3 -> dec x4 (最后一次会导致负数)")
    print("-" * 80)
    
    engine = SelfDeployingEngine()
    
    print("\n启动自部署引擎...")
    print("说明：")
    print("  - 由于当前环境可能未安装 Kubernetes，将使用 Mock 部署器")
    print("  - 运行时监控会检测到违规（计数器变为负数）")
    print("  - 系统会自动分析根因并执行修复")
    print("-" * 80)
    
    result = engine.run_counter_deployment()
    
    print_result(result)
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print(f"\n所有工件已保存到: {result.output_dir}")


if __name__ == "__main__":
    main()