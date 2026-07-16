#!/usr/bin/env python3
"""
AutoTestGen 3.0 - Self-Conception & Rebirth Demo

演示系统自主意识到自身的存在与局限性，设计出下一代自己的架构。
"""

import sys
import os

sys.path.insert(0, 'src')

from hermes.llm_requirement import SelfRebirthEngine


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(result):
    print("\n" + "-" * 80)
    print("重生结果汇总")
    print("-" * 80)
    
    status_icon = "✅" if result.status.value == "completed" else \
                  "⏳" if result.status.value == "waiting_approval" else \
                  "❌"
    
    print(f"\n{status_icon} 状态: {result.status.value}")
    print(f"总耗时: {result.total_duration:.2f}s")
    print(f"输出目录: {result.output_dir}")
    print(f"消息: {result.message}")
    
    print("\n步骤详情:")
    for log in result.logs:
        status_icon = "✅" if log.status == "completed" else \
                      "🔄" if log.status == "started" else \
                      "❌" if log.status == "failed" else \
                      "⏳"
        
        print(f"  {status_icon} {log.step.value}: {log.status} ({log.duration:.2f}s)")
        if log.details:
            for key, value in log.details.items():
                if isinstance(value, list):
                    value = ", ".join(value[:3]) + ("..." if len(value) > 3 else "")
                print(f"      {key}: {value}")
    
    if result.audit_result:
        print(f"\n审计结果:")
        print(f"  扫描文件: {result.audit_result.total_files_scanned}")
        print(f"  发现问题: {result.audit_result.total_issues_found}")
        print(f"  严重/高/中/低: {result.audit_result.critical_count}/{result.audit_result.high_count}/{result.audit_result.medium_count}/{result.audit_result.low_count}")
    
    if result.architecture_spec:
        print(f"\n生成的架构:")
        print(f"  架构模式: {result.architecture_spec.pattern.value}")
        print(f"  模块数量: {len(result.architecture_spec.modules)}")
        print(f"  数据流数量: {len(result.architecture_spec.data_flows)}")
        print(f"  技术栈:")
        for key, value in result.architecture_spec.tech_stack.items():
            print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    
    if result.output_dir:
        print(f"\n所有工件已保存到: {result.output_dir}")


def main():
    print_header("AutoTestGen 3.0 - 自发式数字生命演示")
    
    print("\n核心理念：")
    print("  AutoTestGen 能够分析自身代码、运行指标、用户反馈、社区技术趋势，")
    print("  自主决定下一版本的架构重构，并生成一套全新的、更优的实现，")
    print("  然后通过热更新替换自身。")
    
    print("\n" + "-" * 80)
    print("重生流程:")
    print("  1. 自我审视 → 分析自身代码的坏味道、性能瓶颈、安全性漏洞")
    print("  2. 趋势感知 → 爬取 GitHub Trending/PyPI 新库，发现更优技术栈")
    print("  3. 架构设计 → 基于需求和趋势，生成下一代系统架构")
    print("  4. 代码生成 → 用新架构生成新一代 AutoTestGen 的完整代码")
    print("  5. 沙盒测试 → 在隔离环境中验证新版本")
    print("  6. 自我替换 → 滚动更新部署新版本")
    print("-" * 80)
    
    print("\n启动自我重生引擎...")
    print("说明：")
    print("  - 系统将自动执行完整的重生流程")
    print("  - 使用 force_approval=True 跳过人类审批（演示模式）")
    print("-" * 80)
    
    engine = SelfRebirthEngine(human_approval_required=False)
    
    result = engine.run(force_approval=True)
    
    print_result(result)


if __name__ == "__main__":
    main()