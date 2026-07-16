"""
端到端演示：从自然语言到可验证系统生成 + 自演进

流程：
1. 用户输入自然语言需求
2. 解析需求，提取实体、操作、约束
3. 生成TLA+形式化规范
4. 验证规范结构
5. 生成代码（Python + Flask）
6. 生成测试用例（并发测试）
7. 生成部署文件（Dockerfile等）
8. 执行测试并验证
9. 自演进：基于TLC反例自动修正需求/规范
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hermes.llm_requirement import IntegrationOrchestrator, SelfEvolvingEngine


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_basic_demo():
    print_header("自然语言到可验证系统端到端生成演示")
    
    print("\n示例需求：")
    print('"实现 HTTP 计数器服务，两个操作：POST /inc 增加计数；GET /get 返回当前值。要求并发调用不会丢失更新。"')
    
    orchestrator = IntegrationOrchestrator()
    
    print("\n" + "-" * 80)
    print("开始构建流水线...")
    print("-" * 80)
    
    result = orchestrator.run_counter_example()
    
    print("\n" + "-" * 80)
    print("构建结果汇总")
    print("-" * 80)
    
    print(f"\n状态: {result['status']}")
    print(f"总耗时: {result['total_duration']:.2f}s")
    print(f"输出目录: {result['output_dir']}")
    print(f"生成工件数: {result['artifacts_count']}")
    
    print("\n步骤详情:")
    for i, step in enumerate(result['steps']):
        status_icon = "✅" if step['status'] == "success" else "❌"
        print(f"  {i+1}. {step['step']}: {status_icon} ({step['duration']:.2f}s)")
    
    print("\n" + "-" * 80)
    print("生成的工件")
    print("-" * 80)
    
    if os.path.exists(result['output_dir']):
        for item in os.listdir(result['output_dir']):
            item_path = os.path.join(result['output_dir'], item)
            size = os.path.getsize(item_path) if os.path.isfile(item_path) else "(dir)"
            print(f"  - {item} ({size} bytes)")
    
    print("\n" + "-" * 80)
    print("解析后的需求")
    print("-" * 80)
    
    if result['requirement']:
        req = result['requirement']
        print(f"\n名称: {req['name']}")
        print(f"描述: {req['description']}")
        print(f"\n实体:")
        for e in req['entities']:
            print(f"  - {e['name']} ({e['entity_type']})")
        print(f"\n操作:")
        for op in req['operations']:
            print(f"  - {op['method']} {op['endpoint']}: {op['name']}")
        print(f"\n约束:")
        for c in req['constraints']:
            print(f"  - {c['name']}: {c['description']}")
    
    print("\n" + "-" * 80)
    print("生成的代码预览 (app.py)")
    print("-" * 80)
    
    app_path = os.path.join(result['output_dir'], 'app.py')
    if os.path.exists(app_path):
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(content)
    
    print("\n" + "-" * 80)
    print("测试结果")
    print("-" * 80)
    
    test_output_path = os.path.join(result['output_dir'], 'test_output.json')
    if os.path.exists(test_output_path):
        with open(test_output_path, 'r', encoding='utf-8') as f:
            test_output = json.load(f)
        
        if test_output.get('passed'):
            print("✅ 所有测试通过!")
            print("\n测试输出:")
            print(test_output.get('stdout', ''))
        else:
            print("❌ 测试失败")
            print("\n错误输出:")
            print(test_output.get('stderr', ''))
    else:
        print("测试文件未找到")
    
    print("\n" + "=" * 80)
    print("基础演示完成!")
    print("=" * 80)
    print(f"\n所有工件已保存到: {result['output_dir']}")
    
    return result


def run_self_evolve_demo():
    print_header("自演进系统演示：验证反馈驱动的需求/规范演化")
    
    print("\n示例需求（带缺陷）：")
    print('"实现 HTTP 计数器服务，支持 POST /inc 增加计数，POST /dec 减少计数，GET /get 返回当前值。要求并发调用不会丢失更新。"')
    
    print("\n" + "-" * 80)
    print("启动自演进引擎...")
    print("说明：由于当前环境可能未安装 TLC 模型检查器，")
    print("将使用模拟反例演示自演进流程。")
    print("-" * 80)
    
    engine = SelfEvolvingEngine()
    
    requirement_text = "实现 HTTP 计数器服务，支持 POST /inc 增加计数，POST /dec 减少计数，GET /get 返回当前值。要求并发调用不会丢失更新。"
    
    result = engine.run_with_mock_counterexample(requirement_text)
    
    print("\n" + "-" * 80)
    print("自演进结果汇总")
    print("-" * 80)
    
    print(f"\n状态: {result.status.value}")
    print(f"总耗时: {result.total_duration:.2f}s")
    print(f"迭代次数: {result.iterations}")
    print(f"输出目录: {result.output_dir}")
    print(f"消息: {result.message}")
    
    print("\n演化步骤:")
    for step in result.steps:
        status_icon = "✅" if step.tlc_status == "success" else "❌"
        print(f"\n  迭代 {step.iteration}: {status_icon}")
        print(f"    状态: {step.tlc_status}")
        if step.violation_type:
            print(f"    违反类型: {step.violation_type}")
        print(f"    应用更改: {step.changes_applied} 处")
        for change in step.changes:
            if change.get('applied'):
                print(f"      - {change.get('description')}")
    
    print("\n" + "-" * 80)
    print("最终修正后的需求")
    print("-" * 80)
    
    final_req = result.final_requirement.to_dict()
    print(f"\n名称: {final_req['name']}")
    print(f"描述: {final_req['description']}")
    print(f"\n操作:")
    for op in final_req['operations']:
        print(f"  - {op['method']} {op['endpoint']}: {op['name']}")
        if op.get('description'):
            print(f"    描述: {op['description']}")
    print(f"\n约束:")
    for c in final_req['constraints']:
        print(f"  - {c['name']}: {c['description']}")
    
    print("\n" + "-" * 80)
    print("生成的工件")
    print("-" * 80)
    
    if os.path.exists(result.output_dir):
        for item in os.listdir(result.output_dir):
            item_path = os.path.join(result.output_dir, item)
            size = os.path.getsize(item_path) if os.path.isfile(item_path) else "(dir)"
            print(f"  - {item} ({size} bytes)")
    
    print("\n" + "=" * 80)
    print("自演进演示完成!")
    print("=" * 80)
    print(f"\n所有工件已保存到: {result.output_dir}")
    
    return result


def main():
    print_header("AutoTestGen 端到端生成与自演进系统")
    
    print("\n" + "-" * 80)
    print("请选择演示模式:")
    print("  1. 基础端到端生成")
    print("  2. 自演进演示（模拟反例）")
    print("  3. 运行全部演示")
    print("-" * 80)
    
    choice = input("\n输入选择 [1/2/3]: ").strip()
    
    if choice == "1":
        run_basic_demo()
    elif choice == "2":
        run_self_evolve_demo()
    elif choice == "3":
        run_basic_demo()
        print("\n" + "=" * 80)
        print("按回车键继续自演进演示...")
        print("=" * 80)
        input()
        run_self_evolve_demo()
    else:
        print("无效选择，运行默认演示...")
        run_basic_demo()


if __name__ == "__main__":
    main()