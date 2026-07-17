#!/usr/bin/env python3
"""
AutoTestGen 3.0 主入口脚本

支持的命令:
  - roadmap: 生成进化路线图
  - audit: 执行自我审计
  - trends: 分析技术趋势
  - smells: 检测架构异味
  - metacognition: 生成价值发现白皮书
  - loop: 启动无限自进化闭环
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_roadmap(args):
    from hermes.roadmap_generator import RoadmapGenerator, RoadmapConfig
    
    config = RoadmapConfig(
        output_dir=args.output_dir,
        max_features=args.max_features,
        budget_days=args.budget_days
    )
    
    generator = RoadmapGenerator(config)
    report = generator.generate()
    
    json_path = generator.save_json(report)
    print(f"\n路线图生成完成！")
    print(f"  - Markdown 报告: {os.path.join(args.output_dir, 'roadmap.md')}")
    print(f"  - JSON 数据: {json_path}")
    print(f"  - 候选特性: {report.summary['total_features']} 个")
    print(f"  - P0 特性: {report.summary['p0_count']} 个")


def run_audit(args):
    from hermes.llm_requirement.self_audit import SelfAuditor
    
    auditor = SelfAuditor()
    result = auditor.audit()
    
    print(f"\n审计完成！")
    print(f"  - 扫描文件: {result.total_files_scanned}")
    print(f"  - 发现问题: {result.total_issues_found}")
    print(f"  - 严重: {result.critical_count}")
    print(f"  - 高: {result.high_count}")
    print(f"  - 中: {result.medium_count}")
    print(f"  - 低: {result.low_count}")


def run_trends(args):
    from hermes.llm_requirement.trend_analyzer import TrendAnalyzer
    
    analyzer = TrendAnalyzer()
    result = analyzer.analyze()
    
    print(f"\n趋势分析完成！")
    print(f"  - 扫描源: {result.sources_scanned}")
    print(f"  - 发现项目: {result.total_items_found}")
    print(f"\n推荐技术:")
    for i, trend in enumerate(result.top_recommendations[:5], 1):
        print(f"  {i}. {trend.name} ({trend.category.value})")
        print(f"     相关性: {trend.relevance_score:.2f}")


def run_smells(args):
    from hermes.self_refactor.smell_detector import SmellDetector
    
    detector = SmellDetector()
    smells = detector.detect_all()
    
    print(f"\n异味检测完成！")
    print(f"  - 发现异味: {len(smells)}")
    
    for smell in smells[:10]:
        print(f"  [{smell.severity.value}] {smell.smell_type.value}: {smell.description}")


def run_metacognition(args):
    from hermes.metacognition import (
        IssueCrawler,
        SentimentAnalyzer,
        TopicExtractor,
        ValueDimensionGenerator,
        WhitepaperGenerator,
        WhitepaperConfig
    )
    
    crawler = IssueCrawler(data_dir='./data')
    comments_data = crawler.load_from_file()
    comments = [c.text for c in comments_data]
    
    print(f"\n加载了 {len(comments)} 条评论")
    
    sentiment_analyzer = SentimentAnalyzer()
    sentiments = sentiment_analyzer.analyze_batch(comments)
    sentiment_stats = sentiment_analyzer.get_sentiment_stats(sentiments)
    
    print(f"情感分析完成:")
    print(f"  - 正面: {sentiment_stats['positive']}")
    print(f"  - 负面: {sentiment_stats['negative']}")
    print(f"  - 中性: {sentiment_stats['neutral']}")
    
    topic_extractor = TopicExtractor(n_topics=5, max_features=500)
    topics = topic_extractor.extract(comments)
    
    print(f"\n提取了 {len(topics)} 个主题:")
    for topic in topics:
        print(f"  - {topic.name}: {', '.join(topic.keywords[:3])}")
    
    dimension_generator = ValueDimensionGenerator()
    dimensions = dimension_generator.generate(topics, comments, sentiments)
    
    print(f"\n生成了 {len(dimensions)} 个价值维度:")
    for dim in dimensions:
        print(f"  - {dim.name} (重要性: {dim.importance_score:.2f})")
    
    config = WhitepaperConfig(output_dir='./whitepapers')
    whitepaper_generator = WhitepaperGenerator(config)
    report = whitepaper_generator.generate(dimensions, sentiment_stats, topics, len(comments))
    
    print(f"\n白皮书生成完成！")
    print(f"  - Markdown 报告: {os.path.join('./whitepapers', 'whitepaper.md')}")
    print(f"  - 发现价值维度: {len(report.dimensions)} 个")


def run_loop(args):
    from hermes.infinite_loop import Orchestrator, LoopConfig
    
    config = LoopConfig(
        continuous=args.continuous,
        max_iterations=args.max_iterations,
        iteration_delay_seconds=args.delay,
        auto_deploy=args.auto_deploy,
        auto_self_replace=args.auto_upgrade
    )
    
    orchestrator = Orchestrator(config)
    orchestrator.run(continuous=args.continuous, max_iterations=args.max_iterations)
    
    stats = orchestrator.get_statistics()
    print(f"\n循环统计:")
    print(f"  - 总迭代次数: {stats.get('total_iterations', 0)}")
    print(f"  - 完成次数: {stats.get('completed_iterations', 0)}")
    print(f"  - 成功率: {stats.get('success_rate', 0):.2%}")


def run_cross_repo(args):
    from hermes.infinite_loop import CrossRepoCoordinator, CrossRepoConfig
    
    config = CrossRepoConfig(
        similarity_threshold=args.threshold,
        auto_migrate=args.auto_migrate,
        iteration_delay_seconds=args.delay
    )
    
    coordinator = CrossRepoCoordinator(config)
    coordinator.run(iterations=args.iterations)
    
    stats = coordinator.get_statistics()
    print(f"\n跨仓库统计:")
    print(f"  - 总迁移数: {stats['metrics']['total_migrations']}")
    print(f"  - 成功迁移: {stats['metrics']['successful_migrations']}")
    print(f"  - 成功率: {stats['metrics']['success_rate']:.2%}")
    
    for repo_id, repo_stats in stats['repo_stats'].items():
        print(f"\n  {repo_id} ({repo_stats['name']}):")
        print(f"    - 修复应用: {repo_stats['fixes_applied']}")
        print(f"    - 迁移发送: {repo_stats['migrations_sent']}")
        print(f"    - 迁移接收: {repo_stats['migrations_received']}")


def main():
    parser = argparse.ArgumentParser(description="AutoTestGen 3.0 - 自我进化系统")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    roadmap_parser = subparsers.add_parser('roadmap', help='生成进化路线图')
    roadmap_parser.add_argument('--output-dir', default='./roadmaps', help='输出目录')
    roadmap_parser.add_argument('--max-features', type=int, default=15, help='最大候选特性数')
    roadmap_parser.add_argument('--budget-days', type=float, default=20, help='预算人天数')
    
    subparsers.add_parser('audit', help='执行自我审计')
    
    subparsers.add_parser('trends', help='分析技术趋势')
    
    subparsers.add_parser('smells', help='检测架构异味')
    
    subparsers.add_parser('metacognition', help='生成价值发现白皮书')
    
    loop_parser = subparsers.add_parser('loop', help='启动无限自进化闭环')
    loop_parser.add_argument('--continuous', action='store_true', help='持续运行模式')
    loop_parser.add_argument('--max-iterations', type=int, default=3, help='最大迭代次数')
    loop_parser.add_argument('--delay', type=int, default=60, help='迭代间隔(秒)')
    loop_parser.add_argument('--auto-deploy', action='store_true', help='自动部署')
    loop_parser.add_argument('--auto-upgrade', action='store_true', help='自动自我升级')
    
    cross_repo_parser = subparsers.add_parser('cross-repo', help='启动跨仓库文明协调器')
    cross_repo_parser.add_argument('--iterations', type=int, default=3, help='迭代次数')
    cross_repo_parser.add_argument('--delay', type=int, default=30, help='迭代间隔(秒)')
    cross_repo_parser.add_argument('--threshold', type=float, default=0.5, help='相似度阈值')
    cross_repo_parser.add_argument('--auto-migrate', action='store_true', help='自动迁移')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    commands = {
        'roadmap': run_roadmap,
        'audit': run_audit,
        'trends': run_trends,
        'smells': run_smells,
        'metacognition': run_metacognition,
        'loop': run_loop,
        'cross-repo': run_cross_repo,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"未知命令: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()