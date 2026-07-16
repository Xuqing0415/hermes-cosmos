"""
CLI for CI/CD proof pipeline
"""

import argparse
import sys
import os
from typing import Optional

from hermes.ci.ci_prover import CIProver
from hermes.ci.types import PRInfo
from hermes.ci.proof_cache import ProofCache


def main():
    parser = argparse.ArgumentParser(
        description="CI/CD Proof Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    proof_parser = subparsers.add_parser("proof", help="Run proof verification")
    proof_parser.add_argument("--owner", help="GitHub repository owner")
    proof_parser.add_argument("--repo", help="GitHub repository name")
    proof_parser.add_argument("--pr-number", type=int, help="PR number")
    proof_parser.add_argument("--base-sha", help="Base commit SHA")
    proof_parser.add_argument("--head-sha", help="Head commit SHA")
    proof_parser.add_argument("--token", help="GitHub token")
    proof_parser.add_argument("--repo-path", default=".", help="Repository path")
    proof_parser.add_argument("--cache-path", default="./proof_cache.db", help="Cache path")
    proof_parser.add_argument("--solver", default="z3", help="Theorem prover")
    proof_parser.add_argument("--timeout", type=int, default=30, help="Proof timeout")
    proof_parser.add_argument("--post-comment", action="store_true", help="Post comment to PR")
    
    local_parser = subparsers.add_parser("local", help="Run local proof")
    local_parser.add_argument("--repo-path", default=".", help="Repository path")
    local_parser.add_argument("--cache-path", default="./proof_cache.db", help="Cache path")
    
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    stats_parser.add_argument("--cache-path", default="./proof_cache.db", help="Cache path")
    
    invalidate_parser = subparsers.add_parser("invalidate", help="Invalidate cache")
    invalidate_parser.add_argument("--cache-path", default="./proof_cache.db", help="Cache path")
    invalidate_parser.add_argument("--function", help="Function name to invalidate")
    invalidate_parser.add_argument("--file", help="File to invalidate")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "proof":
        run_proof(args)
    elif args.command == "local":
        run_local(args)
    elif args.command == "stats":
        show_stats(args)
    elif args.command == "invalidate":
        invalidate_cache(args)


def run_proof(args):
    """Run proof verification for a PR"""
    os.environ["GITHUB_TOKEN"] = args.token or os.environ.get("GITHUB_TOKEN", "")
    
    ci_prover = CIProver(
        repo_path=args.repo_path,
        cache_path=args.cache_path,
        solver=args.solver,
        timeout=args.timeout
    )
    
    if args.pr_number and args.base_sha and args.head_sha:
        pr_info = PRInfo(
            owner=args.owner or "",
            repo=args.repo or "",
            pr_number=args.pr_number,
            head_sha=args.head_sha,
            base_sha=args.base_sha,
            title="",
            author=""
        )
        
        report = ci_prover.run_pr_proof(pr_info, args.base_sha, args.head_sha)
        
        if args.post_comment:
            ci_prover.post_pr_comment(pr_info, report)
        
        print_report(report)
    else:
        print("Error: --pr-number, --base-sha, and --head-sha are required")
        sys.exit(1)


def run_local(args):
    """Run local proof verification"""
    ci_prover = CIProver(
        repo_path=args.repo_path,
        cache_path=args.cache_path
    )
    
    report = ci_prover.run_local_proof()
    print_report(report)


def show_stats(args):
    """Show cache statistics"""
    cache = ProofCache(args.cache_path)
    stats = cache.get_stats()
    
    print("=" * 60)
    print("Proof Cache Statistics")
    print("=" * 60)
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Proven:        {stats['proven']}")
    print(f"  Disproven:     {stats['disproven']}")
    print(f"  Timeout:       {stats['timeout']}")
    print(f"  Avg duration:  {stats['avg_duration']}s")
    print(f"  Hit rate:      {stats['hit_rate']:.1%}")
    
    cache.close()


def invalidate_cache(args):
    """Invalidate cache entries"""
    cache = ProofCache(args.cache_path)
    
    if args.function:
        cache.invalidate(args.function)
        print(f"Cache invalidated for function: {args.function}")
    elif args.file:
        cache.invalidate_file(args.file)
        print(f"Cache invalidated for file: {args.file}")
    else:
        print("Error: --function or --file is required")
        sys.exit(1)
    
    cache.close()


def print_report(report):
    """Print proof report"""
    print("=" * 60)
    print("CI Proof Report")
    print("=" * 60)
    print(f"  PR Number:      {report.pr_number}")
    print(f"  Commit:         {report.commit_hash}")
    print(f"  Proven:         {report.total_proven}")
    print(f"  Disproven:      {report.total_disproven}")
    print(f"  Timeout:        {report.total_timeout}")
    print(f"  Unknown:        {report.total_unknown}")
    print(f"  Cached:         {report.total_cached}")
    print(f"  Total duration: {report.total_duration:.2f}s")
    print(f"  Cache hit rate: {report.cache_hit_rate:.1%}")
    
    if report.results:
        print("\n  Results:")
        for result in report.results:
            status = f"{result.status.value}"
            if result.cached:
                status += " (cached)"
            print(f"    - {result.function_name}: {status} ({result.duration:.2f}s)")


if __name__ == "__main__":
    main()
