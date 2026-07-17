#!/usr/bin/env python3
"""
AutoTestGen 3.0 - Plugin-based Domain Adaptation Engine

Usage:
  python autotestgen.py --domain default --repo ./my-project
  python autotestgen.py --domain mlir --path ./llvm-project/mlir/test/
  python autotestgen.py --domain k8s --gitops-repo ./argocd-configs
  python autotestgen.py --list-plugins
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_domain_analysis(domain: str, path: str, **kwargs):
    from hermes.core.plugin_manager import PluginManager
    
    plugin_manager = PluginManager()
    
    print(f"[AutoTestGen Kernel] Loading domain plugin: {domain}...")
    
    plugin = plugin_manager.get_plugin(domain)
    if not plugin:
        print(f"[ERROR] Plugin not found for domain: {domain}")
        print("Available plugins:")
        for p in plugin_manager.list_plugins():
            print(f"  - {p['name']}: {p['domain']} - {p['description']}")
        sys.exit(1)
    
    print(f"[AutoTestGen Kernel] Loaded: {plugin.description}")
    
    context = plugin.create_context(path, **kwargs)
    
    perceiver = plugin.get_perceiver()
    print(f"\n[Perceiver] Detecting pain points in {path}...")
    
    pain_points = perceiver.detect(context)
    print(f"[Perceiver] Found {len(pain_points)} issue(s):")
    for pp in pain_points:
        print(f"  [{pp.severity}] {pp.type}: {pp.message}")
        if pp.location:
            print(f"      Location: {pp.location}")
    
    if not pain_points:
        print("[Perceiver] No issues detected.")
        return
    
    sage = plugin.get_sage()
    
    for pain_point in pain_points[:2]:
        print(f"\n[Sage] Generating patch for: {pain_point.message}")
        
        patch_plan = sage.generate_patch(context, pain_point)
        print(f"[Sage] Patch plan: {patch_plan.description}")
        for op in patch_plan.operations:
            print(f"  - {op['type']}: {op.get('description', op.get('command', ''))}")
        
        knight = plugin.get_knight()
        
        print(f"\n[Knight] Executing patch...")
        exec_result = knight.execute(context, patch_plan)
        print(f"[Knight] Execution: {'SUCCESS' if exec_result.success else 'FAILED'}")
        for result in exec_result.verification_results:
            print(f"  - {result.get('operation', result.get('test', ''))}: {result['status']}")
        
        print(f"\n[Knight] Verifying...")
        verify_result = knight.verify(context, patch_plan)
        print(f"[Knight] Verification: {'SUCCESS' if verify_result.success else 'FAILED'}")
        for result in verify_result.verification_results:
            print(f"  - {result.get('test', result.get('operation', ''))}: {result['status']}")
    
    print(f"\n[Kernel] Domain analysis complete for {domain}")


def list_plugins():
    from hermes.core.plugin_manager import PluginManager
    
    plugin_manager = PluginManager()
    plugins = plugin_manager.list_plugins()
    
    print("Available domain plugins:")
    print("-" * 60)
    for plugin in plugins:
        print(f"\nName: {plugin['name']}")
        print(f"Domain: {plugin['domain']}")
        print(f"Description: {plugin['description']}")


def main():
    parser = argparse.ArgumentParser(description="AutoTestGen 3.0 - Plugin-based Domain Adaptation")
    parser.add_argument('--domain', '-d', help='Target domain (default, mlir, k8s)')
    parser.add_argument('--path', '-p', help='Path to repository or files')
    parser.add_argument('--list-plugins', action='store_true', help='List available plugins')
    parser.add_argument('--repo', help='Alias for --path')
    parser.add_argument('--gitops-repo', help='GitOps repository path (for k8s domain)')
    
    args = parser.parse_args()
    
    if args.list_plugins:
        list_plugins()
        return
    
    if not args.domain:
        print("Error: --domain is required")
        parser.print_help()
        sys.exit(1)
    
    path = args.path or args.repo or args.gitops_repo or "."
    
    kwargs = {}
    if args.gitops_repo:
        kwargs['gitops'] = True
    
    run_domain_analysis(args.domain, path, **kwargs)


if __name__ == "__main__":
    main()