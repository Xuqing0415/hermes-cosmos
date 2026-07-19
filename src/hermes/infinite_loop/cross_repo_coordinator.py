from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time
import random
import hashlib

from .trigger_detector import TriggerDetector, TriggerEvent, TriggerType
from .cross_repo_knowledge import CrossRepoKnowledge, FixTemplate, MatchingResult, create_fix_template
from .safe_migration import SafeMigration, MigrationResult
from .metrics_collector import MetricsCollector, MigrationMetric


@dataclass
class RepoState:
    repo_id: str
    name: str
    current_iteration: int = 0
    last_trigger_time: float = 0.0
    active_issues: int = 0
    fixes_applied: int = 0
    migrations_received: int = 0
    migrations_sent: int = 0


@dataclass
class CrossRepoConfig:
    repos: List[Dict[str, Any]] = field(default_factory=list)
    similarity_threshold: float = 0.5
    auto_migrate: bool = True
    max_concurrent_migrations: int = 3
    iteration_delay_seconds: int = 30
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "security": 1.0,
        "bug": 0.9,
        "performance": 0.7,
        "feature": 0.5,
        "enhancement": 0.4,
        "default": 0.3
    })


class CrossRepoCoordinator:
    def __init__(self, config: Optional[CrossRepoConfig] = None):
        self.config = config or CrossRepoConfig()
        
        if not self.config.repos:
            self.config.repos = [
                {"id": "repo-A", "name": "Service API", "github_repo": "", "github_token": ""},
                {"id": "repo-B", "name": "Data Processor", "github_repo": "", "github_token": ""},
                {"id": "repo-C", "name": "Auth Service", "github_repo": "", "github_token": ""},
            ]
        
        trigger_config = {
            "enabled": ["github_issue"],
            "repos": self.config.repos,
            "self_demand_threshold": 0.7
        }
        self.trigger_detector = TriggerDetector(trigger_config)
        
        self.knowledge = CrossRepoKnowledge("loop_output/cross_repo_knowledge.db")
        self.migration = SafeMigration()
        self.metrics = MetricsCollector("loop_output/cross_repo_metrics.db")
        
        self.repo_states: Dict[str, RepoState] = {}
        for repo in self.config.repos:
            self.repo_states[repo["id"]] = RepoState(
                repo_id=repo["id"],
                name=repo["name"]
            )
        
        self._init_sample_templates()
    
    def _init_sample_templates(self):
        templates = [
            {
                "source_repo_id": "repo-A",
                "issue_data": {
                    "issue_number": 101,
                    "title": "Fix NullPointerException in API handler",
                    "labels": ["bug"]
                },
                "diff_content": """--- a/api/handler.py
+++ b/api/handler.py
@@ -45,6 +45,10 @@ def handle_request(request):
     if not request:
         return error_response("Empty request")
 
+    if request.body is None:
+        return error_response("Null body not allowed")
+
     data = parse_body(request.body)
     return process_data(data)
""",
                "affected_files": ["api/handler.py"]
            },
            {
                "source_repo_id": "repo-A",
                "issue_data": {
                    "issue_number": 102,
                    "title": "Add timeout for HTTP requests",
                    "labels": ["reliability"]
                },
                "diff_content": """--- a/http/client.py
+++ b/http/client.py
@@ -12,7 +12,7 @@ def make_request(url, data):
     try:
         response = requests.post(
             url,
             json=data,
-            timeout=30
+            timeout=10
         )
         return response.json()
     except requests.exceptions.Timeout:
""",
                "affected_files": ["http/client.py"]
            },
            {
                "source_repo_id": "repo-B",
                "issue_data": {
                    "issue_number": 201,
                    "title": "Improve error messages",
                    "labels": ["usability"]
                },
                "diff_content": """--- a/utils/errors.py
+++ b/utils/errors.py
@@ -8,4 +8,6 @@ class AppError(Exception):
     def __init__(self, code, message):
         self.code = code
-        self.message = message
+        self.message = message
+        self.details = f"Error {code}: {message}"
""",
                "affected_files": ["utils/errors.py"]
            }
        ]
        
        for template_data in templates:
            template = create_fix_template(
                source_repo_id=template_data["source_repo_id"],
                issue_data=template_data["issue_data"],
                diff_content=template_data["diff_content"],
                affected_files=template_data["affected_files"]
            )
            self.knowledge.add_fix_template(template)
    
    def run(self, iterations: int = 3):
        print("=" * 70)
        print("Cross-Repository Civilization Coordinator")
        print("=" * 70)
        
        for iteration in range(1, iterations + 1):
            print(f"\n=== Iteration {iteration}/{iterations} ===")
            
            events = self.trigger_detector.check_triggers()
            
            if not events:
                events = [self.trigger_detector.create_manual_trigger(
                    {"title": "Self-evolution cycle", "labels": ["feature"]},
                    repo_id="repo-A"
                )]
            
            for event in events:
                self._process_trigger(event)
            
            time.sleep(self.config.iteration_delay_seconds)
        
        print("\n" + "=" * 70)
        print("Cross-repo cycle completed!")
        print("=" * 70)
        
        stats = self.metrics.get_statistics()
        print(f"\nMigration Statistics:")
        print(f"  Total migrations: {stats['total_migrations']}")
        print(f"  Successful: {stats['successful_migrations']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Active repos: {stats['active_source_repos']} sources, {stats['active_target_repos']} targets")
    
    def _process_trigger(self, event: TriggerEvent):
        repo_id = event.repo_id
        issue_data = event.data
        
        print(f"\n[INFO] {repo_id}: Processing trigger - {event.type.value}")
        
        if "title" in issue_data:
            print(f"[INFO] {repo_id}: Issue: {issue_data['title']}")
        
        self._record_fix(repo_id, issue_data)
        
        if event.type == TriggerType.GITHUB_ISSUE:
            self._evaluate_cross_repo_migration(repo_id, issue_data)
    
    def _record_fix(self, repo_id: str, issue_data: Dict[str, Any]):
        state = self.repo_states.get(repo_id)
        if state:
            state.fixes_applied += 1
        
        self.metrics.update_repo_activity(repo_id, fixes_applied=state.fixes_applied if state else 0)
    
    def _evaluate_cross_repo_migration(self, source_repo_id: str, issue_data: Dict[str, Any]):
        other_repos = [r["id"] for r in self.config.repos if r["id"] != source_repo_id]
        
        for target_repo_id in other_repos:
            matches = self.knowledge.find_matching_templates(
                issue_data,
                target_repo_id,
                self.config.similarity_threshold
            )
            
            if matches:
                best_match = matches[0]
                similarity = best_match.similarity_score
                
                print(f"[INFO] Cross-repo matcher: {similarity:.0%} similarity with template from {best_match.template.source_repo_id}")
                
                if similarity >= 0.7:
                    print(f"[INFO] Generating fix for {target_repo_id}...")
                    result = self.migration.execute_migration(
                        best_match.template,
                        target_repo_id,
                        best_match.matched_files
                    )
                    
                    if result.success:
                        print(f"[INFO] Tests passed. PR created: {result.pr_url}")
                        
                        self._record_successful_migration(source_repo_id, target_repo_id, best_match, result)
                        
                        source_state = self.repo_states.get(source_repo_id)
                        if source_state:
                            source_state.migrations_sent += 1
                        
                        target_state = self.repo_states.get(target_repo_id)
                        if target_state:
                            target_state.migrations_received += 1
                    else:
                        print(f"[INFO] Migration failed: {result.error_message}")
                        
                        self._record_failed_migration(source_repo_id, target_repo_id, best_match, result)
                else:
                    print(f"[INFO] Similarity ({similarity:.0%}) below threshold, skipping")
    
    def _record_successful_migration(self, source_repo_id: str, target_repo_id: str, 
                                     match: MatchingResult, result: MigrationResult):
        test_pass_rate = 0.0
        if result.test_results:
            passed = sum(1 for r in result.test_results if r.get("passed", False))
            test_pass_rate = passed / len(result.test_results)
        
        migration_id = hashlib.md5(f"{match.template.template_id}_{target_repo_id}_{int(time.time())}".encode()).hexdigest()
        
        metric = MigrationMetric(
            migration_id=migration_id,
            template_id=match.template.template_id,
            source_repo_id=source_repo_id,
            target_repo_id=target_repo_id,
            similarity_score=match.similarity_score,
            success=True,
            test_pass_rate=test_pass_rate,
            pr_created=True,
            created_at=time.time()
        )
        
        self.metrics.record_migration(metric)
        self.knowledge.record_migration(match.template.template_id, source_repo_id, target_repo_id, match.similarity_score, "success")
        self.knowledge.update_template_success(match.template.template_id, True)
        
        self.metrics.update_repo_activity(source_repo_id, migrations_sent=self.repo_states[source_repo_id].migrations_sent)
        self.metrics.update_repo_activity(target_repo_id, migrations_received=self.repo_states[target_repo_id].migrations_received)
    
    def _record_failed_migration(self, source_repo_id: str, target_repo_id: str, 
                                 match: MatchingResult, result: MigrationResult):
        test_pass_rate = 0.0
        if result.test_results:
            passed = sum(1 for r in result.test_results if r.get("passed", False))
            test_pass_rate = passed / len(result.test_results)
        
        migration_id = hashlib.md5(f"{match.template.template_id}_{target_repo_id}_{int(time.time())}_failed".encode()).hexdigest()
        
        metric = MigrationMetric(
            migration_id=migration_id,
            template_id=match.template.template_id,
            source_repo_id=source_repo_id,
            target_repo_id=target_repo_id,
            similarity_score=match.similarity_score,
            success=False,
            test_pass_rate=test_pass_rate,
            pr_created=False,
            created_at=time.time()
        )
        
        self.metrics.record_migration(metric)
        self.knowledge.record_migration(match.template.template_id, source_repo_id, target_repo_id, match.similarity_score, "failed")
        self.knowledge.update_template_success(match.template.template_id, False)
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = self.metrics.get_statistics()
        
        repo_stats = {}
        for repo_id, state in self.repo_states.items():
            repo_stats[repo_id] = {
                "name": state.name,
                "fixes_applied": state.fixes_applied,
                "migrations_sent": state.migrations_sent,
                "migrations_received": state.migrations_received
            }
        
        return {
            "metrics": stats,
            "repo_stats": repo_stats
        }