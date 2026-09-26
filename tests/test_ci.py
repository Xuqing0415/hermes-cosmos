"""
Tests for CI/CD Proof Pipeline Module
"""

import os
import shutil
import subprocess
import tempfile
import uuid

import pytest

from hermes.ci.change_analyzer import ChangeAnalyzer, parse_numstat
from hermes.ci.ci_comment_formatter import CICommentFormatter
from hermes.ci.ci_prover import CIProver
from hermes.ci.github_client import GitHubClient
from hermes.ci.incremental_prover import IncrementalProver
from hermes.ci.proof_cache import ProofCache
from hermes.ci.types import (
    ChangedFunction,
    CIProofReport,
    DependencyGraph,
    ProofResult,
    ProofResultStatus,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GIT_ENV = {
    "GIT_AUTHOR_NAME": "ci",
    "GIT_AUTHOR_EMAIL": "ci@example.com",
    "GIT_COMMITTER_NAME": "ci",
    "GIT_COMMITTER_EMAIL": "ci@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


class TestChangeAnalyzer:
    """Tests for ChangeAnalyzer"""

    def test_init(self):
        analyzer = ChangeAnalyzer()
        assert analyzer.repo_path == "."

    def test_build_dependency_graph(self):
        analyzer = ChangeAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""def add(a, b):
    return a + b

def multiply(a, b):
    return add(a, b) * 2
""")
            temp_file = f.name

        try:
            graph = analyzer.build_dependency_graph([temp_file])
            assert graph is not None
            assert isinstance(graph.functions, list)
            assert isinstance(graph.dependencies, dict)
        finally:
            os.unlink(temp_file)

    def test_extract_dependencies(self):
        analyzer = ChangeAnalyzer()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""def helper():
    pass

def main():
    helper()
    return True
""")
            temp_file = f.name
            module_name = os.path.basename(temp_file).replace(".py", "")

        try:
            graph = analyzer.build_dependency_graph([temp_file])
            full_name = f"{module_name}.main"
            assert full_name in graph.dependencies
        finally:
            os.unlink(temp_file)

    def test_compute_function_signature(self):
        analyzer = ChangeAnalyzer()

        signature1 = analyzer._compute_function_signature("def test(): return True")
        signature2 = analyzer._compute_function_signature("def test(): return True")
        signature3 = analyzer._compute_function_signature("def test(): return False")

        assert signature1 == signature2
        assert signature1 != signature3


class TestProofCache:
    """Tests for ProofCache"""

    def test_init(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        cache = None
        try:
            cache = ProofCache(temp_db)
            assert cache._conn is not None

            stats = cache.get_stats()
            assert stats["total_entries"] == 0
        finally:
            if cache:
                cache.close()
            os.unlink(temp_db)

    def test_compute_key(self):
        cache = ProofCache(":memory:")

        key1 = cache.compute_key("test_func", "def test_func(): return True")
        key2 = cache.compute_key("test_func", "def test_func(): return True")
        key3 = cache.compute_key("test_func", "def test_func(): return False")

        assert key1 == key2
        assert key1 != key3

    def test_set_and_get(self):
        cache = ProofCache(":memory:")

        cache.set(
            key="test_key",
            function_name="test_func",
            filename="test.py",
            signature="abc123",
            status=ProofResultStatus.PROVEN,
            duration=0.1,
        )

        entry = cache.get("test_key")
        assert entry is not None
        assert entry.function_name == "test_func"
        assert entry.status == ProofResultStatus.PROVEN
        assert entry.duration == 0.1

    def test_invalidate(self):
        cache = ProofCache(":memory:")

        cache.set(
            key="test_key",
            function_name="test_func",
            filename="test.py",
            signature="abc123",
            status=ProofResultStatus.PROVEN,
        )

        cache.invalidate("test_func")
        entry = cache.get("test_key")
        assert entry is None

    def test_invalidate_file(self):
        cache = ProofCache(":memory:")

        cache.set(
            key="test_key",
            function_name="test_func",
            filename="test.py",
            signature="abc123",
            status=ProofResultStatus.PROVEN,
        )

        cache.invalidate_file("test.py")
        entry = cache.get("test_key")
        assert entry is None

    def test_get_stats(self):
        cache = ProofCache(":memory:")

        cache.set(
            key="key1",
            function_name="func1",
            filename="test.py",
            signature="sig1",
            status=ProofResultStatus.PROVEN,
            duration=0.1,
        )

        cache.set(
            key="key2",
            function_name="func2",
            filename="test.py",
            signature="sig2",
            status=ProofResultStatus.DISPROVEN,
            duration=0.2,
        )

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["proven"] == 1
        assert stats["disproven"] == 1
        assert stats["avg_duration"] == 0.15

    def test_save_and_load(self):
        cache1 = ProofCache(":memory:")

        cache1.set(
            key="test_key",
            function_name="test_func",
            filename="test.py",
            signature="abc123",
            status=ProofResultStatus.PROVEN,
            duration=0.1,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            cache1.save_to_file(temp_file)

            cache2 = ProofCache.load_from_file(temp_file)
            entry = cache2.get("test_key")
            assert entry is not None
            assert entry.function_name == "test_func"
        finally:
            os.unlink(temp_file)


class TestIncrementalProver:
    """Tests for IncrementalProver"""

    def test_init(self):
        cache = ProofCache(":memory:")
        prover = IncrementalProver(cache)

        assert prover.proof_cache is cache
        assert prover.proof_generator is not None

    def test_prove_changed_functions_empty(self):
        cache = ProofCache(":memory:")
        prover = IncrementalProver(cache)

        report = prover.prove_changed_functions([])

        assert report.total_proven == 0
        assert report.total_disproven == 0
        assert len(report.results) == 0

    def test_prove_changed_functions_with_cache(self):
        cache = ProofCache(":memory:")

        func_name = "cached_func"
        func_signature = "abc123"
        cache_key = cache.compute_key(func_name, func_signature)

        cache.set(
            key=cache_key,
            function_name=func_name,
            filename="test.py",
            signature=func_signature,
            status=ProofResultStatus.PROVEN,
            duration=0.1,
        )

        prover = IncrementalProver(cache)

        func = ChangedFunction(name=func_name, filename="test.py", line_start=1, line_end=3, signature=func_signature)

        report = prover.prove_changed_functions([func])

        assert report.total_cached >= 1


class TestGitHubClient:
    """Tests for GitHubClient"""

    def test_init(self):
        client = GitHubClient()
        assert client.base_url == "https://api.github.com"

    def test_init_with_token(self):
        client = GitHubClient(token="test_token")
        assert "token test_token" in client.headers["Authorization"]

    def test_set_repo(self):
        client = GitHubClient()
        client.set_repo("owner", "repo")

        assert client.repo_owner == "owner"
        assert client.repo_name == "repo"

    def test_is_available_no_token(self):
        client = GitHubClient()
        assert not client.is_available()


class TestCICommentFormatter:
    """Tests for CICommentFormatter"""

    def test_format_report(self):
        formatter = CICommentFormatter()

        report = CIProofReport(
            pr_number=123,
            commit_hash="abc123",
            total_proven=2,
            total_disproven=1,
            total_timeout=0,
            total_unknown=0,
            total_cached=1,
            total_duration=0.5,
        )

        markdown = formatter.format_report(report)

        assert isinstance(markdown, str)
        assert "AutoTestGen" in markdown
        assert "" in markdown
        assert "" in markdown

    def test_format_summary(self):
        formatter = CICommentFormatter()

        report = CIProofReport(
            pr_number=123,
            commit_hash="abc123",
            total_proven=5,
            total_disproven=2,
            total_timeout=1,
            total_unknown=0,
            total_duration=1.5,
        )

        markdown = formatter.format_report(report)

        assert "5" in markdown
        assert "2" in markdown
        assert "1.5" in markdown

    def test_format_detailed_report(self):
        formatter = CICommentFormatter()

        report = CIProofReport(pr_number=123, commit_hash="abc123", total_proven=1, total_disproven=0)

        markdown = formatter.format_detailed_report(report)

        assert isinstance(markdown, str)

    def test_format_error(self):
        formatter = CICommentFormatter()

        markdown = formatter.format_error("Test error message")

        assert "" in markdown
        assert "Test error message" in markdown


class TestCIProver:
    """Tests for CIProver"""

    def test_init(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        ci_prover = None
        try:
            ci_prover = CIProver(cache_path=temp_db)

            assert ci_prover.change_analyzer is not None
            assert ci_prover.proof_cache is not None
            assert ci_prover.incremental_prover is not None
            assert ci_prover.github_client is not None
            assert ci_prover.comment_formatter is not None
        finally:
            if ci_prover:
                ci_prover.proof_cache.close()
            os.unlink(temp_db)

    def test_run_local_proof(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        ci_prover = None
        try:
            ci_prover = CIProver(cache_path=temp_db)
            report = ci_prover.run_local_proof()

            assert report is not None
            assert isinstance(report, CIProofReport)
        finally:
            if ci_prover:
                ci_prover.proof_cache.close()
            os.unlink(temp_db)

    def test_get_cache_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        ci_prover = None
        try:
            ci_prover = CIProver(cache_path=temp_db)
            stats = ci_prover.get_cache_stats()

            assert isinstance(stats, dict)
            assert "total_entries" in stats
        finally:
            if ci_prover:
                ci_prover.proof_cache.close()
            os.unlink(temp_db)

    def test_invalidate_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        ci_prover = None
        try:
            ci_prover = CIProver(cache_path=temp_db)
            ci_prover.proof_cache.set(
                key="test_key",
                function_name="test_func",
                filename="test.py",
                signature="abc123",
                status=ProofResultStatus.PROVEN,
            )

            ci_prover.invalidate_cache(function_name="test_func")

            entry = ci_prover.proof_cache.get("test_key")
            assert entry is None
        finally:
            if ci_prover:
                ci_prover.proof_cache.close()
            os.unlink(temp_db)


class TestDependencyGraph:
    """Tests for DependencyGraph"""

    def test_init(self):
        graph = DependencyGraph(functions=["func1", "func2"], dependencies={"func1": [], "func2": ["func1"]})

        assert graph.functions == ["func1", "func2"]
        assert graph.dependencies == {"func1": [], "func2": ["func1"]}

    def test_get_affected_functions_direct(self):
        graph = DependencyGraph(
            functions=["func1", "func2", "func3"], dependencies={"func1": [], "func2": ["func1"], "func3": ["func2"]}
        )

        affected = graph.get_affected_functions(["func1"])

        assert "func1" in affected
        assert "func2" in affected
        assert "func3" in affected

    def test_get_affected_functions_indirect(self):
        graph = DependencyGraph(
            functions=["func1", "func2", "func3", "func4"],
            dependencies={"func1": [], "func2": ["func1"], "func3": ["func2"], "func4": ["func3"]},
        )

        affected = graph.get_affected_functions(["func2"])

        assert "func2" in affected
        assert "func3" in affected
        assert "func4" in affected
        assert "func1" not in affected


class TestIntegration:
    """Integration tests for CI pipeline"""

    def test_full_pipeline(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_db = f.name

        ci_prover = None
        try:
            ci_prover = CIProver(cache_path=temp_db)
            report = ci_prover.run_local_proof()

            assert report is not None
            assert isinstance(report, CIProofReport)
        finally:
            if ci_prover:
                ci_prover.proof_cache.close()
            os.unlink(temp_db)

    def test_comment_formatting(self):
        formatter = CICommentFormatter()

        report = CIProofReport(
            pr_number=42,
            commit_hash="deadbeef",
            total_proven=3,
            total_disproven=0,
            total_timeout=0,
            total_unknown=0,
            total_cached=2,
            total_duration=0.3,
            results=[
                ProofResult(function_name="func1", filename="test.py", status=ProofResultStatus.PROVEN),
                ProofResult(function_name="func2", filename="test.py", status=ProofResultStatus.PROVEN),
                ProofResult(function_name="func3", filename="test.py", status=ProofResultStatus.PROVEN),
            ],
        )

        markdown = formatter.format_report(report)

        assert "[PROVEN]" in markdown
        assert "3" in markdown
        assert "func1" in markdown


class TestBinaryFilesInNumstat:
    """`git diff --numstat` 对二进制文件报的是 ``-``，不是数字。

    改动里只要有一个 .bin/.png/.pyc，``int("-")`` 就抛 ValueError；那个异常不在
    ``except subprocess.CalledProcessError`` 的覆盖范围内，于是整个变更分析崩掉。
    """

    def test_binary_lines_are_kept_with_zero_counts(self):
        files = parse_numstat("-\t-\tassets/logo.bin\n")

        assert len(files) == 1
        assert files[0].filename == "assets/logo.bin"
        assert (files[0].additions, files[0].deletions) == (0, 0)
        assert files[0].changes == 0

    def test_text_and_binary_lines_together(self):
        output = "12\t3\tsrc/app.py\n-\t-\tdata/model.bin\n0\t7\tdocs/readme.md\n"

        by_name = {f.filename: f for f in parse_numstat(output)}

        assert by_name["src/app.py"].changes == 15
        assert by_name["data/model.bin"].changes == 0
        assert by_name["docs/readme.md"].changes == 7

    def test_empty_and_malformed_lines_are_skipped(self):
        assert parse_numstat("") == []
        assert parse_numstat("\n\n") == []
        assert parse_numstat("only-one-column\n") == []


class TestChangeAnalyzerSurvivesBinaryChanges:
    @pytest.fixture
    def repo(self):
        base = os.path.join(REPO_ROOT, ".tmp_test", "ci_binary")
        path = os.path.join(base, "run_" + uuid.uuid4().hex)
        os.makedirs(path)
        try:
            self._git(path, "init", "-q")
            with open(os.path.join(path, "app.py"), "w", encoding="utf-8") as handle:
                handle.write("def add(a, b):\n    return a + b\n")
            with open(os.path.join(path, "blob.bin"), "wb") as handle:
                handle.write(b"\x00\x01\x02")
            self._git(path, "add", "-A")
            self._git(path, "commit", "-q", "-m", "init")
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def _git(self, repo_path, *args):
        completed = subprocess.run(
            ["git", "-C", repo_path, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, **GIT_ENV},
        )
        assert completed.returncode == 0, f"git {' '.join(args)} failed: {completed.stderr}"

    def test_analyze_local_changes_does_not_crash_on_a_binary_edit(self, repo):
        with open(os.path.join(repo, "blob.bin"), "wb") as handle:
            handle.write(b"\x00\xff\x02\x03")

        changed = ChangeAnalyzer(repo_path=repo).analyze_local_changes()

        assert isinstance(changed, list)

    def test_get_changed_files_between_commits_includes_the_binary(self, repo):
        before = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            env={**os.environ, **GIT_ENV},
        ).stdout.strip()
        with open(os.path.join(repo, "blob.bin"), "wb") as handle:
            handle.write(b"\x00\xff\x02\x03")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-q", "-m", "touch binary")

        files = ChangeAnalyzer(repo_path=repo)._get_changed_files(before, "HEAD")

        assert [f.filename for f in files] == ["blob.bin"]
        assert files[0].changes == 0
