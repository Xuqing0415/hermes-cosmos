"""真实缺陷基准：把系统指向真实开源仓库的历史缺陷，测量它真正做到了什么。

`loop_output/` 里的缺陷是我们自己造的（合成的越界、配置漂移、空指针），
所以论文里的成功率只能说明系统在自己出的题上表现如何。本模块换一批题：
用真实项目 git 历史里的修复提交（`Fix …`）构造基准。

每个用例的证据链全部来自 git 与测试输出，没有任何一处由本模块推断：

1. **可复现性**：把修复提交带的测试拿到父提交上跑——真实缺陷应当在这里失败；
2. **对照组**：同一个测试拿到修复提交上跑——应当通过，说明这个测试确实能区分好坏；
3. **检出率**：系统的 Perceiver 有没有报出缺陷所在的文件（而不是别处的假想问题）；
4. **实际修复**：系统的补丁有没有真的改到文件、有没有让失败的测试通过；
5. **自称 vs 实际**：系统声称执行成功/验证通过，而测试依然失败——记一次 `false_claim`。

第 5 条是本模块存在的理由：一个只统计“系统自称成功”的基准，等于没有基准。
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

FIX_SUBJECT_RE = re.compile(r"^\s*(fix|bugfix|hotfix|repair|revert)\b", re.IGNORECASE)
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$")
SOURCE_SUFFIXES = (".py",)
DEFAULT_CASE_TIMEOUT = 300
DEFAULT_SCAN = 600
TAIL_LINES = 12

#: 工作区指纹要忽略的目录：这些是跑测试的副产物，不是系统改动过的源码
CACHE_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

#: Sage 给补丁标注的可信度里，只有这一档表示「依赖关系已被证明」
PROVEN_CONFIDENCE = "proven"


@dataclass
class DefectCase:
    """一个真实历史缺陷：修复提交 + 它的父提交（缺陷现场）。"""

    sha: str
    parent: str
    subject: str
    date: str
    source_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sha": self.sha,
            "parent": self.parent,
            "subject": self.subject,
            "date": self.date,
            "source_files": self.source_files,
            "test_files": self.test_files,
        }


@dataclass
class TestOutcome:
    returncode: int
    passed: int = 0
    failed: int = 0
    errors: int = 0
    tail: str = ""
    timed_out: bool = False

    @property
    def did_fail(self) -> bool:
        return self.timed_out or self.returncode != 0 or self.failed > 0 or self.errors > 0

    @property
    def did_pass(self) -> bool:
        return not self.timed_out and self.returncode == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "returncode": self.returncode,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "timed_out": self.timed_out,
            "tail": self.tail,
        }


@dataclass
class CaseResult:
    case: DefectCase
    reproduced: Optional[bool] = None
    confirmed: Optional[bool] = None
    parent_outcome: Optional[TestOutcome] = None
    control_outcome: Optional[TestOutcome] = None
    after_system_outcome: Optional[TestOutcome] = None
    claimed_pain_points: int = 0
    claimed_locations: List[str] = field(default_factory=list)
    claimed_success: bool = False
    claimed_verifications: List[str] = field(default_factory=list)
    claimed_confidences: List[str] = field(default_factory=list)
    files_touched: List[str] = field(default_factory=list)
    detected: bool = False
    repaired: bool = False
    false_claim: bool = False
    inconclusive: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "reproduced": self.reproduced,
            "confirmed": self.confirmed,
            "detected": self.detected,
            "repaired": self.repaired,
            "false_claim": self.false_claim,
            "inconclusive": self.inconclusive,
            "claimed_pain_points": self.claimed_pain_points,
            "claimed_locations": self.claimed_locations,
            "claimed_success": self.claimed_success,
            "claimed_verifications": self.claimed_verifications,
            "claimed_confidences": self.claimed_confidences,
            "files_touched": self.files_touched,
            "parent_outcome": self.parent_outcome.to_dict() if self.parent_outcome else None,
            "control_outcome": self.control_outcome.to_dict() if self.control_outcome else None,
            "after_system_outcome": self.after_system_outcome.to_dict() if self.after_system_outcome else None,
            "notes": self.notes,
        }


@dataclass
class BenchmarkReport:
    repo: str = ""
    scanned_commits: int = 0
    cases: List[CaseResult] = field(default_factory=list)
    git_available: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    def count(self, attribute: str) -> int:
        return sum(1 for item in self.cases if getattr(item, attribute) is True)

    @property
    def reproduced(self) -> int:
        return self.count("reproduced")

    @property
    def confirmed(self) -> int:
        return self.count("confirmed")

    @property
    def conclusive(self) -> int:
        """既能复现、对照组又通过的用例：只有这些能说明系统行不行。"""

        return self.count("confirmed")

    @property
    def inconclusive(self) -> int:
        return self.count("inconclusive")

    @property
    def detected(self) -> int:
        return self.count("detected")

    @property
    def repaired(self) -> int:
        return self.count("repaired")

    @property
    def false_claims(self) -> int:
        return self.count("false_claim")

    @property
    def unproven_repairs(self) -> int:
        """修好了、但补丁里有未经证明的猜测的用例数。单独报告，不混进修复率。"""

        return sum(
            1
            for item in self.cases
            if item.repaired and any(value != PROVEN_CONFIDENCE for value in item.claimed_confidences)
        )

    @property
    def proven_repairs(self) -> int:
        return self.repaired - self.unproven_repairs

    @staticmethod
    def _rate(part: int, whole: int) -> float:
        return round(part / whole, 4) if whole else 0.0

    def to_dict(self) -> Dict[str, Any]:
        conclusive = self.conclusive
        return {
            "repo": self.repo,
            "scanned_commits": self.scanned_commits,
            "git_available": self.git_available,
            "cases": self.total,
            "reproduced": self.reproduced,
            "confirmed": self.confirmed,
            "conclusive": conclusive,
            "inconclusive": self.inconclusive,
            "detected": self.detected,
            "repaired": self.repaired,
            "proven_repairs": self.proven_repairs,
            "unproven_repairs": self.unproven_repairs,
            "false_claims": self.false_claims,
            "reproduction_rate": self._rate(self.reproduced, self.total),
            # 检出率/修复率只在“有结论”的用例上算：把环境问题算进成功率是编数据
            "detection_rate": self._rate(self.detected, conclusive),
            "repair_rate": self._rate(self.repaired, conclusive),
            # 另一条更严的口径：只算「依赖关系被证明过」的成功修复
            "proven_repair_rate": self._rate(self.proven_repairs, conclusive),
            "false_claim_rate": self._rate(self.false_claims, conclusive),
            "results": [item.to_dict() for item in self.cases],
            "notes": self.notes,
        }

    def summary_line(self) -> str:
        line = (
            f"真实缺陷 {self.total} 例：可复现 {self.reproduced}、有结论 {self.conclusive}"
            f"（对照组也失败的 {self.inconclusive} 例不计入）、系统检出 {self.detected}、"
            f"真正修好 {self.repaired}、自称成功但没修好 {self.false_claims}"
        )
        if self.unproven_repairs:
            line += f"（其中 {self.unproven_repairs} 例靠未经证明的猜测修好，可信修复 {self.proven_repairs} 例）"
        return line

    def markdown(self) -> str:
        lines = ["## 真实缺陷基准（来自真实项目 git 历史）", ""]
        lines.append(f"- 仓库：`{self.repo}`")
        lines.append(f"- 采样范围：最近 {self.scanned_commits} 次提交里的修复提交")
        lines.append(f"- {self.summary_line()}")
        lines.append("")
        lines.append("| 用例 | 修复提交 | 可复现 | 对照组 | 系统检出 | 真正修好 | 自称成功 | 改动的文件 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for item in self.cases:
            lines.append(
                f"| {item.case.subject[:40]} | `{item.case.sha[:8]}` | "
                f"{_yes_no(item.reproduced)} | {_yes_no(item.confirmed)} | {_yes_no(item.detected)} | "
                f"{_yes_no(item.repaired)} | {_yes_no(item.claimed_success)} | "
                f"{'、'.join(item.files_touched) or '（无）'} |"
            )
        lines.append("")
        lines.append(
            "“可复现”指把修复提交带的测试放到父提交上跑确实失败；"
            "“对照组”指同一测试在修复提交上通过；"
            "“真正修好”指系统跑完之后这些测试通过。"
        )
        lines.append(
            f"修复里依据被证明过的有 {self.proven_repairs} 例、依据未经证明（靠猜）的有 "
            f"{self.unproven_repairs} 例；后者即便测试通过，也不能算作可信修复。"
        )
        for note in self.notes:
            lines.append(f"- 说明：{note}")
        lines.append("")
        return "\n".join(lines)


def _yes_no(value: Optional[bool]) -> str:
    if value is None:
        return "未测"
    return "是" if value else "否"


class GitError(RuntimeError):
    pass


class GitRepo:
    """只封装本模块用到的 git 操作，全部走真实 git 命令。"""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        if not shutil.which("git"):
            raise GitError("git 不可用")
        if not os.path.isdir(os.path.join(self.path, ".git")):
            raise GitError(f"{self.path} 不是 git 仓库")

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        completed = subprocess.run(
            ["git", "-C", self.path, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if check and completed.returncode != 0:
            raise GitError(f"git {' '.join(args)} 失败：{completed.stderr.strip()}")
        return completed

    def log(self, limit: int) -> List[Tuple[str, str, str]]:
        fmt = "%H%x1f%ad%x1f%s"
        output = self.run("log", "--no-merges", f"-n{limit}", "--date=short", f"--format={fmt}").stdout
        records = []
        for line in output.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 3:
                records.append((parts[0], parts[1], parts[2]))
        return records

    def changed_files(self, sha: str) -> List[str]:
        output = self.run("show", "--name-only", "--pretty=format:", sha).stdout
        return [line.strip() for line in output.splitlines() if line.strip()]

    def short_sha(self, sha: str) -> str:
        return self.run("rev-parse", "--short", sha).stdout.strip()

    def worktree_add(self, directory: str, sha: str) -> None:
        self.run("worktree", "add", "--detach", directory, sha)

    def worktree_remove(self, directory: str) -> None:
        self.run("worktree", "remove", "--force", directory, check=False)

    def restore_paths(self, directory: str, sha: str, paths: List[str]) -> None:
        """把修复提交里的测试文件取到父提交的检出目录里（复现缺陷的标准做法）。"""

        if not paths:
            return
        subprocess.run(
            ["git", "-C", directory, "checkout", sha, "--", *paths],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )


class DefectSampler:
    """从真实 git 历史里挑出“修 bug 且带测试”的提交作为基准用例。"""

    def __init__(self, repo: GitRepo, scan: int = DEFAULT_SCAN):
        self._repo = repo
        self._scan = max(1, scan)

    def sample(self, limit: int) -> List[DefectCase]:
        cases: List[DefectCase] = []
        for sha, date, subject in self._repo.log(self._scan):
            if len(cases) >= limit:
                break
            if not FIX_SUBJECT_RE.match(subject or ""):
                continue
            files = self._repo.changed_files(sha)
            test_files = [name for name in files if TEST_PATH_RE.search(name)]
            source_files = [name for name in files if name.endswith(SOURCE_SUFFIXES) and not TEST_PATH_RE.search(name)]
            if not test_files or not source_files:
                # 没有测试就无从复现，没有源码就不是代码缺陷：跳过，不当成用例
                continue
            parent = self._repo.run("rev-parse", f"{sha}^", check=False)
            if parent.returncode != 0:
                continue
            cases.append(
                DefectCase(
                    sha=sha,
                    parent=parent.stdout.strip(),
                    subject=(subject or "").strip(),
                    date=date,
                    source_files=source_files,
                    test_files=test_files,
                )
            )
        return cases


class TestRunner:
    """用项目自己的测试来判断缺陷是否存在——这是本基准唯一的事实来源。"""

    def __init__(
        self,
        interpreter: Optional[str] = None,
        timeout: int = DEFAULT_CASE_TIMEOUT,
        temp_dir: Optional[str] = None,
        log=None,
    ):
        self._interpreter = interpreter or sys.executable
        self._timeout = max(1, timeout)
        self._temp_dir = temp_dir
        self._log = log or (lambda message: None)

    def run(self, worktree: str, test_files: List[str]) -> TestOutcome:
        if not test_files:
            return TestOutcome(returncode=0, tail="（该提交没有测试文件，无法判定）")

        env = dict(os.environ)
        # 兼容 src/ 布局：把仓库根与 src/ 都放进 PYTHONPATH，避免污染已安装的包
        env["PYTHONPATH"] = os.pathsep.join([os.path.join(worktree, "src"), worktree, env.get("PYTHONPATH", "")]).strip(
            os.pathsep
        )
        env["PYTHONIOENCODING"] = "utf-8"
        if self._temp_dir:
            # 受限环境下系统临时目录可能不可写；项目测试一碰 tempfile 就会
            # PermissionError，那是环境问题而不是缺陷，必须排除
            os.makedirs(self._temp_dir, exist_ok=True)
            for name in ("TEMP", "TMP", "TMPDIR"):
                env[name] = self._temp_dir
        command = [self._interpreter, "-m", "pytest", *test_files, "-q", "-p", "no:cacheprovider"]
        self._log(f"    $ {' '.join(command)}")
        try:
            completed = subprocess.run(
                command,
                cwd=worktree,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return TestOutcome(returncode=-1, timed_out=True, tail=f"（超时 {self._timeout}s）")

        output = f"{completed.stdout}\n{completed.stderr}"
        return TestOutcome(
            returncode=completed.returncode,
            passed=_extract_count(output, "passed"),
            failed=_extract_count(output, "failed"),
            errors=_extract_count(output, "error"),
            tail="\n".join(output.strip().splitlines()[-TAIL_LINES:]),
        )


COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|error[s]?|skipped)")


def _extract_count(output: str, keyword: str) -> int:
    token = "error" if keyword in ("error", "errors") else keyword
    matches = [int(value) for value, name in COUNT_RE.findall(output) if name.startswith(token)]
    return max(matches) if matches else 0


class RealDefectBenchmark:
    """把系统指向真实仓库的历史缺陷，记录它自称做了什么、实际做到了什么。"""

    def __init__(
        self,
        repo_path: str,
        work_root: str,
        cases: int = 5,
        interpreter: Optional[str] = None,
        timeout: int = DEFAULT_CASE_TIMEOUT,
        control: bool = True,
        scan: int = DEFAULT_SCAN,
        log: Optional[Callable[[str], None]] = None,
    ):
        self._repo_path = os.path.abspath(repo_path)
        self._work_root = os.path.abspath(work_root)
        self._cases = max(1, cases)
        self._timeout = timeout
        self._control = control
        self._scan = scan
        self._log = log or (lambda message: None)
        self._runner = TestRunner(
            interpreter=interpreter,
            timeout=timeout,
            temp_dir=os.path.join(self._work_root, "_tmp"),
            log=self._log,
        )

    def run(self) -> BenchmarkReport:
        os.makedirs(self._work_root, exist_ok=True)
        try:
            repo = GitRepo(self._repo_path)
        except GitError as error:
            report = BenchmarkReport(repo=self._repo_path, git_available=False)
            report.notes.append(f"git 不可用或目标不是仓库：{error}。本基准不做任何推断。")
            return report

        report = BenchmarkReport(repo=self._repo_path, scanned_commits=self._scan)
        selected = DefectSampler(repo, scan=self._scan).sample(self._cases)
        if not selected:
            report.notes.append("没有采到“修复提交同时改源码与测试”的用例，本次不下结论。")
            return report

        for index, case in enumerate(selected, 1):
            self._log(f"  [{index}/{len(selected)}] {case.sha[:8]} {case.subject}")
            report.cases.append(self._run_case(repo, case))
        if report.inconclusive:
            report.notes.append(
                f"{report.inconclusive} 例的对照组（修复提交）也失败：通常是环境或依赖问题"
                "（本机 Python 版本、受限的临时目录、缺依赖），已排除出检出率与修复率的分母。"
            )
        return report

    # ------------------------------------------------------------------ 单例

    def _run_case(self, repo: GitRepo, case: DefectCase) -> CaseResult:
        result = CaseResult(case=case)
        parent_wt = os.path.join(self._work_root, f"parent_{case.sha[:8]}")
        fixed_wt = os.path.join(self._work_root, f"fixed_{case.sha[:8]}")
        self._cleanup(repo, parent_wt, fixed_wt)

        try:
            repo.worktree_add(parent_wt, case.parent)
            # 复现缺陷的标准做法：把修复提交带的测试拿到父提交上跑
            repo.restore_paths(parent_wt, case.sha, case.test_files)
            result.parent_outcome = self._runner.run(parent_wt, case.test_files)
            result.reproduced = result.parent_outcome.did_fail

            before = _snapshot(parent_wt)
            claims = self._run_system(parent_wt)
            after = _snapshot(parent_wt)
            result.claimed_pain_points = claims["pain_points"]
            result.claimed_locations = claims["locations"]
            result.claimed_success = claims["success"]
            result.claimed_verifications = claims["verifications"]
            result.claimed_confidences = claims["confidences"]
            result.files_touched = sorted(set(before) ^ set(after)) + sorted(
                path for path in set(before) & set(after) if before[path] != after[path]
            )

            if result.reproduced:
                # 系统跑完之后缺陷还在吗？——这才是“修好了没有”的唯一判据
                result.after_system_outcome = self._runner.run(parent_wt, case.test_files)
                result.repaired = result.after_system_outcome.did_pass

            result.detected = self._detected(case, claims["locations"], result.files_touched)
            if self._control:
                repo.worktree_add(fixed_wt, case.sha)
                result.control_outcome = self._runner.run(fixed_wt, case.test_files)
                result.confirmed = bool(result.reproduced) and result.control_outcome.did_pass
                if result.control_outcome.did_fail:
                    # 连修复提交上都失败：这个测试区分的不是缺陷，可能是环境/依赖问题，
                    # 该用例不能用来评价系统，标记为无结论并排除出分母
                    result.inconclusive = True
                    result.notes.append(
                        "对照组（修复提交）同样失败，无法证明该测试能区分缺陷，本用例不计入结论："
                        f"{result.control_outcome.tail.splitlines()[-1] if result.control_outcome.tail else ''}"
                    )
            elif result.reproduced:
                result.notes.append("未跑对照组（--no-control）")

            if claims["success"] and not result.repaired and not result.inconclusive:
                result.false_claim = True
                result.notes.append("系统自称执行/验证成功，但这些测试仍然失败")
            if not result.detected and claims["locations"]:
                result.notes.append(f"系统报出的位置是 {claims['locations']}，与缺陷文件无关")
            if not claims["locations"] and claims["pain_points"] == 0:
                result.notes.append("系统的 Perceiver 没有报出任何问题")
            unproven = sorted({value for value in claims["confidences"] if value != PROVEN_CONFIDENCE})
            if unproven:
                result.notes.append(
                    f"补丁的依据未经证明（{', '.join(unproven)}）：即便测试通过，也只是猜对了，" "不能算作可信修复"
                )
        finally:
            self._cleanup(repo, parent_wt, fixed_wt)
        return result

    def _run_system(self, worktree: str) -> Dict[str, Any]:
        """跑一遍系统的 感知->提议->执行/验证 流程，只记录它声称了什么。

        补丁落在基准自己创建的一次性 worktree 里（``apply_to``）：系统不会碰别的地方，
        随后由基准**独立地**重跑测试来判定到底修好没有——那个判据不经过系统自己。

        一次只应用第一个 pain point 的补丁：多个补丁同时落地时，测试失败无法归因，
        会把不是它的账算到它头上。
        """

        from hermes.core.plugin_manager import PluginManager

        plugin = PluginManager().get_plugin("default")
        if plugin is None:
            return {"pain_points": 0, "locations": [], "success": False, "verifications": [], "confidences": []}

        # 把可写的临时目录交给系统：本机 %TEMP% 不可写，一碰 tempfile 的测试会假失败
        context = plugin.create_context(worktree, pytest_tmpdir=os.path.join(self._work_root, "_tmp"))
        perceiver, sage, knight = plugin.get_perceiver(), plugin.get_sage(), plugin.get_knight()
        pain_points = perceiver.detect(context)
        locations = sorted({str(item.location) for item in pain_points if item.location})
        success = False
        verifications: List[str] = []
        confidences: List[str] = []
        for pain_point in list(pain_points)[:1]:
            plan = sage.generate_patch(context, pain_point)
            confidences.extend(str(operation.get("confidence", "unknown")) for operation in plan.operations)
            execution = knight.execute(context, plan, apply_to=worktree)
            verification = knight.verify(context, plan)
            success = success or bool(execution.success) or bool(verification.success)
            verifications.extend(
                f"{item.get('test', item.get('operation', ''))}={item.get('status', '')}"
                for item in verification.verification_results
            )
        return {
            "pain_points": len(pain_points),
            "locations": locations,
            "success": success,
            "verifications": verifications,
            "confidences": confidences,
        }

    @staticmethod
    def _detected(case: DefectCase, locations: List[str], touched: List[str]) -> bool:
        """检出 = 报出的位置确实指向缺陷文件，或者系统真的动到了缺陷文件。"""

        wanted = set(case.source_files)
        for path in list(locations) + list(touched):
            normalized = str(path).replace("\\", "/").lstrip("./")
            if normalized in wanted or any(item.endswith(normalized) for item in wanted):
                return True
        return False

    def _cleanup(self, repo: GitRepo, *worktrees: str) -> None:
        for worktree in worktrees:
            if os.path.isdir(worktree):
                repo.worktree_remove(worktree)
            shutil.rmtree(worktree, ignore_errors=True)


def _snapshot(root: str) -> Dict[str, str]:
    """给工作区拍指纹，用来判断“系统到底有没有改到文件”。

    缓存目录（``__pycache__``、``.pytest_cache``）不算：跑测试本来就会生成它们，
    把它们算成“系统改动的文件”会让检出率虚高。
    """

    snapshot: Dict[str, str] = {}
    for base, directories, files in os.walk(root):
        directories[:] = [name for name in directories if name not in CACHE_DIRECTORIES]
        for name in files:
            path = os.path.join(base, name)
            relative = os.path.relpath(path, root).replace("\\", "/")
            try:
                with open(path, "rb") as handle:
                    snapshot[relative] = hashlib.md5(handle.read()).hexdigest()
            except OSError:
                continue
    return snapshot
