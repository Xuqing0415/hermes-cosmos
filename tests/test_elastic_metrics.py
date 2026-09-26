"""elastic_multiprocess 的指标路径契约，以及 worker 报出来的吞吐是不是真的。

修掉的两个缺陷都长在「路径」上：

1. `coordinator.start_worker` 里写的是 ``sys.executable if 'sys' in dir() else 'python'``。
   该文件当时没有在顶部 import sys（只在文件最底下 import 了一次），而方法里的 ``dir()``
   是**局部**名字，永远不含 sys —— 所以恒定退化成字符串 'python'。
2. worker 把指标写成相对路径，相对的是它自己的 cwd（Popen(cwd='elastic_multiprocess')）；
   coordinator 用同名的相对路径去读，相对的是启动它的目录。两边指向不同文件，
   ``get_metrics()`` 永远返回空，auto_scaler 因此从来没有真正伸缩过。

所以这里断言的是**契约**：谁写、写哪里、谁读、读哪里，必须是同一个路径。

注意：``elastic_multiprocess/`` 整个目录被 .gitignore 忽略（第 53 行），从未进过版本库，
所以本文件在 CI 上会被整体跳过 —— 这是事实的如实反映，不是测试写错了。
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
import uuid

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELASTIC_DIR = os.path.join(REPO_ROOT, "elastic_multiprocess")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ELASTIC_DIR),
    reason="elastic_multiprocess/ 被 .gitignore 忽略、不在版本库里，CI 上不存在这个目录",
)


@pytest.fixture
def metrics_dir():
    """临时指标目录（不用 tmp_path：受限环境下系统临时目录不可写）。"""

    base = os.path.join(REPO_ROOT, ".tmp_test", "elastic_metrics")
    path = os.path.join(base, "run_" + uuid.uuid4().hex)
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _coordinator_module():
    # 用 importlib 而不是 `import a.b as c`：elastic_multiprocess/__init__.py 里有
    # `from .train_worker import train_worker`，包属性名被那个函数盖掉了，
    # 属性访问拿到的是函数不是模块。
    return importlib.import_module("elastic_multiprocess.coordinator")


def _worker_module():
    return importlib.import_module("elastic_multiprocess.train_worker")


class TestMetricsPathContract:
    def test_worker_writes_where_the_coordinator_reads(self, monkeypatch):
        coordinator = _coordinator_module()
        train_worker = _worker_module()

        # coordinator 拉起 worker 时就是这么设的
        monkeypatch.setenv(coordinator.METRICS_ENV, str(coordinator.METRICS_DIR))

        for rank in (0, 1, 7):
            expected = coordinator.METRICS_DIR / coordinator.metrics_filename(rank)
            assert pathlib.Path(train_worker.metrics_path(rank)) == expected

    def test_both_sides_fall_back_to_the_same_directory(self, monkeypatch):
        coordinator = _coordinator_module()
        train_worker = _worker_module()

        monkeypatch.delenv(coordinator.METRICS_ENV, raising=False)

        worker_default = pathlib.Path(train_worker.metrics_path(0)).parent
        assert worker_default == coordinator.METRICS_DIR.parent / "metrics"

    def test_start_worker_uses_sys_executable_and_hands_over_the_directory(self, monkeypatch):
        """旧代码这里会拿到字符串 'python'，且不会传任何指标目录。"""

        coordinator = _coordinator_module()
        captured = {}

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env") or {}
            captured["cwd"] = kwargs.get("cwd")
            return object()

        monkeypatch.setattr(coordinator.subprocess, "Popen", fake_popen)
        instance = coordinator.ElasticCoordinator()
        instance.world_size = 2
        instance.start_worker(1)

        assert captured["cmd"][0] == sys.executable
        assert captured["env"][coordinator.METRICS_ENV] == str(instance.metrics_dir)
        assert captured["env"][coordinator.STOP_ENV] == str(instance.stop_signal)
        assert captured["cwd"] == str(instance.package_dir)

    def test_stop_signal_is_an_absolute_shared_path(self):
        """worker 的 cwd 是包目录，coordinator 的不是 —— 相对路径的停止信号也送不到。"""

        coordinator = _coordinator_module()
        instance = coordinator.ElasticCoordinator()

        assert pathlib.Path(instance.stop_signal).is_absolute()
        assert pathlib.Path(instance.stop_signal).parent == pathlib.Path(ELASTIC_DIR).resolve()


class TestWorkerReportsRealThroughput:
    @pytest.mark.timeout(120)
    def test_a_real_worker_lands_its_metrics_in_the_injected_directory(self, metrics_dir, monkeypatch):
        pytest.importorskip("torch")
        train_worker = _worker_module()

        stop_signal = os.path.join(metrics_dir, "STOP_SIGNAL")
        env = os.environ.copy()
        env[train_worker.METRICS_ENV] = metrics_dir
        env[train_worker.STOP_ENV] = stop_signal
        # 本进程也照同一条规则解析，这样「期望路径」和「worker 实际写的路径」是同源推导的
        monkeypatch.setenv(train_worker.METRICS_ENV, metrics_dir)

        process = subprocess.Popen(
            [
                sys.executable,
                os.path.join(ELASTIC_DIR, "train_worker.py"),
                "--rank",
                "0",
                "--world_size",
                "1",
                "--master_port",
                str(_free_port()),
            ],
            cwd=ELASTIC_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            expected = pathlib.Path(train_worker.metrics_path(0))
            assert expected == pathlib.Path(metrics_dir) / "metrics_worker_0.json"

            deadline = time.time() + 60
            while not expected.exists() and time.time() < deadline and process.poll() is None:
                time.sleep(0.2)

            assert expected.exists(), "worker 没有把指标写进 coordinator 指定的目录"
            payload = json.loads(expected.read_text(encoding="utf-8"))

            # 旧代码这里是写死的 32 / 0.05 = 640.0（不是量出来的）
            assert payload["samples_per_second"] > 0
            assert payload["worker_id"] == 0
        finally:
            with open(stop_signal, "w", encoding="utf-8") as handle:
                handle.write("stop")
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()

    def test_the_fallback_branch_does_not_invent_a_throughput_number(self):
        """没有 PyTorch 时不报吞吐：编一个数字会让 auto_scaler 按不存在的指标伸缩。"""

        source = open(os.path.join(ELASTIC_DIR, "train_worker.py"), encoding="utf-8").read()
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_worker")
        branch = next(
            node for node in ast.walk(function) if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
        )
        keys = {
            key.value
            for node in ast.walk(ast.Module(body=branch.orelse, type_ignores=[]))
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant)
        }

        assert "samples_per_second" not in keys
        # 有 PyTorch 的那一支必须真的报吞吐，否则这个测试就是在测一个空文件
        assert "samples_per_second" in source
        # 旧代码在同名位置写的是写死的 32 / 0.05，两个分支都不许再出现
        assert "32 / 0.05" not in source
