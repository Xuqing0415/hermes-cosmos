"""「模拟」这件事必须能被机器查到，并且真的会改变论文里的标签。

涵盖两层：

1. 标记层：``hermes_unified.self_evolution.IS_SIMULATION is True``，同目录有 MOCKED.md；
   快照的行级 metadata 用 ``source="simulation"`` 自报（``EvolutionSnapshot.is_simulated``）。
2. 传导层：``ResearchDataCollector`` 读到自报标记后，把 ``snapshots`` 等字段从 ``REAL``
   降级为 ``SYNTHETIC``，于是依赖这些字段的发现被评成证据等级 C（而不是 A）。
   对照组：没有标记的快照仍然算 ``REAL``——不能因为引入了这一层就把所有数据都降级。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import uuid

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from hermes.cross_domain.evolution_tracker import (  # noqa: E402
    SIMULATED_SOURCE,
    SOURCE_METADATA_KEY,
    EvolutionSnapshot,
    EvolutionTracker,
)
from hermes.self_research.data_provenance import REAL, SYNTHETIC, DataProvenanceTagger  # noqa: E402
from hermes.self_research.evidence_grader import GRADE_A, GRADE_C, EvidenceGrader  # noqa: E402
from hermes.self_research.research_data_collector import ResearchDataCollector  # noqa: E402
from hermes_unified.self_evolution import IS_SIMULATION  # noqa: E402

SELF_EVOLUTION_DIR = pathlib.Path(REPO_ROOT) / "hermes_unified" / "self_evolution"


@pytest.fixture
def work_dir():
    """临时工作目录（不用 tmp_path：受限环境下系统临时目录不可写）。"""

    base = os.path.join(REPO_ROOT, ".tmp_test", "simulation_provenance")
    path = pathlib.Path(os.path.join(base, "run_" + uuid.uuid4().hex))
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _snapshot(metadata):
    return EvolutionSnapshot(
        id=1,
        snapshot_index=0,
        success_rate=0.8,
        cross_domain_success=0.6,
        pattern_coverage=0.7,
        knowledge_graph_hash="deadbeef",
        total_patterns=1,
        total_relationships=0,
        policy_mutation_rate=0.1,
        policy_exploration_factor=0.15,
        timestamp=__import__("datetime").datetime(2026, 1, 1),
        metadata=metadata,
    )


def _collector_for(work_dir, metadata):
    """造一个只含一条快照的快照库，交给采集器。"""

    db_path = work_dir / "evolution_tracker.db"
    tracker = EvolutionTracker(str(db_path))
    tracker.record_snapshot(
        success_rate=0.8,
        cross_domain_success=0.6,
        pattern_coverage=0.7,
        knowledge_amalgamator=None,
        metadata=metadata,
    )
    return ResearchDataCollector(
        base_dir=str(work_dir),
        snapshot_db=str(db_path),
        policy_path=str(work_dir / "missing_policy.json"),
        notification_log=str(work_dir / "missing.log"),
        graph_path=str(work_dir / "missing_graph.json"),
    )


class TestSimulationIsMachineReadable:
    def test_the_self_evolution_module_declares_itself_a_simulation(self):
        assert IS_SIMULATION is True

    def test_the_mock_marker_file_is_present_and_says_what_to_do(self):
        marker = SELF_EVOLUTION_DIR / "MOCKED.md"

        assert marker.exists()
        text = marker.read_text(encoding="utf-8")
        assert "IS_SIMULATION" in text
        assert "SYNTHETIC" in text

    def test_snapshot_self_reports_simulation_through_its_metadata(self):
        flagged = _snapshot({SOURCE_METADATA_KEY: SIMULATED_SOURCE})
        plain = _snapshot({"iteration": 1})

        assert flagged.is_simulated is True
        assert plain.is_simulated is False
        assert flagged.to_dict()["is_simulated"] is True

    def test_the_demo_writer_stamps_the_flag_on_what_it_writes(self, work_dir):
        """演示数据是硬编码的，写库时必须自报——否则论文会把它当成真实运行结果。"""

        import autotestgen

        db_path = work_dir / "demo.db"
        tracker = EvolutionTracker(str(db_path))

        indices = autotestgen.record_simulated_snapshots(tracker, None)

        assert len(indices) == 5
        snapshots = tracker.get_recent_snapshots(10)
        assert len(snapshots) == 5
        assert all(snapshot.is_simulated for snapshot in snapshots)


class TestSimulatedSnapshotsAreDowngraded:
    def test_simulated_snapshots_stop_being_real(self, work_dir):
        dataset = _collector_for(work_dir, {SOURCE_METADATA_KEY: SIMULATED_SOURCE}).collect()

        assert dataset.provenance["snapshots"] == SYNTHETIC
        assert dataset.provenance["pattern_success_rates"] == SYNTHETIC
        assert dataset.provenance["mental_model"] == SYNTHETIC
        assert "模拟" in dataset.provenance_notes["snapshots"]

    def test_unflagged_snapshots_are_still_real(self, work_dir):
        dataset = _collector_for(work_dir, {"iteration": 1}).collect()

        assert dataset.provenance["snapshots"] == REAL
        assert dataset.provenance["pattern_success_rates"] == REAL

    def test_findings_that_depend_on_simulated_snapshots_are_graded_c(self, work_dir):
        dataset = _collector_for(work_dir, {SOURCE_METADATA_KEY: SIMULATED_SOURCE}).collect()
        report = DataProvenanceTagger().tag(dataset)
        finding = {"finding_id": "F-01", "statement": "成功率从 0.81 提升到 0.90", "data_fields": ["snapshots"]}

        assert EvidenceGrader().grade(finding, report).grade == GRADE_C

    def test_the_same_finding_keeps_grade_a_without_the_marker(self, work_dir):
        dataset = _collector_for(work_dir, {"iteration": 1}).collect()
        report = DataProvenanceTagger().tag(dataset)
        finding = {"finding_id": "F-01", "statement": "成功率从 0.81 提升到 0.90", "data_fields": ["snapshots"]}

        assert EvidenceGrader().grade(finding, report).grade == GRADE_A
