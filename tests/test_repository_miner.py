import os

import pytest

from hermes.cross_domain import RepositoryMiner


class TestRepositoryMinerLocalMining:
    def test_mine_local_repo_honors_max_events(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(os.path.join(repo_root, ".git")):
            pytest.skip("git working tree not available")

        miner = RepositoryMiner()
        report = miner.mine(repo_root, max_events=3)

        assert report.mining_method == "git_log"
        assert 1 <= report.total_events <= 3

    def test_mine_local_repo_falls_back_for_non_repo(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        missing_repo = os.path.join(repo_root, "_no_such_repo_")
        miner = RepositoryMiner()
        report = miner.mine(missing_repo)

        assert report.mining_method != "git_log"
