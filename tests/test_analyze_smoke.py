import json
import os

import pytest

from scripts.analyze import main as analyze
from src.research.config import ExperimentConfig
from src.research.report import FIGURES
from src.research.runner import run_experiment
from tests.test_runner_smoke import TINY


@pytest.fixture(scope="module")
def tiny_exp(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    run_experiment(ExperimentConfig.from_dict({**TINY, "seeds": [0, 1, 2]}), str(out), log=lambda *_: None)
    return os.path.join(str(out), "tiny")


def test_analyze_writes_every_figure_and_report(tiny_exp):
    analyze([tiny_exp])
    out = os.path.join(tiny_exp, "summary")
    for f in FIGURES:
        for ext in (".png", ".svg", ".dark.svg"):
            assert os.path.exists(os.path.join(out, "figures", f + ext)), f + ext
    with open(os.path.join(out, "REPORT.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert all(f"](figures/{f}.png)" in text for f in FIGURES)
    assert "## Reproducibility" in text and "git_commit:" in text and "config_sha256:" in text
    assert "| H1 | cpg > zeros |" in text
    summary = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    assert summary["confirmatory"][0]["verdict"] in ("supported", "not supported", "inconclusive")
    assert set(summary["variants"]) == {"cpg", "zeros"}
    for name in ("summary.csv", "pairwise.csv"):
        assert os.path.getsize(os.path.join(out, name)) > 0
