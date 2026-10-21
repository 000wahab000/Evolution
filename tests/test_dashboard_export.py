import json
import os
import shutil

from app import app
from scripts.analyze import main as analyze
from scripts.export_dashboard import BUDGET_KB, export
from src.research.config import ExperimentConfig
from src.research.runner import run_experiment
from tests.test_runner_smoke import TINY


def test_export_bundle_is_valid_and_small(tmp_path):
    out = str(tmp_path / "results")
    run_experiment(ExperimentConfig.from_dict({**TINY, "seeds": [0, 1, 2]}), out, log=lambda *_: None)
    analyze([os.path.join(out, "tiny")])
    static = str(tmp_path / "static")
    name, kb = export(os.path.join(out, "tiny"), static=static)

    assert name == "tiny" and kb < BUDGET_KB
    bundle = os.path.join(static, "experiments", "tiny")
    idx = json.load(open(os.path.join(bundle, "index.json"), encoding="utf-8"))
    assert set(idx["style"]) == {"cpg", "zeros"}
    assert idx["verdicts"][0]["id"] == "H1"
    for f in idx["figures"].values():
        assert os.path.exists(os.path.join(bundle, f))
    for rel in idx["replays"].values():
        replay = json.load(open(os.path.join(static, "replays", rel + ".json"), encoding="utf-8"))
        assert replay["frames"] and "median seed" in replay["label"]
    curves = json.load(open(os.path.join(bundle, "curves.json"), encoding="utf-8"))
    assert len(curves["episodes"]) == len(curves["cpg"]["iqm"])
    listing = json.load(open(os.path.join(static, "experiments", "index.json"), encoding="utf-8"))
    assert listing["experiments"][0]["name"] == "tiny"


def test_flask_serves_experiment_bundle_and_tab():
    d = os.path.join(app.static_folder, "experiments", "_pytest_bundle")
    os.makedirs(d, exist_ok=True)
    try:
        with open(os.path.join(d, "index.json"), "w") as f:
            json.dump({"experiment": "_pytest_bundle"}, f)
        with app.test_client() as client:
            r = client.get("/static/experiments/_pytest_bundle/index.json")
            ok = r.status_code == 200 and r.get_json()["experiment"] == "_pytest_bundle"
            r.close()  # Windows: release the file before cleanup
            assert ok
            page = client.get("/").data
            assert b'id="tabbtn-experiments"' in page and b'id="tab-experiments"' in page
    finally:
        shutil.rmtree(d)
        parent = os.path.dirname(d)
        if not os.listdir(parent):
            os.rmdir(parent)
