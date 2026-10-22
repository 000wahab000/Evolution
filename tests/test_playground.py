import os

from src.research import playground


def test_checkpoint_resume_continues_generations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    small = dict(seed=0, cpg=True, mutation_rate=0.15, population=4, run_name="t", chart_every=0,
                 checkpoint_every=1, log=lambda *_: None)
    _, _, hist = playground.train(generations=2, **small)
    assert len(hist["best"]) == 2 and os.path.exists(playground.ckpt_path("t", 0))
    _, champ, hist = playground.train(generations=2, keep_training=True, **small)
    assert len(hist["best"]) == 4 and len(champ) == 324
    _, _, hist = playground.train(generations=1, **small)  # no keep_training: starts over
    assert len(hist["best"]) == 1


def test_pretrained_start_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(playground, "PRETRAINED", os.path.abspath(playground.PRETRAINED))
    monkeypatch.chdir(tmp_path)
    _, champ, _ = playground.train(seed=0, cpg=True, mutation_rate=0.15, generations=1, population=4,
                                   run_name="p", start_from="pretrained", chart_every=0, log=lambda *_: None)
    assert len(champ) == 324
