import json

import pytest

import app as app_module
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Evolution-1" in response.data
    assert b"simCanvas" in response.data


def test_api_history(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)


def test_api_best(client):
    response = client.get("/api/best")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, dict)


def test_api_random(client):
    response = client.get("/api/random")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "fitness" in data
    assert isinstance(data["fitness"], (int, float))


def test_api_training_status(client):
    response = client.get("/api/training/status")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "is_training" in data
    assert "population_size" in data
    assert "mutation_rate" in data
    assert "training_enabled" in data


def test_training_start_disabled(client, monkeypatch):
    monkeypatch.setitem(app_module.training_state, "training_enabled", False)
    response = client.post("/api/training/start", json={})
    assert response.status_code == 403
    assert app_module.training_state["is_training"] is False


REPLAY_MODELS = ["best", "universal", "random"]  # the dashboard's model switcher


@pytest.mark.parametrize("model", REPLAY_MODELS)
def test_replay_files(model):
    from scripts.record_replay import FIELDS, OUT_DIR

    with open(f"{OUT_DIR}/{model}.json", encoding="utf-8") as f:
        replay = json.load(f)
    assert replay["fields"] == FIELDS
    assert 0 < len(replay["frames"]) == replay["steps"] <= 600
    assert all(len(frame) == len(FIELDS) for frame in replay["frames"])
    assert len(replay["terrain"]) >= 2


def test_replay_static_route(client):
    response = client.get("/static/replays/best.json")
    assert response.status_code == 200
    assert response.cache_control.max_age > 0
    assert json.loads(response.data)["model"] == "best"
    response.close()


def test_live_sim_disabled_in_demo_mode(client, monkeypatch):
    monkeypatch.setitem(app_module.training_state, "training_enabled", False)
    assert client.get("/api/stream/sim").status_code == 403
    assert client.get("/api/random").status_code == 403
