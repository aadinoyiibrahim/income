import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(bundle_path, monkeypatch_module):
    monkeypatch_module.setenv("INCOME_MODEL_PATH", str(bundle_path))
    from src.api import app

    with TestClient(app) as c:  # context manager runs the lifespan (model load)
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _csv(df) -> bytes:
    return df.to_csv(index=False).encode()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"


def test_model_info(client):
    info = client.get("/model").json()
    assert info["trained_through"] == "2016-05" and "metrics" in info


def test_predict_uses_uploaded_file(client, transactions):
    sub = transactions[transactions.user_id.isin(["u001", "u002"])]
    r = client.post("/predict", files={"file": ("tx.csv", io.BytesIO(_csv(sub)), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["forecast_month"] == "2016-06" and body["n_users"] == 2
    assert {p["user_id"] for p in body["predictions"]} == {"u001", "u002"}


def test_predict_missing_columns(client):
    r = client.post("/predict", files={"file": ("tx.csv", io.BytesIO(b"user_id,foo\na,1\n"), "text/csv")})
    assert r.status_code == 422 and "Missing required columns" in r.json()["detail"]


def test_predict_rejects_non_csv(client):
    r = client.post("/predict", files={"file": ("tx.txt", io.BytesIO(b"x"), "text/plain")})
    assert r.status_code == 400


def test_predict_requires_file(client):
    assert client.post("/predict").status_code == 422


def test_503_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("INCOME_MODEL_PATH", str(tmp_path / "missing.joblib"))
    from src.api import app

    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", files={"file": ("a.csv", io.BytesIO(b"a"), "text/csv")}).status_code == 503
