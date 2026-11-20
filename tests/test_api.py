"""API 冒烟测试:页面、示例、计划、续排、失败、校验错误。"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
client = TestClient(app)


def example_body():
    def load(name):
        return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))

    return {
        "topology": load("topology.json"),
        "routes": load("routes.json"),
        "requirements": load("requirements.json"),
    }


def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "换表轮次规划与补测" in r.text
    assert "推演故障场景" in r.text


def test_example_endpoint():
    r = client.get("/api/example")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"topology", "routes", "requirements", "diagnose"}
    assert body["diagnose"]["candidates"]


def test_plan_ok():
    r = client.post("/api/plan", json={**example_body(), "switched": []})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert len(d["rounds"]) == 3
    assert sorted(r0 for rnd in d["rounds"] for r0 in rnd) == sorted(
        ["R1", "R2", "R3", "R4", "R5"]
    )


def test_plan_replan_with_switched():
    r = client.post("/api/plan", json={**example_body(), "switched": ["R4", "R5"]})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["switched"] == ["R4", "R5"]
    assert len(d["rounds"]) == 2


def test_plan_failure_shape():
    body = example_body()
    body["requirements"]["requirements"][0]["allowed_egress"] = ["E2"]
    r = client.post("/api/plan", json={**body, "switched": []})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "failed"
    assert d["reason"] == "current_unsafe"
    assert d["failure_state"] == []
    f0 = d["failures"][0]
    assert f0["verdict"] == "wrong_egress"
    assert f0["verdict_zh"] == "错出口"
    assert f0["path"][0]["router"] == "R1"


def test_validation_error_returns_400():
    body = example_body()
    body["requirements"]["requirements"][0]["allowed_egress"] = ["NOPE"]
    r = client.post("/api/plan", json={**body, "switched": []})
    assert r.status_code == 400
    assert "不存在的出口" in r.json()["detail"]


# ------------------------------------------------------------ 故障定位 API

def diagnose_body(**extra):
    topo, routes = example_body()["topology"], example_body()["routes"]
    body = {
        "topology": topo,
        "routes": routes,
        "switched": ["R4", "R5"],
        "measurements": [{"ingress": "R1", "target": "10.1.0.5", "result": "blackhole"}],
        "candidates": [{"ingress": r, "target": "10.1.0.5"}
                       for r in ["R1", "R2", "R3", "R4", "R5"]],
    }
    body.update(extra)
    return body


def test_diagnose_ok():
    r = client.post("/api/diagnose", json=diagnose_body())
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert len(d["groups"]) == 3
    assert [p["label"] for p in d["min_probes"]] == ["R2 → 10.1.0.5", "R3 → 10.1.0.5"]
    assert len(d["expected"]) == 3


def test_diagnose_no_scenario():
    r = client.post("/api/diagnose", json=diagnose_body(
        measurements=[{"ingress": "R1", "target": "10.1.0.5", "result": "loop"}]))
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "no_scenario"
    assert "观测与假设不符" in d["message"]


def test_diagnose_without_candidates():
    r = client.post("/api/diagnose", json=diagnose_body(candidates=[]))
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok" and d["groups"] == []
    assert "尚未填写可补测项" in d["message"]


def test_diagnose_validation_errors():
    r = client.post("/api/diagnose", json=diagnose_body(
        candidates=[{"ingress": "R1", "target": "10.1.0.5"}] * 13))
    assert r.status_code == 400
    assert "最多 12 条" in r.json()["detail"]

    r = client.post("/api/diagnose", json=diagnose_body(
        measurements=[{"ingress": "R1", "target": "10.x", "result": "blackhole"}]))
    assert r.status_code == 400
    assert "合法 IPv4" in r.json()["detail"]

    r = client.post("/api/diagnose", json=diagnose_body(switched=["R9"]))
    assert r.status_code == 400
    assert "已换设备" in r.json()["detail"]
