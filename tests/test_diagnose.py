"""故障定位与补测规划的自测。

交付前核对项:
- 撤下受影响路由后按最长前缀回退转发(不是简单断路)
- 结果向量相同的场景归组;最少补测集合最小且确实能区分全部组
- 候选无法区分全部组 / 无场景符合时的提示
- 实测记录筛选场景;入口、目标、结果(出口名)都必须对上
"""
import itertools
import ipaddress
import json
from pathlib import Path

import pytest

from app.diagnose import (
    MAX_CANDIDATES,
    Candidate,
    _min_separating_set,
    _prepare,
    _trace,
    build_scenarios,
    diagnose,
    diagnose_payload,
    parse_candidates,
    parse_measurements,
)
from app.planner import InputError, build_network

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_example():
    def load(name):
        return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))

    return load("topology.json"), load("routes.json")


def model():
    topo, routes = load_example()
    return build_network(topo, routes)


def cand(ingress, target="10.1.0.5"):
    return {"ingress": ingress, "target": target}


def meas(ingress, result, target="10.1.0.5", egress=None):
    d = {"ingress": ingress, "target": target, "result": result}
    if egress:
        d["egress"] = egress
    return d


FIVE = [cand(r) for r in ["R1", "R2", "R3", "R4", "R5"]]


def group_of(res, scenario_key):
    return next((g for g in res["groups"] if scenario_key in g["scenarios"]), None)


def expected_of(res, group_id):
    return next(e for e in res["expected"] if e["group"] == group_id)


# ------------------------------------------------------------------ 场景

def test_scenarios_cover_base_links_and_egresses():
    m = model()
    sc = build_scenarios(m)
    keys = [s.key for s in sc]
    assert keys[0] == "base"
    assert len(sc) == 1 + len(m.links) + len(m.egresses)
    assert {"link:R1-R2", "link:R4-R5", "egress:E1", "egress:E2"} <= set(keys)


# ------------------------------------------------------- 撤路后最长前缀回退

def _fallback_routes(**overrides):
    topo, routes = load_example()
    for router, table in overrides.items():
        routes[router]["before"] = table
    return build_network(topo, routes)


def test_withdrawal_falls_back_to_shorter_prefix_then_loops():
    """R3 同时有 /16、/24 指向 E1;E1 故障时两条都被撤下,回退到默认路由。

    R3 默认回 R2,R2 默认回 R1,R1 的 /16 又指回 R2 —— 证明撤路后确实按
    最长前缀继续转发(形成环路),而不是在 R3 直接判断路。
    """
    m = _fallback_routes(**{
        "R3": [
            {"prefix": "10.1.0.0/16", "next_hop": "198.51.100.2"},
            {"prefix": "10.1.0.0/24", "next_hop": "198.51.100.2"},
            {"prefix": "0.0.0.0/0", "next_hop": "10.0.23.1"},
        ],
        "R2": [
            {"prefix": "10.1.0.0/16", "next_hop": "10.0.23.2"},
            {"prefix": "0.0.0.0/0", "next_hop": "10.0.12.1"},
        ],
    })
    c = Candidate("R3", ipaddress.IPv4Network("10.1.0.5"))
    res = diagnose(m, set(), [], [c])
    exp = expected_of(res, group_of(res, "egress:E1")["id"])
    assert exp["results"][0]["verdict"] == "loop"
    # 同一条候选在无故障场景下正常走 E1
    base = expected_of(res, group_of(res, "base")["id"])
    assert base["results"][0]["text"] == "出口 E1"


def test_withdrawal_reaches_other_egress_via_default():
    """E1 故障时 R3 的 E1 路由被撤,R3 回退默认路由回 R2,R2/R1 默认路由

    把流量带回新路 R1→R4→R5→E2。证明撤路后不是断路,而是按最长前缀
    继续逐跳转发。
    """
    topo, routes = load_example()
    # R1/R2/R3 都不再保留指向目标的明细,只留默认路由;新路 R4/R5 已换表
    routes["R3"]["before"] = [
        {"prefix": "10.1.0.0/16", "next_hop": "198.51.100.2"},
        {"prefix": "0.0.0.0/0", "next_hop": "10.0.23.1"},
    ]
    routes["R2"]["before"] = [{"prefix": "0.0.0.0/0", "next_hop": "10.0.12.1"}]
    routes["R1"]["before"] = [{"prefix": "0.0.0.0/0", "next_hop": "10.0.14.2"}]
    m = build_network(topo, routes)
    c = Candidate("R3", ipaddress.IPv4Network("10.1.0.5"))
    res = diagnose(m, {"R4", "R5"}, [], [c])
    exp = expected_of(res, group_of(res, "egress:E1")["id"])
    assert exp["results"][0]["text"] == "出口 E2"
    # 无故障时同一入口仍按明细走 E1
    base = expected_of(res, group_of(res, "base")["id"])
    assert base["results"][0]["text"] == "出口 E1"


def test_no_withdrawal_in_base_scenario():
    """无故障场景不撤表:R2 入口走旧路 E1。"""
    m = model()
    res = diagnose(m, {"R4", "R5"}, [], [Candidate("R2", ipaddress.IPv4Network("10.1.0.5"))])
    exp = expected_of(res, group_of(res, "base")["id"])
    assert exp["results"][0]["text"] == "出口 E1"


def test_loop_is_reported_not_blackhole_when_default_exists():
    m = _fallback_routes(**{
        "R3": [{"prefix": "10.1.0.0/16", "next_hop": "198.51.100.2"},
               {"prefix": "0.0.0.0/0", "next_hop": "10.0.23.1"}],
        "R2": [{"prefix": "10.1.0.0/16", "next_hop": "10.0.23.2"},
               {"prefix": "0.0.0.0/0", "next_hop": "10.0.12.1"}],
    })
    sc = next(s for s in build_scenarios(m) if s.key == "egress:E1")
    resolve, tables = _prepare(m, sc)
    out = _trace(m, resolve, tables, frozenset(),
                 "R3", ipaddress.IPv4Address("10.1.0.5"))
    assert out.verdict == "loop"


# ------------------------------------------------------------ 实测筛选

def test_measurement_filters_scenarios():
    m = model()
    res = diagnose(m, {"R4", "R5"},
                   parse_measurements([meas("R1", "blackhole")], m),
                   parse_candidates(FIVE, m))
    kept = {s["key"] for s in res["scenarios_matched"]}
    # 旧表下 R1 只有一条 10.1/16 指 R2:R1-R2、R2-R3 断或 E1 断都会断路
    assert "base" not in kept
    assert {"link:R1-R2", "link:R2-R3", "egress:E1"} <= kept
    assert "link:R1-R4" not in kept


def test_measurement_exit_name_must_match():
    m = model()
    res = diagnose(m, {"R4", "R5"},
                   parse_measurements([meas("R1", "ok", egress="E2")], m),
                   parse_candidates(FIVE, m))
    # 旧表下 R1 只可能走 E1
    assert res["status"] == "no_scenario"


def test_no_scenario_message():
    m = model()
    res = diagnose(m, {"R4", "R5"},
                   parse_measurements([meas("R1", "loop")], m),
                   parse_candidates(FIVE, m))
    assert res["status"] == "no_scenario"
    assert "观测与假设不符" in res["message"]
    assert res["groups"] == [] and res["min_probes"] == []


def test_measurements_narrow_then_minimal_probe():
    m = model()
    cands = parse_candidates(FIVE, m)
    res = diagnose(m, {"R4", "R5"},
                   parse_measurements([meas("R1", "blackhole")], m), cands)
    labels = {tuple(g["labels"]) for g in res["groups"]}
    assert labels == {("链路 R1-R2 单独失效",), ("链路 R2-R3 单独失效",),
                      ("出口 E1 单独失效",)}
    assert [p["label"] for p in res["min_probes"]] == ["R2 → 10.1.0.5", "R3 → 10.1.0.5"]
    exp = {e["group"]: [x["text"] for x in e["results"]] for e in res["expected"]}
    assert sorted(exp.values()) == sorted([
        ["出口 E1", "出口 E1"],
        ["断路", "出口 E1"],
        ["断路", "断路"],
    ])


# ------------------------------------------------------------ 归组与区分

def test_groups_partition_all_matched_scenarios():
    m = model()
    res = diagnose(m, {"R4", "R5"}, [], parse_candidates(FIVE, m))
    total = sum(g["scenario_count"] for g in res["groups"])
    assert total == res["scenarios_total"]


def _vectors(m, state, cands):
    out = {}
    for sc in build_scenarios(m):
        resolve, tables = _prepare(m, sc)
        out[sc.key] = tuple(
            _trace(m, resolve, tables, state, c.ingress,
                   c.target.network_address).key for c in cands
        )
    return out


def test_min_probes_are_minimal_and_separate():
    m = model()
    cands = parse_candidates(FIVE, m)
    res = diagnose(m, {"R4", "R5"}, [], cands)
    vec = _vectors(m, frozenset({"R4", "R5"}), cands)
    keys = {g["id"]: g["scenarios"] for g in res["groups"]}
    chosen = [i for i, c in enumerate(cands)
              if any(p["label"] == c.label for p in res["min_probes"])]
    # 1) 选中的补测项区分所有组
    for a, b in itertools.combinations(sorted(keys), 2):
        assert any(vec[keys[a][0]][i] != vec[keys[b][0]][i] for i in chosen)
    # 2) 去掉任意一项就无法区分 => 最小
    for drop in chosen:
        smaller = [i for i in chosen if i != drop]
        assert not all(
            any(vec[keys[a][0]][i] != vec[keys[b][0]][i] for i in smaller)
            for a, b in itertools.combinations(sorted(keys), 2)
        ), f"去掉 {cands[drop].label} 仍能区分,不是最小集合"


def test_min_separating_set_direct():
    assert _min_separating_set([[(1,), (2,)], [(1,), (3,)]], 2) == [1]
    assert _min_separating_set([[(1,), (2,)], [(1,), (2,)]], 2) is None
    assert _min_separating_set([[(1,)]], 1) == []
    assert _min_separating_set([[(1,), (2,), (3,)], [(1,), (9,), (3,)]], 3) == [1]


def test_expected_results_only_for_min_probes():
    m = model()
    cands = parse_candidates(FIVE, m)
    res = diagnose(m, {"R4", "R5"}, [], cands)
    probe_labels = {p["label"] for p in res["min_probes"]}
    for exp in res["expected"]:
        assert {r["candidate"] for r in exp["results"]} == probe_labels
        for r in exp["results"]:
            assert r["text"] in {"断路", "环路"} or r["text"].startswith("出口 ")


# ------------------------------------------------------------ 候选无法区分

def test_no_candidates_asks_for_candidates():
    m = model()
    res = diagnose(m, {"R4", "R5"}, [], [])
    assert res["status"] == "ok"
    assert res["groups"] == [] and res["min_probes"] == []
    assert res["fully_separating"] is False
    assert "尚未填写可补测项" in res["message"]


def test_indistinguishable_scenarios_flagged():
    """无故障与 R1-R4 断在 R1~R5 五条候选上结果相同 => 组内仍含 2 个场景。

    这属于"补测项再多也无法区分"的情况,页面必须明确标出,而不是假装已唯一。
    """
    m = model()
    res = diagnose(m, {"R4", "R5"}, [], parse_candidates(FIVE, m))
    assert res["status"] == "ok"
    assert res["indistinguishable"]
    g = next(g for g in res["indistinguishable"] if "无故障" in g["labels"])
    assert "链路 R1-R4 单独失效" in g["labels"]
    assert g["group"] in {x["id"] for x in res["groups"]}
    # 有可补测项、且所有组都已能区分(组内不可分的场景除外)
    assert res["min_probes"]
    assert "无法再细分" in res["message"]


def test_groups_all_separate_when_candidates_sufficient():
    """只取 R2/R3 两条候选时,仅剩的三个组各自只有一个场景。"""
    m = model()
    res = diagnose(m, {"R4", "R5"},
                   parse_measurements([meas("R1", "blackhole")], m),
                   parse_candidates([cand("R2"), cand("R3")], m))
    assert [len(g["labels"]) for g in res["groups"]] == [1, 1, 1]
    assert res["indistinguishable"] == []
    assert res["fully_separating"] is True


# ------------------------------------------------------------ 输入校验

def test_candidate_limit():
    m = model()
    with pytest.raises(InputError, match="最多 12 条"):
        parse_candidates(
            [cand("R1", f"10.1.{i // 256}.{i % 256}") for i in range(13)], m)


def test_candidate_count_at_limit_ok():
    m = model()
    cands = parse_candidates(
        [cand("R1", f"10.1.{i // 256}.{i % 256}") for i in range(MAX_CANDIDATES)], m)
    assert len(cands) == MAX_CANDIDATES


def test_duplicate_candidate_deduped():
    m = model()
    cands = parse_candidates([cand("R1"), cand("R1")], m)
    assert len(cands) == 1


def test_bad_measurement_result():
    m = model()
    with pytest.raises(InputError, match="ok、blackhole、loop"):
        parse_measurements([{"ingress": "R1", "target": "10.1.0.5", "result": "up"}], m)


def test_ok_measurement_requires_egress():
    m = model()
    with pytest.raises(InputError, match="必须给出 egress"):
        parse_measurements([{"ingress": "R1", "target": "10.1.0.5", "result": "ok"}], m)


def test_unknown_ingress_and_egress():
    m = model()
    with pytest.raises(InputError, match="入口路由器不存在"):
        parse_measurements([meas("R9", "blackhole")], m)
    with pytest.raises(InputError, match="不存在的出口"):
        parse_measurements([meas("R1", "ok", egress="E9")], m)


def test_target_must_be_single_host():
    m = model()
    with pytest.raises(InputError, match="必须是单个 IPv4 地址"):
        parse_measurements([meas("R1", "blackhole", target="10.1.0.0/24")], m)
    with pytest.raises(InputError, match="必须是单个 IPv4 地址"):
        parse_candidates([cand("R1", "10.1.0.0/16")], m)
    # /32 写法可以
    cands = parse_candidates([cand("R1", "10.1.0.5/32")], m)
    assert str(cands[0].target) == "10.1.0.5/32"


def test_diagnose_payload_end_to_end():
    topo, routes = load_example()
    res = diagnose_payload(topo, routes, ["R4", "R5"],
                           [meas("R1", "blackhole")], FIVE)
    assert res["status"] == "ok"
    assert res["switched"] == ["R4", "R5"]
    assert res["min_probes"]
