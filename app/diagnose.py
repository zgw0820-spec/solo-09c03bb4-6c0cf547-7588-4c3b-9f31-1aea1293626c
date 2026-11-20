"""故障定位与补测规划核心逻辑。

场景
----
- 无故障,以及每条链路、每个出口**单独失效**——任一物理故障都同时影响
  两台(或一台)路由器上指向该链路(出口)的下一跳解析。
- 每台路由器使用"已换设备勾选项"确定的表:勾选=换版后,未勾选=换版前。
- 下一跳解析沿用现有规则:链路子网 / 出口子网内最长前缀匹配。故障场景下
  受影响的路由表项被撤下(下一跳原本落在被中断的链路/出口上),剩余表项
  重新按最长前缀逐跳转发,可能回退到更粗的路由。

实测
----
每条记录 = 入口 + 目标 IPv4 + 结果(出口名 / 断路 / 环路)。只保留在全部
记录下逐跳仿真结果一致的场景。

补测
----
最多 12 条"入口 + 目标"候选。逐场景仿真每条候选,结果向量完全相同的场景
归为一组。再求能区分所有组的最小候选集合,并给出各组在该集合上的预期结果。
"""
from __future__ import annotations

import ipaddress
import itertools
from dataclasses import dataclass

from .planner import (
    VERDICT_BLACKHOLE,
    VERDICT_LOOP,
    VERDICT_OK,
    InputError,
    Model,
    _longest_prefix_match,
    build_network,
)

MAX_CANDIDATES = 12
MAX_MEASUREMENTS = 200
MAX_TOTAL_PROBES = 20000

RESULT_ZH = {
    "ok": "出口",
    "blackhole": "断路",
    "loop": "环路",
}


# ------------------------------------------------------------------ 场景

@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    kind: str                       # "base" | "link" | "egress"
    down_link: str | None = None    # 链路 "a-b"
    down_egress: str | None = None


def build_scenarios(model: Model) -> list[Scenario]:
    """无故障 + 每条链路单独失效 + 每个出口单独失效。"""
    scenarios = [Scenario("base", "无故障", "base")]
    for lk in model.links:
        scenarios.append(Scenario(
            f"link:{_link_key(lk)}", f"链路 {_link_key(lk)} 单独失效", "link",
            down_link=_link_key(lk),
        ))
    for eg in model.egresses:
        scenarios.append(Scenario(
            f"egress:{eg.name}", f"出口 {eg.name} 单独失效", "egress",
            down_egress=eg.name,
        ))
    return scenarios


def _base_destination(model: Model, router: str,
                      next_hop: ipaddress.IPv4Address) -> tuple[str, str] | None:
    """现有下一跳解析(与 planner 完全一致):最长前缀优先。

    返回 ("link", "a-b") 或 ("egress", 出口名)。
    """
    for net, (kind, name) in model.resolve[router]:
        if next_hop in net:
            return kind, name
    return None


def _link_key(link) -> str:
    return f"{link.a}-{link.b}"


def _prepare(model: Model, scenario: Scenario):
    """撤下受故障影响的路由,并给出剩余表项可用的下一跳解析表。

    受影响 = 该表项的下一跳按现有解析落在被中断的链路/出口上。撤下后
    重新按最长前缀匹配,可能落到更粗的表项上继续转发。
    """
    resolve: dict[str, list[tuple[ipaddress.IPv4Network, tuple[str, str]]]] = {
        r: [] for r in model.routers
    }
    for lk in model.links:
        if _link_key(lk) == scenario.down_link:
            continue
        resolve[lk.a].append((lk.network, ("router", lk.b)))
        resolve[lk.b].append((lk.network, ("router", lk.a)))
    for eg in model.egresses:
        if eg.name == scenario.down_egress:
            continue
        resolve[eg.router].append((eg.network, ("egress", eg.name)))
    for r in model.routers:
        resolve[r].sort(key=lambda x: x[0].prefixlen, reverse=True)

    def affected(router: str, rt) -> bool:
        dest = _base_destination(model, router, rt.next_hop)
        if dest is None:
            return False
        kind, name = dest
        if kind == "link":
            return scenario.down_link is not None and name == scenario.down_link
        return scenario.down_egress is not None and name == scenario.down_egress

    tables: dict[str, tuple[list, list]] = {}
    for r in model.routers:
        tb = model.tables[r]
        tables[r] = (
            [rt for rt in tb.before if not affected(r, rt)],
            [rt for rt in tb.after if not affected(r, rt)],
        )
    return resolve, tables


# ---------------------------------------------------------------- 转发仿真

@dataclass
class ProbeOutcome:
    verdict: str                  # ok / blackhole / loop
    egress: str | None = None

    @property
    def key(self) -> tuple:
        if self.verdict == VERDICT_OK:
            return ("exit", self.egress)
        return (self.verdict,)

    def as_dict(self) -> dict:
        if self.verdict == VERDICT_OK:
            text = f"出口 {self.egress}"
        else:
            text = RESULT_ZH[self.verdict]
        return {"verdict": self.verdict, "egress": self.egress, "text": text}


def _trace(model: Model, resolve, tables, switched: frozenset[str], ingress: str,
           addr: ipaddress.IPv4Address) -> ProbeOutcome:
    visited: set[str] = set()
    cur = ingress
    while True:
        if cur in visited:
            return ProbeOutcome(VERDICT_LOOP)
        visited.add(cur)
        before, after = tables[cur]
        table = after if cur in switched else before
        rt = _longest_prefix_match(table, addr)
        if rt is None:
            return ProbeOutcome(VERDICT_BLACKHOLE)
        target = None
        for net, dest in resolve[cur]:
            if rt.next_hop in net:
                target = dest
                break
        if target is None:
            return ProbeOutcome(VERDICT_BLACKHOLE)
        kind, name = target
        if kind == "egress":
            return ProbeOutcome(VERDICT_OK, name)
        cur = name


# ------------------------------------------------------------------ 输入

def _parse_network(value, what: str) -> ipaddress.IPv4Network:
    try:
        return ipaddress.IPv4Network(str(value), strict=False)
    except (ipaddress.AddressValueError, ipaddress.NetmaskValueError, ValueError) as exc:
        raise InputError(f"{what} 不是合法 IPv4 地址/CIDR: {value!r}") from exc


def _parse_host(value, what: str) -> ipaddress.IPv4Network:
    """实测/补测的目标是单个 IPv4 地址(允许写 /32)。

    不接受网段:补测只探测一个地址,用网段会让"预期结果"含义含糊。
    """
    net = _parse_network(value, what)
    if net.prefixlen != 32:
        raise InputError(f"{what} 必须是单个 IPv4 地址,不能是网段: {value!r}")
    return net


def _parse_ingress(value, model: Model, what: str) -> str:
    ingress = str(value)
    if ingress not in model.tables:
        raise InputError(f"{what} 的入口路由器不存在: {ingress}")
    return ingress


@dataclass
class Measurement:
    ingress: str
    target: ipaddress.IPv4Network
    result: str                    # ok / blackhole / loop
    egress: str | None


def parse_measurements(raw, model: Model) -> list[Measurement]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InputError("实测记录 measurements 必须是数组")
    if len(raw) > MAX_MEASUREMENTS:
        raise InputError(f"实测记录条数超过上限 {MAX_MEASUREMENTS}")
    egress_names = {eg.name for eg in model.egresses}
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError(f"measurements[{i}] 必须是对象")
        if "ingress" not in item or "target" not in item or "result" not in item:
            raise InputError(f"measurements[{i}] 需要 ingress、target、result 字段")
        ingress = _parse_ingress(item["ingress"], model, f"measurements[{i}]")
        target = _parse_host(item["target"], f"measurements[{i}].target")
        result = str(item["result"])
        if result not in RESULT_ZH:
            raise InputError(
                f"measurements[{i}].result 必须是 ok、blackhole、loop 之一: {result!r}"
            )
        egress = None
        if result == "ok":
            egress = str(item.get("egress") or "")
            if not egress:
                raise InputError(f"measurements[{i}] 结果为 ok 时必须给出 egress")
            if egress not in egress_names:
                raise InputError(f"measurements[{i}] 引用了不存在的出口: {egress}")
        out.append(Measurement(ingress, target, result, egress))
    return out


@dataclass(frozen=True)
class Candidate:
    ingress: str
    target: ipaddress.IPv4Network

    @property
    def label(self) -> str:
        return f"{self.ingress} → {self.target.network_address}"

    def as_dict(self) -> dict:
        return {"ingress": self.ingress, "target": str(self.target.network_address),
                "label": self.label}


def parse_candidates(raw, model: Model) -> list[Candidate]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InputError("可补测项 candidates 必须是数组")
    if len(raw) > MAX_CANDIDATES:
        raise InputError(f"可补测项最多 {MAX_CANDIDATES} 条,当前 {len(raw)} 条")
    seen = set()
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "ingress" not in item or "target" not in item:
            raise InputError(f"candidates[{i}] 需要 ingress 和 target 字段")
        ingress = _parse_ingress(item["ingress"], model, f"candidates[{i}]")
        net = _parse_host(item["target"], f"candidates[{i}].target")
        cand = Candidate(ingress, net)
        if cand.label in seen:
            continue
        seen.add(cand.label)
        out.append(cand)
    return out


def _matches(meas: Measurement, outcome: ProbeOutcome) -> bool:
    if meas.result == "ok":
        return outcome.verdict == VERDICT_OK and outcome.egress == meas.egress
    return outcome.verdict == meas.result


# ------------------------------------------------------------------ 诊断

def _min_separating_set(group_results: list[list[tuple]], n_candidates: int) -> list[int]:
    """求区分所有组的最小候选下标集合;同规模取字典序最小。

    若仅凭现有候选无法把任意两组分开(即缺少可区分的补测项),返回 None。
    注意:分组本身是按候选结果向量做的,所以组与组之间在全部候选上必不相同;
    这里返回 None 只可能是 n_candidates == 0。
    """
    n_groups = len(group_results)
    if n_groups <= 1:
        return []
    pairs = list(itertools.combinations(range(n_groups), 2))

    def separates(combo: tuple[int, ...]) -> bool:
        return all(
            any(group_results[g1][i] != group_results[g2][i] for i in combo)
            for g1, g2 in pairs
        )

    for size in range(1, n_candidates + 1):
        for combo in itertools.combinations(range(n_candidates), size):
            if separates(combo):
                return list(combo)
    return None


def diagnose(model: Model, switched: set[str], measurements: list[Measurement],
             candidates: list[Candidate]) -> dict:
    unknown = switched - set(model.routers)
    if unknown:
        raise InputError(f"已换设备不在路由器清单中: {sorted(unknown)}")
    scenarios = build_scenarios(model)
    total_probes = (len(candidates) + len(measurements)) * len(scenarios)
    if total_probes > MAX_TOTAL_PROBES:
        raise InputError("仿真规模过大,请减少实测记录或补测项")

    state = frozenset(switched)
    prepared = {sc.key: _prepare(model, sc) for sc in scenarios}

    # 1) 只保留与全部实测相符的场景
    kept: list[Scenario] = []
    for sc in scenarios:
        resolve, tables = prepared[sc.key]
        ok = True
        for m in measurements:
            if not _matches(m, _trace(model, resolve, tables, state, m.ingress,
                                      m.target.network_address)):
                ok = False
                break
        if ok:
            kept.append(sc)

    base = {
        "switched": sorted(switched),
        "scenarios_total": len(scenarios),
        "scenarios_matched": [{"key": sc.key, "label": sc.label} for sc in kept],
        "measurements": [
            {"ingress": m.ingress, "target": str(m.target.network_address),
             "result": m.result, "egress": m.egress}
            for m in measurements
        ],
        "candidates": [c.as_dict() for c in candidates],
    }

    if not kept:
        return {
            **base,
            "status": "no_scenario",
            "message": "没有任何故障场景与全部实测记录相符,观测与假设不符:"
                       "请核对拓扑、路由表、已换设备勾选项和实测记录。",
            "groups": [],
            "min_probes": [],
            "expected": [],
            "indistinguishable": [],
            "fully_separating": False,
        }

    # 2) 逐场景仿真每条候选,结果向量相同的归为一组。
    #    没有候选补测项时向量全为空,分组没有信息量,直接给出提示。
    def outcome(sc: Scenario, cand: Candidate) -> ProbeOutcome:
        resolve, tables = prepared[sc.key]
        return _trace(model, resolve, tables, state, cand.ingress,
                      cand.target.network_address)

    if not candidates:
        return {
            **base,
            "status": "ok",
            "message": "尚未填写可补测项:现有实测记录无法确定故障组,"
                       "请填写最多 12 条入口与目标。",
            "groups": [],
            "min_probes": [],
            "expected": [],
            "indistinguishable": [],
            "fully_separating": False,
        }

    keys = [(sc, [outcome(sc, cand).key for cand in candidates]) for sc in kept]

    groups: list[dict] = []
    index: dict[tuple, dict] = {}
    for sc, vec in keys:
        vec_t = tuple(vec)
        if vec_t not in index:
            index[vec_t] = {"scenarios": [], "vector": vec_t, "outcomes": []}
            groups.append(index[vec_t])
        index[vec_t]["scenarios"].append(sc)
        if not index[vec_t]["outcomes"]:
            index[vec_t]["outcomes"] = [outcome(sc, cand) for cand in candidates]

    for gi, g in enumerate(groups, start=1):
        g["id"] = gi
        g["labels"] = [sc.label for sc in g["scenarios"]]

    # 3) 求区分所有组的最小补测集合
    vectors = [list(g["vector"]) for g in groups]
    chosen = _min_separating_set(vectors, len(candidates)) or []

    min_probes = [candidates[i].as_dict() for i in chosen]
    expected = []
    for g in groups:
        outcomes = g["outcomes"]
        expected.append({
            "group": g["id"],
            "labels": g["labels"],
            "results": [
                {"candidate": candidates[i].label, **outcomes[i].as_dict()}
                for i in chosen
            ],
        })

    # 组内多于一个场景 <=> 这些场景在全部候选补测项上结果都相同,无法区分
    indistinguishable = [
        {"group": g["id"], "labels": g["labels"]}
        for g in groups if len(g["scenarios"]) > 1
    ]
    fully_separating = not indistinguishable

    # 4) 结论文案
    if not chosen:
        msg = "实测记录已把故障组收敛为 1 组,无需再补测。"
        if indistinguishable:
            msg += "该组内多个场景在所有候选补测项上结果完全相同,现有条件无法再细分。"
    else:
        msg = f"还需补测 {len(chosen)} 项即可唯一确定故障组。"
        if indistinguishable:
            msg += "另有组内含多个场景,它们在所有候选补测项上结果相同,无法再细分。"

    return {
        **base,
        "status": "ok",
        "message": msg,
        "groups": [
            {
                "id": g["id"],
                "labels": g["labels"],
                "scenarios": [sc.key for sc in g["scenarios"]],
                "scenario_count": len(g["scenarios"]),
                "possible_egresses": sorted({
                    o.egress for o in g["outcomes"] if o.verdict == VERDICT_OK
                }),
            }
            for g in groups
        ],
        "min_probes": min_probes,
        "expected": expected,
        "indistinguishable": indistinguishable,
        "fully_separating": fully_separating,
    }


# ------------------------------------------------------------------ API 入口

def diagnose_payload(topology: dict, routes: dict, switched_raw,
                     measurements_raw, candidates_raw) -> dict:
    model = build_network(topology, routes)
    if switched_raw is None:
        switched_raw = []
    if not isinstance(switched_raw, list):
        raise InputError("switched 必须是路由器名数组")
    switched = {str(x) for x in switched_raw}
    measurements = parse_measurements(measurements_raw, model)
    candidates = parse_candidates(candidates_raw, model)
    return diagnose(model, switched, measurements, candidates)
