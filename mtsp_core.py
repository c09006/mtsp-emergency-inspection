"""
応急危険度判定 mTSP コアモジュール（GUI 非依存）

「問題」「解」「解法」を分離したクラス構成:

- InspectionProblem  : 問題インスタンス（建物座標・判定時間・デポ・制約パラメータ）
- InspectionSolution : 解（ルートと未割当）。KPI は自分で計算し、実行可能性を自己検証できる
- SolverBase         : 解法の共通インターフェース。ソルバーを差し替えて比較実験できる
    - GreedySolver     : 最近傍法 + KDTree + min-heap による時間制約付き貪欲法
    - MultiStartSolver : 複数デポ候補の並列試行（マルチスタート）
    - ORToolsSolver    : OR-Tools ルーティングソルバーによる makespan 最小化
                         （貪欲解を初期解に誘導局所探索で改善）
- find_min_inspectors: 全棟割当可能な最小判定士数の二分探索

【制約条件】
1. 各建物は 1 回だけ判定
2. 判定士は毎日、拠点（デポ）から出発し拠点に帰還する
3. 移動時間（距離 / 速度）と判定時間を稼働時間に加算
4. 1 日の稼働は 最大稼働時間 max_work_h 以内
5. 計画は期日 n_days 日以内（日ごとに独立したルートを持つ）

【目的関数（ORToolsSolver）】
    min  M Σ z_q  +  λ Σ T_q  +  μ Σ p_i a_i
    - z_q : 判定士 q を使用するか (0/1)          → 人数の最小化
    - T_q : 判定士 q の総活動時間（移動+判定）[h] → 総活動時間の最小化
    - a_i : 建物 i の判定開始時刻 [h]、p_i: 優先度 → 優先建築物の早期判定
GreedySolver は makespan を平準化する構築ヒューリスティック（初期解生成用）。
"""
from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.spatial import KDTree


# ── 問題インスタンス ──────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class InspectionProblem:
    """
    応急危険度判定 mTSP の問題インスタンス。

    座標は [0,1]^2 の正規化座標で保持し、area_km でスケールして
    実距離・移動時間に換算する。イミュータブル（frozen）なので
    ソルバー間で安全に共有できる。

    Attributes:
        coords        : (n, 2) 建物座標（正規化 [0,1]）
        inspect_times : (n,) 建物ごとの判定時間 [秒]
        depot_idx     : デポ（拠点）の建物インデックス
        area_km       : エリア一辺の実寸 [km]
        speed_kmh     : 判定士の移動速度 [km/h]
        max_work_h    : 判定士 1 人・1 日あたりの最大稼働時間 [時間]
        priorities    : (n,) 建物ごとの優先度 p_i（0 = 通常）。省略時は全建物 0
        n_days        : 期日（計画日数）。判定士は毎日デポから出発・帰還する
    """
    coords: np.ndarray
    inspect_times: np.ndarray
    depot_idx: int
    area_km: float
    speed_kmh: float
    max_work_h: float
    priorities: Optional[np.ndarray] = None
    n_days: int = 1

    def __post_init__(self):
        coords = np.asarray(self.coords, dtype=np.float64)
        times = np.asarray(self.inspect_times, dtype=np.float64)
        prios = (np.zeros(len(coords)) if self.priorities is None
                 else np.asarray(self.priorities, dtype=np.float64))
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "inspect_times", times)
        object.__setattr__(self, "priorities", prios)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords は (n, 2) の配列である必要があります")
        if len(times) != len(coords):
            raise ValueError("inspect_times の長さが coords と一致しません")
        if len(prios) != len(coords):
            raise ValueError("priorities の長さが coords と一致しません")
        if np.any(prios < 0):
            raise ValueError("priorities は非負である必要があります")
        if not (0 <= self.depot_idx < len(coords)):
            raise ValueError(f"depot_idx={self.depot_idx} が範囲外です")
        if self.area_km <= 0 or self.speed_kmh <= 0 or self.max_work_h <= 0:
            raise ValueError("area_km / speed_kmh / max_work_h は正の値が必要です")
        if not (1 <= int(self.n_days) <= 365):
            raise ValueError("n_days（期日日数）は 1〜365 の整数が必要です")
        object.__setattr__(self, "n_days", int(self.n_days))

    @property
    def n_buildings(self) -> int:
        return len(self.coords)

    @property
    def max_work_sec(self) -> float:
        return self.max_work_h * 3600.0

    @property
    def area_m(self) -> float:
        return self.area_km * 1000.0

    @property
    def speed_ms(self) -> float:
        return self.speed_kmh * 1000.0 / 3600.0

    @property
    def sec_per_unit(self) -> float:
        """正規化距離 1.0 あたりの移動時間 [秒]"""
        return self.area_m / self.speed_ms

    def travel_sec(self, i: int, j: int) -> float:
        """建物 i → j の移動時間 [秒]"""
        d = float(np.linalg.norm(self.coords[i] - self.coords[j]))
        return d * self.sec_per_unit

    def dist_km(self, i: int, j: int) -> float:
        """建物 i → j の実距離 [km]"""
        return float(np.linalg.norm(self.coords[i] - self.coords[j])) * self.area_km

    def with_depot(self, depot_idx: int) -> "InspectionProblem":
        """デポだけを差し替えた問題インスタンスを返す（マルチスタート用）"""
        return InspectionProblem(
            self.coords, self.inspect_times, depot_idx,
            self.area_km, self.speed_kmh, self.max_work_h,
            self.priorities, self.n_days)


# ── 解 ────────────────────────────────────────────────────────────────────────

@dataclass
class InspectionSolution:
    """
    mTSP の解。ルート列と未割当建物を保持し、KPI（makespan・稼働時間・
    移動距離）は自分で計算する。ソルバーの集計値を信用せず、解自身が
    is_feasible() / validate() で制約充足を検証できるのが要点。

    Attributes:
        problem          : 対応する問題インスタンス
        routes           : ルートごとの訪問順インデックス（デポ始点・デポ終点）。
                           複数日計画では 1 ルート = 1 判定士の 1 日分
        unassigned       : 期日内に割当できなかった建物インデックスのリスト
        route_days       : 各ルートの実施日（0 始まり）。None なら全て 0 日目
        route_inspectors : 各ルートの担当判定士 ID。None なら通し番号
    """
    problem: InspectionProblem
    routes: list = field(default_factory=list)
    unassigned: list = field(default_factory=list)
    route_days: Optional[list] = None
    route_inspectors: Optional[list] = None

    @property
    def route_day(self) -> list:
        """各ルートの実施日（0 始まり）"""
        return (self.route_days if self.route_days is not None
                else [0] * len(self.routes))

    @property
    def route_inspector(self) -> list:
        """各ルートの担当判定士 ID"""
        return (self.route_inspectors if self.route_inspectors is not None
                else list(range(len(self.routes))))

    @property
    def n_inspectors(self) -> int:
        """判定士数（上限）= 相異なる担当者 ID の数"""
        return len(set(self.route_inspector)) if self.routes else 0

    @cached_property
    def per_time(self) -> list:
        """判定士ごとの稼働時間 [時間]（移動 + 判定をルートから再計算）"""
        p = self.problem
        times = []
        for route in self.routes:
            if len(route) < 2:
                times.append(0.0)
                continue
            rc = p.coords[route]
            travel_units = float(np.linalg.norm(np.diff(rc, axis=0), axis=1).sum())
            travel_s = travel_units * p.sec_per_unit
            inspect_s = float(p.inspect_times[route[1:-1]].sum())
            times.append((travel_s + inspect_s) / 3600.0)
        return times

    @cached_property
    def per_dist(self) -> list:
        """判定士ごとの移動距離 [km]"""
        p = self.problem
        dists = []
        for route in self.routes:
            if len(route) < 2:
                dists.append(0.0)
                continue
            rc = p.coords[route]
            dists.append(
                float(np.linalg.norm(np.diff(rc, axis=0), axis=1).sum()) * p.area_km)
        return dists

    @property
    def makespan(self) -> float:
        """最大終了時間 [時間] = 最も遅い判定士の稼働時間"""
        return max(self.per_time) if self.per_time else 0.0

    @property
    def total_dist(self) -> float:
        """総移動距離 [km]"""
        return sum(self.per_dist)

    @property
    def n_unassigned(self) -> int:
        return len(self.unassigned)

    @property
    def n_used(self) -> int:
        """
        必要判定士数 = 日ごとの使用人数の最大値。
        判定士は日をまたいで使い回せるため、最も人手が要る日の人数が
        「実際に確保すべき人数」になる（単日計画では従来どおり使用人数）。
        """
        per_day = {}
        for d, r in zip(self.route_day, self.routes):
            if len(r) > 2:
                per_day[d] = per_day.get(d, 0) + 1
        return max(per_day.values()) if per_day else 0

    @property
    def person_days(self) -> int:
        """延べ人日 = 建物が割り当てられたルート（判定士×日）の数"""
        return sum(1 for r in self.routes if len(r) > 2)

    @property
    def n_days_used(self) -> int:
        """実際に使われた日数（建物が割り当てられた最終日 + 1）"""
        used = [d for d, r in zip(self.route_day, self.routes) if len(r) > 2]
        return max(used) + 1 if used else 0

    @property
    def total_time(self) -> float:
        """全判定士の総活動時間 Σ T_q [時間]（移動 + 判定）"""
        return sum(self.per_time)

    @cached_property
    def arrival_times(self) -> np.ndarray:
        """
        建物ごとの判定開始時刻 a_i [時間]。未訪問（デポ・未割当）は NaN。
        a_i = 実施日のオフセット（d 日目 = 24d 時間）+ その日のデポ出発
        からの累積（移動 + それまでの判定）。複数日計画では「後の日ほど
        遅い」が自然に表現される。
        """
        p = self.problem
        a = np.full(p.n_buildings, np.nan)
        for day, route in zip(self.route_day, self.routes):
            t = day * 86400.0   # 日オフセット [秒]
            for k in range(1, len(route) - 1):
                t += p.travel_sec(route[k - 1], route[k])
                a[route[k]] = t / 3600.0
                t += float(p.inspect_times[route[k]])
        return a

    @property
    def priority_cost(self) -> float:
        """優先度重み付き判定開始時刻の合計 Σ p_i a_i [時間]（未割当は除外）"""
        a = self.arrival_times
        mask = ~np.isnan(a)
        return float((self.problem.priorities[mask] * a[mask]).sum())

    @property
    def n_priority(self) -> int:
        """優先建物の総数（デポを除く）"""
        p = self.problem
        mask = p.priorities > 0
        mask[p.depot_idx] = False
        return int(mask.sum())

    @property
    def n_priority_done(self) -> int:
        """検査された（ルートに含まれる）優先建物の数"""
        a = self.arrival_times
        return int(((self.problem.priorities > 0) & ~np.isnan(a)).sum())

    @property
    def priority_unassigned(self) -> list:
        """未検査（未割当）の優先建物インデックスのリスト"""
        p = self.problem
        a = self.arrival_times
        return [i for i in range(p.n_buildings)
                if p.priorities[i] > 0 and i != p.depot_idx
                and np.isnan(a[i])]

    def objective(self, weight_m: float, weight_total: float,
                  weight_priority: float) -> float:
        """目的関数値 M Σz_q + λ ΣT_q + μ Σp_i a_i を返す"""
        return (weight_m * self.n_used
                + weight_total * self.total_time
                + weight_priority * self.priority_cost)

    def validate(self, tol: float = 1e-6) -> list:
        """制約違反のリストを返す（空なら実行可能解）"""
        p = self.problem
        errors = []
        visited = []
        for s, route in enumerate(self.routes):
            if len(route) < 2:
                continue
            if route[0] != p.depot_idx or route[-1] != p.depot_idx:
                errors.append(f"判定士 #{s+1}: ルートがデポ始点・終点になっていません")
            visited.extend(route[1:-1])
            if self.per_time[s] > p.max_work_h + tol:
                errors.append(
                    f"判定士 #{s+1}: 稼働時間 {self.per_time[s]:.3f}h が"
                    f"上限 {p.max_work_h}h を超過")
        if len(visited) != len(set(visited)):
            errors.append("同じ建物が複数回訪問されています")
        covered = set(visited) | set(self.unassigned) | {p.depot_idx}
        missing = set(range(p.n_buildings)) - covered
        if missing:
            errors.append(f"{len(missing)} 棟が訪問リストにも未割当リストにも含まれていません")
        return errors

    def is_feasible(self, tol: float = 1e-6) -> bool:
        return not self.validate(tol)

    def summary(self) -> str:
        days = (f" | {self.n_days_used}/{self.problem.n_days} 日"
                if self.problem.n_days > 1 else "")
        return (f"判定士 {self.n_used}/{self.n_inspectors} 人{days} | "
                f"makespan {self.makespan:.3f}h | 総活動 {self.total_time:.2f}h | "
                f"優先コスト {self.priority_cost:.2f} | "
                f"総距離 {self.total_dist:.1f}km | 未割当 {self.n_unassigned} 棟")


# ── 貪欲法の実体（ProcessPoolExecutor で並列実行するためトップレベルに定義）──

def _greedy_routes(problem: InspectionProblem, m: int, visited=None):
    """
    時間制約付き mTSP 貪欲法（1 日分）。

    【割当戦略】
    min-heap で「現在時刻が最も小さい（最も暇な）判定士」が次の建物を選ぶ。
    これにより各判定士の終了時刻が平準化され、makespan が抑えられる。

    【実行フロー】
    1. 全判定士を時刻 0・デポ位置でヒープに積む
    2. ヒープから最小時刻の判定士 s を取り出す
    3. KDTree で現在地に近い未訪問建物を候補として取得
    4. 制約チェック: t_now + 移動 + 判定 + デポ帰還 ≤ max_work_sec
    5. 条件を満たす最近傍を割当 → ヒープに再投入
    6. どの建物も割当不可なら判定士 s は終了
    7. 全判定士終了 or 全建物割当完了で終了

    Returns:
        (routes, unassigned)
        - routes     : 判定士ごとの訪問順（デポ始点・デポ終点）
        - unassigned : 割当できなかった建物インデックスのリスト
    """
    coords = problem.coords
    inspect_times = problem.inspect_times
    depot_idx = problem.depot_idx
    n = problem.n_buildings
    max_work_sec = problem.max_work_sec
    sec_per_unit = problem.sec_per_unit

    def travel_sec(i, j):
        return float(np.linalg.norm(coords[i] - coords[j])) * sec_per_unit

    # KDTree を構築（近傍探索を O(log n) で実現）
    tree = KDTree(coords)

    if visited is None:
        visited = np.zeros(n, dtype=bool)
    visited[depot_idx] = True   # デポは最初から訪問済み扱い

    # min-heap: (現在時刻[秒], 判定士ID)
    heap = [(0.0, s) for s in range(m)]
    heapq.heapify(heap)

    routes   = [[depot_idx] for _ in range(m)]
    cur_pos  = [depot_idx] * m
    cur_time = [0.0] * m
    done     = [False] * m

    # 初回クエリの近傍数。未訪問が見つからなければ段階的に拡大
    k_base = min(30, n)
    remaining = int((~visited).sum())   # 未割当建物数

    while remaining > 0:
        if all(done):
            break   # 全判定士が稼働時間不足で終了 → 残りは未割当

        cur_t, s = heapq.heappop(heap)
        if done[s]:
            continue

        pos   = cur_pos[s]
        t_now = cur_time[s]

        # 近傍を段階的に拡大しながら割当可能な建物を探す
        found = False
        for k_try in [k_base, k_base * 5, n]:
            k_try = min(k_try, n)
            _, idxs = tree.query(coords[pos], k=k_try)
            idxs = np.atleast_1d(idxs)
            for nxt in idxs:
                if visited[nxt]:
                    continue
                t_travel  = travel_sec(pos, int(nxt))
                t_inspect = inspect_times[nxt]
                t_back    = travel_sec(int(nxt), depot_idx)

                # 制約チェック: 建物訪問 + デポ帰還後も最大稼働時間内か
                if t_now + t_travel + t_inspect + t_back <= max_work_sec:
                    routes[s].append(int(nxt))
                    visited[nxt]  = True
                    cur_pos[s]    = int(nxt)
                    cur_time[s]   = t_now + t_travel + t_inspect
                    remaining    -= 1
                    heapq.heappush(heap, (cur_time[s], s))
                    found = True
                    break
            if found:
                break

        if not found:
            done[s] = True   # どの未訪問建物も時間制約を満たせない

    # 全判定士をデポに帰還させる
    for s in range(m):
        routes[s].append(depot_idx)

    unassigned = [i for i in range(n) if not visited[i]]
    return routes, unassigned


def _greedy_multi_day(problem: InspectionProblem, m: int):
    """
    期日 n_days 日の複数日貪欲法。

    日ごとに「残りの建物」へ 1 日分の貪欲法を適用する。判定士は毎日
    デポから出発・帰還し、期日を使い切っても残った建物が未割当になる。

    Returns:
        (routes, route_days, route_inspectors, unassigned)
    """
    n = problem.n_buildings
    visited = np.zeros(n, dtype=bool)
    routes, route_days, route_inspectors = [], [], []

    for d in range(problem.n_days):
        day_routes, _ = _greedy_routes(problem, m, visited)
        assigned = sum(len(r) - 2 for r in day_routes)
        routes += day_routes
        route_days += [d] * m
        route_inspectors += list(range(m))
        if assigned == 0 or visited.all():
            break   # 割当できる建物が残っていない

    unassigned = [i for i in range(n)
                  if not visited[i] and i != problem.depot_idx]
    return routes, route_days, route_inspectors, unassigned


def _greedy_worker(args):
    """ProcessPoolExecutor 用ワーカー（picklable にするためトップレベル定義）"""
    problem, m = args
    return _greedy_multi_day(problem, m)


# ── 解法 ──────────────────────────────────────────────────────────────────────

class SolverBase(ABC):
    """解法の共通インターフェース。solve() が問題と判定士数から解を返す。"""

    @abstractmethod
    def solve(self, problem: InspectionProblem, m: int) -> InspectionSolution:
        ...


class GreedySolver(SolverBase):
    """最近傍法 + KDTree + min-heap による時間制約付き貪欲法（単一デポ）"""

    def solve(self, problem: InspectionProblem, m: int) -> InspectionSolution:
        if m < 1:
            raise ValueError("判定士数 m は 1 以上が必要です")
        routes, days, inspectors, unassigned = _greedy_multi_day(problem, m)
        return InspectionSolution(problem, routes, unassigned,
                                  days, inspectors)


class MultiStartSolver(SolverBase):
    """
    複数のデポ候補で貪欲法を並列試行し、最良解を返すマルチスタート法。

    デポ位置によってルートの質が変わるため、複数候補を試すことで
    貪欲法の解を確率的に改善する。候補は重複を除去してから実行する
    （貪欲法は決定的なので同一デポの再計算は無駄）。

    最良解の選択基準: (未割当数, makespan, 総距離) の辞書式最小。
    """

    def __init__(self, depot_candidates: Optional[Sequence[int]] = None,
                 n_workers: int = 1):
        self.depot_candidates = depot_candidates
        self.n_workers = max(1, n_workers)

    def solve(self, problem: InspectionProblem, m: int) -> InspectionSolution:
        depots = list(dict.fromkeys(
            self.depot_candidates if self.depot_candidates else [problem.depot_idx]))
        problems = [problem.with_depot(d) for d in depots]

        if len(problems) == 1 or self.n_workers == 1:
            solutions = [GreedySolver().solve(p, m) for p in problems]
        else:
            solutions = []
            with ProcessPoolExecutor(max_workers=self.n_workers) as ex:
                futs = {ex.submit(_greedy_worker, (p, m)): p for p in problems}
                for f in as_completed(futs):
                    routes, days, inspectors, unassigned = f.result()
                    solutions.append(InspectionSolution(
                        futs[f], routes, unassigned, days, inspectors))

        return min(solutions,
                   key=lambda s: (s.n_unassigned, s.makespan, s.total_dist))


class ORToolsSolver(SolverBase):
    """
    OR-Tools ルーティングソルバーによる重み付き多目的最適化。

    【目的関数】
        min  M Σ z_q  +  λ Σ T_q  +  μ Σ p_i a_i
        - M Σ z_q     : 使用判定士数（人数の最小化）
        - λ Σ T_q     : 総活動時間 [h]（移動 + 判定）
        - μ Σ p_i a_i : 優先度重み付き判定開始時刻 [h]（優先建築物の早期判定）

    【OR-Tools での実装】
    - M z_q      → SetFixedCostOfAllVehicles（使用車両ごとの固定費）
    - λ T_q      → アークコスト（移動 + 判定時間の推移に係数 λ）
    - μ p_i a_i  → Time 次元の SetCumulVarSoftUpperBound(i, 0, μ p_i)
                   （上限 0 の soft 制約 → コスト = 係数 × 判定開始時刻）
    - 稼働時間制約 ≤ max_work_h → Time 次元の capacity
    - 未割当の許容              → AddDisjunction（ペナルティ付きドロップ）

    m は「使用できる判定士数の上限」であり、何人使うかはソルバーが決める。
    重みの単位: T_q・a_i は時間 [h]。M は「1 人追加 = M 時間分のコスト」に相当。
    内部では秒単位・整数コスト（重みは WEIGHT_SCALE 倍で量子化、精度 0.01）。

    【解法】
    貪欲解（initial に渡す）を初期解として誘導局所探索 (GUIDED_LOCAL_SEARCH)
    で時間制限まで改善する。初期解がなければ PATH_CHEAPEST_ARC で構築する。

    時間行列を密行列で持つため対象は max_nodes 棟まで。それ以上の規模は
    GreedySolver / MultiStartSolver を使用する。
    """

    WEIGHT_SCALE = 100   # 重みの量子化精度（1/WEIGHT_SCALE 刻み）

    def __init__(self, time_limit_s: float = 10.0,
                 weight_m: float = 1000.0,
                 weight_total: float = 1.0,
                 weight_priority: float = 1.0,
                 max_nodes: int = 2000,
                 log_search: bool = False):
        self.time_limit_s = time_limit_s
        self.weight_m = weight_m               # M: 人数の重み [時間相当]
        self.weight_total = weight_total       # λ: 総活動時間の重み
        self.weight_priority = weight_priority # μ: 優先建物の早期判定の重み
        self.max_nodes = max_nodes
        self.log_search = log_search

    def solve(self, problem: InspectionProblem, m: int,
              initial: Optional[InspectionSolution] = None) -> InspectionSolution:
        # ortools は重い依存なので、このソルバーを使うときだけ import する
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2

        n = problem.n_buildings
        if m < 1:
            raise ValueError("判定士数 m は 1 以上が必要です")
        if n > self.max_nodes:
            raise ValueError(
                f"ORToolsSolver は {self.max_nodes:,} 棟までです（指定: {n:,} 棟）。"
                f"大規模問題には GreedySolver を使用してください")

        depot = problem.depot_idx
        n_days = problem.n_days
        n_vehicles = m * n_days   # 1 車両 = 1 判定士の 1 日分
        DAY_SEC = 86400

        # 時間行列 [秒, 整数]。切り上げにより「整数モデルで実行可能なら
        # 実数値の稼働時間制約も必ず満たす」ことを保証する
        diff = problem.coords[:, None, :] - problem.coords[None, :, :]
        travel = np.ceil(
            np.sqrt((diff ** 2).sum(axis=2)) * problem.sec_per_unit
        ).astype(np.int64)
        service = np.ceil(problem.inspect_times).astype(np.int64)
        service[depot] = 0          # デポ（拠点）自体は判定しない
        max_work_sec = int(problem.max_work_h * 3600)

        manager = pywrapcp.RoutingIndexManager(n, n_vehicles, depot)
        routing = pywrapcp.RoutingModel(manager)

        # 重みを整数化（内部コスト単位: [秒 × WEIGHT_SCALE]）
        scale   = self.WEIGHT_SCALE
        m_int   = int(round(self.weight_m * 3600 * scale))       # M [h] → 秒相当
        lam_int = int(round(self.weight_total * scale))          # λ
        mu_int  = np.round(
            self.weight_priority * problem.priorities * scale
        ).astype(np.int64)                                       # μ p_i

        # 推移 = 出発地での判定時間 + 移動時間（Time 次元用、秒）
        time_mat = (travel + service[:, None]).tolist()

        def time_cb(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return time_mat[i][j]

        transit_idx = routing.RegisterTransitCallback(time_cb)

        # λ Σ T_q: 総活動時間をアークコストとして最小化
        cost_mat = ((travel + service[:, None]) * lam_int).tolist()

        def cost_cb(from_index, to_index):
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            return cost_mat[i][j]

        cost_idx = routing.RegisterTransitCallback(cost_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cost_idx)

        # M Σ z: 使用した判定士 1 人日ごとの固定費（人数・日数の最小化）。
        # 複数日計画では「使う人日」が最小化され、日ごとの使用人数が減る
        routing.SetFixedCostOfAllVehicles(m_int)

        # 稼働時間制約: 各判定士・各日の累積時間 ≤ その日の開始 + max_work_sec。
        # 車両 v の日 = v // m。開始時刻を日オフセット (86400×日) に固定し、
        # 終了時刻の上限を日オフセット + max_work_sec とすることで
        # 「毎日 max_work_h まで働く」を表現する
        horizon = (n_days - 1) * DAY_SEC + max_work_sec
        routing.AddDimension(transit_idx, 0, horizon, n_days == 1, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        if n_days > 1:
            for v in range(n_vehicles):
                off = (v // m) * DAY_SEC
                time_dim.CumulVar(routing.Start(v)).SetRange(off, off)
                time_dim.CumulVar(routing.End(v)).SetRange(
                    off, off + max_work_sec)

        # μ Σ p_i a_i: 優先建物の判定開始時刻（= Time 次元の累積値）にコスト。
        # 上限 0 の soft 制約なので「超過分 = a_i そのもの」に係数がかかる。
        # 複数日では日オフセットが乗るため「後の日ほど高コスト」になり、
        # 優先建物は自然と初日の早い時間帯に組み込まれる
        for node in range(n):
            if node != depot and mu_int[node] > 0:
                time_dim.SetCumulVarSoftUpperBound(
                    manager.NodeToIndex(node), 0, int(mu_int[node]))

        # 未割当の許容: 大きなペナルティ付きで建物をドロップ可能にする
        # （期日内に全棟不可能な場合でも解が返るようにする）。
        # ペナルティは「1 棟ドロップで節約できる最大コスト」を確実に上回る値
        max_mu = int(mu_int.max()) if len(mu_int) else 0
        drop_penalty = (m_int + (lam_int + max_mu) * horizon) * 10
        for node in range(n):
            if node != depot:
                routing.AddDisjunction([manager.NodeToIndex(node)], drop_penalty)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        params.time_limit.FromMilliseconds(int(self.time_limit_s * 1000))
        params.log_search = self.log_search

        assignment = None
        if initial is not None and initial.routes:
            # 貪欲解を初期解として渡す（デポ端点と未割当は除く）。
            # 車両インデックスは「日 × m + 判定士 ID」の順に対応させる
            init_routes = [[] for _ in range(n_vehicles)]
            for day, insp, r in zip(initial.route_day,
                                    initial.route_inspector, initial.routes):
                v = day * m + insp
                if 0 <= v < n_vehicles:
                    init_routes[v] = r[1:-1]
            routing.CloseModelWithParameters(params)
            init_assignment = routing.ReadAssignmentFromRoutes(init_routes, True)
            if init_assignment is not None:
                assignment = routing.SolveFromAssignmentWithParameters(
                    init_assignment, params)
        if assignment is None:
            assignment = routing.SolveWithParameters(params)
        if assignment is None:
            raise RuntimeError("OR-Tools が解を見つけられませんでした")

        # 解の取り出し（車両 v = 日 v//m の判定士 v%m の 1 日分ルート）
        routes, route_days, route_inspectors = [], [], []
        for v in range(n_vehicles):
            idx = routing.Start(v)
            route = []
            while not routing.IsEnd(idx):
                route.append(manager.IndexToNode(idx))
                idx = assignment.Value(routing.NextVar(idx))
            route.append(manager.IndexToNode(idx))   # 終端 = デポ
            routes.append(route)
            route_days.append(v // m)
            route_inspectors.append(v % m)

        unassigned = [
            node for node in range(n)
            if node != depot
            and assignment.Value(
                routing.NextVar(manager.NodeToIndex(node)))
                == manager.NodeToIndex(node)
        ]
        return InspectionSolution(problem, routes, unassigned,
                                  route_days, route_inspectors)


# ── 最小判定士数の探索 ────────────────────────────────────────────────────────

def find_min_inspectors(
    problem: InspectionProblem,
    solver: Optional[SolverBase] = None,
    lo: int = 1,
    hi: Optional[int] = None,
    progress: Optional[Callable[[int], None]] = None,
    should_continue: Optional[Callable[[], bool]] = None,
):
    """
    二分探索で全棟割当可能な最小判定士数を求める。

    探索範囲 [lo, hi] で unassigned == 0 となる最小の m を O(log n) 回の
    solve() 呼び出しで確定する。

    Args:
        problem         : 問題インスタンス
        solver          : 使用する解法（省略時 GreedySolver）
        lo, hi          : 探索範囲（hi 省略時は建物数）
        progress        : 各試行前に m を受け取るコールバック（GUI 進捗表示用）
        should_continue : False を返すと探索を中断するコールバック

    Returns:
        (min_m, solution)
        solution は最小 m での実行可能解。実行可能解が見つからなければ None。
    """
    solver = solver or GreedySolver()
    hi = hi if hi is not None else problem.n_buildings

    best_m = hi
    best_sol = None

    while lo <= hi:
        if should_continue is not None and not should_continue():
            break
        mid = (lo + hi) // 2
        if progress is not None:
            progress(mid)
        sol = solver.solve(problem, mid)
        if sol.n_unassigned == 0:
            best_m, best_sol = mid, sol   # mid で全棟対応可能 → さらに少ない m を試す
            hi = mid - 1
        else:
            lo = mid + 1                  # mid では足りない → m を増やす

    return best_m, best_sol
