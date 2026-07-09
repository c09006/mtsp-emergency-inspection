"""
応急危険度判定 mTSP コアモジュール（GUI 非依存）

「問題」「解」「解法」を分離したクラス構成:

- InspectionProblem  : 問題インスタンス（建物座標・判定時間・デポ・制約パラメータ）
- InspectionSolution : 解（ルートと未割当）。KPI は自分で計算し、実行可能性を自己検証できる
- SolverBase         : 解法の共通インターフェース。ソルバーを差し替えて比較実験できる
    - GreedySolver     : 最近傍法 + KDTree + min-heap による時間制約付き貪欲法
    - MultiStartSolver : 複数デポ候補の並列試行（マルチスタート）
- find_min_inspectors: 全棟割当可能な最小判定士数の二分探索

【制約条件】
1. 各建物は 1 回だけ判定
2. 判定士は拠点（デポ）から出発・拠点に帰還
3. 移動時間（距離 / 速度）と判定時間を稼働時間に加算
4. 現在時刻 + 移動 + 判定 + デポ帰還 ≤ 最大稼働時間 を満たす場合のみ割当
5. 目的関数: 最大終了時間（makespan）の最小化
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
        max_work_h    : 判定士 1 人あたりの最大稼働時間 [時間]
    """
    coords: np.ndarray
    inspect_times: np.ndarray
    depot_idx: int
    area_km: float
    speed_kmh: float
    max_work_h: float

    def __post_init__(self):
        coords = np.asarray(self.coords, dtype=np.float64)
        times = np.asarray(self.inspect_times, dtype=np.float64)
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "inspect_times", times)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords は (n, 2) の配列である必要があります")
        if len(times) != len(coords):
            raise ValueError("inspect_times の長さが coords と一致しません")
        if not (0 <= self.depot_idx < len(coords)):
            raise ValueError(f"depot_idx={self.depot_idx} が範囲外です")
        if self.area_km <= 0 or self.speed_kmh <= 0 or self.max_work_h <= 0:
            raise ValueError("area_km / speed_kmh / max_work_h は正の値が必要です")

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
            self.area_km, self.speed_kmh, self.max_work_h)


# ── 解 ────────────────────────────────────────────────────────────────────────

@dataclass
class InspectionSolution:
    """
    mTSP の解。ルート列と未割当建物を保持し、KPI（makespan・稼働時間・
    移動距離）は自分で計算する。ソルバーの集計値を信用せず、解自身が
    is_feasible() / validate() で制約充足を検証できるのが要点。

    Attributes:
        problem    : 対応する問題インスタンス
        routes     : 判定士ごとの訪問順インデックス（デポ始点・デポ終点）
        unassigned : 時間不足で割当できなかった建物インデックスのリスト
    """
    problem: InspectionProblem
    routes: list = field(default_factory=list)
    unassigned: list = field(default_factory=list)

    @property
    def n_inspectors(self) -> int:
        return len(self.routes)

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
        return (f"判定士 {self.n_inspectors} 人 | makespan {self.makespan:.3f}h | "
                f"総距離 {self.total_dist:.1f}km | 未割当 {self.n_unassigned} 棟")


# ── 貪欲法の実体（ProcessPoolExecutor で並列実行するためトップレベルに定義）──

def _greedy_routes(problem: InspectionProblem, m: int):
    """
    時間制約付き mTSP 貪欲法。

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
    remaining = n - 1   # デポを除いた未割当建物数

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


def _greedy_worker(args):
    """ProcessPoolExecutor 用ワーカー（picklable にするためトップレベル定義）"""
    problem, m = args
    return _greedy_routes(problem, m)


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
        routes, unassigned = _greedy_routes(problem, m)
        return InspectionSolution(problem, routes, unassigned)


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
                    routes, unassigned = f.result()
                    solutions.append(InspectionSolution(futs[f], routes, unassigned))

        return min(solutions,
                   key=lambda s: (s.n_unassigned, s.makespan, s.total_dist))


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
