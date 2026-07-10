"""
応急危険度判定 mTSP - 実験基盤（必要人数分析・感度分析）

【構成】
- Scenario            : 実験条件（問題パラメータ + ソルバー設定）のデータクラス
- dominance_weight_m  : 人数最小化が最優先になる M の閾値を自動計算
- ExperimentRunner    : 1 条件を解いて指標を記録する共通エンジン
- StaffingAnalyzer    : 規定時間 H ごとの最小必要人数（貪欲法 vs OR-Tools）
- SensitivityAnalyzer : ノイズ床測定・One-at-a-Time 感度分析・トルネード図

【使い方（コマンドライン）】
    python experiments.py noise                        # ノイズ床の測定
    python experiments.py staffing --h 4,6,8,10,12     # H vs 必要人数曲線
    python experiments.py sweep --param speed_kmh --values 20,25,30,35,40
    python experiments.py tornado --rel 0.2            # ±20% トルネード図

    共通オプション: --n 200 --seeds 5 --time-limit 30 など（--help 参照）

結果は results/ フォルダに CSV と PNG で保存される。
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, replace, asdict
from typing import Callable, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'Meiryo'
import matplotlib.pyplot as plt

from mtsp_core import (
    InspectionProblem, GreedySolver, ORToolsSolver, find_min_inspectors,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results")


# ── 実験条件 ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    """
    実験条件のまとまり。感度分析では replace() で 1 項目だけ差し替えて使う。

    weight_m が None のときは dominance_weight_m() で「人数最小化が
    最優先になる閾値 × 安全係数」を自動計算する。
    """
    n: int = 200                 # 建物数
    area_km: float = 10.0        # エリア一辺 [km]
    t_min: float = 15.0          # 判定時間 最小 [分]
    t_max: float = 45.0          # 判定時間 最大 [分]
    prio_pct: float = 10.0       # 優先建物割合 [%]
    speed_kmh: float = 30.0      # 移動速度 [km/h]
    max_work_h: float = 8.0      # 規定時間（稼働上限）H [h]
    time_limit: float = 30.0     # OR-Tools の時間制限 [秒]
    lam: float = 1.0             # λ: 総活動時間の重み
    mu: float = 1.0              # μ: 優先建物の重み
    weight_m: Optional[float] = None   # M（None = 自動計算）

    def make_problem(self, seed: int) -> InspectionProblem:
        """シードから問題インスタンスを再現生成する（デポはエリア重心付近）"""
        rng = np.random.default_rng(seed)
        problem = InspectionProblem(
            coords=rng.random((self.n, 2)),
            inspect_times=rng.uniform(self.t_min * 60, self.t_max * 60, self.n),
            depot_idx=0,
            area_km=self.area_km, speed_kmh=self.speed_kmh,
            max_work_h=self.max_work_h,
            priorities=(rng.random(self.n) < self.prio_pct / 100.0
                        ).astype(float))
        c = problem.coords.mean(axis=0)
        depot = int(np.argmin(np.linalg.norm(problem.coords - c, axis=1)))
        return problem.with_depot(depot)


def dominance_weight_m(problem: InspectionProblem, m_upper: int,
                       lam: float, mu: float, safety: float = 3.0) -> float:
    """
    人数最小化が他の項より必ず優先される M の閾値（安全係数付き）を返す。

    判定士 1 人を減らして節約できる M に対し、他項の増加は最大でも
        λ ×（全員が上限まで働く総時間） + μ ×（全優先建物が最遅で判定）
    を超えないため、これより大きい M なら人数削減が常に優先される。
    """
    total_t_max = lam * m_upper * problem.max_work_h
    prio_max = mu * float(problem.priorities.sum()) * problem.max_work_h
    return safety * max(1.0, total_t_max + prio_max)


# ── 共通実験エンジン ──────────────────────────────────────────────────────────

class ExperimentRunner:
    """
    1 条件（Scenario × シード）を解いて指標の辞書を返す共通エンジン。

    手順:
    1. 貪欲法の二分探索で「全棟検査可能な最小人数 greedy_min_m」を求める
    2. M を自動計算（weight_m=None のとき）し、人数上限 = greedy_min_m で
       OR-Tools を実行 → 使用人数 opt_used ≤ greedy_min_m が OR-Tools の答え
    3. 両者の全指標を記録する
    """

    def __init__(self, status_cb: Optional[Callable[[str], None]] = None,
                 should_continue: Optional[Callable[[], bool]] = None):
        self.status_cb = status_cb or (lambda msg: None)
        self.should_continue = should_continue or (lambda: True)

    def run_one(self, sc: Scenario, seed: int) -> Optional[dict]:
        self.status_cb(f"  seed={seed}: 問題生成・貪欲法二分探索...")
        problem = sc.make_problem(seed)

        t0 = time.perf_counter()
        min_m, greedy = find_min_inspectors(problem)
        t_greedy = time.perf_counter() - t0
        if greedy is None:
            self.status_cb(f"  seed={seed}: 実行可能解なし（スキップ）")
            return None

        w_m = (sc.weight_m if sc.weight_m is not None
               else dominance_weight_m(problem, min_m, sc.lam, sc.mu))

        self.status_cb(f"  seed={seed}: OR-Tools 最適化 "
                       f"(上限 {min_m} 人, M={w_m:,.0f})...")
        t0 = time.perf_counter()
        opt = ORToolsSolver(
            time_limit_s=sc.time_limit, weight_m=w_m,
            weight_total=sc.lam, weight_priority=sc.mu,
        ).solve(problem, min_m, initial=greedy)
        t_opt = time.perf_counter() - t0

        if not opt.is_feasible():
            raise RuntimeError(f"OR-Tools 解が制約違反: {opt.validate()}")

        def prio_start(sol):
            mask = sol.problem.priorities > 0
            if not mask.any():
                return np.nan
            a = sol.arrival_times[mask]
            return float(np.nanmean(a)) if not np.all(np.isnan(a)) else np.nan

        row = dict(asdict(sc))
        row.update(
            seed=seed, auto_M=round(w_m, 1),
            greedy_min_m=min_m,
            greedy_total_h=round(greedy.total_time, 3),
            greedy_prio_start_h=round(prio_start(greedy), 3),
            greedy_makespan_h=round(greedy.makespan, 3),
            greedy_dist_km=round(greedy.total_dist, 1),
            greedy_sec=round(t_greedy, 2),
            opt_used=opt.n_used,
            opt_total_h=round(opt.total_time, 3),
            opt_prio_start_h=round(prio_start(opt), 3),
            opt_makespan_h=round(opt.makespan, 3),
            opt_dist_km=round(opt.total_dist, 1),
            opt_unassigned=opt.n_unassigned,
            opt_prio_done=opt.n_priority_done,
            opt_prio_total=opt.n_priority,
            opt_sec=round(t_opt, 2),
            staff_saved=min_m - opt.n_used,
        )
        return row

    def run_many(self, scenarios: Sequence[tuple],
                 csv_path: Optional[str] = None) -> list:
        """
        (Scenario, seed) のリストを順に実行し、行の辞書リストを返す。
        csv_path を指定すると 1 行完了するごとに追記保存する（中断に強い）。
        """
        rows = []
        total = len(scenarios)
        writer = None
        f = None
        try:
            for i, (sc, seed) in enumerate(scenarios, 1):
                if not self.should_continue():
                    self.status_cb("中止されました（完了分は保存済み）")
                    break
                self.status_cb(f"[{i}/{total}] {self._label(sc)}")
                row = self.run_one(sc, seed)
                if row is None:
                    continue
                rows.append(row)
                if csv_path:
                    if writer is None:
                        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                        new = not os.path.exists(csv_path)
                        f = open(csv_path, "a", newline="",
                                 encoding="utf-8-sig")
                        writer = csv.DictWriter(f, fieldnames=row.keys())
                        if new:
                            writer.writeheader()
                    writer.writerow(row)
                    f.flush()
        finally:
            if f:
                f.close()
        return rows

    @staticmethod
    def _label(sc: Scenario) -> str:
        return (f"n={sc.n} H={sc.max_work_h:g}h v={sc.speed_kmh:g}km/h "
                f"優先{sc.prio_pct:g}% μ={sc.mu:g}")


# ── 必要人数分析 ──────────────────────────────────────────────────────────────

class StaffingAnalyzer:
    """
    「規定時間 H で全棟を終えるには最小何人必要か」を分析する。

    各 H・各シードについて、貪欲法の最小人数（find_min_inspectors）と
    OR-Tools の使用人数（人数最小化を最優先にした最適化の結果）を記録し、
    H vs 必要人数の曲線を描く。
    """

    def __init__(self, base: Scenario, runner: Optional[ExperimentRunner] = None):
        self.base = base
        self.runner = runner or ExperimentRunner(print)

    def run(self, h_values: Sequence[float], seeds: Sequence[int],
            csv_path: Optional[str] = None) -> list:
        jobs = [(replace(self.base, max_work_h=h), seed)
                for h in h_values for seed in seeds]
        return self.runner.run_many(jobs, csv_path)

    @staticmethod
    def plot(rows: list, path: Optional[str] = None, fig=None):
        """
        H vs 必要人数（平均 ± 標準偏差）を描画する。
        fig を渡すとそこへ描画（GUI 埋め込み用）、path を渡すと PNG 保存。
        """
        hs = sorted({r["max_work_h"] for r in rows})

        def stats(key):
            means, stds = [], []
            for h in hs:
                v = [r[key] for r in rows if r["max_work_h"] == h]
                means.append(np.mean(v))
                stds.append(np.std(v))
            return np.array(means), np.array(stds)

        gm, gs = stats("greedy_min_m")
        om, os_ = stats("opt_used")

        own = fig is None
        if own:
            fig = plt.figure(figsize=(8, 5.5))
        ax = fig.add_subplot(111)
        ax.errorbar(hs, gm, yerr=gs, marker="o", capsize=4,
                    label="貪欲法の最小人数", color="#e74c3c")
        ax.errorbar(hs, om, yerr=os_, marker="s", capsize=4,
                    label="OR-Tools の必要人数", color="#2196F3")
        ax.set_xlabel("規定時間 H [時間]")
        ax.set_ylabel("必要判定士数 [人]")
        n = rows[0]["n"]
        n_seeds = len({r["seed"] for r in rows})
        ax.set_title(f"規定時間と最小必要人数（{n}棟, シード{n_seeds}個の平均±SD）")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=120)
        if own:
            plt.close(fig)


# ── 感度分析 ──────────────────────────────────────────────────────────────────

class SensitivityAnalyzer:
    """
    実験的感度分析（パラメトリック分析）を行う。

    - noise_floor : 同一条件でシードのみ変えたばらつき（ノイズ床）を測定。
                    パラメータの効果はこのノイズと比較して判断する
    - sweep       : 1 パラメータだけを動かす One-at-a-Time 分析
    - tornado     : 全パラメータを ±rel% 動かし、出力への影響幅を比較する
                    トルネード図用データを作る
    """

    # sweep / tornado で動かせる Scenario のフィールド
    SWEEPABLE = ["n", "area_km", "t_min", "t_max", "prio_pct",
                 "speed_kmh", "max_work_h", "time_limit", "mu", "weight_m"]

    def __init__(self, base: Scenario, seeds: Sequence[int],
                 runner: Optional[ExperimentRunner] = None):
        self.base = base
        self.seeds = list(seeds)
        self.runner = runner or ExperimentRunner(print)

    def noise_floor(self, csv_path: Optional[str] = None) -> dict:
        """ベースライン条件をシードだけ変えて実行し、ばらつきを返す"""
        rows = self.runner.run_many(
            [(self.base, s) for s in self.seeds], csv_path)
        stats = {}
        for key in ["greedy_min_m", "opt_used", "opt_total_h",
                    "opt_prio_start_h", "opt_makespan_h"]:
            v = np.array([r[key] for r in rows], dtype=float)
            v = v[~np.isnan(v)]
            stats[key] = dict(mean=float(v.mean()), std=float(v.std()),
                              min=float(v.min()), max=float(v.max()))
        return stats

    def sweep(self, param: str, values: Sequence[float],
              csv_path: Optional[str] = None) -> list:
        """param を values に変えながら（他は固定）全シードで実行する"""
        if param not in self.SWEEPABLE:
            raise ValueError(f"{param} は感度分析対象外です: {self.SWEEPABLE}")
        jobs = []
        for v in values:
            cast = int(v) if param == "n" else float(v)
            jobs += [(replace(self.base, **{param: cast}), s)
                     for s in self.seeds]
        return self.runner.run_many(jobs, csv_path)

    def tornado(self, rel: float = 0.2,
                params: Optional[Sequence[str]] = None,
                output: str = "opt_used",
                csv_path: Optional[str] = None) -> list:
        """
        各パラメータを ±rel（相対）動かしたときの出力変化を測る。

        Returns:
            [(param, base_mean, low_mean, high_mean), ...]
            （影響幅 |high-low| の大きい順）
        """
        params = params or ["speed_kmh", "t_min", "t_max",
                            "area_km", "prio_pct", "max_work_h"]
        base_rows = self.runner.run_many(
            [(self.base, s) for s in self.seeds], csv_path)
        base_mean = float(np.nanmean([r[output] for r in base_rows]))

        effects = []
        for p in params:
            v0 = getattr(self.base, p)
            lo, hi = v0 * (1 - rel), v0 * (1 + rel)
            lo_rows = self.sweep(p, [lo], csv_path)
            hi_rows = self.sweep(p, [hi], csv_path)
            lo_mean = float(np.nanmean([r[output] for r in lo_rows]))
            hi_mean = float(np.nanmean([r[output] for r in hi_rows]))
            effects.append((p, base_mean, lo_mean, hi_mean))
        effects.sort(key=lambda e: abs(e[3] - e[2]), reverse=True)
        return effects

    @staticmethod
    def plot_sweep(rows: list, param: str, path: Optional[str] = None,
                   outputs=("opt_used", "opt_total_h", "opt_prio_start_h"),
                   fig=None):
        """sweep の結果（平均±SD）をパラメータ別に描画する"""
        labels = {"opt_used": "必要人数 [人]",
                  "opt_total_h": "総活動時間 [h]",
                  "opt_prio_start_h": "優先 平均判定開始 [h]",
                  "opt_makespan_h": "makespan [h]",
                  "greedy_min_m": "貪欲法 最小人数 [人]"}
        vals = sorted({r[param] for r in rows})
        own = fig is None
        if own:
            fig = plt.figure(figsize=(5 * len(outputs), 4.4))
        axes = [fig.add_subplot(1, len(outputs), i + 1)
                for i in range(len(outputs))]
        for ax, key in zip(axes, outputs):
            means, stds = [], []
            for v in vals:
                arr = np.array([r[key] for r in rows if r[param] == v],
                               dtype=float)
                arr = arr[~np.isnan(arr)]
                means.append(arr.mean() if len(arr) else np.nan)
                stds.append(arr.std() if len(arr) else np.nan)
            ax.errorbar(vals, means, yerr=stds, marker="o", capsize=4,
                        color="#2196F3")
            ax.set_xlabel(param)
            ax.set_ylabel(labels.get(key, key))
            ax.grid(alpha=0.3)
        fig.suptitle(f"感度分析: {param} を変化させた影響（平均±SD）")
        fig.tight_layout()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=120)
        if own:
            plt.close(fig)

    @staticmethod
    def plot_tornado(effects: list, output: str, rel: float,
                     path: Optional[str] = None, fig=None):
        """tornado() の結果を横棒グラフで描画する"""
        names = {"speed_kmh": "移動速度", "t_min": "判定時間 最小",
                 "t_max": "判定時間 最大", "area_km": "エリアサイズ",
                 "prio_pct": "優先建物割合", "max_work_h": "規定時間 H",
                 "mu": "μ（優先の重み）", "n": "建物数"}
        own = fig is None
        if own:
            fig = plt.figure(figsize=(8, 0.7 * len(effects) + 2))
        ax = fig.add_subplot(111)
        for i, (p, base, lo, hi) in enumerate(reversed(effects)):
            ax.barh(i, lo - base, left=base, color="#3498db", height=0.6)
            ax.barh(i, hi - base, left=base, color="#e74c3c", height=0.6)
        ax.axvline([effects[0][1]], color="#333", linewidth=1)
        ax.set_yticks(range(len(effects)))
        ax.set_yticklabels([names.get(p, p) for p, *_ in reversed(effects)])
        ax.set_xlabel(output)
        ax.set_title(f"トルネード図: 各パラメータ ±{rel*100:.0f}% の影響\n"
                     f"（青=下振れ側, 赤=上振れ側, 縦線=ベースライン）")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=120)
        if own:
            plt.close(fig)


# ── コマンドラインインターフェース ────────────────────────────────────────────

def _base_from_args(a) -> Scenario:
    return Scenario(n=a.n, time_limit=a.time_limit, mu=a.mu,
                    max_work_h=a.h_base)


def main():
    ap = argparse.ArgumentParser(description="mTSP 実験基盤")
    ap.add_argument("command",
                    choices=["noise", "staffing", "sweep", "tornado"])
    ap.add_argument("--n", type=int, default=200, help="建物数")
    ap.add_argument("--seeds", type=int, default=5, help="シード数")
    ap.add_argument("--time-limit", type=float, default=30.0,
                    help="OR-Tools 時間制限 [秒]")
    ap.add_argument("--mu", type=float, default=1.0, help="優先の重み μ")
    ap.add_argument("--h-base", type=float, default=8.0,
                    help="ベースラインの規定時間 H [h]")
    ap.add_argument("--h", type=str, default="4,6,8,10,12",
                    help="staffing: H のリスト（カンマ区切り）")
    ap.add_argument("--param", type=str, help="sweep: 動かすパラメータ名")
    ap.add_argument("--values", type=str, help="sweep: 値のリスト（カンマ区切り）")
    ap.add_argument("--rel", type=float, default=0.2,
                    help="tornado: 変動幅（0.2 = ±20%%）")
    ap.add_argument("--output", type=str, default="opt_used",
                    help="tornado: 見る出力指標")
    a = ap.parse_args()

    base = _base_from_args(a)
    seeds = list(range(a.seeds))
    tag = f"n{a.n}_s{a.seeds}"

    if a.command == "noise":
        sa = SensitivityAnalyzer(base, seeds)
        stats = sa.noise_floor(os.path.join(RESULTS_DIR, f"noise_{tag}.csv"))
        print("\n── ノイズ床（同一条件・シードのみ変更）──")
        for k, s in stats.items():
            print(f"{k:<20} mean={s['mean']:.3f}  std={s['std']:.3f}  "
                  f"range=[{s['min']:.3f}, {s['max']:.3f}]")
        print("\n※パラメータの効果がこの std より小さい場合、"
              "有意な感度とは言えません")

    elif a.command == "staffing":
        hs = [float(x) for x in a.h.split(",")]
        st = StaffingAnalyzer(base)
        rows = st.run(hs, seeds,
                      os.path.join(RESULTS_DIR, f"staffing_{tag}.csv"))
        png = os.path.join(RESULTS_DIR, f"staffing_{tag}.png")
        st.plot(rows, png)
        print("\n── H vs 必要人数 ──")
        for h in hs:
            g = [r["greedy_min_m"] for r in rows if r["max_work_h"] == h]
            o = [r["opt_used"] for r in rows if r["max_work_h"] == h]
            print(f"H={h:>5.1f}h  貪欲 {np.mean(g):5.1f} 人  "
                  f"OR-Tools {np.mean(o):5.1f} 人  "
                  f"(削減 {np.mean(g) - np.mean(o):+.1f})")
        print(f"図: {png}")

    elif a.command == "sweep":
        if not a.param or not a.values:
            ap.error("sweep には --param と --values が必要です")
        vals = [float(x) for x in a.values.split(",")]
        sa = SensitivityAnalyzer(base, seeds)
        rows = sa.sweep(a.param, vals,
                        os.path.join(RESULTS_DIR,
                                     f"sweep_{a.param}_{tag}.csv"))
        png = os.path.join(RESULTS_DIR, f"sweep_{a.param}_{tag}.png")
        sa.plot_sweep(rows, a.param, png)
        print(f"図: {png}")

    elif a.command == "tornado":
        sa = SensitivityAnalyzer(base, seeds)
        effects = sa.tornado(
            rel=a.rel, output=a.output,
            csv_path=os.path.join(RESULTS_DIR, f"tornado_{tag}.csv"))
        png = os.path.join(RESULTS_DIR, f"tornado_{a.output}_{tag}.png")
        SensitivityAnalyzer.plot_tornado(effects, a.output, a.rel, png)
        print(f"\n── トルネード分析 (±{a.rel*100:.0f}%, 出力={a.output}) ──")
        for p, base_m, lo, hi in effects:
            print(f"{p:<14} base={base_m:.2f}  low側={lo:.2f}  "
                  f"high側={hi:.2f}  影響幅={abs(hi-lo):.2f}")
        print(f"図: {png}")


if __name__ == "__main__":
    main()
