"""
応急危険度判定 mTSP - ソルバー比較ツール

同一の問題インスタンス（建物配置・判定時間・優先建物）に対して
「貪欲法のみ」と「貪欲法 + OR-Tools最適化」を両方実行し、
ルート図とサマリーを並べて表示して最適化の有用性を確認する。

結果は CSV（comparison_results.csv）に追記でき、条件を変えた
繰り返し実験の記録として利用できる。

【使い方】
    python compare_gui.py
"""

import csv
import os
import threading
import time
import datetime

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mtsp_core import (
    InspectionProblem, GreedySolver, ORToolsSolver, find_min_inspectors,
)

COLORS = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
    "#8bc34a","#ff5722","#607d8b","#795548","#9c27b0",
    "#03a9f4","#cddc39","#ff9800","#673ab7","#009688",
]

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "comparison_results.csv")


# ── 比較実験の本体（GUI 非依存・テスト可能）──────────────────────────────────

def run_comparison(n, area_km, t_min, t_max, prio_pct, speed, max_work,
                   m_limit, time_limit, w_m, w_l, w_p,
                   seed=None, status_cb=None):
    """
    同一インスタンスを貪欲法と OR-Tools で解いて比較データを返す。

    Args:
        m_limit   : 判定士数の上限。0 以下なら「最小必要人数 + 4」を自動設定
        seed      : 乱数シード（None ならランダム）
        status_cb : 進捗テキストを受け取るコールバック

    Returns:
        dict(problem, m_limit, greedy, opt, t_greedy, t_opt)
    """
    def status(msg):
        if status_cb:
            status_cb(msg)

    rng = np.random.default_rng(seed)
    status("問題インスタンスを生成中...")
    problem = InspectionProblem(
        coords=rng.random((n, 2)),
        inspect_times=rng.uniform(t_min * 60, t_max * 60, n),
        depot_idx=0,
        area_km=area_km, speed_kmh=speed, max_work_h=max_work,
        priorities=(rng.random(n) < prio_pct / 100.0).astype(float))
    # デポはエリア重心に最も近い建物（メイン GUI の「中心に自動配置」と同じ）
    c = problem.coords.mean(axis=0)
    depot = int(np.argmin(np.linalg.norm(problem.coords - c, axis=1)))
    problem = problem.with_depot(depot)

    if m_limit <= 0:
        status("最小必要判定士数を探索中...")
        min_m, _ = find_min_inspectors(problem)
        m_limit = min_m + 4   # 人数削減の余地を持たせる

    status("貪欲法で求解中...")
    t0 = time.perf_counter()
    greedy = GreedySolver().solve(problem, m_limit)
    t_greedy = time.perf_counter() - t0

    status(f"OR-Tools で最適化中...（最大 {time_limit:.0f} 秒）")
    t0 = time.perf_counter()
    opt = ORToolsSolver(time_limit_s=time_limit, weight_m=w_m,
                        weight_total=w_l, weight_priority=w_p
                        ).solve(problem, m_limit, initial=greedy)
    t_opt = time.perf_counter() - t0

    return dict(problem=problem, m_limit=m_limit,
                greedy=greedy, opt=opt,
                t_greedy=t_greedy, t_opt=t_opt)


def prio_mean_start(sol):
    """優先建物の平均判定開始時刻 [h]（優先建物がなければ None）"""
    mask = sol.problem.priorities > 0
    if not mask.any():
        return None
    a = sol.arrival_times[mask]
    if np.all(np.isnan(a)):
        return None
    return float(np.nanmean(a))


# ── GUI ───────────────────────────────────────────────────────────────────────

class CompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("mTSP ソルバー比較 — 最適化なし vs あり")
        self.root.geometry("1480x900")
        self.result = None
        self.running = False
        self._build_ui()

    def _build_ui(self):
        ctrl = tk.Frame(self.root, width=250, bg="#f0f0f0", padx=10, pady=8)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="ソルバー比較ツール",
                 font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(0, 6))

        self._sep(ctrl, "問題設定")
        self._row(ctrl, "建物数:",            "n_var",     "200")
        self._row(ctrl, "エリア (km):",       "area_var",  "10")
        self._row(ctrl, "判定時間 最小(分):", "tmin_var",  "15")
        self._row(ctrl, "判定時間 最大(分):", "tmax_var",  "45")
        self._row(ctrl, "優先建物割合 (%):",  "prio_var",  "10")
        self._row(ctrl, "移動速度 (km/h):",   "speed_var", "30")
        self._row(ctrl, "最大稼働時間 (h):",  "work_var",  "8")
        self._row(ctrl, "判定士上限 (0=自動):","m_var",    "0")
        self._row(ctrl, "乱数シード (空=毎回変更):", "seed_var", "")

        self._sep(ctrl, "最適化設定")
        self._row(ctrl, "時間制限 (秒):", "limit_var", "30")
        self._row(ctrl, "重み M:",        "wm_var",    "1000")
        self._row(ctrl, "重み λ:",        "wl_var",    "1")
        self._row(ctrl, "重み μ:",        "wp_var",    "1")

        self.run_btn = tk.Button(
            ctrl, text="比較実行", command=self.run,
            bg="#2196F3", fg="white", relief=tk.FLAT, pady=7,
            font=("Arial", 10, "bold"))
        self.run_btn.pack(fill=tk.X, pady=6)

        self.csv_btn = tk.Button(
            ctrl, text="結果をCSVに追記", command=self.save_csv,
            bg="#00796B", fg="white", relief=tk.FLAT, pady=4,
            state=tk.DISABLED)
        self.csv_btn.pack(fill=tk.X, pady=2)
        tk.Label(ctrl, text="comparison_results.csv に保存",
                 bg="#f0f0f0", fg="#666", font=("Arial", 8)).pack(anchor="w")

        self.status = tk.Label(ctrl, text="待機中", bg="#f0f0f0", fg="#333",
                               anchor="w", wraplength=220, justify=tk.LEFT)
        self.status.pack(fill=tk.X, pady=4)
        self.progress = ttk.Progressbar(ctrl, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=4)

        # 右側: 上にルート図 ×2、下に比較サマリー表
        right = tk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(12, 5.4), dpi=100)
        self.fig.patch.set_facecolor("#16213e")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.table = tk.Text(right, height=14, font=("Courier", 10),
                             state=tk.DISABLED)
        self.table.pack(fill=tk.X, padx=4, pady=4)

    def _sep(self, parent, text):
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=5)
        tk.Label(parent, text=text, font=("Arial", 9, "bold"),
                 bg="#f0f0f0", anchor="w").pack(fill=tk.X)

    def _row(self, parent, label, attr, default):
        frm = tk.Frame(parent, bg="#f0f0f0")
        frm.pack(fill=tk.X, pady=1)
        tk.Label(frm, text=label, bg="#f0f0f0", width=18, anchor="w",
                 font=("Arial", 8)).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        setattr(self, attr, var)
        tk.Entry(frm, textvariable=var, width=7).pack(side=tk.LEFT)

    # ── 実行 ─────────────────────────────────────────────────────────────────

    def run(self):
        if self.running:
            return
        try:
            params = dict(
                n=int(self.n_var.get()),
                area_km=float(self.area_var.get()),
                t_min=float(self.tmin_var.get()),
                t_max=float(self.tmax_var.get()),
                prio_pct=float(self.prio_var.get()),
                speed=float(self.speed_var.get()),
                max_work=float(self.work_var.get()),
                m_limit=int(self.m_var.get()),
                time_limit=float(self.limit_var.get()),
                w_m=float(self.wm_var.get()),
                w_l=float(self.wl_var.get()),
                w_p=float(self.wp_var.get()),
            )
            assert 2 <= params["n"] <= 2000, "建物数は 2〜2000 棟"
            seed_txt = self.seed_var.get().strip()
            params["seed"] = int(seed_txt) if seed_txt else None
        except AssertionError as e:
            messagebox.showerror("エラー", str(e))
            return
        except Exception:
            messagebox.showerror("エラー", "入力値を確認してください")
            return

        self.running = True
        self.run_btn.config(state=tk.DISABLED)
        self.csv_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        threading.Thread(target=self._worker, args=(params,),
                         daemon=True).start()

    def _worker(self, params):
        try:
            res = run_comparison(
                **params,
                status_cb=lambda msg: self.root.after(
                    0, self.status.config, {"text": msg}))
            self.root.after(0, self._on_done, res, params)
        except Exception as e:
            self.root.after(0, self._on_error, f"エラー: {e}")

    def _on_error(self, msg):
        self.running = False
        self.run_btn.config(state=tk.NORMAL)
        self.progress.stop()
        self.status.config(text=msg)

    def _on_done(self, res, params):
        self.running = False
        self.run_btn.config(state=tk.NORMAL)
        self.csv_btn.config(state=tk.NORMAL)
        self.progress.stop()
        self.result = (res, params)
        self._draw_maps(res)
        self._fill_table(res, params)
        obj_g = res["greedy"].objective(params["w_m"], params["w_l"], params["w_p"])
        obj_o = res["opt"].objective(params["w_m"], params["w_l"], params["w_p"])
        gain = (1 - obj_o / obj_g) * 100 if obj_g > 0 else 0.0
        self.status.config(text=f"完了 — 目的関数 {gain:.1f}% 改善")

    # ── 表示 ─────────────────────────────────────────────────────────────────

    def _draw_maps(self, res):
        self.fig.clear()
        problem = res["problem"]
        coords  = problem.coords
        prio_mask = problem.priorities > 0
        depot = problem.depot_idx

        for i, (sol, title) in enumerate([
                (res["greedy"], "① 貪欲法のみ（最適化なし）"),
                (res["opt"],    "② 貪欲法 + OR-Tools最適化")]):
            ax = self.fig.add_subplot(1, 2, i + 1)
            ax.set_facecolor("#1a1a2e")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.tick_params(colors="#aaa", labelsize=7)
            for s, route in enumerate(sol.routes):
                if len(route) <= 2:
                    continue
                rc = coords[route]
                ax.plot(rc[:, 0], rc[:, 1], "-",
                        color=COLORS[s % len(COLORS)],
                        linewidth=1.0, alpha=0.8, zorder=2)
            ax.scatter(coords[:, 0], coords[:, 1], c="#cccccc", s=10,
                       alpha=0.8, zorder=3)
            if prio_mask.any():
                # 優先建物: 検査済み=シアン、未検査=赤で区別
                a = sol.arrival_times
                done = prio_mask & ~np.isnan(a)
                pend = prio_mask & np.isnan(a)
                pend[depot] = False
                if done.any():
                    dc = coords[done]
                    ax.scatter(dc[:, 0], dc[:, 1], facecolors="none",
                               edgecolors="#00e5ff", s=60, linewidths=1.4,
                               zorder=5, label=f"優先 検査済 ({done.sum()})")
                if pend.any():
                    pc = coords[pend]
                    ax.scatter(pc[:, 0], pc[:, 1], facecolors="none",
                               edgecolors="#ff1744", s=95, linewidths=1.8,
                               zorder=5, label=f"優先 未検査 ({pend.sum()})")
            ax.scatter([coords[depot, 0]], [coords[depot, 1]], c="#FFD700",
                       s=220, marker="*", zorder=6, label="デポ")
            ax.legend(loc="upper right", fontsize=8,
                      facecolor="#1a1a2e", labelcolor="white")
            ax.set_title(
                f"{title}\n使用 {sol.n_used}/{res['m_limit']} 人 | "
                f"総活動 {sol.total_time:.1f}h | 未割当 {sol.n_unassigned}",
                color="white", fontsize=10)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _fill_table(self, res, params):
        g, o = res["greedy"], res["opt"]
        w_m, w_l, w_p = params["w_m"], params["w_l"], params["w_p"]

        pg, po = prio_mean_start(g), prio_mean_start(o)
        obj_g = g.objective(w_m, w_l, w_p)
        obj_o = o.objective(w_m, w_l, w_p)

        def diff(gv, ov, better_low=True, pct=False):
            """差分表示（改善は▲、悪化は▽）"""
            d = ov - gv
            mark = "▲" if (d < 0) == better_low and d != 0 else \
                   ("―" if d == 0 else "▽")
            if pct and gv:
                return f"{mark} {d:+.1f} ({d / gv * 100:+.1f}%)"
            return f"{mark} {d:+.2f}"

        rows = [
            ("使用判定士数 [人]",
             f"{g.n_used}", f"{o.n_used}",
             diff(g.n_used, o.n_used)),
            ("総活動時間 ΣT_q [h]",
             f"{g.total_time:.2f}", f"{o.total_time:.2f}",
             diff(g.total_time, o.total_time, pct=True)),
            ("優先建物 検査済み [棟]",
             f"{g.n_priority_done}/{g.n_priority}",
             f"{o.n_priority_done}/{o.n_priority}",
             "―" if g.n_priority_done == o.n_priority_done
             else ("▲" if o.n_priority_done > g.n_priority_done else "▽")
             + f" {o.n_priority_done - g.n_priority_done:+d}"),
            ("優先建物 平均判定開始 [h]",
             "-" if pg is None else f"{pg:.2f}",
             "-" if po is None else f"{po:.2f}",
             "-" if (pg is None or po is None) else diff(pg, po)),
            ("最大終了時間 makespan [h]",
             f"{g.makespan:.2f}", f"{o.makespan:.2f}",
             diff(g.makespan, o.makespan)),
            ("総移動距離 [km]",
             f"{g.total_dist:.1f}", f"{o.total_dist:.1f}",
             diff(g.total_dist, o.total_dist, pct=True)),
            ("未割当建物 [棟]",
             f"{g.n_unassigned}", f"{o.n_unassigned}",
             diff(g.n_unassigned, o.n_unassigned)),
            ("目的関数 合計",
             f"{obj_g:,.1f}", f"{obj_o:,.1f}",
             diff(obj_g, obj_o, pct=True)),
            ("計算時間 [秒]",
             f"{res['t_greedy']:.2f}", f"{res['t_opt']:.2f}", ""),
        ]

        lines = [
            f"問題: {params['n']}棟 / エリア{params['area_km']:g}km / "
            f"優先{params['prio_pct']:g}% / 上限{res['m_limit']}人 / "
            f"M={w_m:g} λ={w_l:g} μ={w_p:g} / "
            f"シード={params['seed'] if params['seed'] is not None else 'ランダム'}",
            "",
            f"{'指標':<26}{'貪欲法':>12}{'OR-Tools':>12}   {'差 (▲=改善)'}",
            "─" * 78,
        ]
        for label, gv, ov, dv in rows:
            lines.append(f"{label:<26}{gv:>12}{ov:>12}   {dv}")

        self.table.config(state=tk.NORMAL)
        self.table.delete("1.0", tk.END)
        self.table.insert(tk.END, "\n".join(lines))
        self.table.config(state=tk.DISABLED)

    # ── CSV 保存 ─────────────────────────────────────────────────────────────

    def save_csv(self):
        if self.result is None:
            return
        res, params = self.result
        g, o = res["greedy"], res["opt"]
        w_m, w_l, w_p = params["w_m"], params["w_l"], params["w_p"]
        pg, po = prio_mean_start(g), prio_mean_start(o)

        header = [
            "datetime", "n", "area_km", "prio_pct", "m_limit",
            "time_limit", "M", "lambda", "mu", "seed",
            "greedy_used", "greedy_total_h",
            "greedy_prio_done", "greedy_prio_total", "greedy_prio_start_h",
            "greedy_makespan_h", "greedy_dist_km", "greedy_unassigned",
            "greedy_objective", "greedy_sec",
            "opt_used", "opt_total_h",
            "opt_prio_done", "opt_prio_total", "opt_prio_start_h",
            "opt_makespan_h", "opt_dist_km", "opt_unassigned",
            "opt_objective", "opt_sec", "objective_gain_pct",
        ]
        obj_g = g.objective(w_m, w_l, w_p)
        obj_o = o.objective(w_m, w_l, w_p)
        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            params["n"], params["area_km"], params["prio_pct"],
            res["m_limit"], params["time_limit"], w_m, w_l, w_p,
            params["seed"] if params["seed"] is not None else "",
            g.n_used, round(g.total_time, 3),
            g.n_priority_done, g.n_priority,
            "" if pg is None else round(pg, 3),
            round(g.makespan, 3), round(g.total_dist, 1),
            g.n_unassigned, round(obj_g, 1), round(res["t_greedy"], 2),
            o.n_used, round(o.total_time, 3),
            o.n_priority_done, o.n_priority,
            "" if po is None else round(po, 3),
            round(o.makespan, 3), round(o.total_dist, 1),
            o.n_unassigned, round(obj_o, 1), round(res["t_opt"], 2),
            round((1 - obj_o / obj_g) * 100, 2) if obj_g > 0 else "",
        ]
        # 列構成が古い CSV が残っていたらバックアップして新規作成する
        new_file = not os.path.exists(CSV_PATH)
        if not new_file:
            with open(CSV_PATH, encoding="utf-8-sig") as f:
                first = f.readline()
            if "greedy_prio_done" not in first:
                backup = CSV_PATH.replace(
                    ".csv",
                    datetime.datetime.now().strftime("_old_%Y%m%d%H%M%S.csv"))
                os.rename(CSV_PATH, backup)
                new_file = True
        with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(header)
            w.writerow(row)
        self.status.config(text=f"CSVに追記しました: {CSV_PATH}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = CompareApp(root)
    root.mainloop()
