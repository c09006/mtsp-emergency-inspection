"""
応急危険度判定 mTSP - 時間制約付き複数巡回セールスマン問題 GUI
Time-Constrained mTSP for Emergency Building Inspection

【概要】
地震等の災害後に複数の判定士が拠点（デポ）から出発し、
エリア内の全建物を手分けして応急危険度判定を行う巡回計画を作成する。

ソルバー本体は mtsp_core モジュールに分離されており、本ファイルは
GUI（tkinter + matplotlib）のみを担当する。

【依存ライブラリ】
    pip install scipy matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
import random
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from mtsp_core import (
    InspectionProblem, GreedySolver, MultiStartSolver, find_min_inspectors,
)


# 判定士ごとのルート描画色（最大20人分）
COLORS = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4",
    "#8bc34a","#ff5722","#607d8b","#795548","#9c27b0",
    "#03a9f4","#cddc39","#ff9800","#673ab7","#009688",
]


class TSPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("応急危険度判定 mTSP — 時間制約付き巡回計画")
        self.root.geometry("1420x900")

        self.nodes        = []    # 建物座標リスト [(x, y), ...]  座標系: [0,1]x[0,1]
        self.inspect_times= []    # 建物ごとの判定時間 [秒]
        self.depot_idx    = 0     # デポの建物インデックス
        self.solution     = None  # mtsp_core.InspectionSolution
        self.solving      = False
        self.n_cpu        = os.cpu_count() or 4
        self.depot_mode_var = tk.StringVar(value="center")

        self._build_ui()

    # ── UI 構築 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # 左側コントロールパネル
        ctrl = tk.Frame(self.root, width=310, bg="#f0f0f0", padx=10, pady=8)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="応急危険度判定 mTSP",
                 font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(0,3))
        tk.Label(ctrl, text=f"CPU: {self.n_cpu} コア",
                 bg="#f0f0f0", fg="#888", font=("Arial", 8)).pack(anchor="w")

        # 建物生成セクション
        self._sep(ctrl, "建物生成")
        self._row(ctrl, "建物数:",            "n_var",    "500", 9)
        self._row(ctrl, "エリア (km):",       "area_var", "10",  6)
        self._row(ctrl, "判定時間 最小(分):", "tmin_var", "15",  6)
        self._row(ctrl, "判定時間 最大(分):", "tmax_var", "45",  6)
        tk.Label(ctrl, text="建物ごとに判定時間をランダム設定",
                 bg="#f0f0f0", fg="#666", font=("Arial", 8)).pack(anchor="w")
        tk.Button(ctrl, text="ランダム建物生成", command=self.generate_random,
                  bg="#4CAF50", fg="white", relief=tk.FLAT, pady=4
                  ).pack(fill=tk.X, pady=2)
        tk.Button(ctrl, text="クリア", command=self.clear_nodes,
                  bg="#f44336", fg="white", relief=tk.FLAT, pady=3
                  ).pack(fill=tk.X, pady=1)

        # デポ設定セクション
        self._sep(ctrl, "デポ（拠点）")
        for val, lbl in [("center","中心に自動配置"),
                          ("random","ランダム"),
                          ("click", "クリックで指定")]:
            tk.Radiobutton(ctrl, text=lbl, variable=self.depot_mode_var,
                           value=val, bg="#f0f0f0").pack(anchor="w")
        self.depot_label = tk.Label(ctrl, text="デポ: 未設定",
                                    bg="#f0f0f0", fg="#555", font=("Arial",8), anchor="w")
        self.depot_label.pack(fill=tk.X)

        # 制約条件セクション
        self._sep(ctrl, "制約条件")
        self._row(ctrl, "判定士数 m:",        "m_var",       "4",              6)
        self._row(ctrl, "移動速度 (km/h):",   "speed_var",  "30",              6)
        self._row(ctrl, "最大稼働時間 (h):",  "maxwork_var", "8",              6)
        self._row(ctrl, "並列試行回数:",      "starts_var",  str(self.n_cpu),  6)
        self._row(ctrl, "並列数:",            "workers_var", str(self.n_cpu),  6)
        tk.Label(ctrl,
                 text="目的関数: 最大終了時間(makespan)を最小化\n"
                      "→ 最も遅い担当者が早く終わるよう割当",
                 bg="#f0f0f0", fg="#1565C0", font=("Arial",8),
                 justify=tk.LEFT).pack(anchor="w", pady=3)

        # 実行ボタン群
        self.solve_btn = tk.Button(
            ctrl, text="計画を実行", command=self.solve_tsp,
            bg="#2196F3", fg="white", relief=tk.FLAT, pady=7,
            font=("Arial", 10, "bold"))
        self.solve_btn.pack(fill=tk.X, pady=4)

        self.min_m_btn = tk.Button(
            ctrl, text="必要判定士数を計算", command=self.calc_min_m,
            bg="#00796B", fg="white", relief=tk.FLAT, pady=5,
            font=("Arial", 9, "bold"))
        self.min_m_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = tk.Button(
            ctrl, text="中止", command=lambda: setattr(self,"solving",False),
            bg="#FF9800", fg="white", relief=tk.FLAT, pady=3,
            state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=1)

        # 結果表示セクション
        self._sep(ctrl, "結果")
        self.stat_nodes    = self._lbl(ctrl, "建物数: 0")
        self.stat_m        = self._lbl(ctrl, "判定士数: -")
        self.stat_min_m    = self._lbl(ctrl, "最小必要判定士数: -",
                                       bold=True, color="#00796B")
        self.stat_makespan = self._lbl(ctrl, "最大終了時間: -", bold=True)
        self.stat_unassign = self._lbl(ctrl, "未割当建物: -", color="#e74c3c")
        self.stat_dist     = self._lbl(ctrl, "総移動距離: -")
        self.stat_time     = self._lbl(ctrl, "計算時間: -")
        self.stat_status   = tk.Label(ctrl, text="状態: 待機中", bg="#f0f0f0",
                                      fg="#333", anchor="w", wraplength=270,
                                      justify=tk.LEFT)
        self.stat_status.pack(fill=tk.X)

        # 判定士ごとの稼働時間・移動距離テーブル
        self.per_text = tk.Text(ctrl, height=7, font=("Courier", 8),
                                state=tk.DISABLED)
        self.per_text.pack(fill=tk.X, pady=4)

        self.progress = ttk.Progressbar(ctrl, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=4)

        self._sep(ctrl, "ベンチマーク")
        self.bench_btn = tk.Button(
            ctrl, text="ベンチマーク実行", command=self.run_benchmark,
            bg="#9C27B0", fg="white", relief=tk.FLAT, pady=4)
        self.bench_btn.pack(fill=tk.X, pady=3)

        # 右側キャンバス（matplotlib）
        right = tk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1a1a2e")
        self.fig.patch.set_facecolor("#16213e")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self._redraw()

    def _sep(self, parent, text):
        """セパレータ＋セクションラベルを追加するヘルパー"""
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=5)
        tk.Label(parent, text=text, font=("Arial", 9, "bold"),
                 bg="#f0f0f0", anchor="w").pack(fill=tk.X)

    def _row(self, parent, label, attr, default, width):
        """ラベル＋入力フィールドの行を追加するヘルパー"""
        frm = tk.Frame(parent, bg="#f0f0f0")
        frm.pack(fill=tk.X, pady=2)
        tk.Label(frm, text=label, bg="#f0f0f0", width=16, anchor="w").pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        setattr(self, attr, var)
        tk.Entry(frm, textvariable=var, width=width).pack(side=tk.LEFT)

    def _lbl(self, parent, text, bold=False, color="#333"):
        """統計表示ラベルを追加するヘルパー"""
        font = ("Arial", 9, "bold") if bold else ("Arial", 9)
        lbl = tk.Label(parent, text=text, bg="#f0f0f0", fg=color,
                       anchor="w", font=font)
        lbl.pack(fill=tk.X)
        return lbl

    # ── 建物管理 ─────────────────────────────────────────────────────────────

    def generate_random(self):
        """指定棟数の建物をランダム配置し、判定時間をランダム設定する"""
        try:
            n     = int(self.n_var.get());      assert 2 <= n <= 2_000_000
            t_min = float(self.tmin_var.get()); assert t_min > 0
            t_max = float(self.tmax_var.get()); assert t_max >= t_min
        except Exception:
            messagebox.showerror("エラー", "入力値を確認してください")
            return

        xy = np.random.random((n, 2))
        self.nodes = list(map(tuple, xy.tolist()))
        # 判定時間を [t_min, t_max] 分の範囲でランダム設定（秒換算で保持）
        self.inspect_times = np.random.uniform(
            t_min * 60, t_max * 60, n).tolist()
        self.solution = None
        self.depot_idx = self._auto_depot()
        self.stat_nodes.config(text=f"建物数: {n:,}")
        self.depot_label.config(text=f"デポ: 拠点 #{self.depot_idx}")
        self._reset_result_stats()
        self._redraw()

    def clear_nodes(self):
        self.nodes = []
        self.inspect_times = []
        self.solution = None
        self.stat_nodes.config(text="建物数: 0")
        self._reset_result_stats()
        self._redraw()

    def _reset_result_stats(self):
        self.stat_m.config(text="判定士数: -")
        self.stat_min_m.config(text="最小必要判定士数: -")
        self.stat_makespan.config(text="最大終了時間: -")
        self.stat_unassign.config(text="未割当建物: -")
        self.stat_dist.config(text="総移動距離: -")
        self.stat_time.config(text="計算時間: -")
        self.stat_status.config(text="状態: 待機中")
        self._update_per_text([])

    def _make_problem(self, nodes, inspect_times, depot_idx,
                      area_km, speed, max_work):
        """GUI 入力値から InspectionProblem を構築する"""
        return InspectionProblem(
            coords=np.array(nodes, dtype=np.float64),
            inspect_times=np.array(inspect_times, dtype=np.float64),
            depot_idx=depot_idx,
            area_km=area_km,
            speed_kmh=speed,
            max_work_h=max_work)

    def calc_min_m(self):
        """全棟割当可能な最小判定士数を二分探索で求める（mtsp_core に委譲）"""
        if len(self.nodes) < 2:
            messagebox.showwarning("警告", "建物を2棟以上追加してください")
            return
        if self.solving:
            return
        try:
            speed    = float(self.speed_var.get());    assert speed > 0
            max_work = float(self.maxwork_var.get());  assert max_work > 0
            area_km  = float(self.area_var.get());     assert area_km > 0
        except Exception:
            messagebox.showerror("エラー", "移動速度・最大稼働時間・エリアを確認してください")
            return

        self.solving = True
        self.solve_btn.config(state=tk.DISABLED)
        self.min_m_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.stat_status.config(text="状態: 最小判定士数を探索中...")
        self.progress.start(10)

        threading.Thread(
            target=self._min_m_worker,
            args=(list(self.nodes), list(self.inspect_times),
                  speed, max_work, area_km),
            daemon=True).start()

    def _min_m_worker(self, nodes, inspect_times, speed, max_work, area_km):
        """二分探索のバックグラウンドスレッド本体"""
        t0 = time.perf_counter()
        try:
            problem = self._make_problem(
                nodes, inspect_times, self._auto_depot(),
                area_km, speed, max_work)
            min_m, sol = find_min_inspectors(
                problem, GreedySolver(),
                progress=lambda mid: self.root.after(
                    0, self.stat_status.config,
                    {"text": f"状態: 探索中... m={mid} を試行"}),
                should_continue=lambda: self.solving)
            elapsed = time.perf_counter() - t0
            self.root.after(0, self._on_min_m_done, min_m, sol, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.root.after(0, self._on_min_m_done, None, None, elapsed, f"エラー: {e}")

    def _on_min_m_done(self, min_m, sol, elapsed, err=None):
        """最小判定士数探索完了時のコールバック（メインスレッドで実行）"""
        self.solving = False
        self.solve_btn.config(state=tk.NORMAL)
        self.min_m_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()

        if err:
            self.stat_status.config(text=f"状態: {err}")
            return

        self.stat_min_m.config(text=f"最小必要判定士数: {min_m} 人")
        self.stat_time.config(text=f"計算時間: {elapsed:.2f} 秒")
        self.m_var.set(str(min_m))  # 判定士数入力欄に反映

        if sol:
            self.solution  = sol
            self.depot_idx = sol.problem.depot_idx
            h = int(sol.makespan); mn = int((sol.makespan - h) * 60)
            self.stat_makespan.config(text=f"最大終了時間: {h}h{mn:02d}m")
            self.stat_unassign.config(text="未割当建物: 0 棟 (全棟完了)",
                                      fg="#2e7d32")
            self.stat_dist.config(text=f"総移動距離: {sol.total_dist:.1f} km")
            self.stat_m.config(text=f"判定士数: {min_m} (最小)")
            self._update_per_text(sol.per_time, sol.per_dist)
            self._redraw()

        self.stat_status.config(
            text=f"状態: 完了 — 最小 {min_m} 人で全棟対応可能")

    def _auto_depot(self):
        """デポ設定モードに応じてデポのインデックスを返す"""
        mode = self.depot_mode_var.get()
        if not self.nodes:
            return 0
        if mode == "center":
            # エリア重心に最も近い建物をデポに設定
            arr = np.array(self.nodes)
            c   = arr.mean(axis=0)
            return int(np.argmin(np.linalg.norm(arr - c, axis=1)))
        return random.randrange(len(self.nodes))

    def on_canvas_click(self, event):
        """キャンバスクリック: デポ指定モードならデポを変更、通常は建物を追加"""
        if event.inaxes != self.ax or self.solving:
            return
        x, y = event.xdata, event.ydata
        if x is None:
            return
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        nx = (x - xlim[0]) / (xlim[1] - xlim[0])
        ny = (y - ylim[0]) / (ylim[1] - ylim[0])

        if self.depot_mode_var.get() == "click" and self.nodes:
            # クリック位置に最も近い既存建物をデポに設定
            arr = np.array(self.nodes)
            self.depot_idx = int(
                np.argmin(np.linalg.norm(arr - [nx, ny], axis=1)))
            self.depot_label.config(
                text=f"デポ: 拠点 #{self.depot_idx} (クリック指定)")
            self.solution = None
            self._redraw()
        else:
            try:
                t_min = float(self.tmin_var.get())
                t_max = float(self.tmax_var.get())
            except Exception:
                t_min, t_max = 15, 45
            self.nodes.append((nx, ny))
            self.inspect_times.append(random.uniform(t_min * 60, t_max * 60))
            self.solution = None
            self.stat_nodes.config(text=f"建物数: {len(self.nodes):,}")
            self._redraw()

    # ── 描画 ─────────────────────────────────────────────────────────────────

    def _redraw(self):
        """
        matplotlib キャンバスを再描画する。
        fig.clear() で全要素（カラーバー含む）をリセットしてから再描画することで
        ランダム生成を繰り返してもカラーバーが重複しない。
        """
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1a1a2e")
        self.ax.set_xlim(-0.02, 1.02)
        self.ax.set_ylim(-0.02, 1.02)
        self.ax.tick_params(colors="#aaa")

        n = len(self.nodes)
        n_unassigned = self.solution.n_unassigned if self.solution else 0
        extra = f"  未割当: {n_unassigned}" if n_unassigned else ""
        self.ax.set_title(
            f"応急危険度判定 mTSP  ({n:,} 建物{extra})",
            color="white", fontsize=11)

        if not self.nodes:
            self.canvas.draw_idle()
            return

        coords = np.array(self.nodes)

        # 判定士ごとのルートを色分けして描画
        if self.solution:
            for s, route in enumerate(self.solution.routes):
                color = COLORS[s % len(COLORS)]
                rc    = coords[route]
                lw    = max(0.4, 1.4 - n / 25000)
                self.ax.plot(rc[:, 0], rc[:, 1], "-",
                             color=color, linewidth=lw, alpha=0.75, zorder=2)

        # 建物を判定時間のヒートマップで描画（黄→赤: 短い→長い）
        if self.inspect_times and len(self.inspect_times) == n:
            it   = np.array(self.inspect_times) / 60.0  # 秒→分
            size = max(2, 25 - n // 1000)
            sc   = self.ax.scatter(coords[:, 0], coords[:, 1],
                                   c=it, cmap="YlOrRd", s=size, alpha=0.75,
                                   zorder=3, vmin=it.min(), vmax=it.max())
            try:
                cb = self.fig.colorbar(sc, ax=self.ax, fraction=0.03, pad=0.01)
                cb.set_label("判定時間 (分)", color="white", fontsize=8)
                cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            except Exception:
                pass
        else:
            size = max(2, 25 - n // 1000)
            self.ax.scatter(coords[:, 0], coords[:, 1],
                            c="#76ff03", s=size, alpha=0.75, zorder=3)

        # デポを金色の星マークで表示
        dep = self.depot_idx if self.nodes else 0
        if 0 <= dep < n:
            self.ax.scatter([coords[dep, 0]], [coords[dep, 1]],
                            c="#FFD700", s=200, marker="*", zorder=6,
                            label="デポ")
            self.ax.legend(loc="upper right", fontsize=9,
                           facecolor="#1a1a2e", labelcolor="white")

        self.canvas.draw_idle()

    def _update_per_text(self, per_time, per_dist=None):
        """判定士ごとの稼働時間・移動距離テーブルを更新する"""
        self.per_text.config(state=tk.NORMAL)
        self.per_text.delete("1.0", tk.END)
        if per_time:
            self.per_text.insert(
                tk.END, f"{'担当者':>5}  {'稼働時間(h)':>11}  {'移動距離(km)':>12}\n")
            self.per_text.insert(tk.END, "-" * 34 + "\n")
            for i, t in enumerate(per_time):
                d_str = f"{per_dist[i]:>12.2f}" if per_dist else ""
                self.per_text.insert(
                    tk.END, f"  #{i+1:2d}   {t:>11.3f}  {d_str}\n")
        self.per_text.config(state=tk.DISABLED)

    # ── ソルバー実行 ─────────────────────────────────────────────────────────

    def solve_tsp(self):
        """「計画を実行」ボタンのハンドラ。バックグラウンドスレッドで解を求める。"""
        if len(self.nodes) < 2:
            messagebox.showwarning("警告", "建物を2棟以上追加してください")
            return
        if self.solving:
            return
        try:
            m         = max(1, int(self.m_var.get()))
            speed     = float(self.speed_var.get());    assert speed > 0
            max_work  = float(self.maxwork_var.get());  assert max_work > 0
            area_km   = float(self.area_var.get());     assert area_km > 0
            n_workers = max(1, int(self.workers_var.get()))
            n_starts  = max(1, int(self.starts_var.get()))
        except Exception:
            messagebox.showerror("エラー", "入力値を確認してください")
            return

        if m > len(COLORS):
            messagebox.showerror("エラー", f"判定士数は {len(COLORS)} 以下にしてください")
            return

        # デポ候補リストを生成（並列試行回数分）
        mode = self.depot_mode_var.get()
        if mode == "click":
            depots = [self.depot_idx]
        elif mode == "center":
            depots = [self._auto_depot()]
        else:
            depots = random.sample(range(len(self.nodes)),
                                   min(n_starts, len(self.nodes)))

        self.solving = True
        self.solve_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.stat_status.config(text="状態: 計算中...")
        self.stat_m.config(text=f"判定士数: {m}")
        self.progress.start(10)

        threading.Thread(
            target=self._solve_worker,
            args=(list(self.nodes), list(self.inspect_times),
                  m, speed, max_work, area_km, n_workers, depots),
            daemon=True).start()

    def _solve_worker(self, nodes, inspect_times, m, speed,
                      max_work, area_km, n_workers, depots):
        """ソルバーのバックグラウンドスレッド本体（mtsp_core に委譲）"""
        t0 = time.perf_counter()
        try:
            problem = self._make_problem(
                nodes, inspect_times, depots[0], area_km, speed, max_work)
            solver  = MultiStartSolver(depot_candidates=depots,
                                       n_workers=n_workers)
            sol     = solver.solve(problem, m)
            elapsed = time.perf_counter() - t0
            self.root.after(0, self._on_done, sol, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.root.after(0, self._on_done, None, elapsed, f"エラー: {e}")

    def _on_done(self, sol, elapsed, err=None):
        """ソルバー完了時のコールバック（メインスレッドで実行）"""
        self.solving = False
        self.solve_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.stop()

        if err:
            self.stat_status.config(text=f"状態: {err}")
            return
        if sol is None:
            self.stat_status.config(text="状態: 失敗")
            return

        self.solution  = sol
        self.depot_idx = sol.problem.depot_idx
        n_unassigned   = sol.n_unassigned

        h = int(sol.makespan); mn = int((sol.makespan - h) * 60)
        self.stat_makespan.config(text=f"最大終了時間: {h}h{mn:02d}m")

        # 未割当があれば赤字で警告
        color  = "#e74c3c" if n_unassigned > 0 else "#2e7d32"
        ua_txt = (f"未割当建物: {n_unassigned:,} 棟 ← 時間不足"
                  if n_unassigned else "未割当建物: 0 棟 (全棟完了)")
        self.stat_unassign.config(text=ua_txt, fg=color)
        self.stat_dist.config(text=f"総移動距離: {sol.total_dist:.1f} km")
        self.stat_time.config(text=f"計算時間: {elapsed:.3f} 秒")
        self.stat_status.config(text="状態: 完了")
        self._update_per_text(sol.per_time, sol.per_dist)
        self._redraw()

    # ── ベンチマーク ─────────────────────────────────────────────────────────

    def run_benchmark(self):
        """建物数を段階的に増やして計算時間・makespan を計測する"""
        if self.solving:
            return
        try:
            m        = max(1, int(self.m_var.get()))
            speed    = float(self.speed_var.get())
            max_work = float(self.maxwork_var.get())
            area_km  = float(self.area_var.get())
        except Exception:
            m, speed, max_work, area_km = 4, 30, 8, 10
        self.bench_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        threading.Thread(target=self._bench_worker,
                         args=(m, speed, max_work, area_km),
                         daemon=True).start()

    def _bench_worker(self, m, speed, max_work, area_km):
        sizes   = [100, 500, 1000, 5000, 10000, 50000, 100000]
        solver  = GreedySolver()
        results = []
        for n in sizes:
            self.root.after(0, self.stat_status.config,
                            {"text": f"ベンチマーク中: n={n:,}"})
            problem = InspectionProblem(
                coords=np.random.random((n, 2)).astype(np.float64),
                inspect_times=np.random.uniform(15*60, 45*60, n),
                depot_idx=0,
                area_km=area_km, speed_kmh=speed, max_work_h=max_work)
            t0      = time.perf_counter()
            sol     = solver.solve(problem, m)
            elapsed = time.perf_counter() - t0
            results.append((n, elapsed, sol.makespan, sol.n_unassigned))
        self.root.after(0, self._show_bench, results, m, max_work)

    def _show_bench(self, results, m, max_work):
        self.progress.stop()
        self.bench_btn.config(state=tk.NORMAL)
        self.stat_status.config(text="状態: ベンチマーク完了")

        win = tk.Toplevel(self.root)
        win.title(f"ベンチマーク結果 (m={m} 人, 最大{max_work}h)")
        win.geometry("700x520")

        fig = Figure(figsize=(7, 5))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ns       = [r[0] for r in results]
        times    = [r[1] for r in results]
        makespan = [r[2] for r in results]
        unassign = [r[3] for r in results]

        ax1.loglog(ns, times, "o-", color="#2196F3")
        ax1.set_xlabel("建物数")
        ax1.set_ylabel("計算時間 (秒, log)")
        ax1.set_title(f"建物数 vs 計算時間 (m={m}人)")
        ax1.grid(True, alpha=0.3, which="both")

        ax2_r = ax2.twinx()
        ax2.semilogx(ns, makespan, "s-",  color="#e74c3c", label="最大終了時間 (h)")
        ax2_r.semilogx(ns, unassign, "^--", color="#9b59b6", label="未割当棟数")
        ax2.set_xlabel("建物数")
        ax2.set_ylabel("最大終了時間 (h)", color="#e74c3c")
        ax2_r.set_ylabel("未割当棟数",     color="#9b59b6")
        ax2.set_title("Makespan と未割当建物数")
        ax2.grid(True, alpha=0.3)
        lines1, lbls1 = ax2.get_legend_handles_labels()
        lines2, lbls2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1+lines2, lbls1+lbls2, fontsize=8)

        fig.tight_layout(pad=2)
        c = FigureCanvasTkAgg(fig, master=win)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        txt = tk.Text(win, height=7, font=("Courier", 9))
        txt.pack(fill=tk.X, padx=8, pady=4)
        txt.insert(tk.END,
            f"{'建物数':>8}  {'計算時間(秒)':>12}  {'Makespan(h)':>12}  {'未割当':>8}\n")
        txt.insert(tk.END, "-" * 48 + "\n")
        for n, t, ms, ua in results:
            txt.insert(tk.END,
                f"{n:>8,}  {t:>12.3f}  {ms:>12.3f}  {ua:>8,}\n")
        txt.config(state=tk.DISABLED)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = TSPApp(root)
    root.mainloop()
