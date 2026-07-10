"""
応急危険度判定 mTSP - 時間制約付き複数巡回セールスマン問題 GUI
Time-Constrained mTSP for Emergency Building Inspection

【概要】
地震等の災害後に複数の判定士が拠点（デポ）から出発し、
エリア内の全建物を手分けして応急危険度判定を行う巡回計画を作成する。

ソルバー本体は mtsp_core モジュールに分離されており、本ファイルは
GUI（tkinter + matplotlib）のみを担当する。

【依存ライブラリ】
    pip install scipy matplotlib numpy ortools
    （ortools は「OR-Tools改善」ソルバーを使う場合のみ必要）
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
    InspectionProblem, GreedySolver, MultiStartSolver, ORToolsSolver,
    find_min_inspectors,
)

# ORToolsSolver が密行列で扱える上限（これ以上は貪欲法のみ）
ORTOOLS_MAX_NODES = 2000


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
        self.priorities   = []    # 建物ごとの優先度 p_i (0=通常, 1=優先)
        self.depot_idx    = 0     # デポの建物インデックス
        self.solution     = None  # mtsp_core.InspectionSolution
        self.solving      = False
        self.n_cpu        = os.cpu_count() or 4
        self.depot_mode_var = tk.StringVar(value="center")
        self.solver_var     = tk.StringVar(value="greedy")
        self.last_elapsed   = None   # 直近の計算時間 [秒]
        self.last_solver    = None   # 直近に使ったソルバー名

        self._build_ui()

    # ── UI 構築 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # 左側コントロールパネル（縦スクロール対応）
        # Canvas の中に Frame を埋め込み、ウィンドウが小さいときは
        # スクロールバー / マウスホイールで下部の項目にアクセスできる
        outer = tk.Frame(self.root, width=330, bg="#f0f0f0")
        outer.pack(side=tk.LEFT, fill=tk.Y)
        outer.pack_propagate(False)

        ctrl_canvas = tk.Canvas(outer, bg="#f0f0f0", highlightthickness=0)
        ctrl_bar = ttk.Scrollbar(outer, orient="vertical",
                                 command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_bar.set)
        ctrl_bar.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ctrl = tk.Frame(ctrl_canvas, bg="#f0f0f0", padx=10, pady=8)
        ctrl_win = ctrl_canvas.create_window((0, 0), window=ctrl, anchor="nw")

        # 内部 Frame のサイズ変化に合わせてスクロール範囲を更新
        ctrl.bind("<Configure>", lambda e: ctrl_canvas.configure(
            scrollregion=ctrl_canvas.bbox("all")))
        # Canvas の幅に内部 Frame の幅を追従させる
        ctrl_canvas.bind("<Configure>", lambda e: ctrl_canvas.itemconfigure(
            ctrl_win, width=e.width))

        # マウスホイール: ポインタがパネル上にあるときだけスクロール
        def _on_mousewheel(event):
            w = self.root.winfo_containing(event.x_root, event.y_root)
            while w is not None:
                if w is outer:
                    ctrl_canvas.yview_scroll(
                        int(-event.delta / 120), "units")
                    break
                w = w.master
        self.root.bind_all("<MouseWheel>", _on_mousewheel)

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
        self._row(ctrl, "優先建物割合 (%):",  "prio_var", "10",  6)
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
                 text="目的関数: min MΣz + λΣT + μΣp·a\n"
                      "(人数・総活動時間・優先建物の早期判定)",
                 bg="#f0f0f0", fg="#1565C0", font=("Arial",8),
                 justify=tk.LEFT).pack(anchor="w", pady=3)

        # ソルバー選択セクション
        self._sep(ctrl, "ソルバー")
        for val, lbl in [("greedy",  "貪欲法（高速・大規模対応）"),
                          ("ortools", "貪欲法 + OR-Tools最適化")]:
            tk.Radiobutton(ctrl, text=lbl, variable=self.solver_var,
                           value=val, bg="#f0f0f0").pack(anchor="w")
        self._row(ctrl, "改善時間制限 (秒):", "limit_var", "10", 6)
        # 目的関数の重み M / λ / μ を 1 行にまとめて入力
        wfrm = tk.Frame(ctrl, bg="#f0f0f0")
        wfrm.pack(fill=tk.X, pady=2)
        tk.Label(wfrm, text="重み", bg="#f0f0f0", width=4,
                 anchor="w").pack(side=tk.LEFT)
        for sym, attr, default in [("M:", "wm_var", "1000"),
                                    ("λ:", "wl_var", "1"),
                                    ("μ:", "wp_var", "1")]:
            tk.Label(wfrm, text=sym, bg="#f0f0f0").pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            tk.Entry(wfrm, textvariable=var, width=5).pack(side=tk.LEFT, padx=(0,4))
        tk.Label(ctrl,
                 text=f"OR-Tools最適化は {ORTOOLS_MAX_NODES:,} 棟まで",
                 bg="#f0f0f0", fg="#666", font=("Arial",8),
                 justify=tk.LEFT).pack(anchor="w")

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

        self.summary_btn = tk.Button(
            ctrl, text="サマリーを表示", command=self.show_summary,
            bg="#455A64", fg="white", relief=tk.FLAT, pady=3)
        self.summary_btn.pack(fill=tk.X, pady=2)

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
            p_pct = float(self.prio_var.get()); assert 0 <= p_pct <= 100
        except Exception:
            messagebox.showerror("エラー", "入力値を確認してください")
            return

        xy = np.random.random((n, 2))
        self.nodes = list(map(tuple, xy.tolist()))
        # 判定時間を [t_min, t_max] 分の範囲でランダム設定（秒換算で保持）
        self.inspect_times = np.random.uniform(
            t_min * 60, t_max * 60, n).tolist()
        # 指定割合の建物を優先建物 (p_i = 1) に設定
        self.priorities = (
            np.random.random(n) < p_pct / 100.0).astype(float).tolist()
        self.solution = None
        self.depot_idx = self._auto_depot()
        self.stat_nodes.config(text=f"建物数: {n:,}")
        self.depot_label.config(text=f"デポ: 拠点 #{self.depot_idx}")
        self._reset_result_stats()
        self._redraw()

    def clear_nodes(self):
        self.nodes = []
        self.inspect_times = []
        self.priorities = []
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
                      area_km, speed, max_work, priorities=None):
        """GUI 入力値から InspectionProblem を構築する"""
        return InspectionProblem(
            coords=np.array(nodes, dtype=np.float64),
            inspect_times=np.array(inspect_times, dtype=np.float64),
            depot_idx=depot_idx,
            area_km=area_km,
            speed_kmh=speed,
            max_work_h=max_work,
            priorities=(np.array(priorities, dtype=np.float64)
                        if priorities else None))

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
            self.solution     = sol
            self.depot_idx    = sol.problem.depot_idx
            self.last_elapsed = elapsed
            self.last_solver  = "貪欲法（最小人数の二分探索）"
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
                p_pct = float(self.prio_var.get())
            except Exception:
                t_min, t_max, p_pct = 15, 45, 10
            self.nodes.append((nx, ny))
            self.inspect_times.append(random.uniform(t_min * 60, t_max * 60))
            self.priorities.append(
                1.0 if random.random() < p_pct / 100.0 else 0.0)
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

        # 優先建物を二重丸で強調表示（検査済み=シアン、未検査=赤）
        if self.priorities and len(self.priorities) == n:
            pr = np.array(self.priorities)
            if pr.any():
                size = max(30, 60 - n // 100)
                prio_mask = pr > 0
                if (self.solution is not None
                        and self.solution.problem.n_buildings == n):
                    a = self.solution.arrival_times
                    done = prio_mask & ~np.isnan(a)
                    pend = prio_mask & np.isnan(a)
                    pend[self.depot_idx] = False
                    if done.any():
                        dc = coords[done]
                        self.ax.scatter(dc[:, 0], dc[:, 1],
                                        facecolors="none", edgecolors="#00e5ff",
                                        s=size, linewidths=1.2, zorder=5,
                                        label=f"優先建物 検査済 ({done.sum()})")
                    if pend.any():
                        pc = coords[pend]
                        self.ax.scatter(pc[:, 0], pc[:, 1],
                                        facecolors="none", edgecolors="#ff1744",
                                        s=size * 1.6, linewidths=1.8, zorder=5,
                                        label=f"優先建物 未検査 ({pend.sum()})")
                else:
                    pc = coords[prio_mask]
                    self.ax.scatter(pc[:, 0], pc[:, 1],
                                    facecolors="none", edgecolors="#00e5ff",
                                    s=size, linewidths=1.2, zorder=5,
                                    label=f"優先建物 ({prio_mask.sum()})")

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

    # ── サマリー ─────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_hm(hours):
        """時間 [h] を「XhYYm」形式の文字列にする"""
        total_min = int(round(hours * 60))
        return f"{total_min // 60}h{total_min % 60:02d}m"

    def show_summary(self):
        """直近の計画のサマリー（建物・体制・時間・日数・コスト）を表示する"""
        if self.solution is None:
            messagebox.showinfo("サマリー", "先に「計画を実行」してください")
            return
        sol = self.solution
        p   = sol.problem
        try:
            w_m = float(self.wm_var.get())
            w_l = float(self.wl_var.get())
            w_p = float(self.wp_var.get())
        except Exception:
            w_m, w_l, w_p = 1000.0, 1.0, 1.0

        n_total     = p.n_buildings - 1            # デポを除いた検査対象数
        n_inspected = sum(len(r) - 2 for r in sol.routes)
        n_unassign  = sol.n_unassigned
        rate        = n_inspected / n_total * 100 if n_total else 0.0

        # 優先建物の統計
        prio_mask  = p.priorities > 0
        n_prio     = sol.n_priority
        prio_done  = sol.n_priority_done
        a          = sol.arrival_times
        prio_mean  = (float(np.nanmean(a[prio_mask]))
                      if prio_done > 0 else None)
        all_mean   = (float(np.nanmean(a)) if n_inspected > 0 else None)

        # 所要日数の推定: この計画を 1 日分として、同じペースで
        # 全棟を検査し終えるまでの日数（未割当ゼロなら 1 日）
        if n_unassign == 0:
            days_txt = "1 日で全棟完了"
        elif n_inspected > 0:
            est_days = int(np.ceil(n_total / n_inspected))
            days_txt = (f"約 {est_days} 日 "
                        f"(1日 {n_inspected:,} 棟のペースで全 {n_total:,} 棟)")
        else:
            days_txt = "算出不可（検査済み 0 棟）"

        # 目的関数コストの内訳
        cost_m   = w_m * sol.n_used
        cost_l   = w_l * sol.total_time
        cost_p   = w_p * sol.priority_cost
        cost_sum = cost_m + cost_l + cost_p

        used  = [t for t in sol.per_time if t > 0]
        avg_t = (sum(used) / len(used)) if used else 0.0
        avg_d = (sol.total_dist / sol.n_used) if sol.n_used else 0.0

        prio_rate = prio_done / n_prio * 100 if n_prio else 0.0
        prio_left = n_prio - prio_done
        lines = [
            "■ 建物",
            f"  検査対象（デポ除く）: {n_total:,} 棟",
            f"  検査済み            : {n_inspected:,} 棟 ({rate:.1f}%)",
            f"  未割当              : {n_unassign:,} 棟",
            f"  優先建物            : {n_prio:,} 棟",
            f"    検査済み          : {prio_done:,} 棟 ({prio_rate:.1f}%)"
            + ("  ← 全て検査済み" if n_prio and prio_left == 0 else ""),
        ]
        if prio_left > 0:
            lines.append(f"    未検査            : {prio_left:,} 棟 "
                         f"※地図上で赤丸表示")
        if prio_mean is not None:
            lines.append(f"  優先建物の平均判定開始: {prio_mean:.2f} h"
                         + (f"（全体平均 {all_mean:.2f} h）" if all_mean else ""))
        lines += [
            "",
            "■ 体制・時間",
            f"  判定士              : 使用 {sol.n_used} 人 / 上限 {sol.n_inspectors} 人",
            f"  最大終了時間        : {self._fmt_hm(sol.makespan)}"
            f"（稼働上限 {p.max_work_h:g} h）",
            f"  総活動時間 ΣT_q     : {sol.total_time:.2f} h",
            f"  平均稼働時間        : {avg_t:.2f} h/人",
            f"  総移動距離          : {sol.total_dist:.1f} km"
            f"（平均 {avg_d:.1f} km/人）",
            "",
            "■ 所要日数（推定）",
            f"  {days_txt}",
            "",
            "■ コスト（目的関数）  min MΣz + λΣT + μΣp·a",
            f"  人数     M Σz_q  (M={w_m:g}) : {cost_m:,.1f}",
            f"  総活動   λ ΣT_q  (λ={w_l:g}) : {cost_l:,.1f}",
            f"  優先     μ Σp·a  (μ={w_p:g}) : {cost_p:,.1f}",
            f"  {'─' * 30}",
            f"  合計                : {cost_sum:,.1f}",
            "",
            "■ 計算",
            f"  ソルバー            : {self.last_solver or '-'}",
            f"  計算時間            : "
            + (f"{self.last_elapsed:.2f} 秒" if self.last_elapsed else "-"),
        ]

        win = tk.Toplevel(self.root)
        win.title("計画サマリー")
        win.geometry("460x560")
        txt = tk.Text(win, font=("Courier", 10), padx=10, pady=8)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert(tk.END, "\n".join(lines))
        txt.config(state=tk.DISABLED)

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
            time_limit = float(self.limit_var.get()); assert time_limit > 0
            w_m = float(self.wm_var.get()); assert w_m >= 0
            w_l = float(self.wl_var.get()); assert w_l >= 0
            w_p = float(self.wp_var.get()); assert w_p >= 0
        except Exception:
            messagebox.showerror("エラー", "入力値を確認してください")
            return

        if m > len(COLORS):
            messagebox.showerror("エラー", f"判定士数は {len(COLORS)} 以下にしてください")
            return

        use_ortools = self.solver_var.get() == "ortools"
        if use_ortools and len(self.nodes) > ORTOOLS_MAX_NODES:
            messagebox.showerror(
                "エラー",
                f"OR-Tools改善は {ORTOOLS_MAX_NODES:,} 棟までです。\n"
                f"建物数を減らすか、貪欲法を選択してください")
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
                  list(self.priorities),
                  m, speed, max_work, area_km, n_workers, depots,
                  use_ortools, time_limit, (w_m, w_l, w_p)),
            daemon=True).start()

    def _solve_worker(self, nodes, inspect_times, priorities, m, speed,
                      max_work, area_km, n_workers, depots,
                      use_ortools, time_limit, weights):
        """ソルバーのバックグラウンドスレッド本体（mtsp_core に委譲）"""
        t0 = time.perf_counter()
        w_m, w_l, w_p = weights
        try:
            problem = self._make_problem(
                nodes, inspect_times, depots[0], area_km, speed, max_work,
                priorities)
            # まず貪欲法（マルチスタート）で構築解を得る
            solver = MultiStartSolver(depot_candidates=depots,
                                      n_workers=n_workers)
            sol  = solver.solve(problem, m)
            note = None
            if use_ortools:
                # 貪欲解を初期解として OR-Tools で最適化する
                self.root.after(0, self.stat_status.config,
                                {"text": f"状態: OR-Tools で最適化中... "
                                         f"(最大 {time_limit:.0f} 秒)"})
                obj_g = sol.objective(w_m, w_l, w_p)
                sol = ORToolsSolver(
                    time_limit_s=time_limit, weight_m=w_m,
                    weight_total=w_l, weight_priority=w_p,
                ).solve(sol.problem, m, initial=sol)
                obj_s = sol.objective(w_m, w_l, w_p)
                gain = (1 - obj_s / obj_g) * 100 if obj_g > 0 else 0.0
                note = (f"目的関数 {obj_g:,.1f} → {obj_s:,.1f} "
                        f"(改善 {gain:.1f}%)")
            elapsed = time.perf_counter() - t0
            self.root.after(0, self._on_done, sol, elapsed, None, note)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.root.after(0, self._on_done, None, elapsed, f"エラー: {e}")

    def _on_done(self, sol, elapsed, err=None, note=None):
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

        self.solution     = sol
        self.depot_idx    = sol.problem.depot_idx
        self.last_elapsed = elapsed
        self.last_solver  = ("貪欲法 + OR-Tools最適化"
                             if self.solver_var.get() == "ortools" else "貪欲法")
        n_unassigned      = sol.n_unassigned

        self.stat_m.config(
            text=f"判定士数: 使用 {sol.n_used} / 上限 {sol.n_inspectors} "
                 f"(総活動 {sol.total_time:.1f}h)")
        h = int(sol.makespan); mn = int((sol.makespan - h) * 60)
        self.stat_makespan.config(text=f"最大終了時間: {h}h{mn:02d}m")

        # 未割当があれば赤字で警告
        color  = "#e74c3c" if n_unassigned > 0 else "#2e7d32"
        ua_txt = (f"未割当建物: {n_unassigned:,} 棟 ← 時間不足"
                  if n_unassigned else "未割当建物: 0 棟 (全棟完了)")
        self.stat_unassign.config(text=ua_txt, fg=color)
        self.stat_dist.config(text=f"総移動距離: {sol.total_dist:.1f} km")
        self.stat_time.config(text=f"計算時間: {elapsed:.3f} 秒")
        self.stat_status.config(
            text=f"状態: 完了 — {note}" if note else "状態: 完了")
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
