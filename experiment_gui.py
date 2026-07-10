"""
応急危険度判定 mTSP - 実験ツール GUI

experiments.py（実験基盤）のフロントエンド。4種類の実験を
フォーム入力で設定・実行し、結果のグラフを画面内に表示する。

- ノイズ床測定    : 同一条件でシードのみ変えたばらつき（感度判定の基準）
- 必要人数曲線    : 規定時間 H ごとの最小必要人数（貪欲法 vs OR-Tools）
- スイープ        : 1 パラメータを動かす One-at-a-Time 感度分析
- トルネード図    : 全パラメータ ±X% の影響幅の比較

結果は自動的に results/ フォルダへ CSV + PNG 保存される。

【使い方】
    python experiment_gui.py
"""

import os
import threading
import queue

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from experiments import (
    Scenario, ExperimentRunner, StaffingAnalyzer, SensitivityAnalyzer,
    RESULTS_DIR,
)

# スイープ対象パラメータ（表示名）
PARAM_LABELS = {
    "speed_kmh": "移動速度 [km/h]",
    "t_min": "判定時間 最小 [分]",
    "t_max": "判定時間 最大 [分]",
    "area_km": "エリアサイズ [km]",
    "prio_pct": "優先建物割合 [%]",
    "max_work_h": "規定時間 H [h]",
    "mu": "μ（優先の重み）",
    "n": "建物数",
    "time_limit": "OR-Tools 時間制限 [秒]",
}
OUTPUT_LABELS = {
    "opt_used": "必要人数 [人]",
    "opt_total_h": "総活動時間 [h]",
    "opt_prio_start_h": "優先 平均判定開始 [h]",
    "opt_makespan_h": "makespan [h]",
}


class ExperimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("mTSP 実験ツール — 必要人数分析・感度分析")
        self.root.geometry("1480x900")
        self.running = False
        self.msg_queue = queue.Queue()
        self.exp_var = tk.StringVar(value="staffing")
        self._build_ui()
        self._poll_queue()

    # ── UI 構築 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        ctrl = tk.Frame(self.root, width=280, bg="#f0f0f0", padx=10, pady=8)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="実験ツール", font=("Arial", 12, "bold"),
                 bg="#f0f0f0").pack(pady=(0, 4))

        # 共通設定
        self._sep(ctrl, "共通設定")
        self._row(ctrl, "建物数:",              "n_var",     "200")
        self._row(ctrl, "シード数:",            "seeds_var", "5")
        self._row(ctrl, "OR-Tools制限 (秒):",   "limit_var", "30")
        self._row(ctrl, "優先の重み μ:",        "mu_var",    "1")
        self._row(ctrl, "基準の規定時間 H (h):","hbase_var", "8")
        tk.Label(ctrl, text="M は人数最小化が最優先になる\n閾値を自動計算して設定",
                 bg="#f0f0f0", fg="#666", font=("Arial", 8),
                 justify=tk.LEFT).pack(anchor="w")

        # 実験の種類
        self._sep(ctrl, "実験の種類")
        for val, lbl in [("noise",    "① ノイズ床測定"),
                          ("staffing", "② 必要人数曲線 (H vs 人数)"),
                          ("sweep",    "③ スイープ (1パラメータ)"),
                          ("tornado",  "④ トルネード図 (感度の序列)")]:
            tk.Radiobutton(ctrl, text=lbl, variable=self.exp_var, value=val,
                           bg="#f0f0f0", command=self._on_type_change
                           ).pack(anchor="w")

        # 実験別の設定（種類に応じて表示切替）
        self._sep(ctrl, "実験別の設定")
        self.type_frames = {}

        f = tk.Frame(ctrl, bg="#f0f0f0")
        tk.Label(f, text="同一条件をシード数だけ繰り返し、\n"
                         "結果のばらつき（ノイズ）を測ります。\n"
                         "感度分析の前に必ず実行してください。",
                 bg="#f0f0f0", fg="#555", font=("Arial", 8),
                 justify=tk.LEFT).pack(anchor="w")
        self.type_frames["noise"] = f

        f = tk.Frame(ctrl, bg="#f0f0f0")
        self._row(f, "H リスト (h):", "hlist_var", "4,6,8,10,12")
        self.type_frames["staffing"] = f

        f = tk.Frame(ctrl, bg="#f0f0f0")
        frm = tk.Frame(f, bg="#f0f0f0")
        frm.pack(fill=tk.X, pady=1)
        tk.Label(frm, text="パラメータ:", bg="#f0f0f0", width=16,
                 anchor="w", font=("Arial", 8)).pack(side=tk.LEFT)
        self.param_var = tk.StringVar(value="speed_kmh")
        cb = ttk.Combobox(frm, textvariable=self.param_var, width=12,
                          values=list(PARAM_LABELS.keys()), state="readonly")
        cb.pack(side=tk.LEFT)
        self._row(f, "値リスト:", "values_var", "20,25,30,35,40")
        self.type_frames["sweep"] = f

        f = tk.Frame(ctrl, bg="#f0f0f0")
        self._row(f, "変動幅 ±(%):", "rel_var", "20")
        frm = tk.Frame(f, bg="#f0f0f0")
        frm.pack(fill=tk.X, pady=1)
        tk.Label(frm, text="見る出力:", bg="#f0f0f0", width=16,
                 anchor="w", font=("Arial", 8)).pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value="opt_used")
        cb = ttk.Combobox(frm, textvariable=self.output_var, width=14,
                          values=list(OUTPUT_LABELS.keys()), state="readonly")
        cb.pack(side=tk.LEFT)
        tk.Label(f, text="速度・判定時間・エリア・優先割合・\n規定時間H の6項目を±X%動かします",
                 bg="#f0f0f0", fg="#555", font=("Arial", 8),
                 justify=tk.LEFT).pack(anchor="w")
        self.type_frames["tornado"] = f

        self.type_holder = tk.Frame(ctrl, bg="#f0f0f0")
        self.type_holder.pack(fill=tk.X)
        self._on_type_change()

        # 実行ボタン
        self._sep(ctrl, "実行")
        self.estimate_lbl = tk.Label(ctrl, text="", bg="#f0f0f0",
                                     fg="#1565C0", font=("Arial", 8),
                                     justify=tk.LEFT, anchor="w")
        self.estimate_lbl.pack(fill=tk.X)
        self.run_btn = tk.Button(
            ctrl, text="実験開始", command=self.run,
            bg="#2196F3", fg="white", relief=tk.FLAT, pady=7,
            font=("Arial", 10, "bold"))
        self.run_btn.pack(fill=tk.X, pady=3)
        self.stop_btn = tk.Button(
            ctrl, text="中止（完了分は保存されます）", command=self.stop,
            bg="#FF9800", fg="white", relief=tk.FLAT, pady=3,
            state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=1)

        self.progress = ttk.Progressbar(ctrl, mode="determinate")
        self.progress.pack(fill=tk.X, pady=4)
        self.prog_lbl = tk.Label(ctrl, text="待機中", bg="#f0f0f0",
                                 fg="#333", anchor="w", font=("Arial", 8))
        self.prog_lbl.pack(fill=tk.X)

        # ログ
        self.log = tk.Text(ctrl, height=9, font=("Courier", 8),
                           state=tk.DISABLED, bg="#fafafa")
        self.log.pack(fill=tk.BOTH, expand=True, pady=4)

        # 右側: グラフ + 結果サマリー
        right = tk.Frame(self.root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(11, 5.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.summary = tk.Text(right, height=12, font=("Courier", 10),
                               state=tk.DISABLED)
        self.summary.pack(fill=tk.X, padx=4, pady=4)

        self._show_placeholder()

    def _sep(self, parent, text):
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=5)
        tk.Label(parent, text=text, font=("Arial", 9, "bold"),
                 bg="#f0f0f0", anchor="w").pack(fill=tk.X)

    def _row(self, parent, label, attr, default):
        frm = tk.Frame(parent, bg="#f0f0f0")
        frm.pack(fill=tk.X, pady=1)
        tk.Label(frm, text=label, bg="#f0f0f0", width=16, anchor="w",
                 font=("Arial", 8)).pack(side=tk.LEFT)
        var = tk.StringVar(value=default)
        setattr(self, attr, var)
        tk.Entry(frm, textvariable=var, width=12).pack(side=tk.LEFT)

    def _on_type_change(self):
        for f in self.type_frames.values():
            f.pack_forget()
        self.type_frames[self.exp_var.get()].pack(
            in_=self.type_holder, fill=tk.X)

    def _show_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5,
                "実験を設定して「実験開始」を押してください\n\n"
                "推奨の順序: ①ノイズ床 → ②必要人数曲線 → ③④感度分析",
                ha="center", va="center", fontsize=13, color="#888")
        self.canvas.draw_idle()

    # ── 実行 ─────────────────────────────────────────────────────────────────

    def _read_params(self):
        base = Scenario(
            n=int(self.n_var.get()),
            time_limit=float(self.limit_var.get()),
            mu=float(self.mu_var.get()),
            max_work_h=float(self.hbase_var.get()))
        assert 2 <= base.n <= 2000, "建物数は 2〜2000 棟"
        seeds = list(range(int(self.seeds_var.get())))
        assert seeds, "シード数は 1 以上"
        return base, seeds

    def _estimate_jobs(self, kind, base, seeds):
        if kind == "noise":
            return len(seeds)
        if kind == "staffing":
            return len(self.hlist_var.get().split(",")) * len(seeds)
        if kind == "sweep":
            return len(self.values_var.get().split(",")) * len(seeds)
        if kind == "tornado":
            return (1 + 2 * 6) * len(seeds)   # base + 6項目 × 上下
        return 0

    def run(self):
        if self.running:
            return
        try:
            base, seeds = self._read_params()
        except AssertionError as e:
            messagebox.showerror("エラー", str(e))
            return
        except Exception:
            messagebox.showerror("エラー", "入力値を確認してください")
            return

        kind = self.exp_var.get()
        n_jobs = self._estimate_jobs(kind, base, seeds)
        est_min = n_jobs * (base.time_limit + 3) / 60
        self.estimate_lbl.config(
            text=f"実行数: {n_jobs} 回 / 目安 約{est_min:.0f} 分")

        self.running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress.config(maximum=n_jobs, value=0)
        self._log_clear()
        threading.Thread(target=self._worker, args=(kind, base, seeds),
                         daemon=True).start()

    def stop(self):
        self.running = False
        self.stop_btn.config(state=tk.DISABLED)

    def _status(self, msg):
        """ワーカースレッドからのメッセージをキューに積む"""
        self.msg_queue.put(("log", msg))
        if msg.startswith("["):   # "[i/total]" 形式なら進捗を更新
            try:
                done = int(msg[1:msg.index("/")])
                self.msg_queue.put(("progress", done - 1))
            except Exception:
                pass

    def _poll_queue(self):
        """メインスレッド側: キューを定期的に処理して UI を更新する"""
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._log_append(payload)
                    self.prog_lbl.config(text=payload[:60])
                elif kind == "progress":
                    self.progress.config(value=payload)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def _worker(self, kind, base, seeds):
        try:
            runner = ExperimentRunner(
                status_cb=self._status,
                should_continue=lambda: self.running)
            tag = f"n{base.n}_s{len(seeds)}"

            if kind == "noise":
                sa = SensitivityAnalyzer(base, seeds, runner)
                csv_p = os.path.join(RESULTS_DIR, f"noise_{tag}.csv")
                stats = sa.noise_floor(csv_p)
                self.root.after(0, self._show_noise, stats, csv_p)

            elif kind == "staffing":
                hs = [float(x) for x in self.hlist_var.get().split(",")]
                st = StaffingAnalyzer(base, runner)
                csv_p = os.path.join(RESULTS_DIR, f"staffing_{tag}.csv")
                rows = st.run(hs, seeds, csv_p)
                png = os.path.join(RESULTS_DIR, f"staffing_{tag}.png")
                self.root.after(0, self._show_staffing, rows, hs, csv_p, png)

            elif kind == "sweep":
                param = self.param_var.get()
                vals = [float(x) for x in self.values_var.get().split(",")]
                sa = SensitivityAnalyzer(base, seeds, runner)
                csv_p = os.path.join(RESULTS_DIR, f"sweep_{param}_{tag}.csv")
                rows = sa.sweep(param, vals, csv_p)
                png = os.path.join(RESULTS_DIR, f"sweep_{param}_{tag}.png")
                self.root.after(0, self._show_sweep, rows, param, csv_p, png)

            elif kind == "tornado":
                rel = float(self.rel_var.get()) / 100.0
                output = self.output_var.get()
                sa = SensitivityAnalyzer(base, seeds, runner)
                csv_p = os.path.join(RESULTS_DIR, f"tornado_{tag}.csv")
                effects = sa.tornado(rel=rel, output=output, csv_path=csv_p)
                png = os.path.join(RESULTS_DIR,
                                   f"tornado_{output}_{tag}.png")
                self.root.after(0, self._show_tornado,
                                effects, output, rel, csv_p, png)
        except Exception as e:
            self.msg_queue.put(("log", f"エラー: {e}"))
            self.root.after(0, self._finish)

    def _finish(self):
        self.running = False
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress.config(value=self.progress["maximum"])
        self.prog_lbl.config(text="完了")

    # ── 結果表示 ─────────────────────────────────────────────────────────────

    def _show_noise(self, stats, csv_p):
        self._finish()
        self.fig.clear()
        keys = list(stats.keys())
        labels = {"greedy_min_m": "貪欲 最小人数", "opt_used": "必要人数",
                  "opt_total_h": "総活動時間", "opt_prio_start_h": "優先開始",
                  "opt_makespan_h": "makespan"}
        ax = self.fig.add_subplot(111)
        stds = [stats[k]["std"] for k in keys]
        ax.bar(range(len(keys)), stds, color="#2196F3", width=0.5)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([labels.get(k, k) for k in keys], fontsize=9)
        ax.set_ylabel("標準偏差（ノイズ床）")
        ax.set_title("ノイズ床: シードのみ変えたときの出力のばらつき\n"
                     "（感度分析ではこの幅を超える変化のみ有意とみなす）")
        ax.grid(alpha=0.3, axis="y")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        lines = ["── ノイズ床（同一条件・シードのみ変更）──", ""]
        lines.append(f"{'指標':<22}{'平均':>10}{'標準偏差':>10}{'最小':>10}{'最大':>10}")
        lines.append("─" * 64)
        for k, s in stats.items():
            lines.append(f"{labels.get(k, k):<22}{s['mean']:>10.3f}"
                         f"{s['std']:>10.3f}{s['min']:>10.3f}{s['max']:>10.3f}")
        lines += ["", f"CSV: {csv_p}"]
        self._set_summary("\n".join(lines))

    def _show_staffing(self, rows, hs, csv_p, png):
        self._finish()
        if not rows:
            self._set_summary("結果がありません（中止された可能性があります）")
            return
        self.fig.clear()
        StaffingAnalyzer.plot(rows, path=png, fig=self.fig)
        self.canvas.draw_idle()

        lines = ["── 規定時間 H vs 最小必要人数 ──", ""]
        lines.append(f"{'H [h]':>7}{'貪欲法 [人]':>14}{'OR-Tools [人]':>14}{'削減':>8}")
        lines.append("─" * 46)
        for h in hs:
            g = [r["greedy_min_m"] for r in rows if r["max_work_h"] == h]
            o = [r["opt_used"] for r in rows if r["max_work_h"] == h]
            if g:
                lines.append(f"{h:>7.1f}{np.mean(g):>14.1f}"
                             f"{np.mean(o):>14.1f}{np.mean(g)-np.mean(o):>+8.1f}")
        lines += ["", f"CSV: {csv_p}", f"PNG: {png}"]
        self._set_summary("\n".join(lines))

    def _show_sweep(self, rows, param, csv_p, png):
        self._finish()
        if not rows:
            self._set_summary("結果がありません（中止された可能性があります）")
            return
        self.fig.clear()
        SensitivityAnalyzer.plot_sweep(rows, param, path=png, fig=self.fig)
        self.canvas.draw_idle()

        vals = sorted({r[param] for r in rows})
        lines = [f"── スイープ: {PARAM_LABELS.get(param, param)} ──", ""]
        lines.append(f"{'値':>8}{'必要人数':>10}{'総活動[h]':>11}{'優先開始[h]':>12}")
        lines.append("─" * 44)
        for v in vals:
            sub = [r for r in rows if r[param] == v]
            u = np.mean([r["opt_used"] for r in sub])
            t = np.mean([r["opt_total_h"] for r in sub])
            p = np.nanmean([r["opt_prio_start_h"] for r in sub])
            lines.append(f"{v:>8g}{u:>10.1f}{t:>11.1f}{p:>12.2f}")
        lines += ["", f"CSV: {csv_p}", f"PNG: {png}"]
        self._set_summary("\n".join(lines))

    def _show_tornado(self, effects, output, rel, csv_p, png):
        self._finish()
        if not effects:
            self._set_summary("結果がありません（中止された可能性があります）")
            return
        self.fig.clear()
        SensitivityAnalyzer.plot_tornado(effects, output, rel,
                                         path=png, fig=self.fig)
        self.canvas.draw_idle()

        lines = [f"── トルネード分析 (±{rel*100:.0f}%, "
                 f"出力={OUTPUT_LABELS.get(output, output)}) ──", ""]
        lines.append(f"{'パラメータ':<16}{'base':>8}{'low側':>8}{'high側':>8}{'影響幅':>8}")
        lines.append("─" * 52)
        for p, base_m, lo, hi in effects:
            lines.append(f"{PARAM_LABELS.get(p, p):<16}{base_m:>8.2f}"
                         f"{lo:>8.2f}{hi:>8.2f}{abs(hi-lo):>8.2f}")
        lines += ["", "影響幅の大きい順。ノイズ床の標準偏差より小さい差は",
                  "有意な感度とは言えない点に注意。",
                  "", f"CSV: {csv_p}", f"PNG: {png}"]
        self._set_summary("\n".join(lines))

    # ── ログ/サマリーヘルパー ────────────────────────────────────────────────

    def _log_clear(self):
        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.config(state=tk.DISABLED)

    def _log_append(self, msg):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _set_summary(self, text):
        self.summary.config(state=tk.NORMAL)
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, text)
        self.summary.config(state=tk.DISABLED)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = ExperimentApp(root)
    root.mainloop()
