# 応急危険度判定 mTSP — 時間制約付き巡回計画ソルバー

Time-Constrained mTSP for Emergency Building Inspection

地震等の災害後に、複数の判定士が拠点（デポ）から出発してエリア内の全建物を
手分けして応急危険度判定を行うための巡回計画を作成するツールです。
時間制約付きの複数巡回セールスマン問題（mTSP）として定式化し、
GUI 上で建物の生成・計画の実行・結果の可視化ができます。

## 問題設定

- 各建物は 1 回だけ判定する
- 判定士は全員デポから出発し、デポに帰還する
- 移動時間（距離 ÷ 速度）と建物ごとの判定時間を稼働時間に加算する
- 「現在時刻 + 移動 + 判定 + デポ帰還 ≤ 最大稼働時間」を満たす場合のみ割当てる
- 時間内に回りきれない建物は「未割当」として報告する

**目的関数**（重み付き多目的）:

```
min  M Σ z_q  +  λ Σ T_q  +  μ Σ p_i a_i
```

| 項 | 意味 |
|---|---|
| `M Σ z_q` | 使用判定士数の最小化（z_q: 判定士 q を使うか 0/1） |
| `λ Σ T_q` | 総活動時間の最小化（T_q: 判定士 q の移動+判定時間 [h]） |
| `μ Σ p_i a_i` | 優先建築物の早期判定（p_i: 優先度、a_i: 判定開始時刻 [h]） |

判定士数 m は「使用できる人数の上限」であり、実際に何人使うかはソルバーが
重み M とのバランスで決定する。

## インストール

```bash
pip install numpy scipy matplotlib ortools
```

Python 3.9 以降を推奨します（tkinter は標準ライブラリに含まれます）。
`ortools` は「OR-Tools改善」ソルバーを使う場合のみ必要です。

## 使い方

```bash
python tsp_gui.py
```

1. 建物数・エリアサイズ・判定時間の範囲を指定して「ランダム建物生成」
   （キャンバスをクリックして手動で建物を追加することもできます）
2. デポの配置方法（中心 / ランダム / クリック指定）を選択
3. 判定士数・移動速度・最大稼働時間を設定して「計画を実行」
4. 「必要判定士数を計算」で、全棟を割当てられる最小の判定士数を二分探索で求められます
5. 「ベンチマーク実行」で建物数 100〜100,000 のスケーリングを計測できます

## 構成

| ファイル | 役割 |
|---|---|
| `mtsp_core.py` | ソルバー本体（GUI 非依存）。単体で import して利用可能 |
| `tsp_gui.py` | tkinter + matplotlib による計画作成 GUI |
| `compare_gui.py` | ソルバー比較ツール。同一インスタンスを貪欲法と OR-Tools の両方で解き、ルート図と指標を並べて表示。結果を `comparison_results.csv` に追記できる |
| `experiments.py` | 実験基盤（CLI）。規定時間 H ごとの最小必要人数分析（`StaffingAnalyzer`）と、ノイズ床測定・One-at-a-Time・トルネード図による感度分析（`SensitivityAnalyzer`）。結果は `results/` に CSV + PNG で保存 |
| `experiment_gui.py` | 実験ツールの GUI フロントエンド。4種類の実験（ノイズ床・必要人数曲線・スイープ・トルネード）をフォームで設定・実行し、進捗ログとグラフを画面内に表示。中止しても完了分は CSV に保存される |

`mtsp_core.py` は「問題」「解」「解法」を分離したクラス構成になっています:

- **`InspectionProblem`** — 問題インスタンス（建物座標・判定時間・デポ・制約パラメータ）。
  イミュータブルで、入力値の検証と単位換算を担当
- **`InspectionSolution`** — 解。makespan・稼働時間・移動距離をルートから自分で計算し、
  `validate()` / `is_feasible()` で制約充足を自己検証できる
- **`SolverBase`** — 解法の共通インターフェース。ソルバーを差し替えて比較実験できる
  - **`GreedySolver`** — 最近傍法 + KDTree + min-heap による時間制約付き貪欲法
  - **`MultiStartSolver`** — 複数デポ候補の並列試行（マルチスタート）
  - **`ORToolsSolver`** — OR-Tools ルーティングソルバーによる makespan 最小化
    （貪欲解を初期解に誘導局所探索で改善、〜2,000 棟）
- **`find_min_inspectors()`** — 全棟割当可能な最小判定士数の二分探索

### ライブラリとしての利用例

```python
import numpy as np
from mtsp_core import InspectionProblem, GreedySolver, find_min_inspectors

problem = InspectionProblem(
    coords=np.random.random((500, 2)),          # 正規化座標 [0,1]^2
    inspect_times=np.random.uniform(900, 2700, 500),  # 判定時間 [秒]
    depot_idx=0,
    area_km=10.0,      # エリア一辺 [km]
    speed_kmh=30.0,    # 移動速度 [km/h]
    max_work_h=8.0,    # 最大稼働時間 [h]
)

sol = GreedySolver().solve(problem, m=4)
print(sol.summary())          # 判定士 4 人 | makespan 7.98h | ...
assert sol.is_feasible()      # 制約充足の自己検証

min_m, best = find_min_inspectors(problem)
print(f"全棟対応に必要な最小判定士数: {min_m} 人")
```

## アルゴリズム

### 貪欲法（構築ヒューリスティック）

- 最近傍法 (Nearest Neighbor) + KDTree による O(n log n) 近傍探索
- min-heap で「最も暇な判定士」から順に建物を選ばせるラウンドロビン割当
  （各判定士の終了時刻が平準化され makespan が抑えられる）
- ProcessPoolExecutor による複数デポ候補の並列試行
- 最小判定士数は二分探索で O(log n) 回の試行で確定

10 万棟規模でも数秒〜数十秒で計画を作成できます（ベンチマーク機能で計測可能）。

### OR-Tools 最適化

貪欲解を初期解として OR-Tools のルーティングソルバーに渡し、
誘導局所探索 (Guided Local Search) で時間制限まで改善します。
数理モデルとの対応:

| 数理モデルの要素 | OR-Tools での実装 |
|---|---|
| `M Σ z_q`（人数最小化） | `SetFixedCostOfAllVehicles`（使用車両の固定費） |
| `λ Σ T_q`（総活動時間） | アークコスト = λ ×（移動 + 判定時間） |
| `μ Σ p_i a_i`（優先建物の早期判定） | Time 次元の `SetCumulVarSoftUpperBound(i, 0, μp_i)` |
| 稼働時間制約 ≤ H | Time 次元の capacity |
| 未割当の許容 | `AddDisjunction`（ペナルティ付きドロップ） |

時間行列は整数秒に**切り上げ**て構築しているため、整数モデルで実行可能な解は
実数値の稼働時間制約も必ず満たします。密行列を持つ都合上、対象は 2,000 棟までです
（それ以上の規模は貪欲法を使用）。

参考値（200 棟・優先 10%・M=1000, λ=1, μ=1、時間制限 15 秒）:
- 使用判定士数: 上限 20 人 → **14 人**（貪欲法は 20 人全員使用）
- 優先建物の平均判定開始時刻: μ=0 で 3.7h → μ=50 で **0.7h**
- 総活動時間: 貪欲法から 4〜6% 削減

## 今後の予定

- 貪欲法 vs OR-Tools の解品質・計算時間の系統的な比較実験
- 複数デポ（判定士ごとに異なる出発拠点）への対応
