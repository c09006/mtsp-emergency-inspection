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
- **目的関数: 最大終了時間（makespan）の最小化** — 最も遅い判定士が早く終わるように割当てる

## インストール

```bash
pip install numpy scipy matplotlib
```

Python 3.9 以降を推奨します（tkinter は標準ライブラリに含まれます）。

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
| `tsp_gui.py` | tkinter + matplotlib による GUI |

`mtsp_core.py` は「問題」「解」「解法」を分離したクラス構成になっています:

- **`InspectionProblem`** — 問題インスタンス（建物座標・判定時間・デポ・制約パラメータ）。
  イミュータブルで、入力値の検証と単位換算を担当
- **`InspectionSolution`** — 解。makespan・稼働時間・移動距離をルートから自分で計算し、
  `validate()` / `is_feasible()` で制約充足を自己検証できる
- **`SolverBase`** — 解法の共通インターフェース。ソルバーを差し替えて比較実験できる
  - **`GreedySolver`** — 最近傍法 + KDTree + min-heap による時間制約付き貪欲法
  - **`MultiStartSolver`** — 複数デポ候補の並列試行（マルチスタート）
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

- 最近傍法 (Nearest Neighbor) + KDTree による O(n log n) 近傍探索
- min-heap で「最も暇な判定士」から順に建物を選ばせるラウンドロビン割当
  （各判定士の終了時刻が平準化され makespan が抑えられる）
- ProcessPoolExecutor による複数デポ候補の並列試行
- 最小判定士数は二分探索で O(log n) 回の試行で確定

10 万棟規模でも数秒〜数十秒で計画を作成できます（ベンチマーク機能で計測可能）。

## 今後の予定

- OR-Tools ルーティングソルバーによる本格的な最適化
  （makespan 最小化 + 誘導局所探索、貪欲解を初期解として利用）
- 貪欲法 vs OR-Tools の解品質・計算時間の比較実験
