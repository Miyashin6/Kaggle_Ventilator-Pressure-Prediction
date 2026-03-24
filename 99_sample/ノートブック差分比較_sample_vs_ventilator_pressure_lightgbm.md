# ノートブック差分比較：`sample.ipynb` vs `ventilator_pressure_lightgbm.ipynb`

比較日: 2026-03-10  
対象ファイル:

| ファイル | おおよその規模 |
|----------|----------------|
| `sample.ipynb` | セル **10** 個（Markdown 4 + Code 6）。行数は実行結果の埋め込みにより **1000行超** |
| `ventilator_pressure_lightgbm.ipynb` | セル **25** 個。行数 **約415行**（出力ほぼなし） |

---

## 1. 結論（ざっくり）

| 観点 | `sample.ipynb` | `ventilator_pressure_lightgbm.ipynb` |
|------|------------------|--------------------------------------|
| **目的** | ニューラルネット等で出した **提出ファイル（pressure 予測）を後処理**し、P/PI 制御の**逆関数**で一部ステップを書き換える | **ゼロから**前処理・EDA・特徴量・**LightGBM 学習**・提出までを一通り実行する **ベースラインノート** |
| **言語・説明** | 英語（Kaggle カーネル由来） | 日本語（分析設計・手順の説明付き） |
| **前提** | 別途用意した **`submission.csv`（NN 予測）** や、長時間計算結果のデータセットに依存する記述あり | **train/test CSV** と Kaggle の **Input パス** があれば単体で完結（学習〜提出） |
| **モデル** | 機械学習の「学習」は行わない。**物理・制御（PID）仮定に基づくルール＋数値探索** | **LightGBM（勾配ブースティング）** の回帰 |

**重複するのは「コンペの train/test を読む」「pressure を予測結果として出力する」という文脈だけ**で、中身のアプローチは別物です。

---

## 2. 役割の違い

### `sample.ipynb`

- タイトル: *Predicting pressures by inverting the function of PID controllers*
- **入力**: ニューラルネットが生成した提出ファイルを読み込み、その `pressure` を更新する。
- **ロジック**:
  - 訓練データから **圧力の離散値・ステップ幅**（`p_min`, `p_step`）を推定。
  - テストを **(呼吸数 × 80 ステップ)** に reshape。吸気相は `u_out == 0` のマスク `rr` で抽出。
  - **PI 制御**に合致する行を `find_pi_control` で探索し、予測を更新。
  - **P 制御のみ**の行を `find_p_control` で処理。
  - `joblib.Parallel` で並列処理し、`submission_pi.csv`、`pi_parameters.csv`、`p_parameters.csv` 等を保存。
- **メタデータ**: `papermill` の実行ログが付与されている（Kaggle で実行されたカーネルのコピーと思われる）。

### `ventilator_pressure_lightgbm.ipynb`

- **入力**: `train.csv` / `test.csv` を Kaggle の `INPUT_DIR` から読み込み。
- **ロジック**:
  - 分析設計の Markdown → 前処理 → EDA（分布・時系列・相関）→ `add_features`（ラグ・累積・rolling 等）→ **GroupKFold（breath_id 単位）** → **LightGBM** で学習・OOF 評価 → `submission.csv` 出力。
  - 学習データは **`u_out != 0` の行のみ**を使用（吸気相に合わせたフィルタ）。
- **メタデータ**: シンプルな `kernelspec` / `language_info` のみ（実行出力は最小）。

---

## 3. 依存ライブラリの差

| ライブラリ | `sample.ipynb` | `ventilator_pressure_lightgbm.ipynb` |
|------------|----------------|----------------------------------------|
| `pandas`, `numpy`, `matplotlib` | ✓ | ✓ |
| `seaborn` | ✗ | ✓ |
| `sklearn` | コード内で `mean_absolute_error` を使用するが、**先頭の import セルには含まれていない**（単体実行時は `from sklearn.metrics import mean_absolute_error` が必要） | `GroupKFold` と `mean_absolute_error` を明示 import |
| `lightgbm` | ✗ | ✓ |
| `joblib` (Parallel) | ✓ | ✗ |
| `pickle` | ✓ | ✗ |
| `IPython.display` | ✓ | ✗ |

---

## 4. データパスの差

| 項目 | `sample.ipynb` | `ventilator_pressure_lightgbm.ipynb` |
|------|----------------|--------------------------------------|
| パス例 | `../input/ventilator-pressure-prediction/train.csv`（相対パス） | `/kaggle/input/ventilator-pressure-prediction/...`（絶対パス） |
| 提出物の読み込み | **既存の NN 提出 `submission.csv`** を読み、それを更新する流れ | **自前の `test_pred`** から `submission` を新規作成 |

ローカルで `ventilator_pressure_lightgbm.ipynb` を動かす場合は、`INPUT_DIR` を `Dataset/` 等に合わせる必要があります（README の想定どおり）。

---

## 5. セル構成の対応（概要）

### `ventilator_pressure_lightgbm.ipynb`（セクション構造）

1. タイトル・目的  
2. 分析設計（表形式）  
3. ライブラリ・パス  
4. データ前処理（読込・info・ソート）  
5. EDA（ヒスト・散布・時系列・相関）  
6. 特徴量 `add_features`  
7. データセット作成（`u_out` マスク含む）  
8. GroupKFold  
9. LightGBM 学習・OOF  
10. 提出 CSV  

### `sample.ipynb`（セクション構造）

1. PID 逆関数による後処理の説明（英語）  
2. import  
3. データ読込・`uu`/`rr`/`dt_` 準備・圧力離散値の整理  
4. `find_pi_control`（長大な PI 推定・更新）  
5. `find_p_control`（P 制御）  
6. NN 予測の読込・並列更新・`submission_pi.csv` 出力  
7. 空セル  

→ **セクション番号や流れを 1:1 で対応させることはできません**（設計が異なるため）。

---

## 6. 評価指標との関係

- 両方ともコンペの目的は **pressure の予測誤差（MAE）** です。
- `sample.ipynb` は「**既に出た予測を、制御理論に基づき修正**して MAE を改善する」想定。
- `ventilator_pressure_lightgbm.ipynb` は「**特徴量から直接学習**して MAE を下げる」想定。

---

## 7. 使い分けの目安

| 用途 | おすすめ |
|------|----------|
| コンペの流れを日本語で追い、**自分のベースラインを再現**したい | `ventilator_pressure_lightgbm.ipynb` |
| **上位解法の後処理（PID 逆関数）**のコードを読みたい・試したい | `sample.ipynb`（別途 NN の提出ファイルや長時間計算用データの準備が必要な場合あり） |

---

## 8. 補足

- `sample.ipynb` は **実行結果（stdout・DataFrame 表示・画像など）が JSON に大量に含まれる**ため、Git 上の差分やファイルサイズが大きくなりやすいです。
- `ventilator_pressure_lightgbm.ipynb` の **「学習で `u_out==0` を除く」**方針は、コンペの **吸気相のみスコア**という評価と整合的です。`sample.ipynb` 側も処理対象を `u_out==0`（吸気）に絞る記述がありますが、**目的は「後処理での修正」**です。
