# 変更履歴（Ventilator Pressure Prediction）

このドキュメントは、`ventilator_pressure_lightgbm.ipynb` および関連ファイルに対する変更内容をまとめたものです。

---

## 1. 初回作成時

### 作成したファイル
- `ventilator_pressure_lightgbm.ipynb` … メインのノートブック
- `README.md` … 初心者向けの構成説明・実行方法
- `requirements.txt` … 依存パッケージ

### ノートブックの構成
- **分析設計** … 目的変数（pressure）、連続値・回帰、評価指標（MAE）と根拠を明記
- **データ前処理** … 読み込み、型・欠損確認、`breath_id`・`time_step` でソート
- **EDA** … 目的変数の分布、u_in / u_out / time_step の分布、サンプル breath の時系列、相関ヒートマップ
- **特徴量作成** … breath 単位のラグ・累積・差分・rolling 特徴量
- **データセット作成** … 特徴量リストで X, y, X_test を定義
- **バリデーション設計** … GroupKFold（breath_id でグループ化）
- **モデル学習** … LightGBM（5-fold、early_stopping 50、n_estimators 1000 など）
- **提出** … submission.csv の保存

### 実行環境
- ローカルと Kaggle の両方で動くよう、`os.path.exists("/kaggle/input")` でパスを切り替えていた。

---

## 2. Kaggle Web 環境専用への変更

### 変更理由
Web 上の Kaggle 環境だけで動かすようにするため。

### 変更内容

| 対象 | 変更前 | 変更後 |
|------|--------|--------|
| **ノートブック（パス）** | ローカル/Kaggle で分岐（`if os.path.exists("/kaggle/input")`） | Kaggle 専用の固定パス（`/kaggle/input/ventilator-pressure-prediction`）のみ |
| **ノートブック** | `os` の import、`COMPETITION_NAME` 等の変数 | 削除。INPUT_DIR / OUTPUT_DIR を直接指定 |
| **README.md** | データ準備（API・手動DL）、ローカル実行手順 | 「Web 上の Kaggle 環境専用」と明記し、Kaggle Notebook での実行手順のみに変更 |
| **README** | train.csv / test.csv をフォルダに置く構成 | コンペの Data を追加する前提の説明に変更 |

---

## 3. 実行時間の短縮

### 変更理由
全体で約 30 分かかっており、特に LightGBM の実行時間を短くするため。

### 変更内容（LightGBM まわり）

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| **N_FOLDS** | 5 | 3 |
| **n_estimators** | 1000 | 400 |
| **num_leaves** | 31 | 20 |
| **max_depth** | なし | 6 |
| **learning_rate** | 0.05 | 0.1 |
| **early_stopping** | 50 | 25 |

### 想定効果
- 学習回数: 5 回 → 3 回
- 1 fold あたりのイテレーション数・木の複雑さを抑制
- 実行時間を約 30 分 → おおよそ 10 分程度に短縮する想定（環境により変動）

---

## 4. 過学習の判断ができるようにする変更

### 変更理由
LightGBM が過学習しているかを判断できるようにするため。

### 変更内容

#### 4.1 モデル学習セル
- **訓練 MAE の計算**  
  各 fold で `mae_train = mean_absolute_error(y_tr, model.predict(X_tr))` を計算し、`train_scores` リストに保存。
- **出力の追加・変更**
  - 各 fold: **訓練 MAE** / **検証 MAE** / **差（検証 − 訓練）** を表示
  - ループ後: **平均 訓練 MAE**、**CV（検証）MAE**、**差（検証 − 訓練）** を表示
  - 注釈: 「この差が大きいほど過学習の可能性」と記載

#### 4.2 過学習の判断用マークダウンセル（新規）
- 訓練 MAE が検証 MAE より大幅に小さい場合は過学習の可能性があること
- 差（検証 − 訓練）が大きいほど過学習が疑われること
- 対策例: `num_leaves` / `max_depth` を減らす、`min_data_in_leaf` を増やす、`early_stopping` を厳しくする、`learning_rate` を下げて `n_estimators` を増やす

#### 4.3 可視化セル（新規）
- 各 Fold の **訓練 MAE** と **検証 MAE** を並べた棒グラフを追加
- 青: 訓練 MAE、オレンジ: 検証 MAE。検証が訓練より明らかに高いと過学習の目安になる旨を説明

#### 4.4 セクション 8 の説明文
- 「検証 MAE を算出」→「**訓練 MAE** と **検証 MAE** を算出します。両者の差で過学習を判断できます。」に変更

---

## 5. Kaggle Notebook 版 5（u_in 条件まわりの更新）

### 変更理由
Kaggle 上の Notebook 版 5 に合わせ、`u_in` に関する条件を含む内容へ更新するため。

### 変更内容
- `ventilator_pressure_lightgbm.ipynb` を大きく更新（コミット: `3a61b6a`）

---

## 6. EDA のコメントアウトと学習データのフィルタ整理

### 変更理由
実行時間や検証方針の都合で、EDA をオフにし、学習時の `u_out` フィルタを外すため。

### 変更内容
- **EDA** … 該当セルをコメントアウト
- **学習データ** … `u_out` に基づく訓練用フィルタをコメントアウト
- ノートブックの行数・セル構成が整理される（コミット: `652cc40`）

---

## 7. 参考ノートブック・解説ドキュメントの追加とリポジトリ整備

### 変更理由
学習用のサンプルノートブックと解説 Markdown を揃え、README を現状に合わせ、データセットを Git 管理対象外にするため。

### 変更内容
- **追加ノートブック** … `lightgbm_sample.ipynb`、`sample.ipynb`、`simple-lightgbm.ipynb` など
- **解説・ドキュメント** … 各ノート向けの解説 `.md`、コンペ概略、データセット説明、ノートブック差分比較、調査レポートなど
- **README.md** … 内容・表の更新
- **`.gitignore`** … `Dataset/`、`.DS_Store` を除外（コミット: `a8fb295`）

---

## 8. google-brain-optuna 系フローへの寄せ（EDA・FE・Optuna・LGBMRegressor）

### 変更理由
Google Brain ベースのノート（Optuna による探索を含む）と同様の分析・学習フローを `ventilator_pressure_lightgbm.ipynb` に取り込むため。

### 変更内容
- **EDA** … 再導入・整理
- **特徴量エンジニアリング** … 該当フローに合わせて更新
- **Optuna** … 目的関数・Study の例（後続コミットでコメントアウト）
- **学習** … `LGBMRegressor`、train / valid 分割ベースの流れ
- **`requirements.txt`** … `optuna` を追加（コミット: `0dff0c9`）

---

## 9. Optuna チューニングの一時停止と `u_out` を特徴量へ

### 変更理由
毎回のハイパラ探索をオフにしつつ、モデル入力に `u_out` を含めるため。

### 変更内容
- **Optuna** … `import` と目的関数・Study 例のセルをコメントアウト
- **特徴量** … `X` / `X_test` に `u_out` を追加
- **セクション 6 の説明** … 上記に合わせて更新（コミット: `7b4657b`）

---

## 10. 特徴量重要度セクションと LightGBM 警告の解消

### 変更理由
どの特徴が効いているかを可視化し、`bagging_fraction` / `subsample` の重複などによる LightGBM の警告をなくすため。

### 変更内容
- **セクション 9（新規）** … gain / split の重要度表、gain の棒グラフ、sklearn 側の split に関する注記
- **LightGBM** … サンプリング系パラメータの重複を解消、`verbose=-1` を指定
- **セクション番号** … 提出をセクション 10 に繰り下げ
- **セクション 8** … サンプリング関連パラメータの説明を追記（コミット: `b3cc36a`）

### 未コミット（作業中のローカルファイル）
- `google-brain-lightgbm-optuna.ipynb` / `google-brain-lightgbm-optuna.md` … リポジトリに未追加（`git status` 時点）

---

## 11. 施策1（方法 B）: 吸気相のみで学習・呼気は 0 補完

### 変更理由
`スコアアップ施策評価とロードマップ.md` と同じく、呼気相行を訓練から外し吸気相の予測精度を優先するため。

### 変更内容
- **訓練** … `u_out=0` の行のみから `X`, `y` を構築
- **提出** … テストの吸気行はモデル予測、呼気行は `pressure=0` で補完

---

## 12. 施策4: `u_in` ラグ特徴量（1〜4ステップ）の追加

### 変更理由
`スコアアップ施策評価とロードマップ.md` の施策4に沿って、吸気相の時系列変化をより直接的に学習できるようにするため。

### 変更内容
- **特徴量エンジニアリング** … `u_in_lag1`, `u_in_lag2`, `u_in_lag3`, `u_in_lag4` を `breath_id` ごとの `shift(1..4)` で追加
- **既存互換** … 従来の `u_in_lag` は `u_in_lag2` と同値になるように維持
- **ノート本文** … セクション5の説明を「2ステップラグのみ」から「1〜4ステップラグ」へ更新
- **反映経緯** … `origin/main` の更新取り込み後に再適用して push（コミット: `2bce232`）

---

## 13. 施策4: `u_in` 差分特徴量 + `R/C` 交互作用特徴量の追加

### 変更理由
`スコアアップ施策評価とロードマップ.md` の施策4に沿って、時系列の変化量と呼吸器設定値の相互関係をモデルに明示的に与えるため。

### 変更内容
- **u_in 差分特徴量** … `u_in_diff1`, `u_in_diff2` を `breath_id` ごとの `diff(1)`, `diff(2)` で追加（欠損は 0 補完）
- **R/C 交互作用特徴量** … `R_C_mul`（`R * C`）と `R_C_div`（`R / C`）を追加
- **ノート本文** … セクション5の説明に差分特徴量と交互作用特徴量の追加内容を追記
- **反映コミット** … `feat(notebook): add policy-4 diff and RC interaction features`（`65062ea`）

---

## 14. 施策2: Optuna によるハイパーパラメータ再チューニング

### 変更理由
`スコアアップ施策評価とロードマップ.md` の施策2に沿って、施策1/4を反映後の特徴量セットに対して、LightGBM の最適パラメータを再探索するため。

### 変更内容
- **Optuna セクション更新** … セクション7をコメントアウト例から実行可能コードに更新し、`TPESampler(seed=42)`・`n_trials=20` で探索する構成に変更
- **探索空間の見直し** … `learning_rate`, `num_leaves`, `max_depth`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`, `min_child_samples`, `bagging_fraction`, `bagging_freq` を探索対象に整理
- **学習パラメータ反映** … セクション8の `lgb_params` をベスト結果へ更新（Best MAE: `1.1589910192692925`）
- **関連追記** … 初心者向け解説 `施策2_Optuna実装意図_初心者向け解説.md` を新規追加

---

## 15. CV設計の実装: `GroupKFold` + OOF 評価へ移行

### 変更理由
`CV設計について.md` の方針に合わせ、`train_test_split` による行ベース分割をやめて、`breath_id` 単位でリークを防ぐ評価に統一するため。

### 変更内容
- **CV方式の変更** … `train_test_split` を廃止し、`GroupKFold(n_splits=5)`（`groups=train_insp["breath_id"]`）へ置き換え
- **Optuna objective の更新** … 単一 holdout MAE ではなく、5-fold の平均MAEを最小化する目的関数に変更
- **学習セクションの更新** … foldごとの MAE を出力し、OOF予測から `OOF MAE` を算出
- **推論ロジックの更新** … テスト吸気相 (`u_out=0`) は各 fold モデルの平均予測（アンサンブル）を採用
- **ノート本文の更新** … セクション7/8/10の説明を GroupKFold 前提に更新

---

## ファイル一覧（現在）

| ファイル | 役割 |
|----------|------|
| `ventilator_pressure_lightgbm.ipynb` | メインノート（Kaggle 想定、EDA/FE、Optuna はコメントアウト、施策1方法Bで吸気相のみ学習、特徴量重要度・過学習判断あり） |
| `lightgbm_sample.ipynb` / `sample.ipynb` / `simple-lightgbm.ipynb` | 参考・学習用ノートブック |
| `README.md` | 構成説明・実行の手引き |
| `requirements.txt` | 依存パッケージ（`optuna` 含む） |
| `CHANGELOG.md` | 本変更履歴 |
| 各種 `.md`（解説・コンペ概略・差分比較・調査レポートなど） | ノートブックやコンペの補足資料 |
| `.gitignore` | `Dataset/` 等の除外設定 |
