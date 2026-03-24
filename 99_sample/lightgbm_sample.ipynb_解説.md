# `lightgbm_sample.ipynb` の解説（わかりやすく）

対象ファイル: **`lightgbm_sample.ipynb`**（Kaggle コンペ初期の LightGBM ベースライン系カーネル）

---

## 1. このノートブックは何をするか

**Google Brain Ventilator Pressure Prediction** コンペで、**気道圧 `pressure` を予測する**ためのノートです。

やっていることはシンプルに言うと次のとおりです。

1. `train.csv` / `test.csv` を読む  
2. **たくさんの特徴量**を作る（時系列・呼吸 ID 単位の加工を含む）  
3. **LightGBM** で学習し、**同じ呼吸が学習と検証に分かれない**ように分割して評価する  
4. テストを予測して **`submission.csv`** を出力する  

自前の `ventilator_pressure_lightgbm.ipynb` も同じコンペ用ですが、こちらは **英語・本番寄りの構成**（設定クラス・ログ・図の保存・特徴量が多い）になっています。

---

## 2. 全体の流れ（セクション）

| セクション                   | 内容                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| **Config**              | モデル名、LightGBM のパラメータ、fold 数、乱数シード、目的変数名などを **`Config` クラス**にまとめる                                         |
| **Library**             | `pandas` / `numpy` / `matplotlib` / `seaborn` / `plotly` / `sklearn` / `lightgbm` / `tqdm` / `joblib` など |
| **ユーティリティ**             | ログ保存（`Logger`）、モデルの保存読込（`Util` + joblib）、**メモリ削減**（`reduce_mem_usage`）など                                 |
| **SetUp**               | 入力フォルダ `../input/ventilator-pressure-prediction`、実験出力用に `model` / `fig` / `preds` フォルダを作成                |
| **Load Data**           | `train` / `test` / `sample_submission` を読む。`Config.debug = True` なら呼吸をランダムに 100 本だけに絞って**動作確認用**にできる     |
| **Simple EDA**          | 表の表示、件数、`breath_id` や `R`・`C` の分布、訓練とテストの列の重なり（ベン図）、サンプル呼吸の時系列プロットなど                                     |
| **Feature Engineering** | 下記「特徴量」のとおり                                                                                              |
| **Model**               | LightGBM を包んだ **`LGBM` クラス**（学習・予測・保存）                                                                   |
| **Main**                | 前処理 → 特徴量重要度の図 → 学習（OOF）→ 推論 → `submission.csv`                                                          |

---

## 3. 特徴量（詳しく）

### 3.1 前処理の順序（`preprocess`）

特徴量は **2 段階**で積み上がります。

1. **`get_whole_features(train, test)`**  
   - `whole_df = pd.concat([train, test])` として **訓練とテストを縦に結合**したうえで、**`breath_id` 単位の集約**だけを先に計算します。  
   - 理由: テスト側の呼吸も「その呼吸全体の `u_in` の平均」などに**一貫した定義**で触れるようにするため（リークに注意しつつ、コンペではよく使う手法）。

2. **`get_features(train)` / `get_features(test)`**  
   - 訓練・テスト**それぞれ**に対して、ラグ・累積・ピボットなどを計算し、上の集約特徴と **横（列方向）に連結**します。

最終的に実行ログでは **`(6036000, 104)`** のように、**約 104 列**の特徴行列になっています（行数は元 CSV と同じ＝全タイムステップ）。

各ブロックを追加したあと **`reduce_mem_usage`** で整数・浮動小数の型を詰め、**メモリ使用量を削減**しています。

---

### 3.2 関数ごとの内容

#### `get_raw_features`

- そのまま使う列: **`R`, `C`, `time_step`, `u_in`, `u_out`**（5 列）。

#### `get_cross_features`

- 列名 **`R+C`**: `R` と `C` を **文字列にして連結**し、再度 **整数**にしている（例: R=20, C=50 → `"20"+"50"` → `2050`）。  
- **組み合わせ 1 本のカテゴリ的な列**として、木モデルが「この R と C のペア」を扱いやすくする意図。

#### `get_shift_grpby_breath_id_features`

- `breath_id` ごとに **`u_in` を shift**。`shift_times = [-1, -2, 1, 2]` なので **4 本**の列。  
- 列名例: `shift=-1_u_in_grpby_breath_id`（1 ステップ**後**の `u_in` が前の行に来る、など pandas の `shift` の向きに注意）。  
- **同じ呼吸の前後ステップの入力**を特徴にする＝時系列の近さを表形式で渡す。

#### `get_diff_grpby_breath_id_features`

- 名前は「diff」だが、**実装は `get_shift_grpby_breath_id_features` と同じく `groupby(...).shift(t)`** だけ。  
- 列名だけ `diff=...` になっており、**真の差分（`u_in - u_in.shift(1)`）ではない**。コピペミスと考えてよいです。  
- 結果として **シフト特徴と中身が重複**している可能性が高いです。

#### `get_cumsum_grpby_breath_id_features`

- `breath_id` ごとの **`cumsum`** を `time_step`, `u_in`, `u_out` に対して取る（3 列）。  
- 呼吸の**序盤〜該当ステップまでの積み上げ**を表す。

#### `get_time_step_cat_features`

- `time_step` を **3 段階**にビン分割した `time_step_cat`（0 / 1 / 2）。  
  - `time_step < 1` → 0  
  - `1 < time_step < 1.5` → 1（**ちょうど 1 や 1.5 の行**は上の条件に当てはまらないため、初期値のまま残るなど**境界の取り方に隙**がある）  
  - `1.5 < time_step` → 2  
- 吸気の**早い／中盤／遅い**のざっくり区間を木が分岐しやすくする意図。

#### `get_breath_id_pivot_features`

- 各 `breath_id` 内で `time_step` の **順位**（`rank`）を `time_step_id` とみなし、**`u_in` を「ステップ番号（1〜80）」でピボット**してワイド化。  
- 列名例: `time_step_id=01_u_in`, …, `time_step_id=80_u_in`（最大 **80 列前後**になり、**次元は大きい**）。  
- 「この呼吸の k ステップ目の `u_in`」を**列として固定**するので、**同じステップ位置同士**のパターンを木が拾いやすい。

#### `get_agg_breath_id_whole_features`（`whole_df` 用）

- `breath_id` ごとに `u_in` の **`mean`, `std`, `median`, `max`, `sum`**（`aggregation` ヘルパー）。  
- さらに **`agg_z-score_u_in_grpby_breath_id`** という列を追加しているが、式は  
  `(u_in - mean) / (mean + 1e-8)`  
  となっており、**通常の z-score（分母は std）ではない**。**「瞬間の u_in が、その呼吸の平均に比べてどれだけか」**のような相対量に近い。  
- 変数名は「z-score」だが、**統計的な意味の z ではない**点に注意。

---

### 3.3 まとめ（特徴量）

| 観点        | 内容                                                                 |
| --------- | ------------------------------------------------------------------ |
| **リーク**   | train+test 結合は **集約特徴**の計算に使う典型的パターン。ピボット・シフトは train/test 別計算。     |
| **重み**    | ピボット列が多く、**メモリ・学習時間**を押し上げる。                                       |
| **改善の余地** | `get_diff_*` の重複、境界の粗い `time_step_cat`、疑似 z-score の定義などを直すと整理しやすい。 |

---

## 4. 学習設定（詳しく）

### 4.1 `Config` に書かれている値（コードそのまま）

```python
class Config:
    name_v1 = "lgb baseline"
    model_params = dict(
        objective="mae",
        n_estimators=5000,
        num_leaves=31,
        random_state=2021,
        importance_type="gain",
        colsample_bytree=0.3,
        learning_rate=0.5,
    )
    fit_params = dict(early_stopping_rounds=100, verbose=100)
    n_fold = 3
    seeds = [2021]
    target_col = "pressure"
    debug = False
```

---

### 4.2 各パラメータの意味

| パラメータ                       | 値        | 説明                                                                    |
| --------------------------- | -------- | --------------------------------------------------------------------- |
| **`objective`**             | `"mae"`  | LightGBM の sklearn ラッパーでは **L1（平均絶対誤差）を目的**にした回帰。コンペの評価指標（MAE）と揃える意図。 |
| **`n_estimators`**          | `5000`   | ブースティングの**最大反復回数**（木を足し足しする上限）。                                       |
| **`num_leaves`**            | `31`     | 1 本の木の**複雑さ**の目安。大きいほど表現力は上がるが過学習しやすい。                                |
| **`learning_rate`**         | `0.5`    | **非常に高い**。通常は 0.01〜0.1 程度が多く、0.5 は「少ない反復で大きく動く」設定。ベースライン実験・荒い探索向き。    |
| **`colsample_bytree`**      | `0.3`    | 各木を作るとき**列の 30% だけ**をランダムに使う（特徴が 100 本超あるので**過学習抑制・多様性**のため）。         |
| **`importance_type`**       | `"gain"` | 特徴量重要度を **利得（gain）** で見る設定。                                           |
| **`random_state`**          | `2021`   | 乱数の再現用。`seeds` ループで上書きもされる。                                           |
| **`early_stopping_rounds`** | `100`    | 検証データの指標が **100 ラウンド改善しなければ**学習停止。                                    |
| **`verbose`**               | `100`    | 100 イテレーションごとにログ出力。                                                   |

実際のログでは **5000 イテレーションまで打ち切られず**「Did not meet early stopping」と出る場合があり、**検証誤差がまだ下がり続けている**状態で終わっている例があります（その場合は過学習や計算コストのバランスを別途調整する必要あり）。

---

### 4.3 モデル API（`LGBM` クラス）

- 内部で **`lightgbm.LGBMModel`** を `**model_params` で生成。  
- **`fit`**: `eval_set=[(va_x, va_y)]` を渡し、**検証データで early stopping** が効く形。  
- 学習後、`model/` 下に **`{name}_fold{k}.pkl`** として保存。

---

### 4.4 評価指標 `metrics`

```python
def metrics(y_true, y_pred):
    # todo: expiratory phase is not scored (?)
    score = mean_absolute_error(y_true, y_pred)
    return score
```

- **全行**に対して MAE を計算。  
- コメントどおり、コンペ本番は **呼気相（`u_out=1`）はスコアに含まれない**のに対し、ここでは **吸気・呼気どちらも同じ重み**で MAE を見ている可能性がある。  
- **OOF の数字と LB の相関**を取るときは、この点のギャップに注意。

---

### 4.5 交差検証（カスタム `GroupKFold`）

- **`breath_id` のユニーク値のリスト**に対して、通常の **`KFold`（shuffle=True）** で「どの呼吸 ID を検証に回すか」を決める。  
- その ID に属する**全行**のインデックスを train / valid に振り分けるので、**同じ呼吸のステップが train と valid に割れることはない**。  
- `n_splits=3` なので **3-fold**。  
- `gkf(train, group=train["breath_id"], n_splits=3, random_state=seed)` で **分割リスト**を生成し、`train_cv_v1` に渡す。

---

### 4.6 シードとアンサンブル

- `Config.seeds = [2021]` の **1 本だけ**をループしているので、**乱数の違う複数モデルを平均する**ような **seed アンサンブルはしていない**。  
- `seeds` に `[2021, 2022, ...]` のように増やすと、**OOF / テスト予測を列ごとに足して平均**できる設計になっている。

---

### 4.7 まとめ（学習設定）

| 観点 | 内容 |
|------|------|
| **目的** | MAE 最小化に近い `objective="mae"`。 |
| **強めの学習** | `learning_rate=0.5` と `n_estimators=5000` の組み合わせは攻め。 |
| **汎化** | `colsample_bytree` と GroupKFold で緩和。 |
| **検証の解釈** | OOF は全行 MAE。公式評価（吸気のみ）とは**一致しない**可能性。 |

---

## 5. Main で何が起きるか

1. **`preprocess`**  
   - 上記の特徴量を作り、`train_x`, `train_y`, `test_x` を得る。  
   - 行数は元データと同じ（**全タイムステップ**が対象の想定）。

2. **特徴量重要度**  
   - 一度 LightGBM を回して、どの列が効いているか **図（`fig/importance.png`）** を保存。

3. **学習（`train_cv_v1`）**  
   - fold ごとにモデルを学習し、**OOF（学習データのうち検証に回った行の予測）** を `preds/oof.csv` に保存。  
   - ログに **OOF スコア（MAE 系）** を出す。

4. **推論（`predict_cv_v1`）**  
   - 各 fold のモデルでテストを予測し、**平均**して最終予測にする。

5. **提出**  
   - `sample_submission['pressure']` に予測を入れて **`submission.csv`** を保存。

※ 実行ログでは **約 3 時間以上**かかるような記述があり、フルデータでは**重い**処理です。

---

## 6. 自前の `ventilator_pressure_lightgbm.ipynb` との違い（目安）

| 項目    | `lightgbm_sample.ipynb`            | `ventilator_pressure_lightgbm.ipynb` |
| ----- | ---------------------------------- | ------------------------------------ |
| 言語・説明 | 英語、セクション見出しは短い                     | 日本語で分析設計つき                           |
| 特徴量   | **多い**（ピボット・集約・plotly EDA など）      | **シンプル**（ラグ・累積・rolling など）           |
| 学習データ | 行の除外の記述はノート本文では**明示的でない**（全行学習の想定） | **`u_out != 0` の行だけ学習**（吸気相に合わせる）    |
| API   | `lightgbm.LGBMModel` + 自作ラッパー      | `lightgbm.train`（低レベル API）           |
| パス    | `../input/...`（Kaggle）             | `/kaggle/input/...`                  |

コンペの評価は **吸気相のみ MAE** であることが多いので、**学習で呼気行を除くかどうか**はスコア設計とセットで検討する必要があります。

---

## 7. ローカルで触るときの注意

- **入力パス**を `Dataset/` など自分の環境に合わせて変更する。  
- **`plotly`** や **`matplotlib_venn`** など、環境によっては `pip install` が必要。  
- ノートは **実行済み出力が JSON に大量に含まれる**とファイルサイズが大きくなる（Git 管理時は注意）。

---

## 8. まとめ

**`lightgbm_sample.ipynb` は、Ventilator コンペ向けの「特徴量多め・実験フォルダ構成付き」の LightGBM パイプライン**です。  
内容を理解したうえで、自前ノートに取り入れたい部分（集約特徴、メモリ削減、ログ保存など）だけを移植する使い方が現実的です。
