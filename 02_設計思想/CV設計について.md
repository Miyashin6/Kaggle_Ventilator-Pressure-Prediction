# CV設計まとめ：Ventilator Pressure Prediction

## CV（交差検証）とは

モデルの「本当の実力」を測るための仕組み。

```
全データで学習 → テストデータで評価
  ↑ 問題: 過学習しても気づけない

データを分割して「疑似テスト」を複数回作る
  ↑ これがCV（Cross-Validation）
```

---

## CVの主な種類

### 普通の KFold（基本形）

データをK等分し、1つを検証用・残りを訓練用としてローテーション。

```
fold1: [検証] [訓練] [訓練] [訓練] [訓練]
fold2: [訓練] [検証] [訓練] [訓練] [訓練]
fold3: [訓練] [訓練] [検証] [訓練] [訓練]
...
```

### GroupKFold（このプロジェクトで使うべき手法）

グループ単位（今回は `breath_id`）でまとめて分割する。

```
fold1: [breath 1〜15090 → 検証] [breath 15091〜75450 → 訓練]
fold2: [breath 15091〜30180 → 検証] [それ以外 → 訓練]
...
```

---

## このプロジェクトのデータ構造

```
訓練データ: 6,036,000 行
breath_id のユニーク数: 75,450 個
1呼吸あたりの行数: 必ず 80 行
```

1回の呼吸 = 80行がセットになった時系列データ：

```
breath_id=1  → 80行（time_step 0〜79）  ← 1回の呼吸
breath_id=2  → 80行（time_step 0〜79）  ← 別の呼吸
...
breath_id=75450 → 80行
```

---

## なぜGroupKFoldが重要か

### 問題1：現在のコードはリークしている

```python
# 現在の実装（問題あり）
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
```

`train_test_split` はランダムに行を分割するため：

```
breath_id=1 の 80行 → ランダムに散らばる
  行1〜64  → X_train（学習用）
  行65〜80 → X_valid（検証用）
       ↑
  同じ呼吸の行が train と valid に混在！
```

### 問題2：特徴量が「同じ呼吸」に依存している

このプロジェクトで使っている特徴量はすべて `breath_id` 内で計算：

```python
train["u_in_cumsum"] = train.groupby("breath_id")["u_in"].cumsum()
train["u_in_lag"]    = train.groupby("breath_id")["u_in"].shift(2)
train["u_in_diff1"]  = train.groupby("breath_id")["u_in"].diff(1)
train["u_in_diff2"]  = train.groupby("breath_id")["u_in"].diff(2)
```

例：`breath_id=1` の `time_step=5` の `u_in_cumsum` = `time_step` 0〜5 の `u_in` の合計

### 両方合わさるとデータリークが起きる

```
breath_id=1 の状況（ランダム分割の場合）

  time_step=3 → X_train（学習済み）
  time_step=4 → X_train（学習済み）
  time_step=5 → X_valid（これを予測したい）← u_in_cumsum に上の2行の値が入っている！
  time_step=6 → X_train（学習済み）
  time_step=7 → X_train（学習済み）
```

モデルは time_step=3,4,6,7 の `u_in` を「知った」状態で、
time_step=5 の `u_in_cumsum`（それらを含む）を使って検証している。
= **答えを知った状態で検証している**のと同じ。

### テスト時は全く別の呼吸を予測する

```
テストデータの breath_id は訓練データに存在しない
（完全に未知の呼吸）

テスト時:
  breath_id=99999 の time_step=5 の u_in_cumsum
  → 訓練データに breath_id=99999 は一切ない
  → 「他の行を知っている」というチートは使えない
```

そのため：

| | ランダム分割（現在） | GroupKFold |
|--|--|--|
| 同じ呼吸が分割されるか | される（リーク） | されない |
| 検証スコアの信頼性 | 低い（楽観的すぎ） | 高い（本番に近い） |
| LBとのギャップ | 大きい | 小さい |
| チューニングの効果 | ずれる | 正確 |

---

## OOF（Out-of-Fold）スコア

各foldで「モデルが見たことないデータ」に対する予測を集め、全体スコアを算出する仕組み。

```
fold1の検証予測 ──┐
fold2の検証予測 ──┤→ 全行が「見たことない」予測値 → OOF予測
fold3の検証予測 ──┘

OOF MAE = mean_absolute_error(y_全行, OOF予測_全行)
```

OOFスコアはPublic LBスコアとの相関が高く、信頼できる指標になる。

---

## 実装例

```python
from sklearn.model_selection import GroupKFold
import numpy as np

N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)

oof_preds = np.zeros(len(train_insp))

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(X, y, groups=train_insp["breath_id"])
):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr)

    oof_preds[val_idx] = model.predict(X_val)
    fold_mae = mean_absolute_error(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} MAE: {fold_mae:.4f}")

# 全fold合計のOOFスコア
oof_mae = mean_absolute_error(y, oof_preds)
print(f"OOF MAE: {oof_mae:.4f}")
```

---

## このプロジェクトの推奨CV設計

| 項目       | 選択          | 理由                 |
| -------- | ----------- | ------------------ |
| CV手法     | GroupKFold  | breath_id単位でのリーク防止 |
| groups引数 | `breath_id` | 呼吸単位で完全分離          |
| N_FOLDS  | 5           | 精度と計算時間のバランス       |
| 評価指標     | MAE         | コンペの公式評価指標         |

---

## 一言まとめ

> 「テスト時は**見たことない呼吸**を予測するのに、検証では**見た呼吸の残りの行**を予測している」
> という矛盾を解消するのがGroupKFold。
