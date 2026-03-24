# Optuna実行結果

## 概要

- 最適化対象: LightGBM のハイパーパラメータ
- 最適化指標: MAE（最小化）
- 試行回数: `20`

## ベストスコア

- Best MAE: `1.1589910192692925`

## ベストパラメータ

| パラメータ | 値 |
|---|---:|
| `learning_rate` | `0.03829428196449332` |
| `num_leaves` | `162` |
| `max_depth` | `12` |
| `colsample_bytree` | `0.8775206571361439` |
| `reg_alpha` | `1.3166506372889528` |
| `reg_lambda` | `0.00012124153515675034` |
| `min_child_weight` | `5` |
| `min_child_samples` | `73` |
| `bagging_fraction` | `0.8912541028492775` |
| `bagging_freq` | `5` |

## 実行時データ情報

- チューニング対象行数: `151605`
- 使用特徴量数: `11`

## 使用特徴量一覧

- `R`
- `C`
- `time_step`
- `u_in`
- `u_out`
- `u_in_cumsum`
- `u_in_lag`
- `u_in_diff1`
- `u_in_diff2`
- `R_C_mul`
- `R_C_div`
