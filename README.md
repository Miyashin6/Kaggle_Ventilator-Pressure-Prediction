# Google Brain Ventilator Pressure Prediction Challenge - 初心者向け

## このリポジトリの構成（初心者向けの流れ）

このコンペでは、**ベースライン作成 → 特徴エンジニアリング → モデルチューニング** の順で進めると理解しやすくなります。

| 段階                | 内容                                                          | ノートブックでの対応                           |
| ----------------- | ----------------------------------------------------------- | ------------------------------------ |
| **1. ベースライン作成**   | 生データを最小限の前処理で扱い、シンプルなモデルで予測する。スコアの「基準」を作る。                  | 分析設計・データ前処理・EDA・最小限の特徴量で LightGBM 学習 |
| **2. 特徴エンジニアリング** | 時系列（1 breath あたり 80 ステップ）を活かした特徴量（ラグ、累積、差分など）を追加し、予測精度を上げる。 | 特徴量作成・データセット作成のセクションで実施              |
| **3. モデルチューニング**  | LightGBM のハイパーパラメータやバリデーション方法を調整し、スコアをさらに改善する。              | バリデーション設計・モデル学習のセクションで実施             |

**推奨の進め方**  
まず `ventilator_pressure_lightgbm.ipynb` を上から順に実行してベースラインを出し、その後「特徴量の追加」と「LightGBM のチューニング」を繰り返してスコアを伸ばしていきます。

---

## ノートブックで扱う項目（目次）

ノートブック内では、次の項目が分かるようにセクション分けしています。

| 項目 | 内容 |
|------|------|
| **分析設計** | 目的変数・変数の種類（連続/カテゴリ/二値）・評価指標を明確にし、選んだ根拠を記載 |
| **データ前処理** | 読み込み、欠損・型の確認、breath 単位の並びの整理など |
| **特徴量作成** | ラグ、累積、差分、rolling など時系列を活かした特徴量の設計・実装 |
| **データセット作成** | 学習用・検証用・テスト用の特徴量テーブルの組み立て |
| **バリデーション設計** | breath_id をまたがないようにする GroupKFold など、リークを防ぐ検証方法 |
| **モデル学習** | LightGBM による学習・予測・提出用 CSV の作成 |

---

## 実行環境

**このノートブックは Web 上の Kaggle 環境専用です。** ローカルや Colab ではパスが異なるためそのままでは動きません。

---

## 実行方法（Kaggle Notebook）

1. [Ventilator Pressure Prediction](https://www.kaggle.com/c/ventilator-pressure-prediction) にアクセスし「Join Competition」で参加
2. **Notebooks** タブで「New Notebook」を選択
3. 右側 **Data** で「Add data」→ コンペの **Ventilator Pressure Prediction** のデータを追加（`train.csv`, `test.csv` が含まれるデータセット）
4. このリポジトリの `ventilator_pressure_lightgbm.ipynb` の内容をコピーして新規ノートブックに貼るか、ファイルをアップロード
5. **Run All** で実行 → 出力の `submission.csv` を **Submit** から提出

データはコンペの「Data」を追加すれば自動で `/kaggle/input/ventilator-pressure-prediction/` にマウントされます。追加のダウンロードは不要です。

```
02_ventilator-pressure/
├── ventilator_pressure_lightgbm.ipynb  ← Kaggle にアップロードして実行
├── requirements.txt
└── README.md
```

---

## 評価指標について

- **MAE（Mean Absolute Error）**: 予測した pressure と正解の pressure の絶対誤差の平均です。コンペの公式評価指標です。
- ノートブック内の「分析設計」で、目的変数が **連続値** であることと、**MAE を選んだ根拠** を記載しています。
