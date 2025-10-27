# Phoneme Sequence Learning

音素シーケンスを学習するためのGPT-2ベースの言語モデルです。

## データ

- `persona_text.json`: 人物テキストデータ
- `sentence0-3.json`: 音素データ

各JSONファイルには `text` と `phonemes` (音素の配列) が含まれています。

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

### 0. ダミーモデルの作成（テスト用）

学習せずに検証コードをテストする場合：

```bash
python create_dummy_model.py
```

これで未学習のモデルと語彙が `./model` に作成されます。

### 1. 学習

```bash
python train.py
```

モデルは `./model` に保存されます。

### 2. 検証・評価

学習済みモデルを評価・検証します。

#### データセットで評価

```bash
python validate.py eval 100
```
（100サンプルで評価）

#### 対話的な生成

```bash
python validate.py interactive
```

音素を入力すると、モデルが続きを生成します。

#### 特定のシーケンスを生成

```bash
python validate.py generate y o r o
```

指定した音素から続きを生成します。

### デフォルト設定

- モデルサイズ: 約0.5Mパラメータ（2層、4ヘッド、128次元）
- エポック数: 3
- 学習率: 5e-4
- バッチサイズ: 8

設定は `train.py` の `training_args` で変更できます。

## 出力ファイル

- `./model/`: 学習済みモデル
- `./model/vocab.json`: 語彙マッピング
- `./results/`: チェックポイント
- `./logs/`: 学習ログ