# 手書き数字認識のバッチ

「[PyTorchニューラルネットワーク実装ハンドブック](www.amazon.co.jp/dp/4798055476)」の[4.1節 AlexNetで画像分類するタスク](https://github.com/miyamotok0105/pytorch_handbook/blob/master/chapter4/section4_1.ipynb)をバッチ化したプロジェクトです。

## インストール
1. `$ pip install -r requirements.txt`

## コマンドライン実行方法
1. `$ python handwritten_num_classifier/train_model.py  <config_file> <working_root>`
    - config_file: コンフィグ用のyamlファイル
    - working_root: どこを起点としてデータを参照するか

## コンフィグファイルのフォーマット

`./config/sample.yml`にサンプルを置いています。

```yaml
# 学習エポック数
num_epochs: 20
# GPUを使用するか
cuda: true
# 成果物を出力するディレクトリ。working rootからの相対パス。nullの場合は .myoutput に作られる
output_data:

data:
  # 学習用
  train:
    # データセット
    dataset:
      # データセット
      # MNISTのdatasetクラスを取得するためのパラメータ
      mnist_train:
        # データの置き場
        dir: &dataset data/images
        train_mode: true
        # 画像の前処理。上から下へ順番にかける。
        transform: &transform
          - Resize:
            width: 32
            height: 32
          - Gray2BGR:
          - Invert:
          - ToTensor:
    dataloader: &dataloader
      batch_size: 64
      shuffle: true
  # 評価
  eval:
    dataset:
      mnist_eval:
        dir: *dataset
        train_mode: false
        transform: *transform
    dataloader: 
      batch_size: 64
      shuffle: false

# ネットワークアーキテクチャ
model:
  model_name: alexnet
  # クラス数
  num_classes: 10
  # クラスラベル
  class_labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

loss:
  loss_name: cross_entropy_loss

optimizer:
  optimizer_name: sgd
  lr: 0.01
  momentum: 0.9
  weight_decay: 0.0005

```


## 成果物
`<working_root>/<output_data>/yyyy-mm-dd_hhMMss`という、ディレクトリを作成その中に下記の要素を保存するディレクトリを作成

- code: 学習したコード
- config: コンフィグ  
- detail: アーティファクト(重み)  
- metrics: メトリクス  
- model: 評価の詳細ファイル
