# Stella Bilingual TTS Based on VITS

## 想定環境
* windows
* Python 3.7.2
* torch
* Cython

ライブラリの詳細は`requirements.txt`を参照。  
ライブラリはpipによるインストールを推奨します。

(1)torchをインストール
pip install torch torchvision torchaudio

(2)ライブラリをインストール
pip install -r requirements.txt

(3)コンパイル
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace

### データセットの用意
1. <a href="https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus">JVS corpus</a>をダウンロード、解凍します。  


## 使い方
一括日本語音声作成
python tts.py データセット保存パス\jvs_ver1\jvs005\falset10\transcripts_utf8.txt
