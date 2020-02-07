# bert-knn

bertでテキストembeddingをした結果を利用して、類似文書探索を行う。

## 結論

あまり良い性能は示さない。gensimなどを利用して、tfi-df, lsiでモデルを作成するほうが良いと感がられる。
結果はcompare.ipynbに記載。

encodeには、２つの手法を使用してみる。

- [bert-as-service](https://github.com/hanxiao/bert-as-service) Multilingual Casedの結果をそのまま利用したもの
- [transformersのBertMode](https://github.com/huggingface/transformers)のラストレイヤーの平均をとったもの。日本語のトークナイザーを利用。

近似最近傍探索には[nmslib](https://github.com/nmslib/nmslib)を利用する。CPU環境では一番早いらしい。[source](https://github.com/erikbern/ann-benchmarks)

利用する[テキスト](https://www.kaggle.com/team-ai/shinzo-abe-japanese-prime-minister-twitter-nlp/version/1)。
特に意味はない。



## bert-seving-client

tensorflow2は動かなかった。server起動にメモリを食う。

worker数は8に設定。

## BertModel

transformersがtokenization_bert_japanese.pyを提供し始めたため、日本語の扱いが便利になった。実装にはmecabを利用しているため、mecabのインストールが必要。

bert-serving-clientの方が早い。

## nmslib

いずれfaissも利用してみたい。cpu環境しか使えない場合は現状ベストだと考えられる。