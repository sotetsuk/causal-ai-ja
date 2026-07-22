# 因果AI ―コードファーストで学ぶ因果推論―

本リポジトリは、共立出版より刊行の訳書
[『因果AI ―コードファーストで学ぶ因果推論―』](https://www.kyoritsu-pub.co.jp/book/b10169134.html)(Robert Osazuwa Ness 著・小山田創哲 訳)のサポートページです。
原著 *Causal AI* (Manning Publications, 2025) の公式リポジトリ [altdeep/causalAI](https://github.com/altdeep/causalAI) の fork です。

<p align="center">
  <a href="https://www.kyoritsu-pub.co.jp/book/b10169134.html"><img src="https://hondana-image.s3.amazonaws.com/book/image/10169134/normal_c95b7911-0adb-4b36-b173-ed29cfe7c3fa.jpg" alt="『因果AI ―コードファーストで学ぶ因果推論―』書影" width="200"></a>
</p>

<p align="center">(書影は<a href="https://www.kyoritsu-pub.co.jp/book/b10169134.html">共立出版の書籍ページ</a>より引用)</p>

## Notebook

[notebooks](./notebooks) 以下に原著リポジトリから fork した書籍中のコードを収録した Jupyter ノートブックを置いています。[uv](https://docs.astral.sh/uv/) で各章のノートブックを実行できます(Python 3.10)。章ごとに必要なパッケージのバージョンが互いに非互換なため、章単位の dependency-group に分かれています。実行したい章の group を `uv sync` してから
Jupyter を起動してください。

```sh
uv sync --group ch03  # 第3章の例
uv run --group ch03 jupyter lab notebooks/ch03.ipynb
```

| group | Notebook | 原著 |
|---|---|---|
| `ch02` | [ch02.ipynb](./notebooks/ch02.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%202/Chapter_2_Primer_on_Probability_Modeling.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%202/Chapter_2_Primer_on_Probability_Modeling.ipynb) |
| `ch03` | [ch03.ipynb](./notebooks/ch03.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%203/Chapter_3_Building_a_Causal_Graph.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%203/Chapter_3_Building_a_Causal_Graph.ipynb) |
| `ch04-markov` | [ch04_markov.ipynb](./notebooks/ch04_markov.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%204/Testing_Markov_Property_on_Transportation_DAG.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%204/Testing_Markov_Property_on_Transportation_DAG.ipynb) |
| `ch04-functional` | [ch04_functional.ipynb](./notebooks/ch04_functional.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%204/Testing_a_causal_DAG_with_functional_constraints.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%204/Testing_a_causal_DAG_with_functional_constraints.ipynb) |
| `ch05` | [ch05.ipynb](./notebooks/ch05.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%205/chapter_5_Connecting_Causality_and_Deep_Learning.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%205/chapter_5_Connecting_Causality_and_Deep_Learning.ipynb) |
| `ch06` | [ch06.ipynb](./notebooks/ch06.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%206/chapter_6_notebook.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%206/chapter_6_notebook.ipynb) |
| `ch07` | [ch07.ipynb](./notebooks/ch07.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%207/chapter_7_online_gaming_example.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%207/chapter_7_online_gaming_example.ipynb) |
| `ch09` | [ch09.ipynb](./notebooks/ch09.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%209/Chapter_9_notebooks.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%209/Chapter_9_notebooks.ipynb) |
| `ch10` | [ch10.ipynb](./notebooks/ch10.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2010/Chapter_10_Identification_notebook.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2010/Chapter_10_Identification_notebook.ipynb) |
| `ch11-dowhy` | [ch11_dowhy.ipynb](./notebooks/ch11_dowhy.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2011/Chapter_11_DoWhy_Causal_Effect_Workflow.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2011/Chapter_11_DoWhy_Causal_Effect_Workflow.ipynb) |
| `ch11-bayesian` | [ch11_bayesian.ipynb](./notebooks/ch11_bayesian.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2011/Chapter_11_Bayesian_Causal_Graphical_Inference.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2011/Chapter_11_Bayesian_Causal_Graphical_Inference.ipynb) |
| `ch12` | [ch12.ipynb](./notebooks/ch12.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2012/chapter_12_causal_decision.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2012/chapter_12_causal_decision.ipynb) |
| `ch13` | [ch13.ipynb](./notebooks/ch13.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2013/Chapter_13_Causality_LLMs.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2013/Chapter_13_Causality_LLMs.ipynb) |

> [!WARNING]
> - `ch09` / `ch10` / `ch11-dowhy` は pygraphviz を使うため、システムに graphviz が必要です。
> - ノートブックは実行時に原著リポジトリの raw URL からデータを取得するため、ネットワーク接続が必要です。
> - `ch05` / `ch06` / `ch09` / `ch11-bayesian` / `ch13` は深層学習の訓練・SVI・MCMC・GPT-2 の fine-tuning を含み、CPU での完走には長時間かかります(CI では import 検証のみ行っています)。
> - `ch10` / `ch12` は、取得したコードの実行前確認(`exec` のアンコメントや `input()` への応答)を読者が対話的に行う必要があります。

## 正誤表

| 該当箇所 | 誤(Before) | 正(After) | 説明 |
|---|---|---|---|
| p.73 | pgmpy バージョン 0.1.24 | pgmpy バージョン 0.1.25 | pgmpy 0.1.24 以前の EM には既知のバグがあり、リスト3.6 の出力が影響を受けます。このバグは 0.1.25 で修正されています([CHANGELOG](https://github.com/pgmpy/pgmpy/blob/dev/CHANGELOG.md))。 |


## 関連リンク

- [Manning 書籍ページ](https://www.manning.com/books/causal-ai): 原著 *Causal AI* (Robert Osazuwa Ness, Manning Publications, 2025)
- [altdeep/causalAI](https://github.com/altdeep/causalAI): 原著の公式コードリポジトリ(本リポジトリの fork 元)
- [著者による書籍紹介ページ](https://www.robertosazuwaness.com/causal-ai-book/): 各章の補足資料・参考文献へのリンクあり
