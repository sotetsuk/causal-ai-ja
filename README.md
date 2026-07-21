# 因果AI ―コードファーストで学ぶ因果推論―

本リポジトリは、共立出版より刊行の訳書
[『因果AI ―コードファーストで学ぶ因果推論―』](https://www.kyoritsu-pub.co.jp/book/b10169134.html)(Robert Osazuwa Ness 著・小山田創哲 訳)のサポートページです。

<p align="center">
  <a href="https://www.kyoritsu-pub.co.jp/book/b10169134.html"><img src="https://hondana-image.s3.amazonaws.com/book/image/10169134/normal_c95b7911-0adb-4b36-b173-ed29cfe7c3fa.jpg" alt="『因果AI ―コードファーストで学ぶ因果推論―』書影" width="200"></a>
</p>

<p align="center">(書影は<a href="https://www.kyoritsu-pub.co.jp/book/b10169134.html">共立出版の書籍ページ</a>より引用)</p>

原著 *Causal AI* (Manning Publications, 2025) の公式リポジトリ [altdeep/causalAI](https://github.com/altdeep/causalAI) の fork です。

## 内容

- [notebooks](./notebooks) – 書籍中のコードを収録した Jupyter ノートブック
- [src](./src) – notebooks/ の生成元(jupytext percent format の .py ファイル、管理用。
  notebooks/ は CI が src/ から自動生成します)

## ノートブックの起動

[uv](https://docs.astral.sh/uv/) で各章のノートブックを実行できます(Python 3.10)。

章ごとに必要なパッケージのバージョンが互いに非互換なため、章単位の
dependency-group に分かれています。実行したい章の group を `uv sync` してから
Jupyter を起動してください。

```sh
# 例: 第3章
uv sync --group ch03
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
| `ch11-bayesian` | [ch11_bayesian.ipynb](./notebooks/ch11_bayesian.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2011/Chapter_11_Bayesian_Causal_Graphical_Inference.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2011/Chapter_11_Bayesian_Causal_Graphical_Inference.ipynb) |
| `ch11-dowhy` | [ch11_dowhy.ipynb](./notebooks/ch11_dowhy.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2011/Chapter_11_DoWhy_Causal_Effect_Workflow.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2011/Chapter_11_DoWhy_Causal_Effect_Workflow.ipynb) |
| `ch12` | [ch12.ipynb](./notebooks/ch12.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2012/chapter_12_causal_decision.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2012/chapter_12_causal_decision.ipynb) |
| `ch13` | [ch13.ipynb](./notebooks/ch13.ipynb) | [GitHub](https://github.com/altdeep/causalAI/blob/master/book/chapter%2013/Chapter_13_Causality_LLMs.ipynb) / [Colab](https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2013/Chapter_13_Causality_LLMs.ipynb) |

注意:

- `ch09` / `ch10` / `ch11-dowhy` は pygraphviz を使うため、システムに graphviz が必要です
  (macOS: `brew install graphviz`、Ubuntu: `apt install graphviz libgraphviz-dev pkg-config`。
  macOS でビルドに失敗する場合は `CFLAGS=-I$(brew --prefix graphviz)/include`
  `LDFLAGS=-L$(brew --prefix graphviz)/lib` を指定してください)。
- ノートブックは実行時に原著リポジトリ
  [altdeep/causalML](https://github.com/altdeep/causalML) の raw URL からデータを
  取得するため、ネットワーク接続が必要です。
- `ch05` / `ch06` / `ch09` / `ch11-bayesian` / `ch13` は深層学習の訓練・SVI・MCMC・GPT-2 の
  fine-tuning を含み、CPU での完走には長時間かかります(CI ではこれらの章は依存関係の
  import 検証のみ行っています)。
- `ch10` / `ch12` は、取得したコードの実行前確認(`exec` のアンコメントや `input()` への応答)を
  読者が対話的に行う設計のため、CI では import 検証のみです。
