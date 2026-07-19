# 因果AI ―コードファーストで学ぶ因果推論―

本リポジトリは、共立出版より刊行の訳書
[『因果AI ―コードファーストで学ぶ因果推論―』](https://www.kyoritsu-pub.co.jp/book/b10169134.html)(Robert Osazuwa Ness 著・小山田創哲 訳)のサポートページです。

原著 *Causal AI* (Manning Publications, 2025) の公式リポジトリ [altdeep/causalAI](https://github.com/altdeep/causalAI) の fork です。

## 内容

- [book](./book) – 書籍中のコードを収録した Jupyter ノートブック(jupytext percent format の .py ファイル)

## 実行環境

[uv](https://docs.astral.sh/uv/) で各章のノートブックを再現実行できます(Python 3.10)。

章ごとに必要なパッケージのバージョンが互いに非互換なため、章単位の
dependency-group に分かれています。実行したい章の group を `uv sync` してから
jupytext で実行してください。

```sh
# 例: 第3章
uv sync --group ch3
uv run --group ch3 jupytext --to notebook --execute "book/chapter 3/Chapter_3_Building_a_Causal_Graph.py"
```

`bash scripts/run_chapter.sh ch3` でも同じことができます(CI と同一の入口)。

| group | ノートブック |
|---|---|
| `ch2` | `book/chapter 2/Chapter_2_Primer_on_Probability_Modeling.py` |
| `ch3` | `book/chapter 3/Chapter_3_Building_a_Causal_Graph.py` |
| `ch4-markov` | `book/chapter 4/Testing_Markov_Property_on_Transportation_DAG.py` |
| `ch4-functional` | `book/chapter 4/Testing_a_causal_DAG_with_functional_constraints.py` |
| `ch5` | `book/chapter 5/chapter_5_Connecting_Causality_and_Deep_Learning.py` |
| `ch6` | `book/chapter 6/chapter_6_notebook.py` |
| `ch7` | `book/chapter 7/chapter_7_online_gaming_example.py` |
| `ch9` | `book/chapter 9/Chapter_9_notebooks.py` |
| `ch10` | `book/chapter 10/Chapter_10_Identification_notebook.py` |
| `ch11-bayesian` | `book/chapter 11/Chapter_11_Bayesian_Causal_Graphical_Inference.py` |
| `ch11-dowhy` | `book/chapter 11/Chapter_11_DoWhy_Causal_Effect_Workflow.py` |
| `ch12` | `book/chapter 12/chapter_12_causal_decision.py` |
| `ch13` | `book/chapter 13/Chapter_13_Causality_LLMs.py` |

注意:

- `ch9` / `ch10` / `ch11-dowhy` は pygraphviz を使うため、システムに graphviz が必要です
  (macOS: `brew install graphviz`、Ubuntu: `apt install graphviz libgraphviz-dev pkg-config`。
  macOS でビルドに失敗する場合は `CFLAGS=-I$(brew --prefix graphviz)/include`
  `LDFLAGS=-L$(brew --prefix graphviz)/lib` を指定してください)。
- ノートブックは実行時に原著リポジトリ
  [altdeep/causalML](https://github.com/altdeep/causalML) の raw URL からデータを
  取得するため、ネットワーク接続が必要です。
- `ch5` / `ch6` / `ch9` / `ch11-bayesian` / `ch13` は深層学習の訓練・SVI・MCMC・GPT-2 の
  fine-tuning を含み、CPU での完走には長時間かかります(CI ではこれらの章は依存関係の
  import 検証のみ行っています)。
- `ch10` / `ch12` は、取得したコードの実行前確認(`exec` のアンコメントや `input()` への応答)を
  読者が対話的に行う設計のため、CI では import 検証のみです。
