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
uv sync --group ch03
uv run --group ch03 jupytext --to notebook --execute book/ch03/ch03_building_a_causal_graph.py
```

`bash scripts/run_chapter.sh ch03` でも同じことができます(CI と同一の入口)。

| group | ノートブック |
|---|---|
| `ch02` | `book/ch02/ch02_primer_on_probability_modeling.py` |
| `ch03` | `book/ch03/ch03_building_a_causal_graph.py` |
| `ch04-markov` | `book/ch04/ch04_testing_markov_property_on_transportation_dag.py` |
| `ch04-functional` | `book/ch04/ch04_testing_a_causal_dag_with_functional_constraints.py` |
| `ch05` | `book/ch05/ch05_connecting_causality_and_deep_learning.py` |
| `ch06` | `book/ch06/ch06_notebook.py` |
| `ch07` | `book/ch07/ch07_online_gaming_example.py` |
| `ch09` | `book/ch09/ch09_notebooks.py` |
| `ch10` | `book/ch10/ch10_identification_notebook.py` |
| `ch11-bayesian` | `book/ch11/ch11_bayesian_causal_graphical_inference.py` |
| `ch11-dowhy` | `book/ch11/ch11_dowhy_causal_effect_workflow.py` |
| `ch12` | `book/ch12/ch12_causal_decision.py` |
| `ch13` | `book/ch13/ch13_causality_llms.py` |

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
