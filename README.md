# 因果AI ―コードファーストで学ぶ因果推論―

本リポジトリは、共立出版より刊行の訳書
[『因果AI ―コードファーストで学ぶ因果推論―』](https://www.kyoritsu-pub.co.jp/book/b10169134.html)(Robert Osazuwa Ness 著・小山田創哲 訳)のサポートページです。

原著 *Causal AI* (Manning Publications, 2025) の公式リポジトリ [altdeep/causalAI](https://github.com/altdeep/causalAI) の fork です。

## 内容

- [src](./src) – 書籍中のコードを収録した Jupyter ノートブック(jupytext percent format の .py ファイル)

## 実行環境

[uv](https://docs.astral.sh/uv/) で各章のノートブックを再現実行できます(Python 3.10)。

章ごとに必要なパッケージのバージョンが互いに非互換なため、章単位の
dependency-group に分かれています。実行したい章の group を `uv sync` してから
jupytext で実行してください。

```sh
# 例: 第3章
uv sync --group ch03
uv run --group ch03 jupytext --to notebook --execute src/ch03/ch03.py
```

`bash scripts/run_chapter.sh ch03` でも同じことができます(CI と同一の入口)。

| group | ノートブック |
|---|---|
| `ch02` | `src/ch02/ch02.py` |
| `ch03` | `src/ch03/ch03.py` |
| `ch04-markov` | `src/ch04/ch04_markov.py` |
| `ch04-functional` | `src/ch04/ch04_functional.py` |
| `ch05` | `src/ch05/ch05.py` |
| `ch06` | `src/ch06/ch06.py` |
| `ch07` | `src/ch07/ch07.py` |
| `ch09` | `src/ch09/ch09.py` |
| `ch10` | `src/ch10/ch10.py` |
| `ch11-bayesian` | `src/ch11/ch11_bayesian.py` |
| `ch11-dowhy` | `src/ch11/ch11_dowhy.py` |
| `ch12` | `src/ch12/ch12.py` |
| `ch13` | `src/ch13/ch13.py` |

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
