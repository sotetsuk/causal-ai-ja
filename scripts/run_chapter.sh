#!/usr/bin/env bash
# 章 group を指定してノートブックを検証する。
#
# 使い方: bash scripts/run_chapter.sh <group>
#   例:   bash scripts/run_chapter.sh ch03
#
# mode=execute       : jupytext でノートブックを末尾まで実行する
# mode=execute-sim   : 読者の対話操作(exec のアンコメントや input() への応答)を
#                      scripts/ci_patch.py で模擬した上で末尾まで実行する
# mode=execute-smoke : フル実行が CI では非現実的な章について、scripts/ci_patch.py で
#                      計算量を縮小した上で末尾まで実行する(完走性のみ検証)
#
# 事前に `uv sync --group <group>` 済みであること(uv run が自動で sync する)。
set -euo pipefail

group="${1:?usage: $0 <group>}"

case "$group" in
  ch02)           file="src/ch02.py";            mode=execute ;;
  ch03)           file="src/ch03.py";            mode=execute ;;
  ch04-markov)    file="src/ch04_markov.py";     mode=execute ;;
  ch04-functional) file="src/ch04_functional.py"; mode=execute ;;
  # ch05 はフル実行(2500 エポック)が約 8 時間かかるためエポック数を縮小する
  # (macOS は multiprocessing が spawn 方式のため num_workers=0 への変更も必要)
  ch05)           file="src/ch05.py";            mode=execute-smoke ;;
  ch06)           file="src/ch06.py";            mode=execute ;;
  ch07)           file="src/ch07.py";            mode=execute ;;
  ch09)           file="src/ch09.py";            mode=execute ;;
  # ch10 はリモート取得コードの exec を読者が手動でアンコメントする設計のため CI では模擬する
  ch10)           file="src/ch10.py";            mode=execute-sim ;;
  # ch11-bayesian は SVI 学習(500_000 ステップ)が CPU で 2 時間超かかるため縮小する
  ch11-bayesian)  file="src/ch11_bayesian.py";   mode=execute-smoke ;;
  ch11-dowhy)     file="src/ch11_dowhy.py";      mode=execute ;;
  # ch12 は input() による対話確認を含むため CI では yes 応答を模擬する
  ch12)           file="src/ch12.py";            mode=execute-sim ;;
  # ch13 は seq2seq モデルの学習に CPU で数日かかるため学習をスキップし標本数を縮小する
  ch13)           file="src/ch13.py";            mode=execute-smoke ;;
  *) echo "unknown group: $group" >&2; exit 2 ;;
esac

echo "== $group: $mode: $file"
case "$mode" in
  execute)
    out="${RUNNER_TEMP:-/tmp}/${group}.ipynb"
    uv run --exact --group "$group" jupytext --to notebook --execute -o "$out" "$file"
    ;;
  execute-sim|execute-smoke)
    tmp="${RUNNER_TEMP:-/tmp}/${group}_patched.py"
    uv run --exact --group "$group" python scripts/ci_patch.py "$group" "$file" "$tmp"
    out="${RUNNER_TEMP:-/tmp}/${group}.ipynb"
    uv run --exact --group "$group" jupytext --to notebook --execute -o "$out" "$tmp"
    ;;
esac
