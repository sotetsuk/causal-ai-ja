#!/usr/bin/env bash
# 章 group を指定してノートブックを検証する。
#
# 使い方: bash scripts/run_chapter.sh <group>
#   例:   bash scripts/run_chapter.sh ch3
#
# mode=execute      : jupytext でノートブックを末尾まで実行する
# mode=import-check : 依存の import が通ることのみ検証する(重い章向け)
#
# 事前に `uv sync --group <group>` 済みであること(uv run が自動で sync する)。
set -euo pipefail

group="${1:?usage: $0 <group>}"

case "$group" in
  ch2)            file="book/chapter 2/Chapter_2_Primer_on_Probability_Modeling.py";         mode=execute ;;
  ch3)            file="book/chapter 3/Chapter_3_Building_a_Causal_Graph.py";                mode=execute ;;
  ch4-markov)     file="book/chapter 4/Testing_Markov_Property_on_Transportation_DAG.py";    mode=execute ;;
  ch4-functional) file="book/chapter 4/Testing_a_causal_DAG_with_functional_constraints.py"; mode=execute ;;
  ch5)            file="book/chapter 5/chapter_5_Connecting_Causality_and_Deep_Learning.py"; mode=import-check ;;
  ch6)            file="book/chapter 6/chapter_6_notebook.py";                               mode=import-check ;;
  ch7)            file="book/chapter 7/chapter_7_online_gaming_example.py";                  mode=execute ;;
  ch9)            file="book/chapter 9/Chapter_9_notebooks.py";                              mode=import-check ;;
  # ch10 はリモート取得コードの exec を読者が手動でアンコメントする設計のため非対話実行できない
  ch10)           file="book/chapter 10/Chapter_10_Identification_notebook.py";              mode=import-check ;;
  # ch11-bayesian は SVI 学習が CPU で数時間かかるため import 検証のみ
  ch11-bayesian)  file="book/chapter 11/Chapter_11_Bayesian_Causal_Graphical_Inference.py";  mode=import-check ;;
  ch11-dowhy)     file="book/chapter 11/Chapter_11_DoWhy_Causal_Effect_Workflow.py";         mode=execute ;;
  # ch12 は input() による対話確認を含むため非対話実行できない
  ch12)           file="book/chapter 12/chapter_12_causal_decision.py";                      mode=import-check ;;
  ch13)           file="book/chapter 13/Chapter_13_Causality_LLMs.py";                       mode=import-check ;;
  *) echo "unknown group: $group" >&2; exit 2 ;;
esac

echo "== $group: $mode: $file"
case "$mode" in
  execute)
    out="${RUNNER_TEMP:-/tmp}/${group}.ipynb"
    uv run --exact --group "$group" jupytext --to notebook --execute -o "$out" "$file"
    ;;
  import-check)
    uv run --exact --group "$group" python scripts/check_imports.py "$file"
    ;;
esac
