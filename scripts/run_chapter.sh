#!/usr/bin/env bash
# 章 group を指定してノートブックを検証する。
#
# 使い方: bash scripts/run_chapter.sh <group>
#   例:   bash scripts/run_chapter.sh ch03
#
# mode=execute      : jupytext でノートブックを末尾まで実行する
# mode=import-check : 依存の import が通ることのみ検証する(重い章向け)
#
# 事前に `uv sync --group <group>` 済みであること(uv run が自動で sync する)。
set -euo pipefail

group="${1:?usage: $0 <group>}"

case "$group" in
  ch02)           file="book/ch02/ch02_primer_on_probability_modeling.py";                  mode=execute ;;
  ch03)           file="book/ch03/ch03_building_a_causal_graph.py";                         mode=execute ;;
  ch04-markov)    file="book/ch04/ch04_testing_markov_property_on_transportation_dag.py";   mode=execute ;;
  ch04-functional) file="book/ch04/ch04_testing_a_causal_dag_with_functional_constraints.py"; mode=execute ;;
  ch05)           file="book/ch05/ch05_connecting_causality_and_deep_learning.py";          mode=import-check ;;
  ch06)           file="book/ch06/ch06_notebook.py";                                        mode=import-check ;;
  ch07)           file="book/ch07/ch07_online_gaming_example.py";                           mode=execute ;;
  ch09)           file="book/ch09/ch09_notebooks.py";                                       mode=import-check ;;
  # ch10 はリモート取得コードの exec を読者が手動でアンコメントする設計のため非対話実行できない
  ch10)           file="book/ch10/ch10_identification_notebook.py";                         mode=import-check ;;
  # ch11-bayesian は SVI 学習が CPU で数時間かかるため import 検証のみ
  ch11-bayesian)  file="book/ch11/ch11_bayesian_causal_graphical_inference.py";             mode=import-check ;;
  ch11-dowhy)     file="book/ch11/ch11_dowhy_causal_effect_workflow.py";                    mode=execute ;;
  # ch12 は input() による対話確認を含むため非対話実行できない
  ch12)           file="book/ch12/ch12_causal_decision.py";                                 mode=import-check ;;
  ch13)           file="book/ch13/ch13_causality_llms.py";                                  mode=import-check ;;
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
