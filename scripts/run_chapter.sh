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
  ch02)           file="src/ch02/ch02.py";            mode=execute ;;
  ch03)           file="src/ch03/ch03.py";            mode=execute ;;
  ch04-markov)    file="src/ch04/ch04_markov.py";     mode=execute ;;
  ch04-functional) file="src/ch04/ch04_functional.py"; mode=execute ;;
  ch05)           file="src/ch05/ch05.py";            mode=import-check ;;
  ch06)           file="src/ch06/ch06.py";            mode=import-check ;;
  ch07)           file="src/ch07/ch07.py";            mode=execute ;;
  ch09)           file="src/ch09/ch09.py";            mode=import-check ;;
  # ch10 はリモート取得コードの exec を読者が手動でアンコメントする設計のため非対話実行できない
  ch10)           file="src/ch10/ch10.py";            mode=import-check ;;
  # ch11-bayesian は SVI 学習が CPU で数時間かかるため import 検証のみ
  ch11-bayesian)  file="src/ch11/ch11_bayesian.py";   mode=import-check ;;
  ch11-dowhy)     file="src/ch11/ch11_dowhy.py";      mode=execute ;;
  # ch12 は input() による対話確認を含むため非対話実行できない
  ch12)           file="src/ch12/ch12.py";            mode=import-check ;;
  ch13)           file="src/ch13/ch13.py";            mode=import-check ;;
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
