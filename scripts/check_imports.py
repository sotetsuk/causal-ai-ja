"""ノートブック (.py, jupytext percent format) の import が通ることを検証する。

使い方: python scripts/check_imports.py book/ch05/ch05_....py

ノートブック内の全 import 文からサードパーティのトップレベルモジュールを抽出し、
importlib で順に import する。失敗があればまとめて報告して非ゼロ終了する。
"""

import ast
import importlib
import sys
from pathlib import Path


def collect_top_level_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                modules.add(node.module.split(".")[0])
    return modules


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <notebook.py>", file=sys.stderr)
        return 2

    notebook = Path(sys.argv[1])
    modules = collect_top_level_modules(notebook.read_text(encoding="utf-8"))

    # 標準ライブラリと、ノートブックと同じディレクトリにあるローカルモジュールは除外
    local_modules = {p.stem for p in notebook.parent.glob("*.py")}
    targets = sorted(
        m
        for m in modules
        if m not in sys.stdlib_module_names and m not in local_modules
    )

    failures: list[tuple[str, Exception]] = []
    for name in targets:
        try:
            importlib.import_module(name)
            print(f"ok: {name}")
        except Exception as exc:  # noqa: BLE001 - import 時の全例外を報告対象とする
            failures.append((name, exc))
            print(f"FAIL: {name}: {exc}")

    if failures:
        print(f"\n{len(failures)}/{len(targets)} imports failed", file=sys.stderr)
        return 1
    print(f"\nall {len(targets)} imports ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
