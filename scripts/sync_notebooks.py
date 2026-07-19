"""src/ 以下の jupytext percent format の .py から notebooks/ に .ipynb を生成する。

notebooks/ は毎回フル再生成する(手動編集しないこと)。
使い方: uv run python scripts/sync_notebooks.py
"""

import shutil
from pathlib import Path

import jupytext

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    for py in sorted(SRC.rglob("*.py")):
        # jupytext ヘッダを持つファイルのみ notebook として変換
        head = py.read_text(encoding="utf-8").splitlines()[:5]
        if not any(line.startswith("# jupyter:") for line in head):
            continue
        nb = jupytext.read(py, fmt="py:percent")
        # セル ID はデフォルトでランダム生成され再生成のたびに diff が出るため、
        # 決定的な ID を割り当てる
        for i, cell in enumerate(nb.cells):
            cell.id = f"cell-{i}"
        out = OUT / py.relative_to(SRC).with_suffix(".ipynb")
        out.parent.mkdir(parents=True, exist_ok=True)
        jupytext.write(nb, out, fmt="ipynb")
        print(f"{py.relative_to(ROOT)} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
