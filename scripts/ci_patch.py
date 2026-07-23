#!/usr/bin/env python
"""CI 用に src のノートブックへ最小限のパッチを当てて書き出すスクリプト。

使い方: python scripts/ci_patch.py <group> <src> <out>

src 本体は変更せず、CI での実行(読者操作の模擬・計算量の縮小)に必要な
置換だけを適用したコピーを out に書き出す。各置換は適用前に old の存在を
assert し、src 側の変更で空振りした場合は即座に失敗して CI で気づけるようにする。
"""
import sys

PATCHES = {
    # ch10: 本文はリモート取得したコードを読者が確認してから exec の行を
    # アンコメントして実行する設計。CI では確認済みとみなしてアンコメントを模擬する
    "ch10": [
        (
            "#exec(utilities_code)",
            "exec(utilities_code)",
        ),
    ],
    # ch12: 本文はリモート取得したスクリプトの実行可否を input() で読者に確認する。
    # CI では yes 応答を模擬する(プロンプト自体は出力に残す)
    "ch12": [
        (
            'confirm = input("\\nDo you want to execute this script? (yes/no): ")',
            'print("\\nDo you want to execute this script? (yes/no): ")\n'
            'confirm = "yes"',
        ),
    ],
    # ch05: フル実行(2500 エポック)は約 8 時間かかるためエポック数を縮小する。
    # また macOS ランナーは multiprocessing が spawn 方式のため、num_workers=1 の
    # ままではノートブック上で定義した Dataset をワーカーが復元できずハングする
    "ch05": [
        (
            "kwargs = {'num_workers': 1, 'pin_memory': use_cuda}",
            "kwargs = {'num_workers': 0, 'pin_memory': use_cuda}",
        ),
        (
            "NUM_EPOCHS = 2500",
            "NUM_EPOCHS = 2",
        ),
    ],
    # ch11-bayesian: SVI 学習 500_000 ステップは CPU で 2 時間超かかるため縮小する
    "ch11-bayesian": [
        (
            "for step in range(500_000):",
            "for step in range(2_000):",
        ),
    ],
    # ch13: リスト13.12/13.13 の seq2seq 学習は CPU で数日かかるためスキップする。
    # リスト13.14 が Hugging Face Hub から学習済みモデルを取得して同名変数に
    # 代入し直すため、後続のセルには影響しない(df["King and Prince"] の代入は残す)
    "ch13": [
        (
            'king_dataset = create_king_dataset(df["King"])\n'
            "king_model = train_king_model(king_model_path, king_dataset)\n"
            "\n"
            'datasets = create_seq2seq_datasets(df["King"], df["Prince"])\n'
            "train_dataset_prince, val_dataset_prince = datasets\n"
            "prince_model = train_seq2seq_model(\n"
            "    prince_model_path,\n"
            "    train_dataset_prince,\n"
            "    val_dataset_prince,\n"
            "    epochs=6\n"
            ")",
            "# CI: 学習呼び出しをスキップ(リスト13.14 で学習済みモデルを取得する)",
        ),
        (
            "train_dataset_kingdom, val_dataset_kingdom = create_seq2seq_datasets(\n"
            '    df["King and Prince"], df["Kingdom"]\n'
            ")\n"
            "kingdom_model = train_seq2seq_model(\n"
            "    kingdom_model_path,\n"
            "    train_dataset_kingdom,\n"
            "    val_dataset_kingdom,\n"
            "   epochs=6\n"
            ")",
            "# CI: 学習呼び出しをスキップ(リスト13.14 で学習済みモデルを取得する)",
        ),
        (
            "p2k_data = create_seq2seq_datasets(\n"
            '    df["Prince"], df["King"])\n'
            "train_dataset_prince2king, val_dataset_prince2king = p2k_data\n"
            "prince2king_model = train_seq2seq_model(\n"
            "    prince2king_model_path,\n"
            "    train_dataset_prince2king,\n"
            "    val_dataset_prince2king,\n"
            "    epochs=6\n"
            ")",
            "# CI: 学習呼び出しをスキップ(リスト13.14 で学習済みモデルを取得する)",
        ),
        # 重点サンプリングの標本数を縮小する(フルの 1000 標本では数時間かかる)
        (
            "num_samples = 1000",
            "num_samples = 20",
        ),
    ],
}


def main():
    group, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if group not in PATCHES:
        raise SystemExit(f"no patches defined for group: {group}")
    with open(src, encoding="utf-8") as f:
        text = f.read()
    for old, new in PATCHES[group]:
        assert old in text, (
            f"patch target not found in {src} (src が変更された可能性があります):\n{old}"
        )
        text = text.replace(old, new, 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"patched {src} -> {out} ({len(PATCHES[group])} patches)")


if __name__ == "__main__":
    main()
