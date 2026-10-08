import os
TARGETS = {
    "src/engine_a/filters.py": [
        ('format(b_start, ".0f")', 'format(b_start - 0.001, ".3f")'),
        ('format(b_end, ".0f")', 'format(b_end + 0.001, ".3f")'),
        ("無応答率", "データ応答なし率"),
    ],
    "src/engine_a/other_attacks.py": [
        ('format(b_start, ".0f")', 'format(b_start - 0.001, ".3f")'),
        ('format(b_end, ".0f")', 'format(b_end + 0.001, ".3f")'),
    ],
    "src/engine_b/triage.py": [
        ('format(first_epoch, ".0f")', 'format(first_epoch - 0.001, ".3f")'),
        ('format(last_epoch, ".0f")', 'format(last_epoch + 0.001, ".3f")'),
    ],
}
for path, reps in TARGETS.items():
    if not os.path.exists(path):
        print(path, ": ファイルが見つかりません")
        continue
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    orig = text
    for old, new in reps:
        n = text.count(old)
        text = text.replace(old, new)
        print(path, "|", old, "->", n, "件を変更")
    if text != orig:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
print("完了（0件の行は、適用済みか該当なしです）")
