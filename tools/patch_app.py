import re
path = "app_v5.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()
orig = text

# ---------- (1) None → ― ----------
if "_dash_table" in text:
    print("[1] 統合表の表示: すでに適用済み")
else:
    tail_old = '"B_c2_peers": "B：C2の疑いがある相手"}).round(3), hide_index=True)'
    anchor = "uploaded_file = st.file_uploader("
    n_tail, n_anchor = text.count(tail_old), text.count(anchor)
    print(f"[1] 該当箇所: dataframe = {n_tail}件 / uploader = {n_anchor}件")
    if n_tail == 1 and n_anchor == 1:
        helper = (
            "def _dash_table(df):\n"
            '    """表示用：欠損値（None/NaN）を「―」にし、数値は見やすく整える。"""\n'
            "    import pandas as pd\n"
            "    out = df.copy()\n"
            "    for c in out.columns:\n"
            "        s = out[c]\n"
            "        if pd.api.types.is_numeric_dtype(s):\n"
            '            out[c] = s.map(lambda v: "―" if pd.isna(v) else (str(int(v)) if float(v).is_integer() else format(v, ".3f")))\n'
            "        else:\n"
            '            out[c] = s.map(lambda v: "―" if (pd.isna(v) or str(v).strip() == "" or str(v) == "None") else str(v))\n'
            "    return out\n\n"
        )
        text = text.replace(anchor, helper + anchor, 1)
        text = text.replace(tail_old, '"B_c2_peers": "B：C2の疑いがある相手"}).pipe(_dash_table), hide_index=True)', 1)
        print("[1] 統合表の表示: 適用しました")
    else:
        print("[1] 該当箇所が想定と違うため変更していません")

# ---------- (2) サイドバーの説明 ----------
if "社内外の通信" in text:
    print("[2] サイドバーの説明: すでに適用済み")
else:
    pat = re.compile(r'^([ \t]+)st\.caption\(f"リフレクション：.*?\n', re.M)
    found = pat.findall(text)
    print("[2] 該当箇所:", len(found), "件")
    if len(found) == 1:
        def repl(m):
            ind = m.group(1)
            return ind + 'st.caption(f"リフレクション：増幅率≥{REFLECTION_RULE.get(\'min_amp_ratio\', 10):.0f}倍、{REFLECTION_RULE.get(\'min_flow_rate\', 1):.0f}フロー/秒以上、社内外の通信（DNS/NTP/SSDP等の既知ポート）")\n'
        text = pat.sub(repl, text, count=1)
        print("[2] サイドバーの説明: 適用しました")
    else:
        print("[2] 該当箇所が想定と違うため変更していません")

if text != orig:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
