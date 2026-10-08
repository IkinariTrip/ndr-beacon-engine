import re
path = "app_v5.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()
orig = text

# [1] アプリ名の変更
n = len(re.findall(r"Incidence Cockpit(?! 2606)", text))
text = re.sub(r"Incidence Cockpit(?! 2606)", "Incidence Cockpit 2606", text)
print("[1] アプリ名:", n, "か所を「Incidence Cockpit 2606」に変更")

# CSVボタンの関数追加
HELPER = (
    "def csv_download(label, df, filename, key):\n"
    '    """CSV保存ボタン。大きな表でも再描画のたびに重くならないよう、「作成」を押したときだけCSVを作る。"""\n'
    "    if df is None or len(df) == 0:\n"
    "        return\n"
    '    state_key = "csv" + key\n'
    '    if st.button("📥 " + label + "のCSVを作成", key="mk_" + key):\n'
    '        st.session_state[state_key] = df.to_csv(index=False).encode("utf-8-sig")\n'
    "    if state_key in st.session_state:\n"
    '        st.download_button("⬇ " + label + "をダウンロード", st.session_state[state_key],\n'
    '                           file_name=filename, mime="text/csv", key="dl_" + key)\n'
    "\n\n"
)

# HELPERを一番上に挿入
if "def csv_download(" not in text:
    anchor = "uploaded_file = st.file_uploader("
    if text.count(anchor) == 1:
        text = text.replace(anchor, HELPER + anchor, 1)
        print("[2] CSVボタンの部品を追加しました")

# セッションステート初期化追加
anchor2 = 'st.session_state["result"] = result'
if anchor2 in text and 'st.session_state.keys() if str(k).startswith("csv")' not in text:
    lines = [
        '        for k in [k for k in st.session_state.keys() if str(k).startswith("csv")]:\n',
        '            del st.session_state[k]\n'
    ]
    text = text.replace(anchor2, anchor2 + "\n" + "".join(lines), 1)
    print("[2a] 新しい解析でCSVの作成済みデータを消す処理を追加しました")

# 統合判定のCSVボタン
anchor3 = 'st.dataframe(show.rename('
if anchor3 in text and '"integrated"' not in text:
    button3 = '    csv_download("統合判定の表", show, "integrated_" + res["name"] + ".csv", "integrated")\n'
    text = text.replace(anchor3, button3 + anchor3, 1)
    print("[2b] 統合判定のCSVボタンを追加しました")

# エンジンAのCSVボタン
anchor4 = 'col5.metric("その他攻撃の疑い"'
if anchor4 in text and '"a_hits"' not in text:
    button4 = '    csv_download("エンジンAの検知ブロック", attack_blocks, "engineA_hits_" + res["name"] + ".csv", "a_hits")\n'
    text = text.replace(anchor4, button4 + anchor4, 1)
    print("[2c] エンジンAのCSVボタンを追加しました")

# エンジンBのCSVボタン
anchor5 = 'render_engine_b_results(summary_b, blocks_b, stats_b, ENGINE_B_MODEL, res["name"])'
if anchor5 in text and '"b_pairs"' not in text:
    button5 = '    csv_download("エンジンBの通信ペア", pairs_b.drop(columns=[c for c in ("filters",) if c in pairs_b.columns]), "engineB_pairs_" + res["name"] + ".csv", "b_pairs")\n'
    text = text.replace(anchor5, button5 + anchor5, 1)
    print("[2d] エンジンBのCSVボタンを追加しました")

if text != orig:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
