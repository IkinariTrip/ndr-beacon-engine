"""app_v5.py に (1) DoS注記 と (2) アプリ名「Incidence Cockpit」+アイコン を適用する。"""
import re, sys
path = "app_v5.py"
text = open(path, encoding="utf-8").read()
orig = text
APP_NAME = "Incidence Cockpit"

def find_call(src, start_pat):
    m = re.search(start_pat, src)
    if not m:
        return None
    i, depth, q = m.end(), 1, None
    while i < len(src) and depth:
        c = src[i]
        if q:
            if c == "\\":
                i += 1
            elif c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    return (m.start(), i) if depth == 0 else None

# ---------- (1) 引き下げペアの注記 ----------
if "_dos_label" in text:
    print("[1] DoS注記: すでに適用済み")
else:
    old_def = "def correlate(attack_blocks, pairs_b):"
    old_peer = 'alert_b["peer"] = alert_b["peer_ip"] + ":" + alert_b["peer_port"].astype(str)'
    n_def, n_peer = text.count(old_def), text.count(old_peer)
    print("[1] DoS注記 該当箇所: def =", n_def, "/ peer =", n_peer)
    if n_def == 1 and n_peer == 1:
        helper = (
            "def _dos_label(df):\n"
            '    """エンジンAのフラッドの疑いと一致して引き下げられたペアに、表示用の注記を付ける。"""\n'
            '    if "a_flood_match" in df.columns:\n'
            '        m = df["a_flood_match"].fillna(False).astype(bool)\n'
            '    elif "level_reason" in df.columns:\n'
            '        m = df["level_reason"].astype(str).str.contains("フラッド")\n'
            "    else:\n"
            "        m = pd.Series(False, index=df.index)\n"
            '    return m.map(lambda x: "（DoS疑い・Aと一致）" if x else "")\n'
            "\n\n"
        )
        text = text.replace(old_def, helper + old_def, 1)
        text = text.replace(old_peer, old_peer + '\n        alert_b["peer"] = alert_b["peer"] + _dos_label(alert_b)', 1)
        print("[1] DoS注記: 適用しました")
    else:
        print("[1] DoS注記: 該当箇所が想定と違うため変更していません")

# ---------- (2) アプリ名とアイコン ----------
if "_APP_ICON_PATH" in text:
    print("[2] アプリ名/アイコン: すでに適用済み")
else:
    cfg = find_call(text, r"st\.set_page_config\(")
    ttl = find_call(text, r"st\.title\(")
    n_cfg = len(re.findall(r"st\.set_page_config\(", text))
    n_ttl = len(re.findall(r"st\.title\(", text))
    print("[2] set_page_config =", n_cfg, "件 / st.title =", n_ttl, "件")
    if n_cfg == 1 and n_ttl == 1 and cfg and ttl and cfg[0] < ttl[0]:
        ls = text.rfind("\n", 0, ttl[0]) + 1
        indent = re.match(r"[ \t]*", text[ls:ttl[0]]).group(0)
        if text[ls:ttl[0]].strip():
            print("[2] st.title が行頭にないため変更していません"); sys.exit(1)
        block = (
            f"{indent}_ic1, _ic2 = st.columns([1, 12])\n"
            f"{indent}with _ic1:\n"
            f"{indent}    st.image(_APP_ICON_PATH, width=64)\n"
            f"{indent}with _ic2:\n"
            f"{indent}    st.title(\"{APP_NAME}\")"
        )
        text = text[:ls] + block + text[ttl[1]:]
        call = text[cfg[0]:cfg[1]]
        inner = call[len("st.set_page_config("):-1]
        inner = re.sub(r"page_title\s*=\s*(\"[^\"]*\"|'[^']*')\s*,?\s*", "", inner)
        inner = re.sub(r"page_icon\s*=\s*(\"[^\"]*\"|'[^']*')\s*,?\s*", "", inner)
        inner = inner.strip().rstrip(",")
        args = f'page_title="{APP_NAME}", page_icon=_PILImage.open(_APP_ICON_PATH)'
        if inner:
            args += ", " + inner
        head = (
            "from pathlib import Path as _Path\n"
            "from PIL import Image as _PILImage\n"
            '_APP_ICON_PATH = str(_Path(__file__).resolve().parent / "assets" / "incidence_cockpit_icon.png")\n\n'
        )
        text = text[:cfg[0]] + head + f"st.set_page_config({args})" + text[cfg[1]:]
        print("[2] アプリ名/アイコン: 適用しました")
    else:
        print("[2] st.set_page_config / st.title が1件ずつ見つからないため変更していません")
        print("    → 下の「候補行」を貼り付けてください")

if text != orig:
    open(path, "w", encoding="utf-8").write(text)
