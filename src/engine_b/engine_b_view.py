"""
engine_b_view.py（v3：統合アプリ対応）
エンジンB（C2ビーコン検知）の結果画面。src/engine_b/ に配置する。

v2からの変更点:
  - 解析と表示を分離した。app_v5.py は「解析ボタン押下時に1回だけ analyze_pcap を実行 →
    結果を st.session_state に保存 → render_engine_b_results で表示」という流れで使う。
    （v2の render_engine_b はPCAPのパスを受け取って内部で解析していたが、
     アプリ側では解析後に一時ファイルを削除するため、パス経由の再解析ができない）
  - render_engine_b(pcap_path) は単体利用のために残している。
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import streamlit as st

from inference import analyze_pcap, load_model, DEFAULT_MODEL
from triage import apply_triage, top_reasons, CRITICAL_TH, SAFE_TH

LEVEL_ICON = {"CRITICAL": "🔴", "WARNING": "🟠", "SAFE": "🟢", "FILTERED": "⚪"}
PAIR_KEYS = ["host_ip", "peer_ip", "peer_port", "proto"]


@st.cache_resource(show_spinner=False)
def _model(model_path):
    return load_model(model_path)


def triage_pairs(summary, stats):
    """ペア単位の判定結果に4段階トリアージを付ける（アプリの相関判定でも使う）。"""
    return apply_triage(summary, stats.get("n_filtered_pairs", 0))


def render_engine_b_results(summary, blocks, stats, model_path=DEFAULT_MODEL, pcap_name="pcap"):
    st.caption(f"通信間隔・接続回数のリズムから、C2サーバーへの定期通信の疑いを判定します"
               f"（CatBoost・タイミング系5特徴量／CRITICAL ≥{CRITICAL_TH}、SAFE <{SAFE_TH}）")
    pairs, counts = triage_pairs(summary, stats)

    c = st.columns(4)
    for col, lv in zip(c, ["CRITICAL", "WARNING", "SAFE", "FILTERED"]):
        col.metric(f"{LEVEL_ICON[lv]} {lv}", f"{counts[lv]:,} ペア")
    st.caption(f"社外との通信フロー {stats.get('n_north_south_flows', 0):,}。"
               "FILTERED＝同じ相手との通信が10回未満で判定できないペア")

    alerts = pairs[pairs["level"].isin(["CRITICAL", "WARNING"])] if len(pairs) else pairs
    if len(alerts):
        reason_map = {}
        try:
            model, meta = _model(model_path)
            for key, g in blocks.groupby(PAIR_KEYS):
                reason_map[key] = top_reasons(g.loc[[g["c2_proba"].idxmax()]], model, meta)[0]
        except Exception as e:
            st.caption(f"（判定根拠の計算をスキップしました：{type(e).__name__}）")

        st.markdown("#### 要対応の通信ペア（優先度順）")
        for r in alerts.itertuples():
            ioc = "　🏷️ Suspect Port" if r.suspect_port else ""
            with st.expander(f"{LEVEL_ICON[str(r.level)]} {r.level}｜{r.host_ip} → {r.peer_ip}:{r.peer_port}/{r.proto}"
                             f"　C2確率 {r.max_c2_proba:.2f}、間隔 約{r.iat_median_sec:.1f}秒{ioc}"):
                st.write(f"区分の理由：{r.level_reason}　／　警告ブロック {r.n_alert_blocks} / {r.n_blocks}"
                         f"　／　期間：{r.first_seen} 〜 {r.last_seen}")
                key = (r.host_ip, r.peer_ip, r.peer_port, r.proto)
                if key in reason_map:
                    st.write(f"判定根拠（C2寄りに効いた特徴量）：{reason_map[key]}")
                st.markdown("**Wireshark表示フィルタ**（用途に応じてコピー）")
                for name, f in r.filters.items():
                    st.caption(name)
                    st.code(f, language="text")
    else:
        st.success("CRITICAL／WARNINGに該当する通信ペアはありませんでした。")

    if len(pairs):
        with st.expander("すべての通信ペアの判定結果"):
            cols = ["level", "host_ip", "peer_ip", "peer_port", "proto", "max_c2_proba", "n_alert_blocks",
                    "n_blocks", "iat_median_sec", "suspect_port", "first_seen", "last_seen"]
            show = pairs[[c for c in cols if c in pairs.columns]].copy()
            show["level"] = show["level"].astype(str)
            st.dataframe(show.round(3), hide_index=True)
            st.download_button("結果をCSVで保存", show.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{pcap_name}_engine_b_triage.csv", mime="text/csv",
                               key="engine_b_csv")

    st.caption("注意：本モデルは公開データ（IoT-23のMirai）で学習した原理実証版です。"
               "リズムの異なるC2は見逃す可能性があり、間隔の短い正常通信（NTP・DNS等）を"
               "WARNING／CRITICALとすることがあります。区分は調査の優先順位として使用してください。")
    return pairs


def render_engine_b(pcap_path, model_path=DEFAULT_MODEL):
    """単体利用向け（PCAPのパスを渡すと解析から表示まで行う）。"""
    st.subheader("エンジンB：外部C2ビーコン通信の検知")
    if not pcap_path or not os.path.exists(pcap_path):
        st.info("PCAPファイルを読み込むと解析を開始します。")
        return
    with st.spinner("解析中…"):
        summary, blocks, stats = analyze_pcap(pcap_path, model_path=model_path)
    render_engine_b_results(summary, blocks, stats, model_path,
                            os.path.splitext(os.path.basename(pcap_path))[0])
