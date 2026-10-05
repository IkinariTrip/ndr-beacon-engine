"""
engine_b_view.py（v5.2：C2フィルタに周期性・応答有無チェックを追加）
エンジンB（C2ビーコン検知）の結果画面。src/engine_b/ に配置する。
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import streamlit as st

from inference import analyze_pcap, load_model, DEFAULT_MODEL
from triage import apply_triage, top_reasons, enrich_c2_filters, CRITICAL_TH, SAFE_TH

LEVEL_ICON = {"CRITICAL": "🔴", "WARNING": "🟠", "SAFE": "🟢", "FILTERED": "⚪"}
PAIR_KEYS = ["host_ip", "peer_ip", "peer_port", "proto"]


@st.cache_resource(show_spinner=False)
def _model(model_path):
    return load_model(model_path)


def triage_pairs(summary, stats):
    return apply_triage(summary, stats.get("n_filtered_pairs", 0))


def render_engine_b_results(summary, blocks, stats, model_path=DEFAULT_MODEL, pcap_name="pcap"):
    st.caption(f"通信間隔・接続回数のリズムから、C2サーバーへの定期通信の疑いを判定します"
               f"（CatBoost・タイミング5特徴量／CRITICAL ≥{CRITICAL_TH}、SAFE <{SAFE_TH}）")
    pairs, counts = triage_pairs(summary, stats)

    c = st.columns(4)
    for col, lv in zip(c, ["CRITICAL", "WARNING", "SAFE", "FILTERED"]):
        col.metric(f"{LEVEL_ICON[lv]} {lv}", f"{counts[lv]:,} ペア")
    st.caption(f"社外との通信フロー {stats.get('n_north_south_flows', 0):,}。"
               "FILTERED＝同じ相手との通信が10回未満で判定できないペア")

    alerts = pairs[pairs["level"].isin(["CRITICAL", "WARNING"])] if len(pairs) else pairs
    if len(alerts):
        reason_map, repr_block = {}, {}
        # 代表ブロック（フィルタの具体値に使う）はモデル読込とは独立に、必ず取得する
        try:
            for key, g in blocks.groupby(PAIR_KEYS):
                repr_block[key] = g.loc[g["c2_proba"].idxmax()]
        except Exception:
            pass
        # 判定根拠（SHAP）はCatBoostが必要なため、失敗しても上の代表ブロック取得には影響させない
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
                filters = r.filters
                if key in repr_block:
                    rb = repr_block[key]
                    filters = enrich_c2_filters(
                        filters, r.host_ip, r.peer_ip, r.peer_port, r.proto,
                        iat_median=rb.get("IAT_Median"), iat_mad=rb.get("IAT_MAD"),
                        byte_ratio_median=rb.get("Byte_Ratio_Median"))
                st.markdown("**Wireshark表示フィルタ**（用途に応じてコピー。④は②適用後に追記）")
                for name, f in filters.items():
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


def render_engine_b(pcap_path, model_path=DEFAULT_MODEL):
    st.subheader("エンジンB：外部C2ビーコン通信の検知")
    if not pcap_path or not os.path.exists(pcap_path):
        st.info("PCAPファイルを読み込むと解析を開始します。")
        return
    with st.spinner("解析中…"):
        summary, blocks, stats = analyze_pcap(pcap_path, model_path=model_path)
    render_engine_b_results(summary, blocks, stats, model_path,
                            os.path.splitext(os.path.basename(pcap_path))[0])
