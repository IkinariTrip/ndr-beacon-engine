"""
engine_b_view.py
インシデンス・コックピット（app_v4.py）にエンジンB（C2ビーコン検知）の結果画面を
組み込むためのStreamlit部品。src/engine_b/ に配置する。

app_v4.py 側での使い方（2行追加するだけ）:
    from engine_b_view import render_engine_b
    render_engine_b(pcap_path)      # pcap_path = アップロード済みPCAPのパス

特徴:
  - 重い処理（tshark〜特徴量計算〜判定）は1回だけ実行し、結果をキャッシュする
  - しきい値スライダーを動かしても再解析せず、判定だけをその場で引き直す
  - 警告ペアごとに Wireshark 用フィルタを表示（コピーしてそのまま使える）
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pandas as pd
import streamlit as st

from inference import analyze_pcap, summarize, DEFAULT_MODEL


@st.cache_data(show_spinner=False)
def _run_analysis(pcap_path, mtime, model_path):
    """PCAPの更新時刻もキーにして、同じファイルなら再解析しない。"""
    summary, blocks, stats = analyze_pcap(pcap_path, model_path=model_path)
    return blocks, stats


def rethreshold(blocks, threshold):
    """保存済みのC2確率から、しきい値だけを変えて判定し直す（再解析なし）。"""
    if len(blocks) == 0:
        return blocks, pd.DataFrame()
    b = blocks.copy()
    b["is_alert"] = b["c2_proba"] >= threshold
    return b, summarize(b)


def render_engine_b(pcap_path, model_path=DEFAULT_MODEL):
    st.subheader("エンジンB：C2ビーコン通信の検知")
    st.caption("通信間隔・頻度のリズムから、C2サーバーへの定期通信の疑いがあるペアを判定します"
               "（採用モデル：CatBoost／タイミング系5特徴量）")

    if not pcap_path or not os.path.exists(pcap_path):
        st.info("PCAPファイルを読み込むと解析を開始します。")
        return

    with st.spinner("解析中…（パケット抽出 → 通信ペア化 → 特徴量計算 → 判定）"):
        try:
            blocks, stats = _run_analysis(pcap_path, os.path.getmtime(pcap_path), model_path)
        except Exception as e:  # tshark未導入・壊れたPCAPなど
            st.error(f"解析に失敗しました：{type(e).__name__}: {e}")
            return

    threshold = st.slider("判定しきい値（C2確率がこの値以上で警告）",
                          0.05, 0.95, float(stats.get("threshold", 0.5)), 0.05,
                          key="engine_b_threshold")
    blocks, summary = rethreshold(blocks, threshold)

    n_alert = 0 if len(summary) == 0 else int(summary["is_alert"].sum())
    c = st.columns(4)
    c[0].metric("パケット数", f"{stats.get('n_packets', 0):,}")
    c[1].metric("社外との通信フロー", f"{stats.get('n_north_south_flows', 0):,}")
    c[2].metric("判定した通信ペア", f"{stats.get('n_pairs', 0):,}")
    c[3].metric("C2の疑いがあるペア", f"{n_alert:,}")

    if len(summary) == 0:
        st.info("判定できる通信ペアがありませんでした（同じ相手との通信が20回以上ないと判定できません）。")
        return

    alerts = summary[summary["is_alert"]]
    if len(alerts):
        st.markdown("#### 🚨 C2ビーコンの疑いがある通信")
        for r in alerts.itertuples():
            with st.expander(f"{r.host_ip} → {r.peer_ip}:{r.peer_port}/{r.proto}　"
                             f"（最大C2確率 {r.max_c2_proba:.2f}、間隔 約{r.iat_median_sec:.1f}秒）",
                             expanded=False):
                st.write(f"警告ブロック {r.n_alert_blocks} / {r.n_blocks}"
                         f"（{r.alert_ratio:.0%}）　期間：{r.first_seen} 〜 {r.last_seen}")
                st.markdown("**Wiresharkの表示フィルタ**（コピーして貼り付け）")
                st.code(r.wireshark_filter, language="text")
                pb = blocks[(blocks.host_ip == r.host_ip) & (blocks.peer_ip == r.peer_ip)
                            & (blocks.peer_port == r.peer_port) & (blocks.proto == r.proto)]
                chart = pb.sort_values("window_index").set_index("window_index")[["c2_proba"]]
                st.line_chart(chart, height=160)
    else:
        st.success("しきい値以上のC2確率を示した通信ペアはありませんでした。")

    with st.expander("すべての通信ペアの判定結果"):
        cols = ["is_alert", "host_ip", "peer_ip", "peer_port", "proto", "n_blocks",
                "n_alert_blocks", "max_c2_proba", "iat_median_sec", "first_seen", "last_seen"]
        st.dataframe(summary[cols].round(3), hide_index=True)
        base = os.path.splitext(os.path.basename(pcap_path))[0]
        st.download_button("結果をCSVで保存", summary.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{base}_engine_b_pairs.csv", mime="text/csv")

    st.caption("注意：本モデルは公開データ（IoT-23のMirai）で学習した原理実証版です。"
               "リズムが大きく異なるC2（例：揺らぎゼロの固定タイマー）は見逃す可能性があり、"
               "警告は「調査の優先順位付け」として使用してください。")
