"""
cleansing_report_engine_b.py
エンジンAの「実トラフィックログの整備」スライドと同じ型で、
エンジンBのデータクレンジング過程を段階別に報告するスクリプト。
実行前に build_dataset_engine_b_v25.py を実行しておくこと
（段階2・3の中間ファイルを読み込むため）。
出力:
    raw_pcap/engine_b_iot23/output/cleansing_report.csv
    標準出力に、スライド用の整形済みテキストを表示
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "src", "engine_b"))

import pandas as pd  # noqa: E402

from config import CAPTURES, INTERMEDIATE_DIR, OUTPUT_DIR  # noqa: E402


def stage_counts(name):
    cap = CAPTURES[name]
    bi_path = os.path.join(INTERMEDIATE_DIR, f"biflows_{name}.parquet")
    lb_path = os.path.join(INTERMEDIATE_DIR, f"labeled_{name}.parquet")

    bi = pd.read_parquet(bi_path) if os.path.exists(bi_path) else pd.DataFrame()
    lb = pd.read_parquet(lb_path) if os.path.exists(lb_path) else pd.DataFrame()

    row = dict(capture=name, split=cap["split"])
    row["①_Biflow総数（読み込み直後）"] = len(bi)

    if len(lb):
        row["②_Zeek照合_成功"] = int(lb["matched"].sum())
        row["②_Zeek照合_失敗"] = int((~lb["matched"]).sum())

        reason_counts = lb["exclude_reason"].value_counts(dropna=True)
        for reason in [
            "unmatched", "reference_only", "not_north_south",
            "known_c2_peer", "tcp_no_response",
        ]:
            row[f"③_除外_{reason}"] = int(reason_counts.get(reason, 0))
        
        excluded_label_total = int(
            reason_counts.filter(like="excluded_label:").sum()
            + reason_counts.filter(like="unknown_label:").sum()
        )
        row["③_除外_ラベル対象外（スキャン/DoS等）"] = excluded_label_total
        row["④_確定_Label1（C2）"] = int((lb["label_final"] == 1).sum())
        row["④_確定_Label0（正常）"] = int((lb["label_final"] == 0).sum())
    else:
        for c in [
            "②_Zeek照合_成功", "②_Zeek照合_失敗",
            "③_除外_unmatched", "③_除外_reference_only", "③_除外_not_north_south",
            "③_除外_known_c2_peer", "③_除外_tcp_no_response",
            "③_除外_ラベル対象外（スキャン/DoS等）",
            "④_確定_Label1（C2）", "④_確定_Label0（正常）",
        ]:
            row[c] = None
    return row


def main():
    rows = [stage_counts(name) for name in CAPTURES]
    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "cleansing_report.csv")
    df.to_csv(out_path, index=False)
    
    print("=" * 70)
    print("エンジンB データクレンジング報告（スライド用）")
    print("=" * 70)
    print(df.to_string(index=False))

    total_bi = df["①_Biflow総数（読み込み直後）"].sum()
    total_l1 = df["④_確定_Label1（C2）"].sum()
    total_l0 = df["④_確定_Label0（正常）"].sum()
    total_final = total_l1 + total_l0

    print("\n----- 全体サマリー -----")
    print(f"読み込み直後のBiflow総数 : {total_bi:,.0f}")
    print(f"確定後のLabel=1（C2）総数 : {total_l1:,.0f}")
    print(f"確定後のLabel=0（正常）総数: {total_l0:,.0f}")
    if total_final > 0:
        print(f"確定後の比率（C2:正常）   : {total_l1/total_final*100:.1f}% : {total_l0/total_final*100:.1f}%")
    print(
        "\n[注記] この比率は、エンジンAのCIC-IDS2017のような自然発生比率ではなく、"
        "仕様書v2.5のルールに基づき意図的に調整（North-South絞り込み・除外ルール・"
        "アンダーサンプリング）した結果である。スライドに明記すること。"
    )
    print(f"\n[+] 詳細表を保存しました: {out_path}")


if __name__ == "__main__":
    main()