import pandas as pd
import time

def analyze_zeek_labels(filepath):
    print(f"[*] Zeekログのラベル集計を開始します: {filepath}")
    start_time = time.time()
    
    try:
        # 1. ヘッダのカラム名を取得
        columns = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#fields'):
                    # 先頭の '#fields' を除外して列名のリストを作成
                    columns = line.strip().split('\t')[1:]
                    break
                    
        if not columns:
            print("❌ ヘッダ行 (#fields) が見つかりませんでした。")
            return
            
        print(f"[*] {len(columns)}個のカラムを検出。データを読み込んでいます...")
        
        # 2. Pandasで高速集計 (コメント行 '#' は自動スキップ)
        df = pd.read_csv(filepath, sep='\t', comment='#', names=columns, low_memory=False)
        
        label_col = 'label'
        detailed_col = 'detailed-label' if 'detailed-label' in df.columns else 'detailedlabel'
        
        print(f"\n[+] 集計完了 ({time.time() - start_time:.2f} 秒) - 総通信（フロー）数: {len(df):,} 件")
        
        print("\n" + "="*50)
        print("■ 1. 総合ラベル (label) 集計")
        print("="*50)
        print(df[label_col].value_counts().to_string())
        
        print("\n" + "="*50)
        print("■ 2. 詳細ラベル (detailed-label) 集計")
        print("="*50)
        print(df[detailed_col].value_counts().to_string())
        
        print("\n" + "="*50)
        print("■ 3. ラベル組み合わせ (label | detailedlabel)")
        print("="*50)
        combo = df[label_col].astype(str) + " | " + df[detailed_col].astype(str)
        print(combo.value_counts().to_string())
        print("="*50)
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    target_file = "raw_pcap/2018-07-20-17-31-20-192.168.100.108-zeek-conn-log.labeled"
    analyze_zeek_labels(target_file)