### PCAP Anomaly Detector (NDR Beacon Engine)

ネットワーク検知・対応 (NDR) のためのパケット解析および、機械学習ベースの異常検知（C2ビーコン・通信パターン検知）開発プロジェクトです。

#### 概要

本プロジェクトでは、Mirai C2トラフィック（正例）と、IoT実機（Philips Hue、Amazon Echo等）および通常のWebブラウジングトラフィック（負例）を組み合わせたハイブリッドデータセットを用い、パケット到着間隔 (IAT) やパケット長などの統計的特徴量抽出から異常検知を行うパイプラインを構築しています。

#### 主要スクリプトとファイル構成
- **app_v4.py**: **【エンジンAの完成形（単独版）】** 特徴量抽出から機械学習モデルによる判定ロジックまでが統合されたメイン実行スクリプト。
- **src/**
  - flow_generator.py: パケットをBiflow（双方向フロー）に分割・生成するモジュール。
  - feature_engineering.py: 統計的特徴量（中央値、MAD等）を抽出するモジュール。
- **models/**: 学習済みモデル（LightGBM、CatBoost等）のバイナリ（.joblib）を格納。
- **raw_pcap/**: 実PCAPファイル格納ディレクトリ（※数100MB〜1GB超の巨大ファイルのため、.gitignore によりGitHub管理外としローカル/Google Driveで保持）。

#### 実行方法（エンジンA単独版）

仮想環境を有効化した上で、エンジンAの完成形である app_v4.py を実行します。

```
# 仮想環境の有効化（必要な場合）
source venv/bin/activate
# エンジンA（完成形・単独）の実行
python app_v4.py
```

### 更新履歴 v2.5（エンジンB再構築）

仕様書v2.5準拠のデータ検証（ファイル取り違えの検出、ラベル表記ゆれ対応、近道学習の検証等）を反映し、エンジンBのデータ構築パイプラインを再構築しました。

#### 旧エンジンBファイルの扱い（今後使用しない）

以下は初期検証段階で作成したファイルです。**削除はせず残していますが、今後の開発では使用しません。** 新パイプライン（下記）に一本化します。
- build_dataset_engine_b.py
- src/feature_engineering_engine_b.py

理由:
- src/flow_generator.py（scapy.PcapReaderベース）は、巨大PCAPでメモリ不足・強制終了が発生する既知の問題があるため、新パイプラインでは tshark＋pandasベクトル化処理に切り替えた。
- ラベル付与がファイル名の決め打ちで行われており、過去にファイル取り違え（Somfyのはずが中身はMirai等）が実際に発生したため、Zeekラベルとの時刻照合・対応表方式に変更した。
- Connection_Count の定義（確立セッション総数）が、学習データのC2通信がほぼ未確立（応答なし）である実態と合わず、近道学習の要因になりうるため、「単位時間あたりの接続試行数」に変更した。

src/flow_generator.py 自体は **エンジンAが依存しているため削除しないこと**（app_v4.py の動作に影響する）。

#### 新エンジンBパイプラインの構成
```
pcap-anomaly-detector/
├── app_v4.py                      ← 既存・変更なし（エンジンA単独版）
├── app_v5.py                      ← 【新規】エンジンA×B統合アプリ（下記v3.0参照）
├── src/
│   ├── flow_generator.py          ← 既存・変更なし（エンジンA依存のため削除しない）
│   ├── feature_engineering.py     ← 既存・変更なし（エンジンA用）
│   ├── feature_engineering_engine_b.py  ← 旧エンジンB（今後使用しない）
│   └── engine_b/                  ← エンジンB v2.5 本体
│       ├── __init__.py
│       ├── config.py              # キャプチャ台帳・ラベル対応表・除外ルール（設定はここだけ見ればよい）
│       ├── common.py              # ファイル探索・外部IP判定等の共通関数
│       ├── stage1_extract.py      # 段階1: tsharkでPCAP→Parquet（パケット単位）
│       ├── stage2_biflow.py       # 段階2: パケット→Biflow生成
│       ├── stage3_label.py        # 段階3: Zeekログとの照合・ラベル確定
│       └── stage4_features.py     # 段階4: ブロック化→8特徴量＋生時系列の出力
├── build_dataset_engine_b.py      ← 旧エンジンB（今後使用しない）
├── build_dataset_engine_b_v25.py  ← 新パイプラインの実行スクリプト
├── cleansing_report_engine_b.py   ← データクレンジング段階別件数の報告
├── models/
│   ├── *.joblib                   ← 既存（エンジンA学習済みモデル）
│   └── engine_b_compare/          ← エンジンBモデル比較
│       ├── compare_models.py      # 9モデル比較（A系統6種＋B系統3種）
│       └── dl_models.py           # 1D-CNN / GRU / TCN の定義
├── raw_pcap/
│   ├── heartbeat_philipshue.pcap  ← 既存・変更なし
│   ├── human_web_normal27.pcap    ← 既存・変更なし
│   └── engine_b_iot23/            ← IoT-23 v2（Colabダウンロード分）
│       ├── 01_Train_Positive/
│       │   ├── CTU-IoT-Malware-Capture-7-1/
│       │   └── CTU-IoT-Malware-Capture-48-1/
│       ├── 02_Test_Positive/
│       │   ├── CTU-IoT-Malware-Capture-34-1/
│       │   └── CTU-IoT-Malware-Capture-8-1/
│       ├── 03_Train_Negative/
│       │   ├── CTU-IoT-Malware-Capture-20-1/
│       │   ├── CTU-IoT-Malware-Capture-3-1/
│       │   └── CTU-Honeypot-Capture-4-1/
│       ├── 04_Test_Negative/
│       │   ├── CTU-IoT-Malware-Capture-21-1/
│       │   └── CTU-Honeypot-Capture-5-1/
│       └── data_manifest.csv      # 全ファイルの端末IP・開始日時・列数・MD5台帳
└── requirements.txt
```

#### データ構成（仕様書v2.5 2.2節準拠）

| 区分 | キャプチャ | ファミリ／機器 |
|---|---|---|
| Train 正例 | CTU-IoT-Malware-Capture-7-1, -48-1 | Mirai |
| Train 負例 | 7-1・48-1・20-1・3-1 内の正常通信, CTU-Honeypot-Capture-4-1 | Philips Hue 等 |
| Test 正例 | CTU-IoT-Malware-Capture-34-1（Mirai）, -8-1（**Hakai＝学習していないファミリ**） | 未知ファミリ検知の評価用 |
| Test 負例 | 34-1・8-1・21-1 内の正常通信, CTU-Honeypot-Capture-5-1 | Amazon Echo 等 |
| 参考検証 | 20-1／21-1（Torii） | 約90分周期の低速ビーコン。現設計での検知限界の確認用 |

CTU-Honeypot-Capture-7-1（Somfy）は、中身がMirai感染端末の通信であることが判明したため**使用禁止**。

#### 実行方法（エンジンB v2.5 データ構築）
```
# 仮想環境の有効化
source venv/bin/activate
pip install -r requirements.txt   # tshark は別途 brew install wireshark
# 1. まず小さいキャプチャで動作確認（必須）
python build_dataset_engine_b_v25.py hue-4-1 echo-5-1 3-1 20-1 21-1 8-1 34-1
# 2. データクレンジング報告（段階別件数の確認）
python cleansing_report_engine_b.py
# 3. 問題なければ大きいキャプチャ（7-1, 48-1）も含めて実行
python build_dataset_engine_b_v25.py --all
# 4. モデル比較（9モデル: A系統6種＋B系統3種）
cd models/engine_b_compare
python compare_models.py
```

#### 出力ファイル

| ファイル | 内容 |
|---|---|
| dataset_engine_b_features.csv | 8特徴量＋label＋split＋capture（A系統モデル用） |
| dataset_engine_b_metadata.csv | block_idで結合可能。Src_IP/Dst_IP/conn_state_major等 |
| dataset_engine_b_timeseries.npz | B系統DLモデル用の生時系列（IAT・送信バイト・受信バイト） |
| build_log.csv | キャプチャ別のフロー数・ラベル数・除外理由別件数 |
| cleansing_report.csv | クレンジング過程の段階別件数（①読み込み直後→②Zeek照合→③除外ルール適用→④確定） |
| model_comparison_results.csv | 9モデルの比較結果（Accuracy/F1/PR-AUC/推論時間/モデル容量/近道学習検証） |

#### モデル比較の対象（Transformerは今回対象外）

| 系統 | モデル | 入力 |
|---|---|---|
| A系統（集計特徴量） | Random Forest, XGBoost, LightGBM, CatBoost, MLP, TabNet | 8特徴量 |
| B系統（生の時系列） | 1D-CNN, GRU, TCN | ブロック内20フロー分のIAT・送信バイト・受信バイト（IP・ポート・サービス名は近道学習防止のため不使用） |

Transformerはデータ量に対して時期尚早と判断し、フェーズB・C（CTU-Normal追加後）の発展項目とする。

#### 近道学習の検証（仕様書v2.5 5.3節）

学習用正例（7-1, 48-1）のC2通信は、C2サーバが停止した状態での再接続試行が中心（応答なし）であることが判明している。そのため、model_comparison_results.csv の Recall_34-1_S0（応答なし）と Recall_34-1_S3（接続確立）を分けて評価し、**S3側も検知できて初めて周期性で判定している証拠**とする。S0のみ高くS3が低いモデルは近道学習の疑いがある。

#### 既知の制約・簡略化
- Biflowの終了判定は簡略化規則（FIN/RST直後で区切り、次パケットは無条件に新フロー扱い）であり、LycoSTandのような厳密な状態遷移ではない。
- 時間窓（30分・60分）は未実装。まず20フロー窓で一通り動作確認後に追加する。
- 本パイプラインは合成データでの動作確認のみ実施済み。実データでの初回実行時は、build_log.csv の件数を data_manifest.csv および仕様書付録Aの既知の値と照合すること。

---

### 更新履歴 v3.0（エンジンBモデル選定・エンジンA水平スキャン対応・app_v5統合アプリ）

仕様書v2.5のフェーズA-3（学習と評価）が完了し、エンジンB学習モデルの選定、エンジンAの機能拡張、両エンジンを統合したアプリ（app_v5.py）の実装まで完了した。本節はその最新状態を記録する。

#### エンジンB：9モデル比較と近道学習の検証（完了）

IoT-23 v2（9キャプチャ、約2,616万パケット→4,674ブロック）を用い、A系統（集計8特徴量）4モデル＋MLP・TabNetの6種と、B系統（生の時系列）3モデル（1D-CNN／GRU／TCN）の計9モデルを同一条件で比較した。

- **評価軸**：34-1 S0（応答なしC2）の再現率、34-1 S3（接続確立C2）の再現率、8-1 Hakai（未学習ファミリ）の再現率の3つを分けて評価。S3まで検知できて初めて「応答の有無ではなく周期性で判定している」証拠とした。
- **判定結果**：時系列DL（1D-CNN/GRU/TCN）はAccuracy 99.6〜99.8%だったが、アブレーション検証（入力チャンネルを1つずつ除去）により、Hakai検知は「応答なし」というバイト系特徴に依存した近道学習であったことが判明。IAT（通信間隔）のみでも34-1のS0・S3はともに約1.0を達成でき、周期性のみでの検知が可能なことを確認。
- **決定木系の型分類**：Random Forest・CatBoost・MLPはS3まで検知できる「タイミング型」、LightGBM・TabNetはS3=0でHakai=1.0の「近道型（応答なし＝C2と誤学習）」に分かれた。SHAP監査でLightGBMの近道の正体はByte_Ratio_Median（応答の有無と連動）であると特定し、この特徴量を除くとLightGBMもタイミング型に転換することを確認した。
- **採用モデル**：**CatBoost（モデルB：タイミング系5特徴量＝IAT_Median, IAT_MAD, IAT_CV, Periodic_Peak_Ratio, Connection_Count のみ使用）** を採用。バイト・継続時間系の3特徴量（Burst_Active_Ratio, Payload_Bytes_MAD, Byte_Ratio_Median）を外しても34-1 S3再現率は1.00を維持し、判定根拠がIAT_Median主体（通信間隔そのもの）に入れ替わったことをSHAPで確認。誤検知率（FPR）はモデルAの1.0%からモデルBで2.3%に上昇したが、「C2サーバが応答する環境で応答の有無に頼るリスク」を避けるためモデルBを正式採用した。誤検知増はしきい値調整で対処する方針。
- **未知ファミリ（Hakai）への汎化**：現時点のどのモデル・どの特徴量セットでも、通信間隔のみからのHakai検知はできなかった。原因はモデルではなく、学習用C2がMirai 2本のみというデータの多様性不足と特定（Hakaiは揺らぎゼロという、Miraiとは異なる通信リズムの質を持つ）。

#### エンジンA：水平スキャン検知ルールの追加（完了）

既存の縦方向スキャン検知モデル（CatBoost v4、CIC-IDS2017で学習）は、34-1（Mirai）のPCAPでスキャン送信元を0件と判定した。検証の結果、これは見逃しではなく正しい判定であることを確認（Zeekが「水平スキャン」とラベル付けした5ブロックも、25フロー窓の中では宛先IPが1〜6台に集中しており、スキャンの形をした通信ではなかったため）。

一方、本物の水平スキャンを含む3-1キャプチャで、新たに追加した水平スキャン検知ルール（送信元ごとの宛先IP種類数を用いるルールベース案）を評価したところ、正解3,364ブロックのうち2,339ブロック（約70%）を検知し、誤検知は1件のみだった。既存の縦スキャンモデルは3-1の水平スキャンを0件しか検知できなかったため、ルール追加によりエンジンAの守備範囲を広げられることを確認した。

#### app_v5.py：エンジンA×B統合アプリ（完了）

Streamlit UI上でエンジンA（内部スキャン検知：縦・水平）とエンジンB（外部C2検知）をタブ統合し、PCAPを1回アップロードするだけで両エンジンの解析が走るようにした。

- **統合判定タブ**：両エンジンの結果を端末IPで突き合わせ、A・B両方該当＝最優先、Bのみ該当＝「C2の疑い（CRITICAL）」として優先度付きで提示する相関判定ロジックを実装。
- **エンジンBタブ**：通信ペアを CRITICAL／WARNING／SAFE／FILTERED の4段階にトリアージして表示。
- **調査支援機能（Wiresharkフィルタ自動生成）**：CRITICAL/WARNING判定時に、①ペア全体 ②ペア＋ポート ③ペア＋観測期間 の3段階のWireshark表示フィルタ文字列を自動生成し、コピーして貼るだけで該当通信に絞り込めるようにした（「パケット調査の工数削減」という企画目的に対応）。判定根拠（通信間隔・接続回数等の寄与度）もあわせて表示。
- **デモ検証**：34-1（Mirai）PCAPに対し、エンジンBはC2サーバ185.244.25.235:6667をC2確率1.00でCRITICAL判定、エンジンAはスキャン0件（上記の通り正しい判定）。アプリの出力はZeekの正解ラベルと突き合わせて妥当性を確認済み。

#### 現状のステータス

- データ前処理・モデル学習・モデル選定・SHAP監査・近道学習検証：**完了**（仕様書v2.5 フェーズA-1〜A-3）。
- エンジンA水平スキャン対応、エンジンA×B統合アプリ（app_v5.py）、Wiresharkフィルタ自動生成：**完了**。
- 卒業制作としての位置づけは「公開データによる原理実証（Proof of Concept）」であり、目的は達成。

#### 未着手・次フェーズの課題
1. **ルールベース案①の実装（進行中／未完了）**：ポートスキャン・C2以外にも疑いのある通信（クレンジング時に分離した「その他C2以外の攻撃疑い」パケット）を、断定的でない形で総合判定タブに表示する機能。あわせてWiresharkの表示フィルタ候補（対象IPフィルタ以外の確認推奨項目、具体的な数値付き）を可能な限り列挙する。
2. **未知ファミリへの汎化**：Mirai以外のC2ファミリ（Torii、Cobalt Strike等）の学習データ追加、ファミリ単位の交差検証（Leave-one-family-out）。
3. **正常データの拡充**：CTU-Normal（人間操作による不規則な通信）を追加し、NTP中心の偏りを解消、規則的な正常通信と揺らぎゼロのC2を区別できるか検証。
4. **判定しきい値の調整**：モデルB採用に伴う誤検知率2.3%の低減（PR-AUCは0.99以上のため調整余地あり）。
5. **継続的なSHAP監査**：データ追加のたびに近道学習が再発していないか確認する運用の定着。
6. フェーズB（CTU-Normal追加）・フェーズC（Malware-Traffic-Analysis.net のCobalt Strike PCAP、CIC-IDS2017月曜日の生PCAPによるオフィス通信誤検知評価）は未着手。自社・自宅のPCAPはプライバシー上の理由から成果物の範囲外とし使用しない方針。
