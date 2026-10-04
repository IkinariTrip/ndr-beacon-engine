"""
dl_models.py
B系統（生の時系列）向けの深層学習モデル定義（1D-CNN／GRU／TCN）。

入力形状: (batch, 3, 20)  … チャネル=[IAT, 送信バイト, 受信バイト]、系列長=20（窓サイズ）
出力: 1個のロジット（sigmoidでC2確率に変換）

IPアドレス・ポート番号・サービス名は一切使わない（2026-09-30 合意事項）。

【2026-10-02 修正】
- チャンネルごとの正規化(z-score)を追加。IAT・バイト数は桁が大きく異なり、
  未正規化のままだとニューラルネット系が学習しにくい（木構造モデルとの不公平な
  比較になる）ため、Trainの平均・標準偏差で標準化し、同じ基準をTestにも適用する。
- Mac(macOS)でのマルチプロセスに起因するフリーズ対策として、
  DataLoader相当の処理をすべて手動のミニバッチループに置き換え、
  torchのスレッド数を明示的に1に固定した（num_workers由来の問題を回避）。
- 【追加】XGBoost/LightGBM/CatBoost（libomp同梱）と同一プロセスでPyTorchを
  初期化するとOpenMPランタイムの二重初期化でフリーズすることがある既知の
  問題への対策として、importより前に環境変数を設定する
  （compare_models.py側で既に設定済みの場合も、このファイル単体で
   使われる場合に備えて重ねて設定する。setdefaultなので上書きはしない）。
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    torch.set_num_threads(1)  # Mac環境でのフリーズ対策（BLAS/OpenMPの競合回避）
except ImportError:
    TORCH_AVAILABLE = False
    nn = object  # ダミー（クラス定義を壊さないための仮置き）

import numpy as np


if TORCH_AVAILABLE:

    class CNN1D(nn.Module):
        """単純な1次元畳み込みネットワーク"""

        def __init__(self, in_channels=3, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            # x: (batch, 3, 20)
            h = self.net(x).squeeze(-1)
            return self.fc(h).squeeze(-1)

    class GRUNet(nn.Module):
        """GRUによる時系列モデル（最終隠れ状態を使用）"""

        def __init__(self, in_channels=3, hidden=32):
            super().__init__()
            self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            # x: (batch, 3, 20) -> (batch, 20, 3)
            x = x.transpose(1, 2)
            _, h_n = self.gru(x)
            return self.fc(h_n[-1]).squeeze(-1)

    class _TemporalBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
            self.relu = nn.ReLU()
            self.pad = pad

        def forward(self, x):
            out = self.conv(x)
            if self.pad > 0:
                out = out[:, :, :-self.pad]  # 因果性を保つため、未来側のパディング分を切り落とす
            return self.relu(out)

    class TCN(nn.Module):
        """簡易的な時系列畳み込みネットワーク（膨張畳み込みを3段重ねる）"""

        def __init__(self, in_channels=3, hidden=32):
            super().__init__()
            self.block1 = _TemporalBlock(in_channels, hidden, kernel_size=3, dilation=1)
            self.block2 = _TemporalBlock(hidden, hidden, kernel_size=3, dilation=2)
            self.block3 = _TemporalBlock(hidden, hidden, kernel_size=3, dilation=4)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.block1(x)
            h = self.block2(h)
            h = self.block3(h)
            h = self.pool(h).squeeze(-1)
            return self.fc(h).squeeze(-1)

    MODEL_CLASSES = {"cnn1d": CNN1D, "gru": GRUNet, "tcn": TCN}

else:
    MODEL_CLASSES = {}


def compute_channel_stats(X_train):
    """
    X_train: shape (N, C, T) の numpy配列。
    チャンネルごとの平均・標準偏差を返す（Trainのみから計算し、リークを防ぐ）。
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)   # (1, C, 1)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # 分散ゼロのチャンネルでの0除算を防ぐ
    return mean, std


def apply_channel_norm(X, mean, std):
    return (X - mean) / std


def train_torch_model(model_name, X_train, y_train, X_test, epochs=30, lr=1e-3,
                       device="cpu", batch_size=64, seed=42):
    """
    X_train, X_test: shape (N, 3, 20) の numpy配列（正規化前の生の値でよい。
                     本関数内でTrain統計量に基づく正規化を行う）
    y_train: shape (N,) の0/1 numpy配列
    戻り値: (学習済みモデル, テストデータに対する予測確率 numpy配列, 学習時間, 推論時間)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "torch がインストールされていません。`pip install torch` を実行してから"
            "再実行してください。"
        )
    import time

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- 正規化（Trainの統計量のみを使用。Testにも同じ基準を適用） ----
    mean, std = compute_channel_stats(X_train)
    X_train_n = apply_channel_norm(X_train, mean, std).astype(np.float32)
    X_test_n = apply_channel_norm(X_test, mean, std).astype(np.float32)

    model_cls = MODEL_CLASSES[model_name]
    model = model_cls().to(device)

    Xtr = torch.tensor(X_train_n, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test_n, dtype=torch.float32, device=device)

    n_pos = float(ytr.sum().item())
    n_neg = float(len(ytr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = Xtr.shape[0]
    idx_all = np.arange(n)

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        # Mac環境でのフリーズを避けるため、DataLoader/num_workersは使わず
        # 手動でミニバッチに分割してループする（単一プロセス・単一スレッド）
        np.random.shuffle(idx_all)
        for start in range(0, n, batch_size):
            batch_idx = idx_all[start:start + batch_size]
            batch_idx_t = torch.tensor(batch_idx, dtype=torch.long, device=device)
            xb = Xtr.index_select(0, batch_idx_t)
            yb = ytr.index_select(0, batch_idx_t)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    train_time = time.perf_counter() - t0

    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte)).cpu().numpy()
    infer_time_total = time.perf_counter() - t0

    return model, probs, train_time, infer_time_total
