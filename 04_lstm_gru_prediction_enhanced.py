# 04_lstm_gru_prediction_enhanced.py
# 功能：增强版LSTM/GRU股票预测模型（集成stock_prediction-master核心能力）
# 改进点：
# 1. 增强特征工程：对数收益率、差分序列、滑动窗口统计、更多技术指标
# 2. 改进数据预处理：防止数据泄漏、归一化参数保存、更好的NaN处理
# 3. 改进训练策略：EarlyStopping、学习率调度、类别权重、梯度裁剪
# 4. 增强模型架构：Attention机制、双向LSTM选项
# 5. 改进评估指标：balanced accuracy、macro-F1、更详细的报告
#
# 依赖：
# pip install pandas numpy torch scikit-learn matplotlib seaborn tushare python-dotenv

import argparse
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# NumPy兼容性检查
try:
    import numpy as np
    numpy_version = np.__version__
    major_version = int(numpy_version.split('.')[0])
    if major_version >= 2:
        print("⚠️  警告: 检测到 NumPy 2.x 版本，可能与 PyTorch 不兼容")
except ImportError:
    print("❌ 错误: 未安装 NumPy")
    sys.exit(1)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score
)
from dotenv import load_dotenv
from datetime import datetime, timedelta

from tushare_client import get_pro

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "lstm_models")
PLOT_DIR = os.path.join(SCRIPT_DIR, "lstm_plots")
NORM_PARAMS_DIR = os.path.join(SCRIPT_DIR, "norm_params")
MARKET_ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "market_analysis")
BATCH_ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "batch_predictions")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(NORM_PARAMS_DIR, exist_ok=True)
os.makedirs(BATCH_ANALYSIS_DIR, exist_ok=True)

# 模型参数（增强版）
SEQ_LENGTH = 60  # 增加序列长度，使用过去60天的数据
HIDDEN_SIZE = 128  # 增加隐藏层大小
NUM_LAYERS = 3  # 增加层数
BATCH_SIZE = 32
LEARNING_RATE = 5e-4  # 降低学习率
EPOCHS = 150  # 增加训练轮数
TRAIN_RATIO = 0.8

# 训练策略参数
EARLY_STOPPING_PATIENCE = 30  # 早停耐心值
EARLY_STOPPING_MIN_DELTA = 1e-4  # 最小改进阈值
EARLY_STOPPING_WARMUP = 50  # 前50轮不允许早停
USE_CLASS_WEIGHTS = True  # 使用类别权重
GRADIENT_CLIP = 1.0  # 梯度裁剪阈值
WEIGHT_DECAY = 1e-4  # 权重衰减


def get_next_trading_day(pro, last_trade_date, exchange="SSE", lookforward_days=10) -> str:
    """获取下一个交易日"""
    try:
        if isinstance(last_trade_date, pd.Timestamp):
            last_date_str = last_trade_date.strftime('%Y%m%d')
        else:
            last_date_str = str(last_trade_date).replace('-', '')
        
        end_date = str(int(last_date_str) + lookforward_days * 100)
        cal = pro.trade_cal(exchange=exchange, start_date=last_date_str, end_date=end_date,
                            fields="cal_date,is_open")
        if cal is None or cal.empty:
            return None
        
        cal = cal.sort_values("cal_date")
        open_days = cal.loc[cal["is_open"] == 1, "cal_date"]
        next_days = open_days[open_days > last_date_str]
        if not next_days.empty:
            return next_days.iloc[0]
        return None
    except Exception as e:
        print(f"获取下一个交易日失败: {e}")
        return None


def safe_fillna(df: pd.DataFrame) -> pd.DataFrame:
    """安全填充NaN和inf值"""
    df = df.replace([np.inf, -np.inf], np.nan)
    try:
        df = df.bfill().ffill()
    except Exception:
        df = df.fillna(method='bfill').fillna(method='ffill')
    return df.fillna(0)


def fetch_stock_history_extended(pro, ts_code: str, days=1825) -> pd.DataFrame:
    """
    获取个股历史数据（扩展版，获取5年数据）
    参考stock_prediction-master的fetch_stock_history_extended
    """
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    try:
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount"
        )
        if df is None or df.empty:
            return pd.DataFrame()

        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.sort_values("trade_date").reset_index(drop=True)

        # 基础指标
        df["pct_chg"] = (df["close"] - df["pre_close"]) / (df["pre_close"] + 1e-8) * 100
        df["amplitude"] = (df["high"] - df["low"]) / (df["pre_close"] + 1e-8) * 100

        # 移动平均线（多周期）
        for w in [5, 10, 20, 30, 60]:
            df[f"ma{w}"] = df["close"].rolling(w).mean()

        # 成交量移动平均
        df["vol_ma5"] = df["vol"].rolling(5).mean()
        df["vol_ma10"] = df["vol"].rolling(10).mean()
        df["vol_ratio"] = df["vol"] / (df["vol_ma5"] + 1e-8)

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 布林带
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-8)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / ((df["bb_upper"] - df["bb_lower"]) + 1e-8)

        # 多周期收益率（参考stock_prediction-master）
        for p in [1, 3, 5, 10, 20]:
            df[f"return_{p}d"] = df["close"].pct_change(p) * 100

        # 对数收益率（参考stock_prediction-master的log_return）
        df["log_return_1d"] = np.log(df["close"] / (df["close"].shift(1) + 1e-8))
        df["log_return_5d"] = np.log(df["close"] / (df["close"].shift(5) + 1e-8))

        # 差分序列（参考stock_prediction-master的difference）
        df["close_diff_1"] = df["close"].diff(1)
        df["pct_chg_diff_1"] = df["pct_chg"].diff(1)

        # 滑动窗口统计（参考stock_prediction-master的sliding_windows）
        for w in [5, 20]:
            df[f"pct_chg_win{w}_mean"] = df["pct_chg"].rolling(w).mean()
            df[f"pct_chg_win{w}_std"] = df["pct_chg"].rolling(w).std()
            df[f"vol_win{w}_mean"] = df["vol"].rolling(w).mean()
            df[f"vol_win{w}_std"] = df["vol"].rolling(w).std()

        # KDJ
        low_9 = df["low"].rolling(9).min()
        high_9 = df["high"].rolling(9).max()
        rsv = (df["close"] - low_9) / ((high_9 - low_9) + 1e-8) * 100
        df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

        # CCI
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
        df["cci"] = (tp - sma_tp) / (0.015 * (mad + 1e-8))

        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr_atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr_atr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / (df["close"] + 1e-8) * 100

        # OBV
        df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()
        df["obv_ma"] = df["obv"].rolling(20).mean()

        df = safe_fillna(df)
        print(f"✅ 成功获取 {len(df)} 条历史数据")
        return df
    except Exception as e:
        print(f"获取历史数据失败: {e}")
        return pd.DataFrame()


def create_enhanced_features(df_raw: pd.DataFrame, df_scaled: pd.DataFrame):
    """
    创建增强特征和标签（参考stock_prediction-master的create_improved_features）
    使用三分类：上涨/下跌/震荡
    """
    feature_cols = [
        "open", "high", "low", "close", "vol", "amount",
        "pct_chg", "amplitude",
        "ma5", "ma10", "ma20", "ma30", "ma60",
        "vol_ma5", "vol_ma10", "vol_ratio",
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
        "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
        "log_return_1d", "log_return_5d",
        "close_diff_1", "pct_chg_diff_1",
        "pct_chg_win5_mean", "pct_chg_win5_std", "pct_chg_win20_mean", "pct_chg_win20_std",
        "vol_win5_mean", "vol_win5_std", "vol_win20_mean", "vol_win20_std",
        "kdj_k", "kdj_d", "kdj_j",
        "cci", "atr", "atr_pct",
        "obv", "obv_ma"
    ]
    available_cols = [c for c in feature_cols if c in df_scaled.columns]
    if len(available_cols) < 10:
        available_cols = ["open", "high", "low", "close", "vol", "pct_chg", "ma5", "ma10", "ma20", "rsi"]

    features = df_scaled[available_cols].values

    # 三分类标签（参考stock_prediction-master的阈值设计）
    THRESHOLD_UP = 0.8  # 上涨阈值
    THRESHOLD_DOWN = -0.8  # 下跌阈值
    labels = []
    for i in range(len(df_raw) - 1):
        next_pct = float(df_raw.iloc[i+1].get("pct_chg", 0.0))
        if next_pct > THRESHOLD_UP:
            labels.append(1)  # 上涨
        elif next_pct < THRESHOLD_DOWN:
            labels.append(0)  # 下跌
        else:
            labels.append(2)  # 震荡
    labels.append(2)  # 最后一天

    X, y = [], []
    for i in range(SEQ_LENGTH, len(features)):
        X.append(features[i-SEQ_LENGTH:i])
        y.append(labels[i])

    y = np.array(y)
    counts = {0: int((y==0).sum()), 1: int((y==1).sum()), 2: int((y==2).sum())}
    print(f"  标签分布 - 下跌:{counts[0]} 上涨:{counts[1]} 震荡:{counts[2]} (阈值: {THRESHOLD_DOWN}%~{THRESHOLD_UP}%)")

    return np.array(X), y, available_cols


class StockDataset(Dataset):
    """股票数据集"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EnhancedLSTMModel(nn.Module):
    """
    增强版LSTM模型（参考stock_prediction-master的模型设计）
    - 双向LSTM
    - 多头注意力机制
    - 更深的网络结构
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes=3, use_attention=True, bidirectional=True):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 多头注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_size,
                num_heads=4,
                batch_first=True
            )
        
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            last_output = attn_out[:, -1, :]
        else:
            last_output = lstm_out[:, -1, :]
        
        # 全连接层
        last_output = self.dropout1(torch.relu(self.fc1(last_output)))
        output = self.fc2(last_output)
        return output


class EnhancedGRUModel(nn.Module):
    """增强版GRU模型（类似LSTM的增强）"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes=3, use_attention=True, bidirectional=True):
        super(EnhancedGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=gru_output_size,
                num_heads=4,
                batch_first=True
            )
        
        self.fc1 = nn.Linear(gru_output_size, hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        
        if self.use_attention:
            attn_out, _ = self.attention(gru_out, gru_out, gru_out)
            last_output = attn_out[:, -1, :]
        else:
            last_output = gru_out[:, -1, :]
        
        last_output = self.dropout1(torch.relu(self.fc1(last_output)))
        output = self.fc2(last_output)
        return output

class GRULSTMModel(nn.Module):
    """混合版GRU+LSTM模型
    - LSTM 与 GRU 并行编码，序列输出按权重线性融合（默认 70% LSTM + 30% GRU，可用 lstm_weight/gru_weight 调整）
    - 可选多头自注意力
    - 全连接层
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes=3,
        use_attention=True,
        bidirectional=True,
        lstm_weight=0.7,
        gru_weight=0.3,
    ):
        super(GRULSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        dropout = 0.3 if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        embed_dim = hidden_size * 2 if bidirectional else hidden_size
        num_heads = 4
        if embed_dim % num_heads != 0:
            for nh in (2, 1):
                if embed_dim % nh == 0:
                    num_heads = nh
                    break

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
            )
        else:
            self.attention = None

        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        s = float(lstm_weight) + float(gru_weight)
        self.register_buffer("w_lstm", torch.tensor(float(lstm_weight) / s))
        self.register_buffer("w_gru", torch.tensor(float(gru_weight) / s))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        mixed = self.w_lstm * lstm_out + self.w_gru * gru_out

        if self.use_attention:
            attn_out, _ = self.attention(mixed, mixed, mixed)
            last_output = attn_out[:, -1, :]
        else:
            last_output = mixed[:, -1, :]

        last_output = self.dropout1(torch.relu(self.fc1(last_output)))
        output = self.fc2(last_output)
        return output


def normalize_model_type(model_type):
    """将用户输入规范为 lstm / gru / grulstm（其余回退为 lstm）。"""
    if model_type is None:
        return "lstm"
    mt = str(model_type).strip().lower()
    return mt if mt in ("lstm", "gru", "grulstm") else "lstm"


def create_sequence_predictor_model(model_type, input_size, hidden_size, num_layers, num_classes=3):
    """
    按类型创建序列预测模型（与命令行 --model-type、交互输入一致）。
    - lstm: EnhancedLSTMModel
    - gru: EnhancedGRUModel
    - grulstm: GRULSTMModel（并行 LSTM+GRU 融合 + 可选注意力）
    """
    mt = normalize_model_type(model_type)
    if mt == "lstm":
        return EnhancedLSTMModel(input_size, hidden_size, num_layers, num_classes)
    if mt == "gru":
        return EnhancedGRUModel(input_size, hidden_size, num_layers, num_classes)
    return GRULSTMModel(input_size, hidden_size, num_layers, num_classes)


def model_type_display_name(model_type: str) -> str:
    mt = normalize_model_type(model_type)
    return {"lstm": "增强版 LSTM（双向+注意力）", "gru": "增强版 GRU（双向+注意力）", "grulstm": "LSTM+GRU 混合（并行融合+注意力）"}[mt]


def make_class_weights(y, num_classes=3):
    """计算类别权重（参考stock_prediction-master）"""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class EarlyStopping:
    """早停机制（参考stock_prediction-master的EarlyStopping）"""
    def __init__(self, patience=30, min_delta=1e-4, warmup=50, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.epoch_count = 0
    
    def step(self, score):
        self.epoch_count += 1
        
        # Warmup期间不早停
        if self.epoch_count <= self.warmup:
            return False
        
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


def train_enhanced_model(model, train_loader, val_loader, device, epochs=EPOCHS, num_classes=3):
    """
    增强版训练函数（参考stock_prediction-master的train_improved_model）
    - 类别权重
    - 早停机制
    - 学习率调度
    - 梯度裁剪
    """
    # 计算类别权重
    y_train_all = []
    for _, yb in train_loader:
        y_train_all.append(yb.numpy())
    y_train_all = np.concatenate(y_train_all)
    class_weights = make_class_weights(y_train_all, num_classes=num_classes).to(device)
    print(f"类别权重: {class_weights.detach().cpu().numpy()}")

    # 损失函数（带类别权重和标签平滑）
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        warmup=EARLY_STOPPING_WARMUP,
        mode='max'
    )

    best_state = None
    best_macro_f1 = -1.0
    best_val_acc = 0.0
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_macro_f1s = []
    val_balanced_accs = []

    print(f"开始训练，最多 {epochs} 轮；warmup={EARLY_STOPPING_WARMUP}；patience={EARLY_STOPPING_PATIENCE}")

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            tr_loss += float(loss.item())

        tr_loss /= max(1, len(train_loader))

        # 验证阶段
        model.eval()
        va_loss = 0.0
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += float(loss.item())
                pred = torch.argmax(logits, dim=1)
                all_pred.append(pred.cpu().numpy())
                all_true.append(yb.cpu().numpy())

        va_loss /= max(1, len(val_loader))
        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)

        va_acc = accuracy_score(all_true, all_pred)
        va_macro_f1 = f1_score(all_true, all_pred, average='macro')
        va_bal_acc = balanced_accuracy_score(all_true, all_pred)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        val_accuracies.append(va_acc)
        val_macro_f1s.append(va_macro_f1)
        val_balanced_accs.append(va_bal_acc)

        # 早停判断（使用macro-F1）
        improved = va_macro_f1 > best_macro_f1 + EARLY_STOPPING_MIN_DELTA
        if improved:
            best_macro_f1 = va_macro_f1
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step(va_macro_f1)

        if epoch % 10 == 0 or epoch == 1:
            pred_counts = np.bincount(all_pred, minlength=num_classes)
            pred_dist = (pred_counts / pred_counts.sum()).round(3)
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"TrainLoss {tr_loss:.4f} | "
                f"ValLoss {va_loss:.4f} ValAcc {va_acc:.4f} BalAcc {va_bal_acc:.4f} MacroF1 {va_macro_f1:.4f} | "
                f"PredDist(d/u/s) {pred_dist} "
                f"{'✓' if improved else ''}"
            )

        # 早停检查
        if early_stopping.step(va_macro_f1):
            print(f"早停触发：epoch={epoch}，macro-F1连续{EARLY_STOPPING_PATIENCE}轮未改善")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"已加载最佳模型：ValAcc={best_val_acc:.4f} ValLoss={best_val_loss:.4f} MacroF1={best_macro_f1:.4f}")
    else:
        print("⚠️ 未找到best_state，将使用最后一轮模型")

    # 最终验证报告
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    print("\n" + "="*60)
    print("验证集详细评估（最终best模型）")
    print("="*60)
    print(f"Accuracy: {accuracy_score(all_true, all_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(all_true, all_pred):.4f}")
    print(f"Macro F1: {f1_score(all_true, all_pred, average='macro'):.4f}")
    print("\n分类报告:")
    print(classification_report(all_true, all_pred, target_names=["下跌","上涨","震荡"]))

    return model, train_losses, val_losses, val_accuracies, val_macro_f1s, val_balanced_accs


def plot_training_history_enhanced(train_losses, val_losses, val_accuracies, val_macro_f1s, val_balanced_accs, stock_code, model_type):
    """绘制增强的训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 损失曲线
    axes[0, 0].plot(train_losses, label='训练损失', linewidth=2)
    axes[0, 0].plot(val_losses, label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{stock_code} - {model_type} 训练损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(val_accuracies, label='验证准确率', linewidth=2, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title(f'{stock_code} - {model_type} 验证准确率曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Macro-F1曲线
    axes[1, 0].plot(val_macro_f1s, label='Macro F1', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Macro F1')
    axes[1, 0].set_title(f'{stock_code} - {model_type} Macro F1曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Balanced Accuracy曲线
    axes[1, 1].plot(val_balanced_accs, label='Balanced Accuracy', linewidth=2, color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Balanced Accuracy')
    axes[1, 1].set_title(f'{stock_code} - {model_type} Balanced Accuracy曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f'{stock_code}_{model_type}_enhanced_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练历史图已保存: {plot_path}")
    plt.close()


def save_norm_params(scaler, feature_cols, stock_code, model_type):
    """保存归一化参数（参考stock_prediction-master）"""
    norm_params = {
        'mean_list': scaler.data_min_.tolist(),
        'std_list': (scaler.data_max_ - scaler.data_min_).tolist(),
        'feature_cols': feature_cols,
        'stock_code': stock_code,
        'model_type': model_type
    }
    norm_path = os.path.join(NORM_PARAMS_DIR, f'{stock_code}_{model_type}_norm_params.json')
    with open(norm_path, 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, ensure_ascii=False, indent=2)
    print(f"✅ 归一化参数已保存: {norm_path}")


def extract_stock_codes_from_analysis_file(analysis_file_path: str) -> list:
    """
    从市场分析报告中提取股票代码（只提取"所有热门股票代码汇总"部分的25只股票）
    
    参数:
        analysis_file_path: 分析报告文件路径
    
    返回:
        list: 股票代码列表
    """
    stock_codes = []
    
    if not os.path.exists(analysis_file_path):
        print(f"❌ 文件不存在: {analysis_file_path}")
        return stock_codes
    
    try:
        with open(analysis_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        import re
        
        # 查找"所有热门股票代码汇总"部分的起始位置
        start_extracting = False
        found_section = False
        skip_next_separator = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # 检测到汇总部分开始（支持多种可能的标题格式）
            if "所有热门股票代码汇总" in line_stripped or "热门股票代码汇总" in line_stripped:
                found_section = True
                skip_next_separator = True  # 标记下一行是分隔线，需要跳过
                continue
            
            # 如果标记了跳过分隔线，且当前行是分隔线，则跳过
            if skip_next_separator:
                if line_stripped.startswith('=') and len(line_stripped.replace('=', '').replace(' ', '')) == 0:
                    skip_next_separator = False
                    start_extracting = True  # 跳过分隔线后开始提取
                    continue
            
            # 如果已经开始提取
            if start_extracting:
                # 如果遇到分隔线（多个等号），停止提取
                if line_stripped.startswith('=') and len(line_stripped.replace('=', '').replace(' ', '')) == 0:
                    # 如果已经提取到股票代码，遇到分隔线就停止
                    if len(stock_codes) > 0:
                        break
                    else:
                        continue  # 跳过分隔线，继续查找
                
                # 如果遇到AI分析部分，停止提取
                if "AI深度分析结果" in line_stripped or "AI深度分析" in line_stripped:
                    break
                
                # 匹配格式：数字. 股票代码（如：1. 601933.SH）
                # 只匹配行首的数字编号格式，避免匹配板块描述中的股票代码
                match = re.match(r'^\s*\d+\.\s+([0-9]{6}\.[A-Z]{2})', line_stripped)
                if match:
                    stock_code = match.group(1)
                    stock_codes.append(stock_code)
        
        if stock_codes:
            print(f"✅ 从分析报告中提取到 {len(stock_codes)} 只股票代码（来自汇总部分）")
        elif found_section:
            print("⚠️  找到汇总部分但未提取到股票代码，尝试备用方法...")
        else:
            print("⚠️  未找到汇总部分标题，尝试备用方法...")
        
        # 备用方法：如果逐行解析失败，使用正则表达式
        if not stock_codes:
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找"所有热门股票代码汇总"和下一个分隔线之间的内容
            # 使用更灵活的正则表达式
            pattern = r'所有热门股票代码汇总.*?\n.*?={50,}\s*\n(.*?)(?=\s*\n={50,}|\s*\n.*?AI深度分析结果)'
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            
            if match:
                summary_section = match.group(1)
                # 只匹配行首的数字编号格式
                pattern2 = r'^\s*\d+\.\s+([0-9]{6}\.[A-Z]{2})'
                for line in summary_section.split('\n'):
                    line_stripped = line.strip()
                    if not line_stripped:  # 跳过空行
                        continue
                    match2 = re.match(pattern2, line_stripped)
                    if match2:
                        stock_code = match2.group(1)
                        if stock_code not in stock_codes:  # 避免重复
                            stock_codes.append(stock_code)
                
                if stock_codes:
                    print(f"✅ 从分析报告中提取到 {len(stock_codes)} 只股票代码（备用方法）")
                else:
                    print("⚠️  备用方法也未找到股票代码")
            else:
                print("⚠️  备用方法：未找到汇总部分")
        
        if not stock_codes:
            print("❌ 未能从分析报告中提取股票代码")
            print("   提示：请确保分析报告包含'所有热门股票代码汇总'部分")
            print(f"   文件路径: {analysis_file_path}")
            # 调试信息：查找包含"汇总"的行
            try:
                with open(analysis_file_path, 'r', encoding='utf-8') as f:
                    debug_lines = f.readlines()
                print("   查找包含'汇总'的行:")
                for i, line in enumerate(debug_lines, 1):
                    if "汇总" in line:
                        print(f"   第{i}行: {line.rstrip()}")
            except Exception as e:
                print(f"   调试信息获取失败: {e}")
    
    except Exception as e:
        print(f"❌ 读取分析报告失败: {e}")
    
    return stock_codes


def analyze_single_stock(pro, stock_code: str, model_type: str = 'lstm') -> dict:
    """
    分析单只股票（封装原有的分析逻辑）

    model_type: lstm / gru / grulstm（对应 GRULSTMModel 混合结构）

    返回:
        dict: 包含预测结果的字典，如果失败返回None
    """
    try:
        print(f"\n{'='*80}")
        print(f"正在分析股票: {stock_code}")
        print(f"{'='*80}")
        
        # 获取历史数据
        df = fetch_stock_history_extended(pro, stock_code, days=1825)
        if df.empty or len(df) < SEQ_LENGTH + 60:
            print(f"❌ {stock_code} 数据不足（建议至少 200~300 根日K）")
            return None
        
        df_raw = df.copy()
        feature_cols = [c for c in df.columns if c not in ["ts_code", "trade_date"]]
        
        # 防止数据泄漏：按时间切分后fit scaler
        split_raw = int(len(df_raw) * TRAIN_RATIO)
        df_train_raw = df_raw.iloc[:split_raw].copy()
        df_all_raw = df_raw.copy()
        
        scaler = MinMaxScaler()
        df_train_scaled = df_train_raw.copy()
        df_train_scaled[feature_cols] = scaler.fit_transform(df_train_raw[feature_cols])
        
        df_all_scaled = df_all_raw.copy()
        df_all_scaled[feature_cols] = scaler.transform(df_all_raw[feature_cols])
        
        # 创建特征和标签
        X, y, used_cols = create_enhanced_features(df_all_raw, df_all_scaled)
        if len(X) == 0:
            print(f"❌ {stock_code} 无法创建训练样本")
            return None
        
        # 划分数据集
        split_idx = int(len(X) * TRAIN_RATIO)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型（lstm / gru / grulstm 混合）
        model_type = normalize_model_type(model_type)
        model = create_sequence_predictor_model(
            model_type, X.shape[2], HIDDEN_SIZE, NUM_LAYERS, num_classes=3
        ).to(device)
        
        train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
        
        # 训练模型（简化版，减少输出）
        print(f"  训练中...（最多{EPOCHS}轮）")
        model, train_losses, val_losses, val_accuracies, val_macro_f1s, val_balanced_accs = train_enhanced_model(
            model, train_loader, val_loader, device, epochs=EPOCHS, num_classes=3
        )
        
        # 获取日期信息
        last_trade_date = df_raw.iloc[-1]['trade_date']
        if isinstance(last_trade_date, pd.Timestamp):
            last_date_str = last_trade_date.strftime('%Y-%m-%d')
        else:
            last_date_str = str(last_trade_date)
        
        next_trade_date = get_next_trading_day(pro, last_trade_date)
        if next_trade_date:
            next_date_str = pd.to_datetime(str(next_trade_date), format='%Y%m%d').strftime('%Y-%m-%d')
        else:
            next_date_str = "下一个交易日"
        
        # 预测（添加置信度计算）
        model.eval()
        last_seq = torch.tensor(X[-1:], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(last_seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs))  # 最大概率作为置信度
        
        original_close_price = float(df_raw.iloc[-1]['close'])
        trend_map = {0: "下跌", 1: "上涨", 2: "震荡"}
        trend = trend_map[pred]
        
        # 生成投资建议（结合准确率和置信度）
        val_accuracy = float(val_accuracies[-1]) if val_accuracies else 0.0
        CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值
        RANDOM_GUESS_ACCURACY = 0.333  # 三分类随机猜测准确率
        
        # 判断模型可靠性
        if val_accuracy < RANDOM_GUESS_ACCURACY:
            # 准确率低于随机猜测，模型不可靠
            model_reliability = "极低"
            reliability_warning = f"⚠️ 模型准确率({val_accuracy:.1%})低于随机猜测(33.3%)，预测不可靠"
            should_reverse = True  # 考虑反向操作
        elif val_accuracy < 0.45:
            # 准确率略高于随机，但不够可靠
            model_reliability = "低"
            reliability_warning = f"⚠️ 模型准确率({val_accuracy:.1%})较低，预测可靠性有限"
            should_reverse = False
        elif val_accuracy < 0.55:
            # 准确率略高于随机，有一定预测能力
            model_reliability = "中等"
            reliability_warning = f"模型准确率({val_accuracy:.1%})中等，预测有一定参考价值"
            should_reverse = False
        else:
            # 准确率较高，模型较可靠
            model_reliability = "较高"
            reliability_warning = f"模型准确率({val_accuracy:.1%})较高，预测较可靠"
            should_reverse = False
        
        # 生成建议
        if confidence < CONFIDENCE_THRESHOLD:
            # 置信度低，无论准确率如何都建议观望
            suggestion = "观望"
            suggestion_reason = f"预测置信度较低（{confidence:.1%}），建议观望。{reliability_warning}"
            risk_level = "高"
            reliability_adjusted_prediction = trend  # 保持原预测
        else:
            # 置信度足够，根据模型可靠性调整建议
            if should_reverse and val_accuracy < RANDOM_GUESS_ACCURACY:
                # 模型比随机还差，考虑反向操作
                if pred == 1:  # 预测上涨 -> 实际可能下跌
                    reliability_adjusted_prediction = "下跌"
                    suggestion = "卖出/减仓"
                    suggestion_reason = f"模型预测上涨，但准确率极低({val_accuracy:.1%})，建议反向操作（卖出/减仓）。{reliability_warning}"
                    risk_level = "极高"
                elif pred == 0:  # 预测下跌 -> 实际可能上涨
                    reliability_adjusted_prediction = "上涨"
                    suggestion = "买入/加仓"
                    suggestion_reason = f"模型预测下跌，但准确率极低({val_accuracy:.1%})，建议反向操作（买入/加仓）。{reliability_warning}"
                    risk_level = "极高"
                else:  # 预测震荡
                    reliability_adjusted_prediction = "震荡"
                    suggestion = "观望"
                    suggestion_reason = f"预测震荡，但模型准确率极低({val_accuracy:.1%})，建议观望。{reliability_warning}"
                    risk_level = "高"
            else:
                # 正常使用模型预测
                reliability_adjusted_prediction = trend
                if pred == 1:  # 上涨
                    suggestion = "买入/加仓"
                    suggestion_reason = f"预测上涨，置信度较高（{confidence:.1%}）。{reliability_warning}"
                    risk_level = "中等" if val_accuracy >= 0.45 else "高"
                elif pred == 0:  # 下跌
                    suggestion = "卖出/减仓"
                    suggestion_reason = f"预测下跌，置信度较高（{confidence:.1%}）。{reliability_warning}"
                    risk_level = "中等" if val_accuracy >= 0.45 else "高"
                else:  # 震荡
                    suggestion = "持有/观望"
                    suggestion_reason = f"预测震荡，建议持有或观望。{reliability_warning}"
                    risk_level = "低"
        
        result = {
            'stock_code': stock_code,
            'current_price': original_close_price,
            'prediction': trend,  # 原始预测
            'reliability_adjusted_prediction': reliability_adjusted_prediction,  # 可靠性调整后的预测
            'confidence': confidence,
            'prob_down': float(probs[0]),
            'prob_up': float(probs[1]),
            'prob_sideways': float(probs[2]),
            'suggestion': suggestion,
            'suggestion_reason': suggestion_reason,
            'risk_level': risk_level,
            'model_reliability': model_reliability,
            'val_accuracy': val_accuracy,
            'val_macro_f1': float(val_macro_f1s[-1]) if val_macro_f1s else 0.0,
            'val_balanced_acc': float(val_balanced_accs[-1]) if val_balanced_accs else 0.0,
            'last_trade_date': last_date_str,
            'next_trade_date': next_date_str,
            'status': 'success'
        }
        
        print(f"  ✅ {stock_code}: 预测{next_date_str}趋势={trend}, 准确率={result['val_accuracy']:.4f}, Macro-F1={result['val_macro_f1']:.4f}")
        return result
        
    except Exception as e:
        print(f"  ❌ {stock_code} 分析失败: {e}")
        return {
            'stock_code': stock_code,
            'status': 'failed',
            'error': str(e)
        }


def batch_analyze_hot_stocks(pro, analysis_file_path: str, model_type: str = 'lstm'):
    """
    批量分析热门股票
    
    参数:
        pro: Tushare API对象
        analysis_file_path: 市场分析报告文件路径
        model_type: 模型类型（lstm / gru / grulstm 混合）
    """
    print("\n" + "=" * 100)
    print("批量分析热门股票")
    print("=" * 100)
    
    # 提取股票代码
    stock_codes = extract_stock_codes_from_analysis_file(analysis_file_path)
    
    if not stock_codes:
        print("❌ 未能提取到股票代码，无法进行批量分析")
        return
    
    print(f"\n将分析 {len(stock_codes)} 只热门股票...")
    
    # 批量分析
    results = []
    success_count = 0
    failed_count = 0
    
    for idx, stock_code in enumerate(stock_codes, 1):
        print(f"\n[{idx}/{len(stock_codes)}] 分析进度")
        result = analyze_single_stock(pro, stock_code, model_type)
        if result:
            results.append(result)
            if result.get('status') == 'success':
                success_count += 1
            else:
                failed_count += 1
    
    # 保存批量分析结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(BATCH_ANALYSIS_DIR, f'batch_prediction_{timestamp}.txt')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("热门股票批量预测分析报告\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型类型: {model_type.upper()}\n")
        f.write(f"分析股票数: {len(stock_codes)}\n")
        f.write(f"成功: {success_count} 只, 失败: {failed_count} 只\n\n")
        f.write("=" * 100 + "\n")
        f.write("详细预测结果\n")
        f.write("=" * 100 + "\n\n")
        
        # 按置信度阈值分类
        CONFIDENCE_THRESHOLD = 0.5
        high_confidence_results = []
        low_confidence_results = []
        
        for result in results:
            if result.get('status') == 'success':
                if result.get('confidence', 0) >= CONFIDENCE_THRESHOLD:
                    high_confidence_results.append(result)
                else:
                    low_confidence_results.append(result)
        
        # 先输出高置信度预测（可投资建议）
        f.write("=" * 100 + "\n")
        f.write("📊 高置信度预测（可构成投资建议）\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"置信度阈值: {CONFIDENCE_THRESHOLD:.0%}\n")
        f.write(f"高置信度股票数: {len(high_confidence_results)} 只\n\n")
        
        if high_confidence_results:
            for idx, result in enumerate(high_confidence_results, 1):
                f.write(f"【{idx}. {result['stock_code']}】⭐\n")
                f.write(f"  当前收盘价: {result['current_price']:.2f}\n")
                f.write(f"  数据日期: {result['last_trade_date']}\n")
                f.write(f"  预测目标: {result['next_trade_date']}\n")
                f.write(f"  模型原始预测: {result['prediction']}\n")
                if result.get('reliability_adjusted_prediction') != result['prediction']:
                    f.write(f"  ⚠️  可靠性调整后预测: {result.get('reliability_adjusted_prediction', result['prediction'])} (模型准确率低，已反向调整)\n")
                f.write(f"  预测置信度: {result.get('confidence', 0):.1%}\n")
                f.write(f"  模型可靠性: {result.get('model_reliability', '未知')} (准确率: {result['val_accuracy']:.1%})\n")
                f.write(f"  下跌概率: {result['prob_down']:.2%}\n")
                f.write(f"  上涨概率: {result['prob_up']:.2%}\n")
                f.write(f"  震荡概率: {result['prob_sideways']:.2%}\n")
                f.write(f"  💡 投资建议: {result.get('suggestion', '未知')}\n")
                f.write(f"  📝 建议理由: {result.get('suggestion_reason', '')}\n")
                f.write(f"  ⚠️  风险等级: {result.get('risk_level', '未知')}\n")
                f.write(f"  Macro-F1: {result['val_macro_f1']:.4f}\n")
                f.write("\n")
        else:
            f.write("⚠️  没有高置信度预测，建议观望\n\n")
        
        # 再输出低置信度预测（仅供参考）
        f.write("=" * 100 + "\n")
        f.write("📋 低置信度预测（仅供参考，不建议作为投资依据）\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"低置信度股票数: {len(low_confidence_results)} 只\n\n")
        
        for idx, result in enumerate(low_confidence_results, 1):
            f.write(f"【{idx}. {result['stock_code']}】\n")
            f.write(f"  当前收盘价: {result['current_price']:.2f}\n")
            f.write(f"  数据日期: {result['last_trade_date']}\n")
            f.write(f"  预测目标: {result['next_trade_date']}\n")
            f.write(f"  预测趋势: {result['prediction']}\n")
            f.write(f"  预测置信度: {result.get('confidence', 0):.1%} ⚠️ 置信度较低\n")
            f.write(f"  下跌概率: {result['prob_down']:.2%}\n")
            f.write(f"  上涨概率: {result['prob_up']:.2%}\n")
            f.write(f"  震荡概率: {result['prob_sideways']:.2%}\n")
            f.write(f"  💡 投资建议: {result.get('suggestion', '观望')}\n")
            f.write(f"  📝 建议理由: {result.get('suggestion_reason', '')}\n")
            f.write(f"  模型准确率: {result['val_accuracy']:.4f}\n")
            f.write("\n")
        
        # 失败的分析
        failed_results = [r for r in results if r.get('status') != 'success']
        if failed_results:
            f.write("=" * 100 + "\n")
            f.write("❌ 分析失败的股票\n")
            f.write("=" * 100 + "\n\n")
            for idx, result in enumerate(failed_results, 1):
                f.write(f"【{idx}. {result['stock_code']}】\n")
                f.write(f"  状态: 分析失败\n")
                f.write(f"  错误: {result.get('error', '未知错误')}\n")
                f.write("\n")
        
        # 统计汇总
        f.write("=" * 100 + "\n")
        f.write("📈 预测统计汇总\n")
        f.write("=" * 100 + "\n\n")
        
        successful_results = [r for r in results if r.get('status') == 'success']
        if successful_results:
            # 整体统计
            up_count = sum(1 for r in successful_results if r['prediction'] == '上涨')
            down_count = sum(1 for r in successful_results if r['prediction'] == '下跌')
            sideways_count = sum(1 for r in successful_results if r['prediction'] == '震荡')
            
            f.write(f"整体预测趋势分布:\n")
            f.write(f"  上涨: {up_count} 只 ({up_count/len(successful_results)*100:.1f}%)\n")
            f.write(f"  下跌: {down_count} 只 ({down_count/len(successful_results)*100:.1f}%)\n")
            f.write(f"  震荡: {sideways_count} 只 ({sideways_count/len(successful_results)*100:.1f}%)\n\n")
            
            # 高置信度统计
            if high_confidence_results:
                hc_up = sum(1 for r in high_confidence_results if r['prediction'] == '上涨')
                hc_down = sum(1 for r in high_confidence_results if r['prediction'] == '下跌')
                hc_sideways = sum(1 for r in high_confidence_results if r['prediction'] == '震荡')
                
                f.write(f"高置信度预测分布（可投资建议）:\n")
                f.write(f"  上涨: {hc_up} 只 ({hc_up/len(high_confidence_results)*100:.1f}%)\n")
                f.write(f"  下跌: {hc_down} 只 ({hc_down/len(high_confidence_results)*100:.1f}%)\n")
                f.write(f"  震荡: {hc_sideways} 只 ({hc_sideways/len(high_confidence_results)*100:.1f}%)\n\n")
                
                # 投资建议统计
                buy_count = sum(1 for r in high_confidence_results if r.get('suggestion') == '买入/加仓')
                sell_count = sum(1 for r in high_confidence_results if r.get('suggestion') == '卖出/减仓')
                hold_count = sum(1 for r in high_confidence_results if r.get('suggestion') in ['持有/观望', '观望'])
                
                f.write(f"投资建议分布:\n")
                f.write(f"  买入/加仓: {buy_count} 只\n")
                f.write(f"  卖出/减仓: {sell_count} 只\n")
                f.write(f"  持有/观望: {hold_count} 只\n\n")
            
            avg_accuracy = np.mean([r['val_accuracy'] for r in successful_results])
            avg_f1 = np.mean([r['val_macro_f1'] for r in successful_results])
            avg_bal_acc = np.mean([r['val_balanced_acc'] for r in successful_results])
            avg_confidence = np.mean([r.get('confidence', 0) for r in successful_results])
            
            f.write(f"模型性能统计:\n")
            f.write(f"  平均准确率: {avg_accuracy:.4f}\n")
            f.write(f"  平均Macro-F1: {avg_f1:.4f}\n")
            f.write(f"  平均Balanced Accuracy: {avg_bal_acc:.4f}\n")
            f.write(f"  平均预测置信度: {avg_confidence:.1%}\n\n")
            
            # 风险提示
            f.write("=" * 100 + "\n")
            f.write("⚠️  重要风险提示\n")
            f.write("=" * 100 + "\n\n")
            f.write("1. 本预测基于历史数据，不保证未来表现\n")
            f.write("2. 股票投资有风险，请谨慎决策\n")
            f.write("3. 建议结合市场环境、公司基本面等多方面因素综合判断\n")
            f.write("4. 低置信度预测仅供参考，不建议作为投资依据\n")
            f.write("5. 高置信度预测也需结合个人风险承受能力\n")
            f.write("6. 建议设置止损点，控制风险\n\n")
    
    print(f"\n✅ 批量分析完成！")
    print(f"📄 分析报告已保存: {result_file}")
    print(f"   成功: {success_count} 只, 失败: {failed_count} 只")
    
    # 打印简要汇总
    successful_results = [r for r in results if r.get('status') == 'success']
    if successful_results:
        up_count = sum(1 for r in successful_results if r['prediction'] == '上涨')
        down_count = sum(1 for r in successful_results if r['prediction'] == '下跌')
        sideways_count = sum(1 for r in successful_results if r['prediction'] == '震荡')
        
        print(f"\n预测趋势汇总:")
        print(f"  上涨: {up_count} 只, 下跌: {down_count} 只, 震荡: {sideways_count} 只")


def main(mode=None, model_type=None):
    pro = get_pro()
    
    print("=" * 100)
    print("增强版 LSTM / GRU / LSTM+GRU混合 股票涨跌趋势预测")
    print("=" * 100)
    
    # 选择分析模式
    if mode is None:
        print("\n请选择分析模式：")
        print("  1. 单只股票分析（默认）")
        print("  2. 批量分析热门股票（从市场分析报告中提取25只股票）")
        mode_choice = input("请选择（1/2，默认1）: ").strip()
    else:
        mode_choice = str(mode).strip()
    
    if mode_choice == '2':
        # 批量分析模式
        # 查找最新的市场分析报告
        if os.path.exists(MARKET_ANALYSIS_DIR):
            analysis_files = [f for f in os.listdir(MARKET_ANALYSIS_DIR) if f.startswith('market_sector_analysis_') and f.endswith('.txt')]
            if analysis_files:
                # 按文件名排序，取最新的
                analysis_files.sort(reverse=True)
                latest_file = os.path.join(MARKET_ANALYSIS_DIR, analysis_files[0])
                print(f"\n找到市场分析报告: {analysis_files[0]}")
                
                if model_type is None:
                    model_type = input("请选择模型类型 (lstm/gru/grulstm，默认lstm): ").strip().lower()
                else:
                    model_type = str(model_type).strip().lower()
                model_type = normalize_model_type(model_type)
                
                batch_analyze_hot_stocks(pro, latest_file, model_type)
                return
            else:
                print("❌ 未找到市场分析报告文件")
                print(f"   请确保 {MARKET_ANALYSIS_DIR} 目录下有 market_sector_analysis_*.txt 文件")
                return
        else:
            print(f"❌ 市场分析目录不存在: {MARKET_ANALYSIS_DIR}")
            print("   请先运行 03_stock_prediction.py 生成市场分析报告")
            return
    
    # 单只股票分析模式（原有逻辑）
    stock_code = input("\n请输入股票代码（可直接输入数字，如：003018、000001、600000，或完整格式：003018.SZ）: ").strip()
    if not stock_code:
        print("❌ 股票代码不能为空")
        return
    
    # 统一转换为大写（处理大小写混合的情况）
    stock_code = stock_code.upper()
    
    # 自动补全后缀
    import re
    if '.' not in stock_code:
        code_match = re.match(r'^(\d{6})$', stock_code)
        if code_match:
            code_prefix = stock_code[:3]
            if code_prefix in ['000', '001', '002', '003', '300', '301']:
                stock_code = stock_code + '.SZ'
                print(f"✅ 自动识别为深市股票，已添加后缀: {stock_code}")
            elif code_prefix in ['600', '601', '603', '605', '688']:
                stock_code = stock_code + '.SH'
                print(f"✅ 自动识别为沪市股票，已添加后缀: {stock_code}")
            else:
                print(f"❌ 错误: 无法识别股票代码前缀 {code_prefix}")
                return
        else:
            print(f"❌ 错误: 股票代码应为6位数字")
            return
    
    # 验证股票代码格式
    pattern = r'^((000|001|002|003|300|301)\d{3}\.SZ|(600|601|603|605|688)\d{3}\.SH)$'
    if not re.match(pattern, stock_code):
        print(f"❌ 错误: 股票代码格式不正确")
        print(f"💡 提示: 支持的格式：")
        print(f"   - 深市主板: 000001.SZ, 001001.SZ")
        print(f"   - 深市中小板: 002001.SZ, 003001.SZ")
        print(f"   - 创业板: 300001.SZ, 301001.SZ")
        print(f"   - 沪市主板: 600000.SH, 601000.SH, 603000.SH, 605000.SH")
        print(f"   - 科创板: 688001.SH")
        print(f"   您输入的代码: {stock_code}")
        return
    
    if model_type is None:
        model_type = input("请选择模型类型 (lstm/gru/grulstm，默认lstm): ").strip().lower()
    else:
        model_type = str(model_type).strip().lower()
    model_type = normalize_model_type(model_type)
    
    print("\n正在获取历史数据（默认5年）...")
    df = fetch_stock_history_extended(pro, stock_code, days=1825)
    if df.empty or len(df) < SEQ_LENGTH + 60:
        print("❌ 数据不足（建议至少 200~300 根日K）")
        return
    print(f"✅ 获取 {len(df)} 条数据")

    df_raw = df.copy()

    # 特征列（排除非数值列）
    feature_cols = [c for c in df.columns if c not in ["ts_code", "trade_date"]]

    # 防止数据泄漏：按时间切分后fit scaler（参考stock_prediction-master）
    split_raw = int(len(df_raw) * TRAIN_RATIO)
    df_train_raw = df_raw.iloc[:split_raw].copy()
    df_all_raw = df_raw.copy()

    scaler = MinMaxScaler()
    df_train_scaled = df_train_raw.copy()
    df_train_scaled[feature_cols] = scaler.fit_transform(df_train_raw[feature_cols])

    df_all_scaled = df_all_raw.copy()
    df_all_scaled[feature_cols] = scaler.transform(df_all_raw[feature_cols])

    print("正在创建增强特征与标签...")
    X, y, used_cols = create_enhanced_features(df_all_raw, df_all_scaled)
    if len(X) == 0:
        print("❌ 无法创建训练样本")
        return
    print(f"✅ 样本数: {len(X)}, 特征数: {X.shape[2]}")

    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model = create_sequence_predictor_model(
        model_type, X.shape[2], HIDDEN_SIZE, NUM_LAYERS, num_classes=3
    ).to(device)
    print(f"✅ 创建{model_type_display_name(model_type)}")

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)  # 时间序列不shuffle
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    print("\n开始训练模型...")
    model, train_losses, val_losses, val_accuracies, val_macro_f1s, val_balanced_accs = train_enhanced_model(
        model, train_loader, val_loader, device, epochs=EPOCHS, num_classes=3
    )

    # 绘制训练历史
    plot_training_history_enhanced(train_losses, val_losses, val_accuracies, val_macro_f1s, val_balanced_accs, stock_code, model_type.upper())

    # 获取最后一条数据的交易日期
    last_trade_date = df_raw.iloc[-1]['trade_date']
    if isinstance(last_trade_date, pd.Timestamp):
        last_date_str = last_trade_date.strftime('%Y-%m-%d')
        last_date_display = last_trade_date.strftime('%Y年%m月%d日')
    else:
        last_date_str = str(last_trade_date)
        last_date_display = last_date_str
    
    next_trade_date = get_next_trading_day(pro, last_trade_date)
    if next_trade_date:
        next_date_str = pd.to_datetime(str(next_trade_date), format='%Y%m%d').strftime('%Y-%m-%d')
        next_date_display = pd.to_datetime(str(next_trade_date), format='%Y%m%d').strftime('%Y年%m月%d日')
    else:
        next_date_str = "下一个交易日"
        next_date_display = "下一个交易日"
    
    # 预测次日趋势
    print("\n" + "=" * 100)
    print("次日趋势预测")
    print("=" * 100)
    print(f"\n📅 数据日期: {last_date_display} ({last_date_str})")
    print(f"🎯 预测目标: {next_date_display} ({next_date_str})")
    
    model.eval()
    last_seq = torch.tensor(X[-1:], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(last_seq)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    original_close_price = float(df_raw.iloc[-1]['close'])
    trend_map = {0: "下跌", 1: "上涨", 2: "震荡"}
    trend = trend_map[pred]

    print(f"\n当前收盘价: {original_close_price:.2f}")
    print(f"预测{next_date_display}趋势: {trend}")
    print(f"下跌概率: {probs[0]:.2%}")
    print(f"上涨概率: {probs[1]:.2%}")
    print(f"震荡概率: {probs[2]:.2%}")

    # 保存模型和归一化参数
    model_path = os.path.join(OUT_DIR, f'{stock_code}_{model_type}_enhanced_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'scaler': scaler,
        'feature_cols': used_cols,
        'input_size': X.shape[2],
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'seq_length': SEQ_LENGTH,
        'last_trade_date': last_date_str,
        'next_trade_date': next_date_str
    }, model_path)
    print(f"\n✅ 模型已保存: {model_path}")
    
    save_norm_params(scaler, used_cols, stock_code, model_type)
    
    print("\n" + "=" * 100)
    print("✅ 完成！")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增强版 LSTM/GRU/LSTM+GRU混合 股票涨跌趋势预测")
    parser.add_argument("--mode", type=str, default=None, help="分析模式：1=单股，2=批量")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["lstm", "gru", "grulstm"],
        help="模型类型：lstm / gru / grulstm（并行LSTM+GRU混合，对应 GRULSTMModel）",
    )
    args = parser.parse_args()
    main(mode=args.mode, model_type=args.model_type)

