# 05_hybrid_prediction.py
# 功能：混合预测系统 - 深度学习（LSTM+Attention） + Azure OpenAI（JSON输出）
# 关键修复：
# 1) 早停：加入warmup，且用 macro-F1 作为早停/选模指标（比val_loss/acc更适合不平衡三分类）
# 2) LLM空响应：强制JSON输出、降低prompt长度、开启JSON response_format(若支持)、重试与降级
# 3) 时间序列：DataLoader不shuffle；Scaler仅在训练段fit避免泄露
#
# 依赖：
# pip install pandas numpy torch scikit-learn tushare python-dotenv openai

import argparse
import os
import sys
import json
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    balanced_accuracy_score, f1_score
)

from datetime import datetime, timedelta

from tushare_client import get_pro
from openai_client import get_azure_client


# ============== 基础检查 ==============
try:
    numpy_version = np.__version__
    major_version = int(numpy_version.split(".")[0])
    if major_version >= 2:
        print("⚠️ 警告: 检测到 NumPy 2.x，可能与部分PyTorch版本不兼容；如遇异常建议降到 1.26.x")
except Exception:
    pass


# ============== 路径 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_OUT_DIR = os.path.join(SCRIPT_DIR, "hybrid_predictions")
os.makedirs(HYBRID_OUT_DIR, exist_ok=True)


# ============== 超参（你可再调） ==============
SEQ_LENGTH = 60
HIDDEN_SIZE = 128
NUM_LAYERS = 3
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
EPOCHS = 150
TRAIN_RATIO = 0.8

# 早停：warmup + patience
EARLY_STOPPING_WARMUP = 50          # 前50轮不允许早停（避免太早停）
EARLY_STOPPING_PATIENCE = 30        # warmup后，连续30轮无提升则停止

# 选模指标：macro-F1（比accuracy更适合不平衡三分类）
EARLY_STOPPING_METRIC = "macro_f1"  # 可选：macro_f1 / val_loss

# 分类阈值（你当前用三分类）
THRESHOLD_UP = 0.8
THRESHOLD_DOWN = -0.8

# LLM融合：默认关闭（等DL稳定再开）
ENABLE_LLM = True
ENABLE_FUSION = True
DL_WEIGHT = 0.7
LLM_WEIGHT = 0.3

# LLM输出控制
LLM_MAX_TOKENS = 800          # 太大容易reasoning吃掉；太小容易截断；这里取中间
LLM_RETRIES = 3
LLM_TIMEOUT_SEC = 60


# ============== 工具函数 ==============
def get_next_trading_day(pro, last_trade_date, exchange="SSE", lookforward_days=10) -> str:
    """
    获取下一个交易日
    """
    try:
        # 将日期转换为字符串格式（YYYYMMDD）
        if isinstance(last_trade_date, pd.Timestamp):
            last_date_str = last_trade_date.strftime('%Y%m%d')
        else:
            last_date_str = str(last_trade_date).replace('-', '')
        
        # 往前查找未来10个交易日
        end_date = str(int(last_date_str) + lookforward_days * 100)  # 简单估算
        cal = pro.trade_cal(exchange=exchange, start_date=last_date_str, end_date=end_date,
                            fields="cal_date,is_open")
        if cal is None or cal.empty:
            return None
        
        cal = cal.sort_values("cal_date")
        open_days = cal.loc[cal["is_open"] == 1, "cal_date"]
        # 找到last_date_str之后的第一个交易日
        next_days = open_days[open_days > last_date_str]
        if not next_days.empty:
            return next_days.iloc[0]
        return None
    except Exception as e:
        print(f"获取下一个交易日失败: {e}")
        return None


def safe_fillna(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    # 优先前后填充
    try:
        df = df.bfill().ffill()
    except Exception:
        df = df.fillna(method="bfill").fillna(method="ffill")
    # 仍有空则0
    return df.fillna(0)


# ============== 数据获取与特征 ==============
def fetch_stock_history_extended(pro, ts_code: str, days=1825) -> pd.DataFrame:
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

        # 基础
        df["pct_chg"] = (df["close"] - df["pre_close"]) / (df["pre_close"] + 1e-8) * 100
        df["amplitude"] = (df["high"] - df["low"]) / (df["pre_close"] + 1e-8) * 100

        # MA
        for w in [5, 10, 20, 30, 60]:
            df[f"ma{w}"] = df["close"].rolling(w).mean()

        # 量能
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

        # BB
        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-8)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / ((df["bb_upper"] - df["bb_lower"]) + 1e-8)

        # return
        for p in [1, 3, 5, 10, 20]:
            df[f"return_{p}d"] = df["close"].pct_change(p) * 100

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

        # DMI(简化)
        high_diff = df["high"].diff()
        low_diff = -df["low"].diff()
        tr = pd.concat([high_diff.abs(), low_diff.abs(), (df["high"] - df["low"]).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0)

        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-8))
        df["dmi_plus"] = plus_di
        df["dmi_minus"] = minus_di
        df["dmi_adx"] = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8))

        # OBV
        df["obv"] = (np.sign(df["close"].diff()) * df["vol"]).fillna(0).cumsum()
        df["obv_ma"] = df["obv"].rolling(20).mean()

        # ATR pct
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr_atr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr_atr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / (df["close"] + 1e-8) * 100

        df = safe_fillna(df)
        return df
    except Exception as e:
        print(f"获取历史数据失败: {e}")
        return pd.DataFrame()


def create_improved_features(df_raw: pd.DataFrame, df_scaled: pd.DataFrame):
    feature_cols = [
        "open","high","low","close","vol","amount",
        "pct_chg","amplitude",
        "ma5","ma10","ma20","ma30","ma60",
        "vol_ma5","vol_ma10","vol_ratio",
        "rsi","macd","macd_signal","macd_hist",
        "bb_middle","bb_upper","bb_lower","bb_width","bb_position",
        "return_1d","return_3d","return_5d","return_10d","return_20d",
        "kdj_k","kdj_d","kdj_j",
        "cci","dmi_plus","dmi_minus","dmi_adx",
        "obv","obv_ma","atr","atr_pct"
    ]
    available_cols = [c for c in feature_cols if c in df_scaled.columns]
    if len(available_cols) < 10:
        available_cols = ["open","high","low","close","vol","pct_chg","ma5","ma10","ma20","rsi"]

    features = df_scaled[available_cols].values

    # 标签：看下一天pct_chg
    labels = []
    for i in range(len(df_raw) - 1):
        next_pct = float(df_raw.iloc[i+1].get("pct_chg", 0.0))
        if next_pct > THRESHOLD_UP:
            labels.append(1)  # up
        elif next_pct < THRESHOLD_DOWN:
            labels.append(0)  # down
        else:
            labels.append(2)  # sideways
    labels.append(2)

    X, y = [], []
    for i in range(SEQ_LENGTH, len(features)):
        X.append(features[i-SEQ_LENGTH:i])
        y.append(labels[i])

    y = np.array(y)
    counts = {0: int((y==0).sum()), 1: int((y==1).sum()), 2: int((y==2).sum())}
    print(f"  标签分布 - 下跌:{counts[0]} 上涨:{counts[1]} 震荡:{counts[2]} (阈值: {THRESHOLD_DOWN}%~{THRESHOLD_UP}%)")

    return np.array(X), y, available_cols


# ============== Dataset ==============
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============== 模型 ==============
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size*2,
            num_heads=4,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        h = attn_out[:, -1, :]
        h = self.drop1(torch.relu(self.fc1(h)))
        return self.fc2(h)


def make_class_weights(y, num_classes=3):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ============== 训练（修复早停+防塌缩观测） ==============
def train_improved_model(model, train_loader, val_loader, device, epochs=EPOCHS, num_classes=3):
    # class weights
    y_train_all = []
    for _, yb in train_loader:
        y_train_all.append(yb.numpy())
    y_train_all = np.concatenate(y_train_all)
    class_weights = make_class_weights(y_train_all, num_classes=num_classes).to(device)
    print(f"类别权重: {class_weights.detach().cpu().numpy()}")

    # label smoothing 可缓解塌缩（PyTorch>=1.10一般支持）
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    best_state = None
    best_metric = -1e9
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 0

    print(f"开始训练，最多 {epochs} 轮；warmup={EARLY_STOPPING_WARMUP}；patience={EARLY_STOPPING_PATIENCE}；early_stop_metric={EARLY_STOPPING_METRIC}")

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            tr_total += yb.size(0)
            tr_correct += int((pred == yb).sum().item())

        tr_loss /= max(1, len(train_loader))
        tr_acc = tr_correct / max(1, tr_total)

        # val
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
        va_macro_f1 = f1_score(all_true, all_pred, average="macro")
        va_bal_acc = balanced_accuracy_score(all_true, all_pred)

        # 观测塌缩：打印预测分布
        pred_counts = np.bincount(all_pred, minlength=num_classes)
        pred_dist = (pred_counts / pred_counts.sum()).round(3)

        # 选择早停/选模指标
        if EARLY_STOPPING_METRIC == "macro_f1":
            metric = va_macro_f1
            scheduler.step(metric)  # plateau按macro-f1
        else:
            metric = -va_loss
            scheduler.step(va_macro_f1)

        improved = metric > best_metric + 1e-6

        if improved:
            best_metric = metric
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            # warmup期不计patience
            if epoch > EARLY_STOPPING_WARMUP:
                patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"TrainLoss {tr_loss:.4f} TrainAcc {tr_acc:.4f} | "
                f"ValLoss {va_loss:.4f} ValAcc {va_acc:.4f} BalAcc {va_bal_acc:.4f} MacroF1 {va_macro_f1:.4f} | "
                f"PredDist(d/u/s) {pred_dist} "
                f"{'✓' if improved else ''}"
            )

        if epoch > EARLY_STOPPING_WARMUP and patience >= EARLY_STOPPING_PATIENCE:
            print(f"早停触发：epoch={epoch}，在warmup后指标 {EARLY_STOPPING_METRIC} 连续{EARLY_STOPPING_PATIENCE}轮未改善")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"已加载最佳模型：ValAcc={best_val_acc:.4f} ValLoss={best_val_loss:.4f} Best({EARLY_STOPPING_METRIC})={best_metric:.4f}")
    else:
        print("⚠️ 未找到best_state（不正常），将使用最后一轮模型")

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

    return model, best_val_acc


# ============== LLM（修复空响应） ==============
def _extract_json_from_text(text: str):
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            return None
    return None


def llm_predict_stock(stock_code: str, df_raw: pd.DataFrame, market_analysis: str = "") -> dict:
    client, deployment, _ = get_azure_client()
    if not client:
        return {"error": "未配置AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY"}

    recent = df_raw.tail(30)
    latest = df_raw.iloc[-1]

    # 极简summary（关键：缩短prompt，减少reasoning）
    summary = {
        "code": stock_code,
        "close": float(latest.get("close", 0)),
        "pct_chg": float(latest.get("pct_chg", 0)),
        "rsi": float(latest.get("rsi", 50)),
        "macd": float(latest.get("macd", 0)),
        "macd_signal": float(latest.get("macd_signal", 0)),
        "kdj_k": float(latest.get("kdj_k", 50)),
        "kdj_d": float(latest.get("kdj_d", 50)),
        "atr_pct": float(latest.get("atr_pct", 0)),
        "mean_30_pct": float(recent["pct_chg"].mean()) if "pct_chg" in recent else 0.0,
        "vol_ratio": float(latest.get("vol_ratio", 1.0)),
    }

    market_short = (market_analysis or "").strip().replace("\n", " ")
    if len(market_short) > 300:
        market_short = market_short[:300]

    # 强制“只输出JSON”，不要推理，减少gpt-5把token花在reasoning导致content为空的概率
    prompt = f"""
只输出JSON，不要任何多余文本，不要markdown，不要解释。

任务：预测 {stock_code} 次日方向（上涨/下跌/震荡），并给出概率。
数据（近期摘要）：
{json.dumps(summary, ensure_ascii=False)}

市场环境（可为空）：
{market_short}

返回JSON格式：
{{
  "trend": "上涨/下跌/震荡",
  "prob_up": 0.0-1.0,
  "prob_down": 0.0-1.0,
  "prob_sideways": 0.0-1.0,
  "confidence": 0.0-1.0,
  "reasoning": "不超过80字"
}}
""".strip()

    system = "You are a financial analyst. Return ONLY valid JSON. No extra text."

    last_err = None
    for attempt in range(1, LLM_RETRIES+1):
        try:
            kwargs = dict(
                model=deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=LLM_MAX_TOKENS,
            )

            # 如果SDK/部署支持 response_format=json_object，会显著降低解析失败/空内容
            # 不支持会抛异常，自动走except重试（下一次去掉该参数）
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

            resp = client.chat.completions.create(**kwargs)

            if not resp.choices:
                last_err = "API响应无choices"
                continue

            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)

            content = ""
            if hasattr(choice, "message") and choice.message is not None:
                content = choice.message.content or ""

            # gpt-5有时content空但reasoning存在；这时直接判失败重试
            if not content.strip():
                last_err = f"content为空（finish_reason={finish_reason}）"
                # 进一步缩短、降低max tokens再试
                time.sleep(1.0 * attempt)
                continue

            obj = _extract_json_from_text(content)
            if obj is None:
                # 尝试直接json.loads
                try:
                    obj = json.loads(content)
                except Exception:
                    obj = None

            if obj is None:
                last_err = "无法解析JSON"
                time.sleep(1.0 * attempt)
                continue

            # 概率归一化
            pu = float(obj.get("prob_up", 0.33))
            pdn = float(obj.get("prob_down", 0.33))
            ps = float(obj.get("prob_sideways", 0.34))
            s = pu + pdn + ps
            if s <= 0:
                pu, pdn, ps = 0.33, 0.33, 0.34
                s = 1.0
            obj["prob_up"] = pu / s
            obj["prob_down"] = pdn / s
            obj["prob_sideways"] = ps / s
            obj["raw_response"] = content
            obj["finish_reason"] = finish_reason
            return obj

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0 * attempt)

    return {"error": f"LLM调用失败: {last_err}"}


# ============== 主流程 ==============
def main(stock_code=None):
    pro = get_pro()

    print("="*100)
    print("混合预测系统 - 深度学习 + 大模型（修复版）")
    print("="*100)

    if stock_code is None:
        stock_code = input("\n请输入股票代码（格式：000001.SZ、002001.SZ、300001.SZ、600000.SH、603000.SH、688001.SH）: ").strip()
        if not stock_code:
            return
        stock_code = stock_code.upper()
    else:
        stock_code = stock_code.strip().upper()
        if not stock_code:
            print("错误: 股票代码不能为空")
            return
    
    # 验证股票代码格式（支持更多板块）
    import re
    # 深市：000xxx(主板)、001xxx(主板)、002xxx(中小板)、003xxx(中小板)、300xxx(创业板)、301xxx(创业板注册制)
    # 沪市：600xxx(主板)、601xxx(主板)、603xxx(主板)、605xxx(主板)、688xxx(科创板)
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

    print("\n正在获取历史数据（默认5年）...")
    df = fetch_stock_history_extended(pro, stock_code, days=1825)
    if df.empty or len(df) < SEQ_LENGTH + 60:
        print("❌ 数据不足（建议至少 200~300 根日K）")
        return
    print(f"✅ 获取 {len(df)} 条数据")

    df_raw = df.copy()

    # 特征列
    feature_cols = [c for c in df.columns if c not in ["ts_code", "trade_date"]]

    # 避免泄露：按时间切分后fit scaler
    split_raw = int(len(df_raw) * TRAIN_RATIO)
    df_train_raw = df_raw.iloc[:split_raw].copy()
    df_all_raw = df_raw.copy()

    scaler = MinMaxScaler()
    df_train_scaled = df_train_raw.copy()
    df_train_scaled[feature_cols] = scaler.fit_transform(df_train_raw[feature_cols])

    df_all_scaled = df_all_raw.copy()
    df_all_scaled[feature_cols] = scaler.transform(df_all_raw[feature_cols])

    print("正在创建特征与标签...")
    X, y, used_cols = create_improved_features(df_all_raw, df_all_scaled)
    if len(X) == 0:
        print("❌ 无法创建训练样本")
        return
    print(f"✅ 样本数: {len(X)}, 特征数: {X.shape[2]}")

    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model = ImprovedLSTMModel(input_size=X.shape[2], hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=3).to(device)

    # 时间序列：不shuffle
    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    print("\n开始训练模型...")
    model, dl_acc = train_improved_model(model, train_loader, val_loader, device, epochs=EPOCHS, num_classes=3)

    # 获取最后一条数据的交易日期
    last_trade_date = df_raw.iloc[-1]['trade_date']
    if isinstance(last_trade_date, pd.Timestamp):
        last_date_str = last_trade_date.strftime('%Y-%m-%d')
        last_date_display = last_trade_date.strftime('%Y年%m月%d日')
    else:
        last_date_str = str(last_trade_date)
        last_date_display = last_date_str
    
    # 获取下一个交易日
    next_trade_date = get_next_trading_day(pro, last_trade_date)
    if next_trade_date:
        next_date_str = pd.to_datetime(str(next_trade_date), format='%Y%m%d').strftime('%Y-%m-%d')
        next_date_display = pd.to_datetime(str(next_trade_date), format='%Y%m%d').strftime('%Y年%m月%d日')
    else:
        next_date_str = "下一个交易日"
        next_date_display = "下一个交易日"
    
    print(f"\n📅 数据日期: {last_date_display} ({last_date_str})")
    print(f"🎯 预测目标: {next_date_display} ({next_date_str})")
    
    # 对最后一个序列做预测
    model.eval()
    last_seq = torch.tensor(X[-1:], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(last_seq)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    dl_trend = ["下跌", "上涨", "震荡"][pred]
    print(f"\n✅ 深度学习预测 ({next_date_display}): {dl_trend} | prob(d/u/s)={probs[0]:.2%}/{probs[1]:.2%}/{probs[2]:.2%}")

    # 读取市场分析（可选）
    market_analysis = ""
    analysis_path = os.path.join(SCRIPT_DIR, "a_share_out", "analysis_20260205.txt")
    if os.path.exists(analysis_path):
        with open(analysis_path, "r", encoding="utf-8") as f:
            market_analysis = f.read()

    llm_result = {"error": "LLM disabled"}
    if ENABLE_LLM:
        print("\n正在调用大模型...")
        llm_result = llm_predict_stock(stock_code, df_raw, market_analysis)
        if "error" in llm_result:
            print(f"⚠️ 大模型失败: {llm_result['error']}")
        else:
            print(f"✅ 大模型趋势: {llm_result.get('trend')} | "
                  f"prob(d/u/s)={llm_result['prob_down']:.2%}/{llm_result['prob_up']:.2%}/{llm_result['prob_sideways']:.2%} | "
                  f"finish_reason={llm_result.get('finish_reason')}")
            if "reasoning" in llm_result:
                print("  reasoning:", llm_result["reasoning"])

    # 融合
    print("\n" + "="*100)
    print("融合预测结果")
    print("="*100)
    print(f"\n📅 数据日期: {last_date_display} ({last_date_str})")
    print(f"🎯 预测目标: {next_date_display} ({next_date_str})")

    if not ENABLE_FUSION:
        print("⚠️ 融合未启用：仅使用深度学习结果")
        fused_probs = probs
        fused_trend = dl_trend
    elif "error" in llm_result:
        print("⚠️ LLM失败：仅使用深度学习结果")
        fused_probs = probs
        fused_trend = dl_trend
    else:
        llm_probs_3 = np.array([llm_result["prob_down"], llm_result["prob_up"], llm_result["prob_sideways"]], dtype=float)
        llm_probs_3 = llm_probs_3 / (llm_probs_3.sum() + 1e-8)
        fused_probs = DL_WEIGHT * probs + LLM_WEIGHT * llm_probs_3
        fused_trend = ["下跌", "上涨", "震荡"][int(np.argmax(fused_probs))]

    print(f"\n融合预测 ({next_date_display}): {fused_trend}")
    print(f"  下跌概率: {fused_probs[0]:.2%}")
    print(f"  上涨概率: {fused_probs[1]:.2%}")
    print(f"  震荡概率: {fused_probs[2]:.2%}")

    # 保存
    out_path = os.path.join(HYBRID_OUT_DIR, f"{stock_code}_hybrid_prediction.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"混合预测报告 - {stock_code}\n")
        f.write("="*100 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据日期: {last_date_display} ({last_date_str})\n")
        f.write(f"预测目标: {next_date_display} ({next_date_str})\n\n")
        f.write("深度学习预测:\n")
        f.write(f"  trend={dl_trend}\n")
        f.write(f"  prob_down={probs[0]:.6f} prob_up={probs[1]:.6f} prob_sideways={probs[2]:.6f}\n\n")
        f.write("大模型预测:\n")
        f.write(json.dumps(llm_result, ensure_ascii=False, indent=2))
        f.write("\n\n融合预测:\n")
        f.write(f"  trend={fused_trend}\n")
        f.write(f"  prob_down={fused_probs[0]:.6f} prob_up={fused_probs[1]:.6f} prob_sideways={fused_probs[2]:.6f}\n")

    print(f"\n✅ 已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="混合预测系统 - 深度学习 + 大模型")
    parser.add_argument("--stock", type=str, default=None, help="股票代码（如 000001.SZ）")
    args = parser.parse_args()
    main(stock_code=args.stock)