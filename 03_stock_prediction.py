# 03_stock_prediction.py
# 功能：大盘A股行情分析 + 热门板块挖掘
# 1) 读取已有的市场数据和分析
# 2) 分析大盘A股整体行情
# 3) 挖掘当前热门Top-5板块
# 4) 根据换手率、成交量等因素，找出每个板块排名前5的热门股
# 5) 调用大模型进行深度分析
#
# 依赖：
# pip install pandas python-dotenv openai tushare

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
import glob
import argparse

from tushare_client import get_pro
from openai_client import get_azure_client

# 配置
# 自动获取当日日期（格式：YYYYMMDD）
# 会自动查找对应的数据文件，如果不存在则提示先运行 01_data_fetch.py
TARGET_DATE = datetime.now().strftime("%Y%m%d")  # 自动获取今日日期

# 输出目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "a_share_out")
MARKET_OUT_DIR = os.path.join(SCRIPT_DIR, "market_analysis")
LATEST_TRADE_DATE_PATH = os.path.join(OUT_DIR, "_latest_trade_date.txt")


def load_market_data(trade_date: str):
    """
    加载市场数据和分析
    """
    daily_csv_path = os.path.join(OUT_DIR, f"daily_{trade_date}.csv")
    idx_csv_path = os.path.join(OUT_DIR, f"index_{trade_date}.csv")
    summary_csv_path = os.path.join(OUT_DIR, f"summary_{trade_date}.csv")
    analysis_txt_path = os.path.join(OUT_DIR, f"analysis_{trade_date}.txt")
    
    data = {}
    
    # 读取daily数据
    if os.path.exists(daily_csv_path):
        data['daily'] = pd.read_csv(daily_csv_path, encoding='utf-8-sig')
        print(f"[完成] 已加载 daily 数据: {len(data['daily'])} 条")
    else:
        print(f"[错误] 找不到 {daily_csv_path}")
        return None
    
    # 读取index数据
    if os.path.exists(idx_csv_path):
        data['index'] = pd.read_csv(idx_csv_path, encoding='utf-8-sig')
        print(f"[完成] 已加载 index 数据: {len(data['index'])} 条")
    else:
        data['index'] = pd.DataFrame()
        print("[警告] 找不到 index 数据")
    
    # 读取summary数据
    if os.path.exists(summary_csv_path):
        data['summary'] = pd.read_csv(summary_csv_path, encoding='utf-8-sig')
        print("[完成] 已加载 summary 数据")
    else:
        print(f"[错误] 找不到 {summary_csv_path}")
        return None
    
    # 读取分析报告
    if os.path.exists(analysis_txt_path):
        with open(analysis_txt_path, 'r', encoding='utf-8') as f:
            data['analysis'] = f.read()
        print("[完成] 已加载市场分析报告")
    else:
        data['analysis'] = ""
        print("[警告] 找不到市场分析报告")
    
    return data


def get_stock_basic_info(pro, trade_date: str):
    """
    获取股票基本信息（行业分类、流通股本等）
    """
    try:
        # 方法1：尝试在主查询中直接包含float_share字段
        try:
            stock_basic = pro.stock_basic(
                exchange='', 
                list_status='L', 
                fields='ts_code,name,industry,area,float_share'
            )
            if 'float_share' in stock_basic.columns:
                print(f"[完成] 已获取股票基本信息（含流通股本）: {len(stock_basic)} 条")
                return stock_basic
        except Exception as e1:
            print(f"[警告] 尝试获取float_share失败: {e1}")
        
        # 方法2：如果方法1失败，尝试只获取基本字段
        try:
            stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,area')
            print(f"[完成] 已获取股票基本信息: {len(stock_basic)} 条")
        except Exception as e2:
            print(f"[警告] 获取股票基本信息失败: {e2}")
            return pd.DataFrame()
        
        # 方法3：尝试通过daily_basic获取流通市值，然后计算流通股本
        try:
            print("正在尝试通过daily_basic获取流通市值数据...")
            daily_basic_temp = pro.daily_basic(
                trade_date=trade_date,
                fields='ts_code,circ_mv,close'
            )
            
            if not daily_basic_temp.empty and 'circ_mv' in daily_basic_temp.columns:
                # 流通股本 = 流通市值 / 收盘价（单位：万股）
                daily_basic_temp['float_share'] = daily_basic_temp['circ_mv'] / (daily_basic_temp['close'] + 1e-8) / 10000
                # 合并到stock_basic
                stock_basic = stock_basic.merge(
                    daily_basic_temp[['ts_code', 'float_share']],
                    on='ts_code',
                    how='left'
                )
                print(f"[完成] 已通过流通市值计算流通股本: {stock_basic['float_share'].notna().sum()} 只股票有数据")
                return stock_basic
        except Exception as e3:
            print(f"[警告] 通过daily_basic获取流通股本失败: {e3}")
        
        # 如果所有方法都失败，添加空列
        stock_basic['float_share'] = None
        print("[警告] 无法获取流通股本数据，将使用成交量估算换手率")
        print("提示：如果您的Tushare账户有足够积分，可以尝试升级权限")
        
        return stock_basic
    except Exception as e:
        print(f"[警告] 获取股票基本信息失败: {e}")
        return pd.DataFrame()


def get_daily_basic_info(pro, trade_date: str):
    """
    获取每日基本面数据（包含换手率、流通市值等）
    """
    try:
        # 尝试获取更多字段，包括流通市值（可用于计算流通股本）
        try:
            daily_basic = pro.daily_basic(
                trade_date=trade_date,
                fields='ts_code,turnover_rate,volume_ratio,pe,pb,circ_mv,close'
            )
            print(f"[完成] 已获取每日基本面数据（含流通市值）: {len(daily_basic)} 条")
        except Exception as e:
            # 如果获取失败，尝试只获取基本字段
            print(f"[警告] 尝试获取完整字段失败: {e}，改用基本字段")
            try:
                daily_basic = pro.daily_basic(
                    trade_date=trade_date,
                    fields='ts_code,turnover_rate,volume_ratio,pe,pb'
                )
                print(f"[完成] 已获取每日基本面数据: {len(daily_basic)} 条")
            except Exception as e2:
                print(f"[警告] 获取每日基本面数据失败: {e2}")
                print("提示：请检查Tushare账户积分是否足够，daily_basic接口可能需要积分权限")
                return pd.DataFrame()
        
        return daily_basic
    except Exception as e:
        print(f"[警告] 获取每日基本面数据失败: {e}")
        print("提示：请检查Tushare账户积分是否足够，daily_basic接口可能需要积分权限")
        return pd.DataFrame()


def fetch_stock_history_for_analysis(pro, ts_code: str, days=60):
    """
    获取股票历史数据用于技术分析（默认60天）
    """
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount"
        )
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算涨跌幅
        df['pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close'] * 100
        
        return df
    except Exception as e:
        return pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame):
    """
    计算技术指标：均线、MACD、KDJ、RSI、布林线等
    """
    if df.empty or len(df) < 20:
        return df
    
    # 移动平均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean() if len(df) >= 60 else np.nan
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # KDJ
    low_9 = df['low'].rolling(9).min()
    high_9 = df['high'].rolling(9).max()
    rsv = (df['close'] - low_9) / ((high_9 - low_9) + 1e-8) * 100
    df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # 布林带
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / ((df['bb_upper'] - df['bb_lower']) + 1e-8)
    
    # 成交量移动平均
    df['vol_ma5'] = df['vol'].rolling(5).mean()
    df['vol_ma10'] = df['vol'].rolling(10).mean()
    df['vol_ratio'] = df['vol'] / (df['vol_ma5'] + 1e-8)
    
    return df


def analyze_stock_comprehensive(pro, ts_code: str, current_data: pd.Series, trade_date: str, market_context: dict = None):
    """
    综合分析单只股票的多维度指标
    
    参数:
        pro: Tushare API对象
        ts_code: 股票代码
        current_data: 当前交易日的数据（Series）
        trade_date: 交易日期
        market_context: 市场环境数据（包含index、summary、analysis等）
    
    返回:
        dict: 包含各项分析指标的字典
    """
    result = {
        'ts_code': ts_code,
        'comprehensive_score': 0.0,
        'scores': {}
    }
    
    # 提取市场环境信息
    market_sentiment = None
    index_pct_chg = None
    if market_context:
        if 'summary' in market_context and not market_context['summary'].empty:
            summary = market_context['summary'].iloc[0]
            # 计算市场情绪：上涨比例
            market_sentiment = summary['up'] / summary['stocks'] if summary['stocks'] > 0 else 0.5
        
        if 'index' in market_context and not market_context['index'].empty:
            # 计算三大指数平均涨跌幅
            index_pct_chg = market_context['index']['pct_chg'].mean()
    
    # 获取历史数据
    hist_df = fetch_stock_history_for_analysis(pro, ts_code, days=60)
    if hist_df.empty or len(hist_df) < 20:
        return result
    
    # 计算技术指标
    hist_df = calculate_technical_indicators(hist_df)
    
    # 获取最新数据
    latest = hist_df.iloc[-1]
    prev_5 = hist_df.iloc[-5:] if len(hist_df) >= 5 else hist_df
    
    # ========== 1. 量价健康配合分析 ==========
    volume_price_score = 0.0
    turnover_rate = current_data.get('turnover_rate', 0)
    
    # 换手率合理性（3%-15%为佳）
    if 3 <= turnover_rate <= 15:
        volume_price_score += 30
    elif 1 <= turnover_rate < 3:
        volume_price_score += 15  # 偏低但可接受
    elif 15 < turnover_rate <= 25:
        volume_price_score += 20  # 偏高但可能活跃
    else:
        volume_price_score += 5  # 异常
    
    # 上涨放量、回调缩量
    if len(prev_5) >= 5:
        up_days = prev_5[prev_5['pct_chg'] > 0]
        down_days = prev_5[prev_5['pct_chg'] < 0]
        
        if not up_days.empty and not down_days.empty:
            avg_vol_up = up_days['vol'].mean()
            avg_vol_down = down_days['vol'].mean()
            if avg_vol_up > avg_vol_down * 1.1:  # 上涨时成交量更大
                volume_price_score += 20
    
    # 量比（当前成交量 vs 5日均量）
    vol_ratio = current_data.get('volume_ratio', 1.0)
    if 1.2 <= vol_ratio <= 3.0:  # 适度放量
        volume_price_score += 20
    elif vol_ratio > 3.0:  # 过度放量
        volume_price_score += 10
    
    result['scores']['volume_price'] = min(volume_price_score, 100)
    
    # ========== 2. 技术面分析 ==========
    tech_score = 0.0
    
    # 均线多头排列
    if not pd.isna(latest.get('ma5')) and not pd.isna(latest.get('ma10')) and not pd.isna(latest.get('ma20')):
        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            tech_score += 25  # 多头排列
        elif latest['close'] > latest['ma5']:
            tech_score += 15  # 至少站上5日线
    
    # MACD金叉且在零轴上方
    if not pd.isna(latest.get('macd')) and not pd.isna(latest.get('macd_signal')):
        if latest['macd'] > latest['macd_signal'] and latest['macd'] > 0:
            tech_score += 20
        elif latest['macd'] > latest['macd_signal']:
            tech_score += 10
    
    # KDJ低位金叉
    if not pd.isna(latest.get('kdj_k')) and not pd.isna(latest.get('kdj_d')):
        if latest['kdj_k'] > latest['kdj_d'] and latest['kdj_k'] < 80:
            tech_score += 15
    
    # RSI健康区间
    if not pd.isna(latest.get('rsi')):
        if 50 <= latest['rsi'] < 70:
            tech_score += 15
        elif 30 <= latest['rsi'] < 50:
            tech_score += 10
    
    # 布林线位置
    if not pd.isna(latest.get('bb_position')):
        if 0.5 <= latest['bb_position'] <= 0.8:  # 中上轨之间
            tech_score += 15
    
    # 突破关键位（价格创新高或突破均线）
    if len(hist_df) >= 20:
        recent_high = hist_df['high'].iloc[-20:].max()
        if latest['close'] > recent_high * 0.98:  # 接近或突破近期高点
            tech_score += 10
    
    result['scores']['technical'] = min(tech_score, 100)
    
    # ========== 3. 股性活跃度分析 ==========
    activity_score = 0.0
    
    # 近期涨停记录
    limit_up_count = (prev_5['pct_chg'] >= 9.5).sum()
    if limit_up_count >= 1:
        activity_score += 30
    elif limit_up_count >= 0:
        activity_score += 10
    
    # 连板记录（连续上涨）
    consecutive_up = 0
    for i in range(len(prev_5) - 1, -1, -1):
        if prev_5.iloc[i]['pct_chg'] > 0:
            consecutive_up += 1
        else:
            break
    if consecutive_up >= 2:
        activity_score += 20
    
    # 振幅（活跃度）
    amplitude = (latest['high'] - latest['low']) / latest['pre_close'] * 100
    if 3 <= amplitude <= 8:  # 适度活跃
        activity_score += 15
    
    # 抗跌性（大盘调整时相对表现）
    # 这里简化处理，如果当前涨幅为正且换手率合理，认为抗跌
    if current_data.get('pct_chg', 0) > 0 and turnover_rate > 2:
        activity_score += 15
    
    result['scores']['activity'] = min(activity_score, 100)
    
    # ========== 4. 筹码结构分析（简化版） ==========
    # 由于Tushare免费版可能没有详细的筹码分布数据，这里用价格位置和成交量分布来估算
    chip_score = 0.0
    
    # 价格位置（相对近期区间）
    if len(hist_df) >= 20:
        price_range = hist_df['close'].iloc[-20:]
        price_position = (latest['close'] - price_range.min()) / (price_range.max() - price_range.min() + 1e-8)
        
        if 0.3 <= price_position <= 0.7:  # 中位，筹码相对稳定
            chip_score += 30
        elif price_position > 0.7:  # 高位，但可能突破
            chip_score += 20
    
    # 成交量分布（低量说明筹码锁定）
    if len(hist_df) >= 10:
        recent_vol_avg = hist_df['vol'].iloc[-10:].mean()
        if latest['vol'] < recent_vol_avg * 0.8 and latest['pct_chg'] > 0:
            chip_score += 20  # 缩量上涨，筹码锁定
    
    # 回调承接力（下跌时成交量小）
    if len(prev_5) >= 3:
        down_days = prev_5[prev_5['pct_chg'] < 0]
        if not down_days.empty:
            avg_down_vol = down_days['vol'].mean()
            avg_up_vol = prev_5[prev_5['pct_chg'] > 0]['vol'].mean() if (prev_5['pct_chg'] > 0).any() else avg_down_vol
            if avg_down_vol < avg_up_vol * 0.9:
                chip_score += 20  # 下跌缩量，承接力强
    
    result['scores']['chip'] = min(chip_score, 100)
    
    # ========== 5. 主力资金分析（简化版） ==========
    # Tushare免费版可能没有大单数据，这里用成交额和量比来估算
    capital_score = 0.0
    
    # 成交额（大额成交可能有大资金参与）
    amount = current_data.get('amount', 0)
    if amount > 1e8:  # 超过1亿
        capital_score += 25
    elif amount > 5e7:  # 超过5000万
        capital_score += 15
    
    # 持续放量（3-5日）
    if len(prev_5) >= 5:
        vol_trend = prev_5['vol'].iloc[-3:].mean() / prev_5['vol'].iloc[-5:-3].mean() if len(prev_5) >= 5 else 1.0
        if vol_trend > 1.2:  # 近期成交量增加
            capital_score += 20
    
    # 量价配合（价格上涨伴随成交量增加）
    if current_data.get('pct_chg', 0) > 0 and vol_ratio > 1.2:
        capital_score += 15
    
    result['scores']['capital'] = min(capital_score, 100)
    
    # ========== 6. 市场环境因子（新增） ==========
    # 考虑大盘环境和市场情绪对个股的影响
    market_factor_score = 0.0
    
    if market_sentiment is not None:
        # 市场情绪因子：如果市场整体上涨，个股上涨更有意义
        stock_pct_chg = current_data.get('pct_chg', 0)
        if market_sentiment > 0.5:  # 市场整体上涨
            if stock_pct_chg > 0:
                market_factor_score += 20  # 跟随大盘上涨
            elif stock_pct_chg < -2:
                market_factor_score -= 10  # 逆势下跌，可能有问题
        else:  # 市场整体下跌
            if stock_pct_chg > 0:
                market_factor_score += 25  # 逆势上涨，强势
            elif stock_pct_chg < -2:
                market_factor_score += 5  # 跟随大盘下跌，正常
    
    if index_pct_chg is not None:
        stock_pct_chg = current_data.get('pct_chg', 0)
        # 相对强度：个股表现 vs 大盘表现
        relative_strength = stock_pct_chg - index_pct_chg
        if relative_strength > 2:  # 明显强于大盘
            market_factor_score += 15
        elif relative_strength > 0:
            market_factor_score += 10
        elif relative_strength < -2:  # 明显弱于大盘
            market_factor_score -= 5
    
    result['scores']['market_factor'] = max(0, min(market_factor_score, 100))
    
    # ========== 综合得分计算 ==========
    # 权重：量价25% + 技术面25% + 股性20% + 筹码15% + 资金10% + 市场因子5%
    result['comprehensive_score'] = (
        result['scores']['volume_price'] * 0.25 +
        result['scores']['technical'] * 0.25 +
        result['scores']['activity'] * 0.20 +
        result['scores']['chip'] * 0.15 +
        result['scores']['capital'] * 0.10 +
        result['scores'].get('market_factor', 50) * 0.05  # 市场因子权重较低，但能区分强弱
    )
    
    # 添加技术指标到结果中
    result['indicators'] = {
        'ma5': latest.get('ma5', np.nan),
        'ma10': latest.get('ma10', np.nan),
        'ma20': latest.get('ma20', np.nan),
        'macd': latest.get('macd', np.nan),
        'rsi': latest.get('rsi', np.nan),
        'kdj_k': latest.get('kdj_k', np.nan),
        'kdj_d': latest.get('kdj_d', np.nan),
    }
    
    return result


def analyze_sector_performance(pro, daily_df: pd.DataFrame, trade_date: str, market_context: dict = None):
    """
    分析板块表现，找出热门Top-5板块及每个板块的热门Top-5股票
    
    返回:
        dict: {
            'top_sectors': [{'industry': '行业名', 'metrics': {...}, 'top_stocks': [...]}],
            'sector_stats': DataFrame
        }
    """
    print("\n正在分析板块表现...")
    
    # 获取股票基本信息
    stock_basic = get_stock_basic_info(pro, trade_date)
    if stock_basic.empty:
        print("[错误] 无法获取股票基本信息，无法进行板块分析")
        return None
    
    # 获取每日基本面数据（换手率等）
    daily_basic = get_daily_basic_info(pro, trade_date)
    
    # 合并数据（只选择存在的列）
    available_cols = ['ts_code', 'industry']  # 必需列
    optional_cols = ['name', 'float_share']  # 可选列
    
    # 检查哪些列存在
    merge_cols = ['ts_code']  # ts_code是必需的
    for col in optional_cols:
        if col in stock_basic.columns:
            merge_cols.append(col)
    
    # 确保industry列存在
    if 'industry' not in stock_basic.columns:
        print("[错误] stock_basic中缺少industry列")
        return None
    
    merge_cols.append('industry')
    
    # 合并数据
    daily_with_info = daily_df.merge(
        stock_basic[merge_cols], 
        on='ts_code', 
        how='left'
    )
    
    # 合并换手率数据
    if not daily_basic.empty:
        # 选择要合并的列
        merge_cols = ['ts_code', 'turnover_rate', 'volume_ratio']
        
        # 如果daily_basic中有流通市值，也合并进来（可用于后续计算）
        if 'circ_mv' in daily_basic.columns:
            merge_cols.append('circ_mv')
        if 'close' in daily_basic.columns:
            merge_cols.append('close')
        
        daily_with_info = daily_with_info.merge(
            daily_basic[merge_cols],
            on='ts_code',
            how='left'
        )
        
        # 如果换手率为空，尝试计算（优先使用float_share，其次使用circ_mv）
        missing_turnover = daily_with_info['turnover_rate'].isna()
        if missing_turnover.any():
            if 'float_share' in daily_with_info.columns:
                # 方法1：使用流通股本计算
                daily_with_info.loc[missing_turnover, 'turnover_rate'] = (
                    daily_with_info.loc[missing_turnover, 'vol'] / 
                    (daily_with_info.loc[missing_turnover, 'float_share'] * 100 + 1e-8) * 100
                )
            elif 'circ_mv' in daily_with_info.columns and 'close' in daily_with_info.columns:
                # 方法2：使用流通市值和收盘价计算流通股本，再计算换手率
                float_share_calc = (
                    daily_with_info.loc[missing_turnover, 'circ_mv'] / 
                    (daily_with_info.loc[missing_turnover, 'close'] + 1e-8) / 10000
                )
                daily_with_info.loc[missing_turnover, 'turnover_rate'] = (
                    daily_with_info.loc[missing_turnover, 'vol'] / (float_share_calc * 100 + 1e-8) * 100
                )
            else:
                # 方法3：如果都没有，使用0或估算值
                daily_with_info.loc[missing_turnover, 'turnover_rate'] = 0
        
        # 如果量比为空，设置为1.0
        if 'volume_ratio' not in daily_with_info.columns or daily_with_info['volume_ratio'].isna().any():
            daily_with_info['volume_ratio'] = daily_with_info.get('volume_ratio', pd.Series([1.0] * len(daily_with_info))).fillna(1.0)
    else:
        # 如果没有daily_basic数据，尝试计算（需要流通股本）
        if 'float_share' in daily_with_info.columns:
            # 换手率 = 成交量 / 流通股本 * 100
            daily_with_info['turnover_rate'] = (
                daily_with_info['vol'] / (daily_with_info['float_share'] * 100 + 1e-8) * 100
            )
        else:
            daily_with_info['turnover_rate'] = 0
        daily_with_info['volume_ratio'] = 1.0
    
    # 过滤掉没有行业信息的股票
    daily_with_info = daily_with_info[daily_with_info['industry'].notna()]
    
    if daily_with_info.empty:
        print("[错误] 没有有效的行业分类数据")
        return None
    
    # 按行业统计板块指标
    sector_stats = daily_with_info.groupby('industry').agg({
        'pct_chg': ['mean', 'count', lambda x: (x > 0).sum()],  # 平均涨跌幅、股票数量、上涨家数
        'amount': 'sum',  # 总成交额
        'vol': 'sum',  # 总成交量
        'turnover_rate': 'mean',  # 平均换手率
        'volume_ratio': 'mean'  # 平均量比
    }).round(4)
    
    sector_stats.columns = ['avg_pct_chg', 'stock_count', 'up_count', 'total_amount', 'total_vol', 'avg_turnover_rate', 'avg_volume_ratio']
    
    # 计算上涨比例
    sector_stats['up_ratio'] = sector_stats['up_count'] / sector_stats['stock_count'] * 100
    
    # 计算热度综合得分（加权）
    # 热度 = 平均涨跌幅 * 0.4 + 平均换手率 * 0.3 + 上涨比例 * 0.2 + 成交额占比 * 0.1
    total_amount = sector_stats['total_amount'].sum()
    sector_stats['amount_ratio'] = sector_stats['total_amount'] / total_amount * 100
    sector_stats['heat_score'] = (
        sector_stats['avg_pct_chg'] * 0.4 +
        sector_stats['avg_turnover_rate'] * 0.3 +
        sector_stats['up_ratio'] * 0.2 +
        sector_stats['amount_ratio'] * 0.1
    )
    
    # 按热度得分排序
    sector_stats = sector_stats.sort_values('heat_score', ascending=False)
    
    # 找出Top-5热门板块
    top_5_sectors = sector_stats.head(5)
    
    result = {
        'top_sectors': [],
        'sector_stats': sector_stats
    }
    
    # 对每个热门板块，找出Top-5热门股票（使用综合评分）
    print("正在对热门板块股票进行多维度综合分析...")
    
    for industry, sector_row in top_5_sectors.iterrows():
        # 获取该板块的所有股票
        sector_stocks = daily_with_info[daily_with_info['industry'] == industry].copy()
        
        # 对每只股票进行综合分析
        comprehensive_scores = []
        stock_analyses = []
        
        for idx, stock_row in sector_stocks.iterrows():
            try:
                analysis = analyze_stock_comprehensive(pro, stock_row['ts_code'], stock_row, trade_date, market_context)
                comprehensive_scores.append(analysis['comprehensive_score'])
                stock_analyses.append(analysis)
            except Exception as e:
                # 如果分析失败，使用基础得分
                base_score = (
                    stock_row['pct_chg'] * 0.3 +
                    stock_row.get('turnover_rate', 0) * 0.3 +
                    (stock_row['vol'] / sector_stocks['vol'].max() * 100) * 0.2 +
                    (stock_row['amount'] / sector_stocks['amount'].max() * 100) * 0.2
                )
                comprehensive_scores.append(base_score)
                stock_analyses.append({
                    'ts_code': stock_row['ts_code'],
                    'comprehensive_score': base_score,
                    'scores': {}
                })
        
        # 添加综合得分
        sector_stocks['comprehensive_score'] = comprehensive_scores
        
        # 按综合得分排序
        sector_stocks = sector_stocks.sort_values('comprehensive_score', ascending=False)
        
        # 取Top-5
        top_5_stocks_list = []
        for idx, stock_row in sector_stocks.head(5).iterrows():
            # 找到对应的分析结果
            analysis = next((a for a in stock_analyses if a['ts_code'] == stock_row['ts_code']), {})
            
            stock_info = {
                'ts_code': stock_row['ts_code'],
                'pct_chg': stock_row['pct_chg'],
                'vol': stock_row['vol'],
                'amount': stock_row['amount'],
                'turnover_rate': stock_row.get('turnover_rate', 0),
                'comprehensive_score': stock_row['comprehensive_score'],
                'scores': analysis.get('scores', {}),
                'indicators': analysis.get('indicators', {})
            }
            
            if 'name' in stock_row:
                stock_info['name'] = stock_row['name']
            
            top_5_stocks_list.append(stock_info)
        
        result['top_sectors'].append({
            'industry': industry,
            'metrics': {
                'avg_pct_chg': sector_row['avg_pct_chg'],
                'stock_count': int(sector_row['stock_count']),
                'up_ratio': sector_row['up_ratio'],
                'total_amount': sector_row['total_amount'],
                'avg_turnover_rate': sector_row['avg_turnover_rate'],
                'heat_score': sector_row['heat_score']
            },
            'top_stocks': top_5_stocks_list
        })
    
    return result


def analyze_market_with_llm(market_data: dict, sector_analysis: dict, trade_date: str) -> str:
    """
    使用大模型分析大盘行情和热门板块
    """
    client, deployment, _ = get_azure_client()
    if not client:
        return "未配置Azure OpenAI API，跳过分析"
    
    # 构建市场整体情况文本
    market_summary = ""
    if 'summary' in market_data and not market_data['summary'].empty:
        summary = market_data['summary'].iloc[0]
        market_summary = f"""
【市场整体情况】
- 全市场: {int(summary['stocks'])}只股票
- 上涨: {int(summary['up'])}只 ({int(summary['up'])/int(summary['stocks'])*100:.1f}%)
- 下跌: {int(summary['down'])}只 ({int(summary['down'])/int(summary['stocks'])*100:.1f}%)
- 平盘: {int(summary['flat'])}只
- 总成交额: {summary['amt_total_yi_est']:.2f}亿元
- 涨停: {int(summary['limit_up_10_est'] + summary['limit_up_20_est'])}只
- 跌停: {int(summary['limit_dn_10_est'] + summary['limit_dn_20_est'])}只
"""
    
    # 三大指数情况
    index_text = ""
    if 'index' in market_data and not market_data['index'].empty:
        index_text = "\n【三大指数情况】\n"
        for _, row in market_data['index'].iterrows():
            index_name = {"000001.SH": "上证指数", "399001.SZ": "深证成指", "399006.SZ": "创业板指"}.get(row['ts_code'], row['ts_code'])
            index_text += f"- {index_name}: {row['pct_chg']:.2f}%, 成交额: {row['amount']/1e8:.2f}亿元\n"
    
    # 热门板块信息（增强版，包含市场环境对比）
    sector_text = "\n【热门Top-5板块及热门股票】\n"
    
    # 添加市场环境对比信息
    if 'index' in market_data and not market_data['index'].empty:
        avg_index_pct = market_data['index']['pct_chg'].mean()
        sector_text += f"\n【市场环境参考】\n"
        sector_text += f"- 三大指数平均涨跌幅: {avg_index_pct:.2f}%\n"
        for _, row in market_data['index'].iterrows():
            index_name = {"000001.SH": "上证指数", "399001.SZ": "深证成指", "399006.SZ": "创业板指"}.get(row['ts_code'], row['ts_code'])
            sector_text += f"  {index_name}: {row['pct_chg']:.2f}%\n"
    
    if 'summary' in market_data and not market_data['summary'].empty:
        summary = market_data['summary'].iloc[0]
        up_ratio = summary['up'] / summary['stocks'] * 100 if summary['stocks'] > 0 else 0
        sector_text += f"- 市场上涨比例: {up_ratio:.1f}%\n"
        sector_text += f"- 涨停/跌停: {int(summary['limit_up_10_est'] + summary['limit_up_20_est'])}/{int(summary['limit_dn_10_est'] + summary['limit_dn_20_est'])}只\n"
    
    for idx, sector_info in enumerate(sector_analysis['top_sectors'], 1):
        sector_text += f"\n【{idx}. {sector_info['industry']}】\n"
        sector_text += f"- 平均涨跌幅: {sector_info['metrics']['avg_pct_chg']:.2f}%"
        
        # 对比市场平均表现
        if 'index' in market_data and not market_data['index'].empty:
            avg_index_pct = market_data['index']['pct_chg'].mean()
            vs_market = sector_info['metrics']['avg_pct_chg'] - avg_index_pct
            sector_text += f" (vs大盘: {vs_market:+.2f}%)"
        sector_text += "\n"
        
        sector_text += f"- 股票数量: {sector_info['metrics']['stock_count']}只\n"
        sector_text += f"- 上涨比例: {sector_info['metrics']['up_ratio']:.2f}%\n"
        sector_text += f"- 总成交额: {sector_info['metrics']['total_amount']/1e8:.2f}亿元\n"
        sector_text += f"- 平均换手率: {sector_info['metrics']['avg_turnover_rate']:.2f}%\n"
        sector_text += f"- 热度得分: {sector_info['metrics']['heat_score']:.2f}\n"
        sector_text += f"- 板块热门Top-5股票:\n"
        for stock_idx, stock in enumerate(sector_info['top_stocks'], 1):
            stock_name = stock.get('name', '') if 'name' in stock else ''
            name_display = f" {stock_name}" if stock_name else ""
            sector_text += f"  {stock_idx}. {stock['ts_code']}{name_display}: 涨跌幅{stock['pct_chg']:.2f}%, "
            sector_text += f"换手率{stock.get('turnover_rate', 0):.2f}%, 成交额{stock['amount']/1e8:.2f}亿元\n"
            sector_text += f"    综合得分: {stock.get('comprehensive_score', 0):.2f}分"
            scores = stock.get('scores', {})
            if scores:
                sector_text += f" (量价:{scores.get('volume_price', 0):.1f} 技术:{scores.get('technical', 0):.1f} "
                sector_text += f"活跃:{scores.get('activity', 0):.1f} 筹码:{scores.get('chip', 0):.1f} "
                sector_text += f"资金:{scores.get('capital', 0):.1f}"
                if 'market_factor' in scores:
                    sector_text += f" 市场:{scores.get('market_factor', 0):.1f}"
                sector_text += ")\n"
            else:
                sector_text += "\n"
    
    # 市场分析摘要（充分利用02_analysis.py的输出）
    analysis_summary = ""
    if 'analysis' in market_data and market_data['analysis']:
        # 提取关键信息：指数分析、资金流向、风险提示、操作建议等
        analysis_text = market_data['analysis']
        # 提取关键段落（如果分析报告有结构化内容）
        key_sections = []
        
        # 查找关键段落
        if "指数为何跌" in analysis_text or "指数" in analysis_text:
            # 提取指数分析部分（前500字）
            idx_section = analysis_text[:min(500, len(analysis_text))]
            key_sections.append(f"【市场指数分析】\n{idx_section}...")
        
        if "资金在防守还是兑现" in analysis_text or "资金" in analysis_text:
            # 提取资金分析部分
            capital_start = analysis_text.find("资金")
            if capital_start >= 0:
                capital_section = analysis_text[capital_start:min(capital_start+500, len(analysis_text))]
                key_sections.append(f"【资金流向分析】\n{capital_section}...")
        
        if "风险提示" in analysis_text or "风险" in analysis_text:
            # 提取风险提示部分
            risk_start = analysis_text.find("风险")
            if risk_start >= 0:
                risk_section = analysis_text[risk_start:min(risk_start+300, len(analysis_text))]
                key_sections.append(f"【市场风险提示】\n{risk_section}...")
        
        if key_sections:
            analysis_summary = "\n" + "\n\n".join(key_sections) + "\n"
        else:
            # 如果没有找到结构化内容，使用前1500字
            analysis_summary = f"\n【市场整体分析摘要】\n{analysis_text[:1500]}...\n"
    
    prompt = f"""请作为资深股票分析师，基于以下数据对A股大盘行情和热门板块进行深度分析。

{market_summary}
{index_text}
{sector_text}
{analysis_summary}

请从以下维度进行深入分析：

【1. 大盘行情分析】
- 当前A股市场整体表现如何？（上涨/下跌/震荡）
- 三大指数的表现说明了什么？
- 市场情绪如何？（上涨/下跌家数、涨跌停数量）
- 成交额和成交量反映了什么市场特征？
- 当前市场处于什么阶段？（牛市/熊市/震荡市）

【2. 热门板块深度分析】
- 为什么这5个板块成为当前最热门的板块？
- 每个热门板块的上涨逻辑是什么？
- 板块轮动特征如何？
- 是否存在板块联动效应？
- 热门板块的持续性如何？

【3. 热门股票分析】
- 每个板块的热门Top-5股票有什么共同特征？
- 这些股票为什么能在板块中脱颖而出？
- 换手率和成交量的异常说明了什么？
- 是否存在资金集中流入的现象？

【4. 市场机会与风险】
- 当前市场有哪些投资机会？
- 热门板块和热门股票的投资价值如何？
- 需要注意哪些风险？
- 市场是否存在过热或过冷的情况？

【5. 操作建议】
- 对于不同风险偏好的投资者，有哪些操作建议？
- 热门板块和热门股票适合什么样的投资策略？
- 需要注意哪些关键信号？

请用专业但通俗易懂的语言，字数控制在1500-2000字，确保分析有深度、有逻辑、可操作。"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位专业的股票分析师，擅长大盘行情分析和板块轮动研究，能够结合市场整体情况和技术面进行综合分析。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=16384
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API调用失败: {str(e)}"


def get_latest_trade_date_from_files():
    """
    从输出目录中查找最新的数据文件，提取交易日期
    如果找不到，返回None
    """
    if not os.path.exists(OUT_DIR):
        return None
    
    # 查找所有daily_*.csv文件
    daily_files = glob.glob(os.path.join(OUT_DIR, "daily_*.csv"))
    if not daily_files:
        return None
    
    # 提取日期并排序，返回最新的
    dates = []
    for file in daily_files:
        filename = os.path.basename(file)
        # 提取日期部分：daily_YYYYMMDD.csv
        try:
            date_str = filename.replace("daily_", "").replace(".csv", "")
            if len(date_str) == 8 and date_str.isdigit():
                dates.append(date_str)
        except:
            continue
    
    if dates:
        dates.sort(reverse=True)  # 降序排列，最新的在前
        return dates[0]
    return None


def get_trade_date_from_marker() -> str | None:
    if not os.path.exists(LATEST_TRADE_DATE_PATH):
        return None
    try:
        with open(LATEST_TRADE_DATE_PATH, "r", encoding="utf-8") as f:
            s = f.read().strip()
        return s if s else None
    except Exception:
        return None


def main():
    # 创建输出目录
    os.makedirs(MARKET_OUT_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser(description="大盘A股行情分析 + 热门板块挖掘")
    parser.add_argument("--date", type=str, default=None, help="指定交易日YYYYMMDD（优先级最高）")
    args = parser.parse_args()

    trade_date = args.date or get_trade_date_from_marker() or get_latest_trade_date_from_files() or TARGET_DATE
    print(f"[日期] 使用交易日期: {trade_date}")
    
    # 获取Tushare API
    pro = get_pro()
    
    # 加载市场数据
    print("=" * 100)
    print("正在加载市场数据...")
    print("=" * 100)
    market_data = load_market_data(trade_date)
    
    if market_data is None:
        print("[错误] 数据加载失败，请先运行 01_data_fetch.py 获取数据")
        return
    
    # 分析板块表现（传入市场环境数据）
    print("\n" + "=" * 100)
    print("正在分析板块表现和挖掘热门股票...")
    print("=" * 100)
    
    # 构建市场环境数据
    market_context = {
        'index': market_data.get('index', pd.DataFrame()),
        'summary': market_data.get('summary', pd.DataFrame()),
        'analysis': market_data.get('analysis', '')
    }
    
    sector_analysis = analyze_sector_performance(pro, market_data['daily'], trade_date, market_context)
    
    if sector_analysis is None:
        print("[错误] 板块分析失败")
        return
    
    # 打印热门板块和热门股票
    print("\n" + "=" * 100)
    print("热门Top-5板块及热门股票")
    print("=" * 100)
    
    all_hot_stocks = []  # 收集所有热门股票代码
    
    for idx, sector_info in enumerate(sector_analysis['top_sectors'], 1):
        print(f"\n【{idx}. {sector_info['industry']}】")
        print(f"  平均涨跌幅: {sector_info['metrics']['avg_pct_chg']:.2f}%")
        print(f"  股票数量: {sector_info['metrics']['stock_count']}只")
        print(f"  上涨比例: {sector_info['metrics']['up_ratio']:.2f}%")
        print(f"  总成交额: {sector_info['metrics']['total_amount']/1e8:.2f}亿元")
        print(f"  平均换手率: {sector_info['metrics']['avg_turnover_rate']:.2f}%")
        print(f"  热度得分: {sector_info['metrics']['heat_score']:.2f}")
        print(f"  板块热门Top-5股票:")
        
        for stock_idx, stock in enumerate(sector_info['top_stocks'], 1):
            stock_name = stock.get('name', '') if 'name' in stock else ''
            name_display = f" {stock_name}" if stock_name else ""
            print(f"    {stock_idx}. {stock['ts_code']}{name_display}")
            print(f"       涨跌幅: {stock['pct_chg']:.2f}%, "
                  f"换手率: {stock.get('turnover_rate', 0):.2f}%, "
                  f"成交额: {stock['amount']/1e8:.2f}亿元")
            print(f"       综合得分: {stock.get('comprehensive_score', 0):.2f}分")
            
            # 显示各维度得分
            scores = stock.get('scores', {})
            if scores:
                score_str = " | ".join([f"{k}:{v:.1f}" for k, v in scores.items()])
                print(f"       维度得分: {score_str}")
            
            all_hot_stocks.append(stock['ts_code'])
    
    print("\n" + "=" * 100)
    print("所有热门股票代码汇总（共25只）")
    print("=" * 100)
    for i, stock_code in enumerate(all_hot_stocks, 1):
        print(f"{i}. {stock_code}")
    
    # 调用大模型分析
    print("\n" + "=" * 100)
    print("正在调用大模型进行深度分析...")
    print("=" * 100)
    analysis = analyze_market_with_llm(market_data, sector_analysis, trade_date)
    
    # 保存分析结果
    analysis_path = os.path.join(MARKET_OUT_DIR, f"market_sector_analysis_{trade_date}.txt")
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(f"A股大盘行情及热门板块分析报告\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"分析日期: {trade_date}\n\n")
        f.write("=" * 100 + "\n")
        f.write("热门Top-5板块及热门股票\n")
        f.write("=" * 100 + "\n\n")
        
        for idx, sector_info in enumerate(sector_analysis['top_sectors'], 1):
            f.write(f"【{idx}. {sector_info['industry']}】\n")
            f.write(f"  平均涨跌幅: {sector_info['metrics']['avg_pct_chg']:.2f}%\n")
            f.write(f"  股票数量: {sector_info['metrics']['stock_count']}只\n")
            f.write(f"  上涨比例: {sector_info['metrics']['up_ratio']:.2f}%\n")
            f.write(f"  总成交额: {sector_info['metrics']['total_amount']/1e8:.2f}亿元\n")
            f.write(f"  平均换手率: {sector_info['metrics']['avg_turnover_rate']:.2f}%\n")
            f.write(f"  热度得分: {sector_info['metrics']['heat_score']:.2f}\n")
            f.write(f"  板块热门Top-5股票:\n")
            
            for stock_idx, stock in enumerate(sector_info['top_stocks'], 1):
                stock_name = stock.get('name', '') if 'name' in stock else ''
                name_display = f" {stock_name}" if stock_name else ""
                f.write(f"    {stock_idx}. {stock['ts_code']}{name_display}\n")
                f.write(f"       涨跌幅: {stock['pct_chg']:.2f}%, ")
                f.write(f"换手率: {stock.get('turnover_rate', 0):.2f}%, ")
                f.write(f"成交额: {stock['amount']/1e8:.2f}亿元\n")
                f.write(f"       综合得分: {stock.get('comprehensive_score', 0):.2f}分\n")
                
                # 写入各维度得分
                scores = stock.get('scores', {})
                if scores:
                    score_details = []
                    score_details.append(f"量价配合: {scores.get('volume_price', 0):.1f}分")
                    score_details.append(f"技术面: {scores.get('technical', 0):.1f}分")
                    score_details.append(f"股性活跃: {scores.get('activity', 0):.1f}分")
                    score_details.append(f"筹码结构: {scores.get('chip', 0):.1f}分")
                    score_details.append(f"主力资金: {scores.get('capital', 0):.1f}分")
                    if 'market_factor' in scores:
                        score_details.append(f"市场因子: {scores.get('market_factor', 0):.1f}分")
                    f.write(f"       维度得分: {' | '.join(score_details)}\n")
                
                # 写入技术指标
                indicators = stock.get('indicators', {})
                if indicators and not all(pd.isna(v) for v in indicators.values()):
                    indicator_strs = []
                    if not pd.isna(indicators.get('ma5')):
                        indicator_strs.append(f"MA5: {indicators['ma5']:.2f}")
                    if not pd.isna(indicators.get('rsi')):
                        indicator_strs.append(f"RSI: {indicators['rsi']:.1f}")
                    if not pd.isna(indicators.get('macd')):
                        indicator_strs.append(f"MACD: {indicators['macd']:.3f}")
                    if indicator_strs:
                        f.write(f"       技术指标: {' | '.join(indicator_strs)}\n")
                
                f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("所有热门股票代码汇总（共25只）\n")
        f.write("=" * 100 + "\n")
        for i, stock_code in enumerate(all_hot_stocks, 1):
            f.write(f"{i}. {stock_code}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("AI深度分析结果\n")
        f.write("=" * 100 + "\n\n")
        f.write(analysis)
    
    print("\n[完成] 分析完成！")
    print(f"[输出] 分析报告已保存: {analysis_path}")
    
    # 打印分析结果预览
    print("\n" + "=" * 100)
    print("AI深度分析结果预览：")
    print("=" * 100)
    print(analysis[:1000] + "..." if len(analysis) > 1000 else analysis)


if __name__ == "__main__":
    main()
