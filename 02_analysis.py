# 02_analysis.py
# 功能：调用大模型分析市场数据
# 1) 读取已保存的市场数据文件
# 2) 调用Azure OpenAI API进行深度分析
# 3) 保存分析结果
#
# 依赖：
# pip install pandas python-dotenv openai

import os
import pandas as pd

from tushare_client import get_pro_optional as get_pro
from openai_client import get_azure_client
from datetime import datetime
import argparse

# 自动获取当日日期（格式：YYYYMMDD）
# 会自动查找对应的数据文件，如果不存在则提示先运行 01_data_fetch.py
TARGET_DATE = datetime.now().strftime("%Y%m%d")  # 自动获取今日日期

# 输出目录：保存到脚本所在目录下的 a_share_out 文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "a_share_out")
LATEST_TRADE_DATE_PATH = os.path.join(OUT_DIR, "_latest_trade_date.txt")


def analyze_sector_performance(pro, daily: pd.DataFrame, trade_date: str) -> str:
    """
    分析板块表现：获取行业分类数据并统计各板块涨跌情况
    """
    if daily is None or daily.empty or pro is None:
        return ""
    
    try:
        sector_info = "\n\n板块表现分析：\n"
        
        # 按涨跌幅分组统计
        pct_chg = daily['pct_chg']
        
        # 统计不同涨跌幅区间的股票数量
        sector_info += "【涨跌幅分布】\n"
        sector_info += f"- 大涨(>5%): {(pct_chg > 5).sum()}只\n"
        sector_info += f"- 中涨(2%-5%): {((pct_chg >= 2) & (pct_chg <= 5)).sum()}只\n"
        sector_info += f"- 小涨(0-2%): {((pct_chg > 0) & (pct_chg < 2)).sum()}只\n"
        sector_info += f"- 小跌(0--2%): {((pct_chg < 0) & (pct_chg >= -2)).sum()}只\n"
        sector_info += f"- 中跌(-2%--5%): {((pct_chg < -2) & (pct_chg >= -5)).sum()}只\n"
        sector_info += f"- 大跌(<-5%): {(pct_chg < -5).sum()}只\n"
        
        # 尝试获取行业分类数据
        try:
            # 获取所有股票的基本信息（包含行业）
            stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,area')
            
            # 合并行业信息
            daily_with_industry = daily.merge(stock_basic[['ts_code', 'industry']], on='ts_code', how='left')
            
            # 按行业统计
            if 'industry' in daily_with_industry.columns:
                industry_stats = daily_with_industry.groupby('industry').agg({
                    'pct_chg': ['mean', 'count', 'sum'],
                    'amount': 'sum'
                }).round(2)
                
                industry_stats.columns = ['平均涨跌幅', '股票数量', '涨跌幅总和', '成交额']
                industry_stats = industry_stats.sort_values('平均涨跌幅', ascending=False)
                
                # 找出领涨和领跌行业（前10和后10）
                sector_info += "\n【行业板块表现TOP10（按平均涨跌幅）】\n"
                top_industries = industry_stats.head(10)
                for idx, row in top_industries.iterrows():
                    sector_info += f"- {idx}: 平均涨跌{row['平均涨跌幅']:.2f}%, {int(row['股票数量'])}只股票, 成交额{row['成交额']/1e8:.2f}亿元\n"
                
                sector_info += "\n【行业板块表现BOTTOM10（按平均涨跌幅）】\n"
                bottom_industries = industry_stats.tail(10)
                for idx, row in bottom_industries.iterrows():
                    sector_info += f"- {idx}: 平均涨跌{row['平均涨跌幅']:.2f}%, {int(row['股票数量'])}只股票, 成交额{row['成交额']/1e8:.2f}亿元\n"
                
                # 统计各行业涨跌家数
                industry_up_down = daily_with_industry.groupby('industry').apply(
                    lambda x: pd.Series({
                        '上涨': (x['pct_chg'] > 0).sum(),
                        '下跌': (x['pct_chg'] < 0).sum(),
                        '平盘': (x['pct_chg'] == 0).sum()
                    })
                )
                industry_up_down = industry_up_down.sort_values('上涨', ascending=False)
                
                sector_info += "\n【行业涨跌家数统计（上涨家数TOP10）】\n"
                top_up = industry_up_down.head(10)
                for idx, row in top_up.iterrows():
                    sector_info += f"- {idx}: 上涨{int(row['上涨'])}只, 下跌{int(row['下跌'])}只, 平盘{int(row['平盘'])}只\n"
                
        except Exception as e:
            print(f"获取行业分类数据失败: {e}，使用简化分析")
        
        return sector_info
        
    except Exception as e:
        print(f"板块分析失败: {e}")
        return ""


def analyze_summary_with_llm(summary: pd.DataFrame, trade_date: str, idx: pd.DataFrame = None, daily: pd.DataFrame = None, pro=None) -> str:
    """
    使用大模型API分析市场情绪汇总数据
    深入分析指数下跌原因、结构、资金流向和次日关注信号
    """
    client, deployment, _ = get_azure_client()
    if not client:
        return "未配置Azure OpenAI API，跳过分析"
    
    # 构建分析提示
    summary_text = summary.to_string(index=False)
    
    # 添加指数信息（详细）
    index_info = ""
    if idx is not None and not idx.empty:
        index_info = "\n\n三大指数详细情况：\n"
        for _, row in idx.iterrows():
            index_name = {"000001.SH": "上证指数", "399001.SZ": "深证成指", "399006.SZ": "创业板指"}.get(row['ts_code'], row['ts_code'])
            index_info += f"- {index_name}({row['ts_code']}):\n"
            index_info += f"  开盘: {row['open']:.2f}, 最高: {row['high']:.2f}, 最低: {row['low']:.2f}, 收盘: {row['close']:.2f}\n"
            index_info += f"  涨跌: {row['change']:.2f}, 涨跌幅: {row['pct_chg']:.2f}%\n"
            index_info += f"  成交额: {row['amount']/1e8:.2f}亿元, 成交量: {row['vol']/1e8:.2f}亿手\n"
    
    # 添加市场结构信息
    structure_info = ""
    if daily is not None and not daily.empty:
        # 涨跌幅分布
        pct_chg = daily['pct_chg']
        structure_info = "\n\n市场结构分析：\n"
        structure_info += f"- 涨幅>5%: {(pct_chg > 5).sum()}只, 占比 {(pct_chg > 5).sum()/len(daily)*100:.2f}%\n"
        structure_info += f"- 涨幅2%-5%: {((pct_chg >= 2) & (pct_chg <= 5)).sum()}只, 占比 {((pct_chg >= 2) & (pct_chg <= 5)).sum()/len(daily)*100:.2f}%\n"
        structure_info += f"- 涨幅0%-2%: {((pct_chg > 0) & (pct_chg < 2)).sum()}只, 占比 {((pct_chg > 0) & (pct_chg < 2)).sum()/len(daily)*100:.2f}%\n"
        structure_info += f"- 跌幅0%--2%: {((pct_chg < 0) & (pct_chg >= -2)).sum()}只, 占比 {((pct_chg < 0) & (pct_chg >= -2)).sum()/len(daily)*100:.2f}%\n"
        structure_info += f"- 跌幅-2%--5%: {((pct_chg < -2) & (pct_chg >= -5)).sum()}只, 占比 {((pct_chg < -2) & (pct_chg >= -5)).sum()/len(daily)*100:.2f}%\n"
        structure_info += f"- 跌幅<-5%: {(pct_chg < -5).sum()}只, 占比 {(pct_chg < -5).sum()/len(daily)*100:.2f}%\n"
        
        # 成交额分析
        total_amount = daily['amount'].sum()
        top_amount = daily.nlargest(100, 'amount')['amount'].sum()
        structure_info += f"\n- 成交额集中度: 前100只股票成交额占比 {top_amount/total_amount*100:.2f}%\n"
        
        # 涨跌停对比
        limit_up = summary.iloc[0]['limit_up_10_est'] + summary.iloc[0]['limit_up_20_est']
        limit_down = summary.iloc[0]['limit_dn_10_est'] + summary.iloc[0]['limit_dn_20_est']
        structure_info += f"- 涨跌停对比: 涨停{limit_up}只 vs 跌停{limit_down}只, 比例 {limit_up/(limit_down+1):.2f}\n"
    
    # 计算技术面关键数据
    tech_info = ""
    if idx is not None and not idx.empty:
        tech_info = "\n\n技术面关键数据：\n"
        for _, row in idx.iterrows():
            index_name = {"000001.SH": "上证指数", "399001.SZ": "深证成指", "399006.SZ": "创业板指"}.get(row['ts_code'], row['ts_code'])
            # 计算振幅
            amplitude = (row['high'] - row['low']) / row['pre_close'] * 100
            # 计算上影线和下影线
            upper_shadow = (row['high'] - max(row['open'], row['close'])) / row['pre_close'] * 100
            lower_shadow = (min(row['open'], row['close']) - row['low']) / row['pre_close'] * 100
            tech_info += f"- {index_name}:\n"
            tech_info += f"  振幅: {amplitude:.2f}%, 上影线: {upper_shadow:.2f}%, 下影线: {lower_shadow:.2f}%\n"
            tech_info += f"  关键点位: 支撑位(最低) {row['low']:.2f}, 阻力位(最高) {row['high']:.2f}\n"
    
    # 获取板块分析数据
    sector_info = ""
    if pro is not None and daily is not None and not daily.empty:
        print("正在分析板块表现...")
        sector_info = analyze_sector_performance(pro, daily, trade_date)
    
    prompt = f"""请作为资深股票市场分析师，深入分析以下A股市场行情数据，交易日：{trade_date}

【市场情绪汇总】
{summary_text}
{index_info}
{structure_info}
{tech_info}
{sector_info}

请务必从以下八个核心维度进行深入分析，要求逻辑清晰、数据支撑、可操作性强：

【1. 指数为何跌？】
- 结合三大指数的涨跌幅、成交额、涨跌家数比例，分析指数下跌的主要原因
- 是系统性风险还是结构性调整？
- 下跌的力度和速度如何？（温和调整 vs 恐慌性下跌）
- 是否有外部因素影响（政策、消息面、外围市场等）？

【2. 跌在什么结构？】
- **重点分析板块结构**：详细分析哪些行业/板块领涨、哪些领跌
- 板块涨跌幅排名分析（TOP10领涨板块 vs BOTTOM10领跌板块）
- 板块成交额分析：哪些板块资金流入，哪些流出
- 板块涨跌家数对比：哪些板块内部上涨股票多，哪些下跌股票多
- 是大盘股领跌还是小盘股领跌？
- 是权重股拖累还是普跌？
- 哪些板块相对抗跌，哪些板块跌幅较大？
- 涨跌幅分布结构说明了什么？（如：跌幅<-5%的股票占比）
- 风格轮动特征（价值 vs 成长，大盘 vs 小盘）
- 板块轮动逻辑分析：为什么这些板块涨/跌？

【3. 资金在防守还是兑现？】
- 从成交额变化和涨跌停比例判断资金态度
- 成交额是放大还是萎缩？说明资金是恐慌出逃还是观望？
- 涨停板数量 vs 跌停板数量，反映资金是进攻还是防守？
- 成交额集中度（前100只股票占比）说明资金是抱团还是分散？
- 综合判断：资金是在防守（避险）还是在兑现（获利了结）？
- 资金流向特征（流入哪些板块，流出哪些板块）

【4. 技术面关键信号】
- 分析三大指数的技术形态（K线形态、上下影线、振幅）
- 关键支撑位和阻力位的识别
- 技术指标信号（基于当前数据推断）
- 是否出现技术破位或技术修复信号？
- 量价关系分析（价跌量增/量缩的含义）

【5. 市场情绪周期位置】
- 当前市场处于什么情绪阶段？（恐慌、观望、乐观、狂热）
- 情绪指标解读（涨跌家数比、涨跌停比、成交额变化）
- 情绪是否过度悲观或过度乐观？
- 情绪修复的可能性与时间窗口

【6. 风险提示】
- 识别当前市场的主要风险点（系统性风险、结构性风险、流动性风险等）
- 哪些信号需要警惕？（如：跌停数量增加、成交额异常放大/萎缩等）
- 潜在的黑天鹅或灰犀牛事件可能性
- 风险等级评估（低/中/高）

【7. 操作策略建议】
- 基于当前市场状态，给出仓位管理建议（加仓/减仓/观望）
- 适合的操作策略（短线/中线/长线）
- 板块配置建议（哪些板块可以关注，哪些需要回避）
- 止损止盈建议
- 适合不同风险偏好投资者的策略

【8. 第二天应盯哪些信号？】
- 基于当前市场状态，明确第二天需要重点关注的5-8个关键信号
- 包括但不限于：开盘情况、关键点位、成交量变化、板块轮动、资金流向、技术形态等
- 给出具体的观察指标和判断标准
- 如果出现什么信号，可能意味着什么？
- 不同信号组合的应对策略

请用专业但通俗易懂的语言，字数控制在1500-2000字，确保分析有深度、有逻辑、可操作。每个维度都要有具体的数据支撑和明确的结论。"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位专业的股票市场分析师，擅长分析市场情绪和行情数据。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=16384
            # 注意：该模型不支持自定义temperature参数，使用默认值1
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
    import glob
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
    parser = argparse.ArgumentParser(description="调用大模型分析市场数据")
    parser.add_argument("--date", type=str, default=None, help="指定交易日YYYYMMDD（优先级最高）")
    args = parser.parse_args()

    # 优先级：命令行 --date > 01脚本产出的 marker > 输出目录最新文件日期 > 今日
    trade_date = args.date or get_trade_date_from_marker() or get_latest_trade_date_from_files() or TARGET_DATE
    print(f"[日期] 使用交易日期: {trade_date}")
    
    # 检查数据文件是否存在
    daily_csv_path = os.path.join(OUT_DIR, f"daily_{trade_date}.csv")
    idx_csv_path = os.path.join(OUT_DIR, f"index_{trade_date}.csv")
    summary_csv_path = os.path.join(OUT_DIR, f"summary_{trade_date}.csv")
    
    if not os.path.exists(daily_csv_path):
        print(f"[错误] 找不到数据文件 {daily_csv_path}")
        print("提示: 请先运行 01_data_fetch.py 获取数据")
        return
    
    if not os.path.exists(summary_csv_path):
        print(f"[错误] 找不到数据文件 {summary_csv_path}")
        print("提示: 请先运行 01_data_fetch.py 获取数据")
        return
    
    # 读取数据
    print(f"正在读取 {trade_date} 的数据文件...")
    daily = pd.read_csv(daily_csv_path, encoding='utf-8-sig')
    summary = pd.read_csv(summary_csv_path, encoding='utf-8-sig')
    
    idx = pd.DataFrame()
    if os.path.exists(idx_csv_path):
        idx = pd.read_csv(idx_csv_path, encoding='utf-8-sig')
    else:
        print("[警告] 找不到指数数据文件，将跳过指数分析")
    
    print(f"[完成] 数据读取完成: daily {len(daily)}条, index {len(idx)}条, summary 1条")
    
    # 获取Tushare API（用于板块分析）
    pro = get_pro()
    
    # 调用大模型分析
    print("\n正在调用大模型分析市场数据...")
    analysis = analyze_summary_with_llm(summary, trade_date, idx, daily, pro)
    
    # 保存分析结果
    analysis_path = os.path.join(OUT_DIR, f"analysis_{trade_date}.txt")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(f"市场行情分析报告 - 交易日: {trade_date}\n")
        f.write("=" * 100 + "\n\n")
        f.write("市场情绪汇总数据：\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        if idx is not None and not idx.empty:
            f.write("\n三大指数情况：\n")
            f.write(idx.sort_values("ts_code").to_string(index=False))
            f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("AI分析结果：\n")
        f.write("=" * 100 + "\n\n")
        f.write(analysis)
    
    print(f"[完成] 已保存分析结果: {analysis_path}")
    
    # 打印分析结果预览
    print("\n" + "=" * 100)
    print("AI分析结果预览：")
    print("=" * 100)
    print(analysis[:500] + "..." if len(analysis) > 500 else analysis)


if __name__ == "__main__":
    main()

