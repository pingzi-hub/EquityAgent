# 01_data_fetch.py
# 功能：获取A股市场数据
# 1) 自动从目标日期往前找最近一个A股交易日
# 2) 拉取该交易日：全A股日行情、三大指数日行情
# 3) 生成市场情绪汇总（涨跌家数、估算涨跌停、成交额）
# 4) 保存所有数据到文件
#
# 依赖：
# pip install tushare pandas python-dotenv

import os
import sys
import argparse
import glob
import pandas as pd

from tushare_client import get_pro
from datetime import datetime, timedelta

# 自动获取当日日期（格式：YYYYMMDD）
# 如果当日不是交易日，会自动回退到最近交易日
# 可以通过命令行参数 --date 指定日期，例如：python 01_data_fetch.py --date 20260209
TARGET_DATE = datetime.now().strftime("%Y%m%d")  # 自动获取今日日期

# 输出目录：保存到脚本所在目录下的 a_share_out 文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "a_share_out")
LATEST_TRADE_DATE_PATH = os.path.join(OUT_DIR, "_latest_trade_date.txt")


def clear_previous_a_share_outputs() -> None:
    """每次执行前删除 a_share_out 中由本流程生成的旧文件，避免多交易日文件混放。"""
    if not os.path.isdir(OUT_DIR):
        return
    patterns = (
        "daily_*.csv",
        "daily_*.txt",
        "index_*.csv",
        "index_*.txt",
        "summary_*.csv",
        "summary_*.txt",
        "analysis_*.txt",
        "_latest_trade_date.txt",
    )
    n = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(OUT_DIR, pattern)):
            try:
                os.remove(path)
                n += 1
            except OSError:
                pass
    if n:
        print(f"[清理] 已删除 a_share_out 内 {n} 个旧输出文件。")


def get_last_open_date(pro, target_date: str, exchange="SSE", lookback_days=60) -> str:
    """
    从 target_date 往前找最近交易日。lookback_days 给足够窗口避免长假。
    如果 target_date 不是交易日，会自动往前找最近的交易日。
    如果 target_date 是未来日期，会自动回退到最近的已过去交易日。
    """
    try:
        # 使用 datetime 进行正确的日期计算
        target_dt = datetime.strptime(target_date, "%Y%m%d")
        start_dt = target_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y%m%d")
        
        # 获取当前日期
        today = datetime.now()
        today_str = today.strftime("%Y%m%d")
        
        # 确保 end_date 不超过今天（使用日期对象比较更安全）
        if target_dt > today:
            end_date = today_str
            print(f"提示: target_date {target_date} 是未来日期，已自动调整为今天 {end_date}")
        else:
            end_date = target_date
        
        cal = pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date,
                            fields="cal_date,is_open")
        if cal is None or cal.empty:
            raise ValueError(f"交易日历查询返回空，请检查日期格式或API权限")
        cal = cal.sort_values("cal_date")
        open_days = cal.loc[cal["is_open"] == 1, "cal_date"]
        if open_days.empty:
            raise ValueError(f"在区间 {start_date}~{end_date} 未找到交易日，请增大 lookback 或检查日期格式")
        
        # 检查目标日期是否是交易日（确保类型一致）
        open_days_str = [str(d) for d in open_days.values]
        target_is_trade_day = str(target_date) in open_days_str
        
        # 确保返回的日期不超过今天（过滤掉未来日期）
        # 将 open_days 转换为字符串格式以确保比较正确
        open_days_str_series = open_days.astype(str)
        open_days_past = open_days_str_series[open_days_str_series <= today_str]
        if open_days_past.empty:
            raise ValueError(f"在区间 {start_date}~{end_date} 未找到已过去的交易日")
        
        # 返回最近的已过去交易日
        last_trade_date = str(open_days_past.iloc[-1])
        
        # 如果目标日期不是交易日，给出提示
        if not target_is_trade_day and target_date == today_str:
            print(f"提示: 今天 {target_date} 不是交易日，已自动选择最近的交易日 {last_trade_date}")
        elif not target_is_trade_day:
            print(f"提示: {target_date} 不是交易日，已自动选择最近的交易日 {last_trade_date}")
        elif target_date == today_str and last_trade_date == target_date:
            print(f"提示: 今天 {target_date} 是交易日")
        
        return last_trade_date
    except Exception as e:
        print(f"获取交易日失败: {e}")
        raise


def get_prev_open_date(pro, trade_date: str, exchange="SSE", lookback_days=200) -> str | None:
    """
    获取严格早于 trade_date 的上一个交易日。
    """
    try:
        trade_dt = datetime.strptime(trade_date, "%Y%m%d")
        start_dt = trade_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y%m%d")

        cal = pro.trade_cal(exchange=exchange, start_date=start_date, end_date=trade_date, fields="cal_date,is_open")
        if cal is None or cal.empty:
            return None
        cal = cal.sort_values("cal_date")
        open_days = cal.loc[cal["is_open"] == 1, "cal_date"].astype(str)
        prev_days = open_days[open_days < str(trade_date)]
        if prev_days.empty:
            return None
        return str(prev_days.iloc[-1])
    except Exception:
        return None


def save_latest_trade_date(trade_date: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LATEST_TRADE_DATE_PATH, "w", encoding="utf-8") as f:
        f.write(str(trade_date).strip() + "\n")


def fetch_all_a_daily(pro, trade_date: str) -> pd.DataFrame:
    """
    获取指定交易日的全A股日行情数据
    """
    try:
        # 检查是否是今天
        today_str = datetime.now().strftime("%Y%m%d")
        is_today = trade_date == today_str
        
        daily = pro.daily(
            trade_date=trade_date,
            fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        )
        if daily is None or daily.empty:
            if is_today:
                print(f"警告: 今天 {trade_date} 的 daily 数据为空")
                print(f"  可能原因：数据尚未更新（通常在交易日收盘后几小时更新）")
            else:
                print(f"警告: trade_date={trade_date} 的 daily 数据为空")
        else:
            print(f"成功获取 {len(daily)} 条股票日行情数据")
        return daily if daily is not None else pd.DataFrame()
    except Exception as e:
        print(f"获取 daily 数据失败: {e}")
        # 如果是权限错误，给出更明确的提示
        if "权限" in str(e) or "积分" in str(e) or "permission" in str(e).lower():
            print("  提示: 可能是 Tushare token 权限不足，需要足够的积分才能获取该数据")
        return pd.DataFrame()


def fetch_index_daily(pro, trade_date: str) -> pd.DataFrame:
    """
    获取指定交易日的三大指数日行情数据
    """
    # 检查是否是今天
    today_str = datetime.now().strftime("%Y%m%d")
    is_today = trade_date == today_str
    
    index_codes = ["000001.SH", "399001.SZ", "399006.SZ"]  # 上证、深成、创业板
    frames = []
    for code in index_codes:
        try:
            df = pro.index_daily(
                ts_code=code,
                trade_date=trade_date,
                fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
            )
            if df is not None and not df.empty:
                frames.append(df)
                print(f"成功获取指数 {code} 的数据")
            else:
                print(f"警告: 指数 {code} 在 {trade_date} 的数据为空")
        except Exception as e:
            print(f"获取指数 {code} 数据失败: {e}")
            # 如果是权限错误，给出更明确的提示
            if "权限" in str(e) or "积分" in str(e) or "permission" in str(e).lower():
                print(f"  提示: 可能是 Tushare token 权限不足，需要足够的积分才能获取指数数据")
    idx = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if idx.empty:
        if is_today:
            print(f"警告: 今天 {trade_date} 的所有指数数据都为空")
            print(f"  可能原因：数据尚未更新（通常在交易日收盘后几小时更新）")
        else:
            print(f"警告: 所有指数在 {trade_date} 的数据都为空")
    return idx


def make_summary(daily: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame([{
            "trade_date": trade_date,
            "stocks": 0, "up": 0, "down": 0, "flat": 0,
            "amt_total_yi_est": 0.0,
            "limit_up_10_est": 0, "limit_dn_10_est": 0,
            "limit_up_20_est": 0, "limit_dn_20_est": 0
        }])

    up = int((daily["pct_chg"] > 0).sum())
    down = int((daily["pct_chg"] < 0).sum())
    flat = int((daily["pct_chg"] == 0).sum())

    # 估算涨跌停：不同板块涨跌停规则不同，这里仅用阈值粗估
    limit_up_10 = int((daily["pct_chg"] >= 9.8).sum())
    limit_dn_10 = int((daily["pct_chg"] <= -9.8).sum())
    limit_up_20 = int((daily["pct_chg"] >= 19.6).sum())
    limit_dn_20 = int((daily["pct_chg"] <= -19.6).sum())

    # amount口径在不同数据源/字段可能是元/千元；这里先按"元"->亿元估算
    # 若你发现明显不对，把 daily['amount'].head() 发我，我帮你校准
    amt_total_yi_est = float(daily["amount"].sum() / 1e8)

    return pd.DataFrame([{
        "trade_date": trade_date,
        "stocks": int(len(daily)),
        "up": up, "down": down, "flat": flat,
        "amt_total_yi_est": amt_total_yi_est,
        "limit_up_10_est": limit_up_10,
        "limit_dn_10_est": limit_dn_10,
        "limit_up_20_est": limit_up_20,
        "limit_dn_20_est": limit_dn_20,
    }])


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='获取A股市场数据')
    parser.add_argument('--date', type=str, default=None,
                        help='指定目标日期（格式：YYYYMMDD），例如：20260209。如果不指定，则使用当前日期')
    args = parser.parse_args()
    
    # 确定目标日期
    if args.date:
        target_date = args.date
        # 验证日期格式
        try:
            datetime.strptime(target_date, "%Y%m%d")
        except ValueError:
            print(f"错误: 日期格式不正确，应为 YYYYMMDD 格式，例如：20260209")
            sys.exit(1)
    else:
        target_date = TARGET_DATE
    
    print(f"使用目标日期: {target_date}")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    pro = get_pro()

    # 1) 自动选择最近交易日（须先成功再清空旧输出，避免 token 无效时已删光 a_share_out）
    try:
        trade_date = get_last_open_date(pro, target_date, exchange="SSE", lookback_days=200)
        print(f"TARGET_DATE={target_date}  -> use trade_date={trade_date}")
    except Exception as e:
        print(f"获取交易日失败: {e}")
        print("提示: 若含 token invalid，请检查 .env 中 TUSHARE_TOKEN 与 tushare_client 网关；")
        print("      若日期是未来日，请改用最近真实交易日或省略 --date。")
        sys.exit(1)

    clear_previous_a_share_outputs()

    # 2) 拉数据
    tried_dates = []
    for attempt in range(3):
        tried_dates.append(trade_date)
        print(f"\n开始获取 {trade_date} 的数据...")
        daily = fetch_all_a_daily(pro, trade_date)
        idx = fetch_index_daily(pro, trade_date)

        # 如果当日/目标交易日数据为空，优先回退到上一个交易日再试
        if daily is not None and not daily.empty:
            break

        prev_date = get_prev_open_date(pro, trade_date, exchange="SSE", lookback_days=400)
        if not prev_date:
            break
        if prev_date == trade_date:
            break

        print(f"[提示] {trade_date} 数据未就绪或为空，自动回退到上一个交易日 {prev_date} 重试。")
        trade_date = prev_date

    # 记录步骤01最终使用的 trade_date，供步骤02/03对齐
    save_latest_trade_date(trade_date)
    
    # 检查数据是否为空
    today_str = datetime.now().strftime("%Y%m%d")
    is_today = trade_date == today_str
    
    if daily.empty:
        print("\n[警告] daily 数据为空！")
        if is_today:
            print("  可能原因：")
            print("  1. 今天是交易日，但数据尚未更新（通常在收盘后几小时更新）")
            print("  2. Tushare token 权限不足（需要积分）")
            print("\n建议：")
            print("  - 如果是交易日收盘后，请稍后再试（数据通常在收盘后2-4小时更新）")
            print("  - 检查 Tushare 账户积分是否足够")
            print("  - 可以尝试使用昨天的日期获取数据")
        else:
            print("  可能原因：")
            print("  1. 该日期不是交易日或数据尚未更新")
            print("  2. Tushare token 权限不足（需要积分）")
            print("  3. 日期格式或API调用问题")
            print("\n建议：")
            print("  - 使用最近的真实交易日（如今天或昨天）")
            print("  - 检查 Tushare 账户积分是否足够")
            print("  - 尝试手动调用 pro.daily() 测试")
    
    if idx.empty:
        if is_today:
            print("\n[警告] index 数据为空！可能是数据尚未更新")
        else:
            print("\n[警告] index 数据为空！")

    # 3) 输出文件路径
    daily_txt_path = os.path.join(OUT_DIR, f"daily_{trade_date}.txt")
    daily_csv_path = os.path.join(OUT_DIR, f"daily_{trade_date}.csv")
    idx_txt_path = os.path.join(OUT_DIR, f"index_{trade_date}.txt")
    idx_csv_path = os.path.join(OUT_DIR, f"index_{trade_date}.csv")
    summary_txt_path = os.path.join(OUT_DIR, f"summary_{trade_date}.txt")
    summary_csv_path = os.path.join(OUT_DIR, f"summary_{trade_date}.csv")

    # 保存daily数据为TXT和CSV
    if not daily.empty:
        with open(daily_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"全A股日行情数据 - 交易日: {trade_date}\n")
            f.write("=" * 100 + "\n\n")
            f.write(daily.to_string(index=False))
            f.write("\n\n")
            f.write(f"总计: {len(daily)} 条记录\n")
        daily.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")
        print(f"已保存 daily 数据: {daily_txt_path}, {daily_csv_path}")
    else:
        with open(daily_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"全A股日行情数据 - 交易日: {trade_date}\n")
            f.write("=" * 100 + "\n\n")
            f.write("(无数据)\n")

    # 保存index数据为TXT和CSV
    if not idx.empty:
        with open(idx_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"三大指数日行情数据 - 交易日: {trade_date}\n")
            f.write("=" * 100 + "\n\n")
            f.write(idx.sort_values("ts_code").to_string(index=False))
            f.write("\n\n")
            f.write(f"总计: {len(idx)} 条记录\n")
        idx.to_csv(idx_csv_path, index=False, encoding="utf-8-sig")
        print(f"已保存 index 数据: {idx_txt_path}, {idx_csv_path}")
    else:
        with open(idx_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"三大指数日行情数据 - 交易日: {trade_date}\n")
            f.write("=" * 100 + "\n\n")
            f.write("(无数据)\n")

    # 保存summary数据为TXT和CSV
    summary = make_summary(daily, trade_date)
    
    # 保存TXT格式
    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"市场情绪汇总 - 交易日: {trade_date}\n")
        f.write("=" * 100 + "\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        f.write("说明:\n")
        f.write("- stocks: 股票总数\n")
        f.write("- up/down/flat: 上涨/下跌/平盘家数\n")
        f.write("- amt_total_yi_est: 总成交额估算(亿元)\n")
        f.write("- limit_up_10_est/limit_dn_10_est: 10%涨跌停估算家数\n")
        f.write("- limit_up_20_est/limit_dn_20_est: 20%涨跌停估算家数\n")
    
    # 保存CSV格式
    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"已保存 summary 数据: {summary_txt_path}, {summary_csv_path}")

    # 4) 关键输出
    print("\nSUMMARY:")
    print(summary.to_string(index=False))

    print("\nINDEX:")
    if idx.empty:
        print("(empty)")
    else:
        print(idx.sort_values("ts_code").to_string(index=False))

    print("\n[完成] 数据获取完成！")
    print(f"保存的文件：")
    print(f"  - {daily_txt_path}")
    print(f"  - {daily_csv_path}")
    print(f"  - {idx_txt_path}")
    print(f"  - {idx_csv_path}")
    print(f"  - {summary_txt_path}")
    print(f"  - {summary_csv_path}")
    print("\n提示：运行 02_analysis.py 进行大模型分析")


if __name__ == "__main__":
    main()

