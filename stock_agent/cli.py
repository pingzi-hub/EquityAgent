import argparse
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class StepResult:
    name: str
    returncode: int


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _python_exe() -> str:
    return sys.executable or "python"


def _run_script(script_name: str, extra_args: Optional[list[str]] = None) -> StepResult:
    root = _repo_root()
    script_path = os.path.join(root, script_name) #拼接成绝对路径
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"未找到脚本: {script_path}")

    cmd = [_python_exe(), script_path]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(cmd, cwd=root) #在指定目录下运行命令
    return StepResult(name=script_name, returncode=proc.returncode)


def _latest_trade_date_marker_path() -> str:
    return os.path.join(_repo_root(), "a_share_out", "_latest_trade_date.txt")


def _read_latest_trade_date_from_marker() -> Optional[str]:
    p = _latest_trade_date_marker_path()
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            s = f.read().strip()
        return s if s else None
    except Exception:
        return None


def _latest_trade_date_from_daily_csv() -> Optional[str]:
    """从 a_share_out/daily_*.csv 文件名推断最新交易日（作为 marker 的兜底）。"""
    out_dir = os.path.join(_repo_root(), "a_share_out")
    if not os.path.isdir(out_dir):
        return None
    dates: list[str] = []
    for fp in glob.glob(os.path.join(out_dir, "daily_*.csv")):
        base = os.path.basename(fp)
        if base.startswith("daily_") and base.endswith(".csv"):
            d = base[6:-4]
            if len(d) == 8 and d.isdigit():
                dates.append(d)
    if not dates:
        return None
    dates.sort(reverse=True)
    return dates[0]


def _check_env(need_tushare: bool, need_azure: bool) -> None:
    missing = []
    if need_tushare and not os.getenv("TUSHARE_TOKEN"):
        missing.append("TUSHARE_TOKEN")
    if need_azure:
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            missing.append("AZURE_OPENAI_API_KEY")
    if missing:
        msg = "缺少环境变量: " + ", ".join(missing) + "。请在 .env 或系统环境变量中配置后再运行。"
        raise SystemExit(msg)


def main() -> None:
    # 从项目根目录加载 .env，确保入口脚本能读到配置
    load_dotenv(dotenv_path=os.path.join(_repo_root(), ".env"))

    parser = argparse.ArgumentParser(description="股票分析与推荐 Agent（统一入口）")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="目标日期YYYYMMDD。仅作用于数据抓取（01_data_fetch.py）。不传则由脚本自行取今日并回退到最近交易日。",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="01,02,03",
        help="要执行的步骤，逗号分隔：01,02,03,04,05。默认 01,02,03。",
    )
    parser.add_argument(
        "--stock",
        type=str,
        default=None,
        help="股票代码（仅作用于 05_hybrid_prediction.py），如 000001.SZ",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["lstm", "gru"],
        help="模型类型（仅作用于 04_lstm_gru_prediction_enhanced.py），默认 lstm",
    )
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    # 依赖检查：01需要tushare；02/03需要azure +（03也需要tushare）
    need_tushare = any(s in {"01", "03", "04", "05"} for s in steps)
    need_azure = any(s in {"02", "03", "05"} for s in steps)
    _check_env(need_tushare=need_tushare, need_azure=need_azure)

    results: list[StepResult] = []

    # 与 01 实际落盘的 trade_date 对齐：禁止在 marker 丢失时用 args.date 顶替（args 是“目标日”，未必等于最终交易日）。
    forced_date: Optional[str] = None

    if "01" in steps:
        extra = ["--date", args.date] if args.date else None
        results.append(_run_script("01_data_fetch.py", extra_args=extra))
        if results[-1].returncode != 0:
            raise SystemExit(f"步骤01失败（{results[-1].returncode}），已中止。")
        forced_date = _read_latest_trade_date_from_marker() or _latest_trade_date_from_daily_csv()
        if not forced_date:
            raise SystemExit(
                "步骤01未产出有效交易日：未读到 a_share_out/_latest_trade_date.txt，且无法从 daily_*.csv 推断。\n"
                "常见原因：Tushare 返回 token invalid / 网络失败，01 已以非 0 退出；若仍见本提示，请检查项目根目录 .env 与 tushare_client 网关。\n"
                "建议：在项目根执行  python -m stock_agent  （勿依赖已删空的 a_share_out）。"
            )
    else:
        forced_date = _read_latest_trade_date_from_marker() or _latest_trade_date_from_daily_csv() or args.date

    if "02" in steps:
        extra = ["--date", forced_date] if forced_date else None
        results.append(_run_script("02_analysis.py", extra_args=extra))
        if results[-1].returncode != 0:
            raise SystemExit(f"步骤02失败（{results[-1].returncode}），已中止。")

    if "03" in steps:
        extra = ["--date", forced_date] if forced_date else None
        results.append(_run_script("03_stock_prediction.py", extra_args=extra))
        if results[-1].returncode != 0:
            raise SystemExit(f"步骤03失败（{results[-1].returncode}），已中止。")

    if "04" in steps:
        extra: list[str] = ["--mode", "2"]
        if args.model_type:
            extra.extend(["--model-type", args.model_type])
        results.append(_run_script("04_lstm_gru_prediction_enhanced.py", extra_args=extra))
        if results[-1].returncode != 0:
            raise SystemExit(f"步骤04失败（{results[-1].returncode}），已中止。")

    if "05" in steps:
        if not args.stock:
            raise SystemExit("步骤05需要指定 --stock 参数（如 --stock 000001.SZ）")
        extra = ["--stock", args.stock]
        results.append(_run_script("05_hybrid_prediction.py", extra_args=extra))
        if results[-1].returncode != 0:
            raise SystemExit(f"步骤05失败（{results[-1].returncode}），已中止。")

    print("\n执行完成：")
    for r in results:
        print(f"- {r.name}: returncode={r.returncode}")


if __name__ == "__main__":
    main()

