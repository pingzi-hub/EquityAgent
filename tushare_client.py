"""
Tushare 统一初始化（课程/网关方式）。

用法与官方一致：先 pro_api(token)，再设置 pro._DataApi__http_url。
需要复权 K 线等可继续用：import tushare as ts; ts.pro_bar(api=get_pro(), ts_code="000001.SZ", ...)

Token：环境变量 TUSHARE_TOKEN（项目根目录 .env）。
网关：环境变量 TUSHARE_DATAAPI_URL；未设置时默认为教学网关。
"""
import os
from typing import Any, Optional

import tushare as ts
from dotenv import load_dotenv

# 默认 dataapi 网关（与 .env 中 TUSHARE_DATAAPI_URL 二选一）
_DEFAULT_DATAAPI_URL = "http://124.220.22.110:8020/"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_SCRIPT_DIR, ".env")

_pro = None


def _normalize_dataapi_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        u = _DEFAULT_DATAAPI_URL
    if not u.endswith("/"):
        u = u + "/"
    return u


def _new_pro() -> Optional[Any]:
    # override=True：避免系统/终端里已存在空 TUSHARE_TOKEN 时忽略 .env
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    token = (os.getenv("TUSHARE_TOKEN") or "").strip().strip('"').strip("'")
    if not token:
        return None
    base = _normalize_dataapi_url(os.getenv("TUSHARE_DATAAPI_URL") or _DEFAULT_DATAAPI_URL)
    pro = ts.pro_api(token)
    pro._DataApi__http_url = base
    return pro


def get_pro():
    """
    返回已配置网关的 Tushare pro 实例；未配置 TUSHARE_TOKEN 时退出进程。
    """
    global _pro
    if _pro is not None:
        return _pro
    pro = _new_pro()
    if pro is None:
        raise SystemExit(
            "未找到 TUSHARE_TOKEN：请在项目根目录 .env 写入 TUSHARE_TOKEN=你的token"
        )
    _pro = pro
    return pro


def get_pro_optional():
    """
    与 get_pro 相同网关逻辑；无 token 时返回 None 并打印警告（供 02 等可选 Tushare 场景）。
    """
    global _pro
    if _pro is not None:
        return _pro
    pro = _new_pro()
    if pro is None:
        print("警告: 未找到 TUSHARE_TOKEN，板块分析功能可能受限")
        return None
    _pro = pro
    return pro
