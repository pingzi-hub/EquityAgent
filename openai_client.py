"""
统一创建 OpenAI / Azure OpenAI 客户端，供 02/03/05 等脚本复用。

- 官方 Azure（endpoint 含 openai.azure.com）：AzureOpenAI，路径含 /openai/deployments/...
- 其它兼容网关（novai、自建反代等）：OpenAI(base_url=.../v1)，对齐 /v1/chat/completions
"""
import os

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_SCRIPT_DIR, ".env")


def get_azure_client():
    load_dotenv(dotenv_path=_ENV_PATH)

    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
    api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        return None, None, None

    try:
        if "openai.azure.com" in endpoint.lower():
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
        else:
            base_url = endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1"
            client = OpenAI(base_url=base_url, api_key=api_key)
        return client, deployment, api_version
    except Exception as e:
        print(f"创建 OpenAI 客户端失败: {e}")
        return None, None, None
