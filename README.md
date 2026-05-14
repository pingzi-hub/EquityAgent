# 📈 EquityAgent · A股智能分析与推荐 Agent

> **EquityAgent** 是一个面向 A 股市场的一体化智能分析系统。  
> 它将原本分散的 **数据抓取 → 市场分析 → 板块热度研判 → 个股推荐**  
> 整合为一套 **统一、自动化、可扩展的 Agent 工作流**，实现从数据到决策的闭环。<br/>
> 每次运行代码时，会自动删除之前的历史数据，避免历史数据占用内存。<br/>
> LSTM+GRU模型权重自己可以调整，目前采用0.7+0.3比例。

---

## 🚀 一键运行
在项目根目录执行：
python -m stock_agent
### 可选参数

- **指定数据日期**（仅影响数据抓取，格式 `YYYYMMDD`）：
python -m stock_agent --date 20260209
- **只运行指定步骤**（逗号分隔）：
python -m stock_agent --steps 01,02
---

## ⚙️ 环境变量配置

复制项目中的 `env.example` 文件为 `.env`，并填写以下配置  
（也可直接设置为系统环境变量）：
env
TUSHARE_TOKEN=你的Tushare令牌<br/>
AZURE_OPENAI_ENDPOINT=你的Azure OpenAI地址<br/>
AZURE_OPENAI_API_KEY=你的Azure OpenAI密钥<br/>
AZURE_OPENAI_DEPLOYMENT=gpt-5<br/>
AZURE_OPENAI_API_VERSION=2024-12-01-preview<br/>
> ⚠️ **安全提示**  
> `.env` 文件包含敏感信息，**请勿提交至代码仓库**，请确保已加入 `.gitignore`。

---

## 📁 核心目录与文件说明

| 文件 / 目录 | 说明 |
|------------|------|
| `01_data_fetch.py` | 提取 A 股每日行情、指数数据与市场情绪，输出至 `a_share_out/` |
| `02_analysis.py` | 基于交易数据，调用 Azure OpenAI 生成专业市场分析报告 |
| `03_stock_prediction.py` | 行业板块热度分析 + 个股综合评分 + AI 智能推荐报告 |
| `04_lstm_gru_prediction_enhanced.py` | 基于 LSTM/GRU/LSTM+GRU模型的深度时序预测脚本 |
| `05_hybrid_prediction.py` | 融合多因子与传统技术指标的综合预测脚本 |
| `stock_agent/` | 统一入口模块，用于后续 Agent 智能化扩展 |
---

## 📌 适用场景

- A 股每日复盘自动化  
- 行业板块轮动监测  
- 个股量化筛选与 AI 研判  
- LLM + 金融数据的 Agent 实践  

---

## 📄 License

本项目仅供研究与学习使用，不构成任何投资建议。
