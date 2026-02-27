# HerdLLM

> 基于 LLM-Agent 的金融市场羊群效应仿真系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-60a5fa.svg)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

HerdLLM 是一个基于大语言模型 (LLM) 的多智能体金融市场仿真系统。通过让 LLM 扮演具有不同人格的交易者，在社交网络中交互决策，研究**噪声交易**与**羊群效应**的微观形成机制。

核心能力：LLM 驱动的 Agent 决策 · 记忆反思与经验学习 · BA 无标度社交网络 · 2×2 对照实验 · Stylized Facts 验证 · 真实市场基准对比

<details>
<summary>目录</summary>

- [实验结果](#实验结果)
- [研究背景](#研究背景)
- [核心机制](#核心机制)
- [快速开始](#快速开始)
- [运行模式](#运行模式)
- [实验设计](#实验设计)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [引用](#引用)
- [License](#license)

</details>

---

## 实验结果

基于 **30 Agent × 100 轮 × 15 次 Monte Carlo** 的 2×2 对照实验：

| 指标 | Baseline | Full (Memory+Social) | 变化 |
|------|----------|----------------------|------|
| **Return Kurtosis** | 0.81 ± 0.44 | 5.87 ± 3.90 | **↑ 7.2×** |
| Herding Ratio | 74.7% ± 4.9% | 86.7% ± 5.9% | +12.0pp |
| Hurst Exponent | 0.92 ± 0.10 | 0.85 ± 0.14 | -0.07 |
| Volatility | 0.353% | 0.190% | -46% |

**关键发现：**
- 社交网络显著增强了收益率分布的**厚尾特性**（Kurtosis 提升 7.2 倍）
- 羊群效应在引入社交网络后显著增强（+10-12pp）
- 激进型 Agent 在所有条件下收益均优于保守型（+3.7% vs +3.3%）

### 图表展示

<p align="center">
  <img src="results/figures/hurst_comparison.png" width="420"/>
  <img src="results/figures/return_distributions.png" width="420"/>
</p>
<p align="center">
  <em>左：各条件 Hurst 指数对比 &nbsp;|&nbsp; 右：收益率分布与 QQ 图</em>
</p>

<p align="center">
  <img src="results/figures/herding_analysis.png" width="420"/>
  <img src="results/figures/network_topology.png" width="420"/>
</p>
<p align="center">
  <em>左：LSV 羊群指标分析 &nbsp;|&nbsp; 右：BA 无标度社交网络结构</em>
</p>

<p align="center">
  <img src="results/figures/stylized_facts.png" width="420"/>
  <img src="results/figures/real_market_comparison.png" width="420"/>
</p>
<p align="center">
  <em>左：Stylized Facts 金融典型事实验证 &nbsp;|&nbsp; 右：与真实市场 (SPY) 对比</em>
</p>

实验共生成 14 张分析图表，完整列表见 [`results/figures/`](results/figures/)。

---

## 研究背景

**噪声交易与羊群效应的微观起源是什么？**

传统金融理论假设投资者理性，但实证研究表明市场中存在显著的噪声交易和羊群行为。本项目通过 Agent-Based Model (ABM) 从微观层面研究这些现象的形成机制。

| 理论 | 核心观点 | 本项目对应 |
|------|----------|------------|
| **噪声交易者理论** (De Long et al., 1990) | 非理性交易者影响价格 | LLM-Agent 的有限理性决策 |
| **羊群行为理论** (Banerjee, 1992) | 信息级联导致从众 | 社交网络中的行为传播 |
| **行为金融学** (Kahneman & Tversky, 1979) | 认知偏差影响决策 | 记忆模块的经验反思 |
| **复杂系统理论** | 微观交互涌现宏观模式 | ABM 仿真框架 |

---

## 核心机制

### 价格影响模型

采用线性价格影响函数：

$$P_{t+1} = P_t \times \left(1 + \lambda \cdot \frac{N_{buy} - N_{sell}}{N_{total}}\right)$$

其中 $\lambda = 0.02$ 为价格影响系数，反映市场流动性。$\lambda$ 越大，市场越不稳定。

### Agent 记忆模块

每个 Agent 维护最近 20 条决策记录（回合、新闻、操作、盈亏等）。在新决策时，系统检索相似历史经验，生成反思 Prompt，例如：

> "你回顾历史，发现在类似利空新闻下，上次卖出后亏损了 $50。请谨慎决策。"

### 社交网络模型

使用 **Barabási-Albert (BA) 模型** 生成无标度网络（$n$ 个节点，每个新节点连接 $m=3$ 条边）。网络中少数"意见领袖"拥有大量连接，信息快速传播。Agent 决策时参考邻居的买入/卖出/持有比例。

### LLM 决策引擎

基于 Ollama 本地大模型（默认 `qwen2.5:7b`），Prompt 由四部分构成：

1. **System Prompt**：人格定义 + 交易规则
2. **市场状态**：当前价格、趋势
3. **历史反思**：记忆模块检索的相似经验
4. **社交情绪**：邻居的行为分布

输出格式：`JSON {action, quantity, reason}`

---

## 快速开始

### 环境要求

- Python >= 3.11、[Poetry](https://python-poetry.org/)、[Ollama](https://ollama.ai)

### 安装

```bash
git clone https://github.com/Candy-A-Mine/HerdLLM.git
cd HerdLLM
poetry install

# 安装并启动 Ollama，下载模型
ollama serve
ollama pull qwen2.5:7b
```

### 运行

```bash
# 演示模式（默认）- 20 Agent, 15 轮
poetry run python main.py

# 快速测试
poetry run python main.py --mode quick

# 标准实验 - 完整 4 条件对照
poetry run python main.py --mode standard

# 自定义参数
poetry run python main.py --mode custom --agents 50 --rounds 100 --runs 30

# 基于已有结果生成图表
poetry run python main.py --mode figures
```

LLM 配置默认连接 `localhost:11434`，可在 `src/config.py` 中修改。

---

## 运行模式

| 模式 | Agent数 | 回合数 | Monte Carlo | 实验条件 | 用途 |
|------|---------|--------|-------------|----------|------|
| demo | 20 | 15 | 1 | FULL | 功能演示 |
| quick | 20 | 15 | 3 | 2 种 | 快速调试 |
| standard | 50 | 100 | 30 | 全部 4 种 | 正式实验 |
| custom | 自定义 | 自定义 | 自定义 | 自定义 | 灵活配置 |
| figures | - | - | - | - | 生成图表 |

> 本次毕业论文实验配置：`--mode custom --agents 30 --rounds 100 --runs 15`（受算力限制采用精简参数）。

更多 API 用法和数据格式详见 [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)。

---

## 实验设计

### 2×2 对照实验

研究**记忆**和**社交网络**对市场行为的影响：

```
                    社交网络
                    禁用    启用
            ┌────────┬────────┐
    禁用    │BASELINE│SOCIAL_ │
记          │        │ ONLY   │
忆  ────────┼────────┼────────┤
    启用    │MEMORY_ │  FULL  │
            │ ONLY   │        │
            └────────┴────────┘
```

| 条件 | 记忆 | 社交网络 | 研究目的 |
|------|------|----------|----------|
| BASELINE | ✗ | ✗ | 对照组：纯 LLM 决策 |
| MEMORY_ONLY | ✓ | ✗ | 研究个体学习效应 |
| SOCIAL_ONLY | ✗ | ✓ | 研究羊群效应 |
| FULL | ✓ | ✓ | 研究协同效应 |

每个条件运行多次 Monte Carlo 仿真，计算均值、标准误和 95% 置信区间。

### Agent 人格类型

系统支持 4 种人格类型，实验中采用二元分布（各 50%）：

| 人格类型 | 行为特征 | 实验配置 |
|----------|----------|----------|
| 保守型 (Conservative) | 风险厌恶，倾向持有 | 15 (50%) |
| 激进型 (Aggressive) | 风险偏好，积极交易 | 15 (50%) |
| 趋势跟随型 (Trend_Follower) | 追涨杀跌，动量策略 | 未启用 |
| 羊群型 (Herding) | 跟随同伴，从众行为 | 未启用 |

> 完整系统支持 4 种人格的任意组合，可在 `src/config.py` 中配置。

### 模型验证

- **Stylized Facts 验证**：检验厚尾特性、波动聚集、长记忆性三项金融典型事实，详见 [`docs/STYLIZED_FACTS.md`](docs/STYLIZED_FACTS.md)
- **真实市场对比**：与 SPY（标普500 ETF）进行多维度对比，含相似度评分（0-100 分 + A/B/C/D 评级），详见 [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md#真实市场基准对比)

---

## 系统架构

### 目录结构

```
HerdLLM/
├── main.py                      # 主程序入口
├── pyproject.toml               # Poetry 依赖配置
├── poetry.lock                  # 锁定的依赖版本
├── LICENSE
├── dataset/
│   └── news_events.json         # 新闻事件数据集
├── src/
│   ├── agent.py                 # TraderAgent 交易智能体
│   ├── market.py                # Market 市场价格形成
│   ├── memory.py                # AgentMemory 记忆模块
│   ├── social_network.py        # SocialNetwork BA网络
│   ├── llm_client.py            # LLMClient Ollama通信
│   ├── config.py                # 实验配置
│   ├── experiment.py            # 实验运行框架
│   ├── metrics.py               # 金融指标计算
│   ├── stylized_facts.py        # Stylized Facts 验证
│   ├── real_market_benchmark.py # 真实市场基准对比
│   ├── visualization.py         # 可视化模块
│   ├── generate_figures.py      # 独立图表生成
│   └── analysis.py              # 统计分析
├── docs/
│   ├── UML_CLASS_DIAGRAM.md     # UML 类图
│   ├── API_REFERENCE.md         # API 参考与数据格式
│   └── STYLIZED_FACTS.md        # Stylized Facts 详细说明
└── results/
    └── figures/                 # 可视化图表（14 张）
```

### 模块依赖

```
main.py → config.py, experiment.py, visualization.py
              experiment.py → agent.py, market.py, metrics.py
                  agent.py → llm_client.py, memory.py, social_network.py
```

> 完整 UML 类图和数据流图见 [`docs/UML_CLASS_DIAGRAM.md`](docs/UML_CLASS_DIAGRAM.md)。

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.11+ | 主语言 |
| 包管理 | Poetry | 依赖管理 |
| LLM | Ollama + OpenAI SDK | 本地大模型 |
| 网络 | NetworkX | 社交网络建模 |
| 数据 | Pandas + PyArrow | 数据处理 |
| 计算 | NumPy + SciPy | 数值计算 |
| 可视化 | Matplotlib + Seaborn | 图表生成 |
| 金融数据 | yfinance | 真实市场数据 |

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{herdllm,
  title = {HerdLLM: LLM-Agent Based Financial Market Herding Simulation},
  author = {SuZX},
  year = {2025},
  url = {https://github.com/Candy-A-Mine/HerdLLM}
}
```

### 参考文献

1. Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223-236.
2. De Long, J. B., et al. (1990). Noise trader risk in financial markets. *Journal of Political Economy*, 98(4), 703-738.
3. Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509-512.
4. Banerjee, A. V. (1992). A simple model of herd behavior. *The Quarterly Journal of Economics*, 107(3), 797-817.

---

## License

MIT License — 详见 [LICENSE](LICENSE) 文件

---

<p align="center">
  <b>HerdLLM</b> — 探索金融市场的微观世界
</p>
