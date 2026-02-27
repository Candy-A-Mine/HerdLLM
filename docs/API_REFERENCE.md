# API 参考

## SimulationConfig

```python
@dataclass
class SimulationConfig:
    # 基础参数
    num_agents: int = 50              # Agent 数量
    num_rounds: int = 100             # 仿真回合数
    initial_price: float = 100.0      # 初始价格
    initial_cash: float = 10000.0     # 初始现金
    initial_holdings: int = 50        # 初始持仓
    trade_size: int = 10              # 每次交易股数

    # 市场参数
    price_impact: float = 0.02        # 价格影响系数

    # 功能开关
    enable_memory: bool = True        # 启用记忆模块
    enable_social_network: bool = True # 启用社交网络
    memory_capacity: int = 20         # 记忆容量
    social_network_m: int = 3         # BA网络m参数

    # LLM 参数
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"

    # 随机种子
    seed: int | None = None
```

## 预设配置函数

| 函数 | 用途 | 配置特点 |
|------|------|----------|
| `get_quick_test_config()` | 快速测试 | 20 Agent, 15 回合, 3 次运行 |
| `get_standard_config()` | 标准实验 | 50 Agent, 100 回合, 30 次运行 |
| `get_large_scale_config()` | 大规模实验 | 100 Agent, 200 回合, 50 次运行 |
| `get_herding_focus_config()` | 羊群研究 | 40% 羊群型 Agent, m=5 |

## Python API 调用

```python
from src.config import SimulationConfig, get_standard_config
from src.experiment import ExperimentRunner

# 方式1: 使用预设配置
config = get_standard_config()
runner = ExperimentRunner(config)
results = runner.run()

# 方式2: 自定义配置
config = SimulationConfig(
    num_agents=100,
    num_rounds=200,
    enable_memory=True,
    enable_social_network=True,
    price_impact=0.03,
)
```

## Parquet 数据字段

实验决策数据以 Parquet 格式存储，字段如下：

| 字段 | 类型 | 说明 |
|------|------|------|
| `round_num` | int | 回合号 |
| `agent_id` | int | Agent ID |
| `personality` | str | 人格类型 |
| `cash_before` / `cash_after` | float | 决策前后现金 |
| `holdings_before` / `holdings_after` | int | 决策前后持仓 |
| `portfolio_value_before` / `portfolio_value_after` | float | 组合价值 |
| `news` | str | 当轮新闻 |
| `current_price` | float | 当前价格 |
| `action` | str | BUY / SELL / HOLD |
| `quantity` | int | 交易数量 |
| `reason` | str | LLM 决策理由 |
| `memory_reflection` | str | 历史反思内容 |
| `social_buy_pct` / `social_sell_pct` / `social_hold_pct` | float | 社交网络情绪 |

## 数据分析示例

```python
import pandas as pd

# 读取决策数据
df = pd.read_parquet("results/demo/decisions.parquet")

# 各人格类型的平均收益
df.groupby("personality")["portfolio_value_after"].mean()

# 羊群型 Agent 的社交跟随率
herding = df[df["personality"] == "Herding"]
follow_rate = (herding["action"] == herding["social_majority"]).mean()

# 按回合统计买卖分布
df.groupby(["round_num", "action"]).size().unstack()
```

## 真实市场基准对比

将仿真数据与真实市场数据 (默认 SPY - 标普500 ETF) 进行多维度对比：

| 对比维度 | 指标 |
|----------|------|
| 基本统计 | 均值、标准差、偏度、峰度 |
| 风险指标 | VaR (5%)、最大回撤 |
| 分布检验 | Kolmogorov-Smirnov 检验 |
| 相似度评分 | 0-100 分 + A/B/C/D 评级 |

### 相似度评分体系

| 维度 | 权重 | 评分标准 |
|------|------|----------|
| 波动率相似度 | 30% | min(σ_sim, σ_real) / max(σ_sim, σ_real) |
| 峰度相似度 | 25% | 超额峰度比较 |
| 偏度相似度 | 20% | 偏度差异 |
| VaR相似度 | 25% | VaR比较 |

**评级标准：**
- A (Excellent): ≥ 80 分
- B (Good): ≥ 60 分
- C (Fair): ≥ 40 分
- D (Poor): < 40 分

### 使用方法

```python
from src.real_market_benchmark import compare_with_real_market

results = compare_with_real_market(
    sim_prices=simulation_results["price_history"],
    ticker="SPY",           # 标普500 ETF
    period="1y",            # 最近1年数据
    output_path="results/comparison.png",
    print_report=True,
)

# 访问结果
print(f"相似度总分: {results['similarity_score']['total_score']:.1f}")
print(f"评级: {results['similarity_score']['grade']}")
```
