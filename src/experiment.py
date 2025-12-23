"""
实验框架模块 (Experiment Framework Module)

本模块实现了运行受控ABM仿真实验的完整框架。

核心功能：
=========
1. 蒙特卡洛仿真：多次独立运行，统计分析结果的稳定性
2. 受控对比实验：基线 vs 处理组（2×2实验设计）
3. 参数敏感性分析：研究参数变化对结果的影响
4. 并行执行：支持异步并行LLM调用，提升效率

实验流程架构：
=============
    ExperimentRunner (实验管理器)
        │
        ├── Condition 1: baseline
        │   ├── Run 1 → SimulationRunner
        │   ├── Run 2 → SimulationRunner
        │   └── ...
        │
        ├── Condition 2: memory_only
        │   └── ...
        │
        └── ...

    SimulationRunner (单次仿真)
        │
        ├── Round 1
        │   ├── 新闻选择
        │   ├── Agent决策 (并行LLM调用)
        │   ├── 订单处理
        │   └── 记忆更新
        │
        ├── Round 2
        │   └── ...
        └── ...

模块类：
=======
- SimulationRunner: 执行单次仿真，管理Agent、市场、社交网络
- ExperimentRunner: 执行完整实验，支持多条件多次运行
- SensitivityAnalyzer: 参数敏感性分析

作者: SuZX
日期: 2024
"""

import asyncio
import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.agent import DecisionRecord, TraderAgent
from src.config import (
    ExperimentCondition,
    ExperimentConfig,
    SensitivityConfig,
    SimulationConfig,
)
from src.llm_client import LLMClient
from src.market import Market
from src.memory import AgentMemory
from src.metrics import calculate_all_metrics, format_metrics_report
from src.social_network import SocialNetwork


# =============================================================================
# 新闻事件池
# =============================================================================

# 预定义的市场新闻事件列表
# 用于在每轮仿真中随机选择，模拟真实市场的信息流
NEWS_EVENTS = [
    # 货币政策类新闻
    "Central Bank announces surprise interest rate hike of 0.5%",
    "Federal Reserve signals potential rate cuts in coming months",
    "Central Bank maintains current interest rate, markets stable",

    # 企业盈利类新闻
    "Major tech company reports record-breaking quarterly earnings",
    "Corporate earnings season shows mixed results across sectors",

    # 宏观经济类新闻
    "GDP growth falls below market expectations at 1.2%",
    "Inflation data comes in higher than forecasted at 4.5%",
    "Unemployment rate drops to historic low of 3.5%",
    "Consumer confidence index reaches 5-year high",
    "Strong retail sales data beats analyst expectations",
    "Manufacturing sector contracts for third consecutive month",
    "Strong jobs report boosts market sentiment",

    # 地缘政治类新闻
    "Trade tensions escalate between major economies",
    "Oil prices surge 10% due to geopolitical tensions",

    # 行业动态类新闻
    "Breakthrough in AI technology drives tech sector optimism",
    "Housing market shows signs of cooling, prices down 3%",
    "Cryptocurrency market sees major volatility, down 15%",

    # 政策类新闻
    "New fiscal stimulus package announced by government",
    "Supply chain disruptions ease, commodity prices stabilize",

    # 市场信号类新闻
    "Bond yields invert, raising recession concerns",
]


# =============================================================================
# 仿真运行器
# =============================================================================


class SimulationRunner:
    """
    单次仿真运行器

    管理单次仿真的完整生命周期，包括：
    - 组件初始化（LLM客户端、市场、Agent、社交网络）
    - 仿真循环执行（轮次迭代、决策收集、订单处理）
    - 结果编译和指标计算

    仿真流程：
    =========
    ```
    setup() → 初始化所有组件
        ↓
    run() → 执行仿真循环
        │
        ├── Round 0
        │   ├── 选择新闻事件
        │   ├── 所有Agent做决策 (可并行)
        │   ├── 市场处理订单
        │   └── 更新Agent记忆的PnL
        │
        ├── Round 1
        │   └── ...
        │
        └── Round N-1
            ↓
    _compile_results() → 编译结果和计算指标
    ```

    Attributes:
        config: 仿真配置对象
        llm_client: LLM客户端（用于Agent决策）
        market: 市场对象（价格形成和订单处理）
        agents: Agent列表
        social_network: 社交网络（可选）
        decision_records: 决策记录列表

    Example:
        >>> config = SimulationConfig(num_agents=30, num_rounds=100)
        >>> runner = SimulationRunner(config)
        >>> results = runner.run(parallel=True)
        >>> print(f"Final price: {results['market_stats']['final_price']}")
    """

    def __init__(self, config: SimulationConfig):
        """
        初始化仿真运行器

        Args:
            config: 仿真配置对象，包含所有仿真参数
        """
        self.config = config

        # 组件引用（在 setup() 中初始化）
        self.llm_client: LLMClient | None = None
        self.market: Market | None = None
        self.agents: list[TraderAgent] = []
        self.social_network: SocialNetwork | None = None

        # 决策记录存储
        self.decision_records: list[DecisionRecord] = []

    def setup(self) -> None:
        """
        初始化所有仿真组件

        执行顺序：
        1. 设置随机种子（确保可重复性）
        2. 初始化LLM客户端
        3. 初始化社交网络（如果启用）
        4. 初始化市场
        5. 创建Agent

        Note:
            必须在 run() 之前调用
        """
        # =====================================================================
        # 步骤1: 设置随机种子（确保Monte Carlo变异性的同时保持可重复性）
        # =====================================================================
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # =====================================================================
        # 步骤2: 初始化LLM客户端
        # =====================================================================
        self.llm_client = LLMClient(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
            seed=self.config.seed,  # 传递种子确保Monte Carlo变异性
        )

        # =====================================================================
        # 步骤3: 初始化社交网络（如果启用）
        # 使用 Barabasi-Albert 模型生成无标度网络
        # =====================================================================
        if self.config.enable_social_network:
            self.social_network = SocialNetwork(num_agents=self.config.num_agents)
        else:
            self.social_network = None

        # =====================================================================
        # 步骤4: 初始化市场
        # =====================================================================
        self.market = Market(
            initial_price=self.config.initial_price,
            price_impact=self.config.price_impact,
            num_agents=self.config.num_agents,
        )

        # =====================================================================
        # 步骤5: 创建Agent
        # =====================================================================
        self.agents = self._create_agents()

        # 重置决策记录
        self.decision_records = []

    def _create_agents(self) -> list[TraderAgent]:
        """
        根据配置创建Agent

        创建流程：
        1. 根据 personality_distribution 生成人格类型列表
        2. 调整列表长度以匹配 num_agents
        3. 打乱顺序（避免人格聚集）
        4. 为每个人格创建Agent实例

        Returns:
            创建的Agent列表
        """
        agents = []

        # ---------------------------------------------------------------------
        # 步骤1: 构建人格类型列表
        # personality_distribution 示例: {"Conservative": 8, "Aggressive": 7, ...}
        # ---------------------------------------------------------------------
        personalities = []
        for personality, count in self.config.personality_distribution.items():
            personalities.extend([personality] * count)

        # ---------------------------------------------------------------------
        # 步骤2: 调整列表长度以匹配目标Agent数量
        # 如果人格分布总数不足，随机补充；如果过多，截断
        # ---------------------------------------------------------------------
        while len(personalities) < self.config.num_agents:
            personalities.append(random.choice(list(self.config.personality_distribution.keys())))
        personalities = personalities[: self.config.num_agents]

        # 打乱顺序，避免同一人格的Agent在网络中聚集
        random.shuffle(personalities)

        # ---------------------------------------------------------------------
        # 步骤3: 为每个人格创建Agent实例
        # ---------------------------------------------------------------------
        for i, personality in enumerate(personalities):
            # 创建记忆模块（如果启用）
            if self.config.enable_memory:
                memory = AgentMemory(max_memories=self.config.memory_capacity)
            else:
                # 禁用记忆：设置容量为0
                memory = AgentMemory(max_memories=0)

            # 创建Agent实例
            agent = TraderAgent(
                id=i,
                cash=self.config.initial_cash,
                holdings=self.config.initial_holdings,
                personality=personality,
                llm_client=self.llm_client,
                trade_size=self.config.trade_size,
                memory=memory,
                # 只在启用社交网络时传入网络引用
                social_network=self.social_network if self.config.enable_social_network else None,
            )
            agents.append(agent)

        return agents

    def run(self, progress_bar: bool = True, parallel: bool = True) -> dict:
        """
        执行仿真

        支持两种执行模式：
        1. 并行模式（推荐）：使用 asyncio 并行调用LLM，大幅提升效率
        2. 同步模式：顺序执行Agent决策，用于调试

        Args:
            progress_bar: 是否显示进度条
            parallel: 是否使用并行LLM调用（默认True）

        Returns:
            包含仿真结果和指标的字典
        """
        # 初始化组件
        self.setup()

        try:
            # 根据模式选择执行方法
            if parallel:
                # 并行模式：使用 asyncio 事件循环
                return asyncio.run(self._run_async(progress_bar))
            else:
                # 同步模式：顺序执行
                return self._run_sync(progress_bar)
        finally:
            # ================================================================
            # 清理：重置异步客户端，避免 "Event loop is closed" 错误
            # 这在多次运行时尤其重要
            # ================================================================
            if self.llm_client is not None:
                self.llm_client.reset_async_client()

    def _run_sync(self, progress_bar: bool = True) -> dict:
        """
        同步执行仿真（顺序调用LLM）

        此方法按顺序执行每个Agent的决策，适用于：
        - 调试和问题排查
        - 不支持异步的环境

        Args:
            progress_bar: 是否显示进度条

        Returns:
            仿真结果字典
        """
        # 创建轮次迭代器
        iterator = range(self.config.num_rounds)
        if progress_bar:
            iterator = tqdm(iterator, desc="Simulation", leave=False)

        # =====================================================================
        # 主仿真循环：逐轮执行
        # =====================================================================
        for round_num in iterator:
            # -----------------------------------------------------------------
            # 步骤1: 选择本轮的新闻事件
            # 添加随机偏移以增加变异性
            # -----------------------------------------------------------------
            news_idx = (round_num + random.randint(0, 5)) % len(NEWS_EVENTS)
            news = NEWS_EVENTS[news_idx]

            # -----------------------------------------------------------------
            # 步骤2: 收集所有Agent的交易订单（同步执行）
            # -----------------------------------------------------------------
            orders = []
            for agent in self.agents:
                # 每个Agent根据当前状态做出决策
                order, record = agent.act(
                    news=news,
                    current_price=self.market.current_price,
                    price_history=self.market.get_price_history(),
                    round_num=round_num,
                )
                orders.append(order)
                self.decision_records.append(record)

            # -----------------------------------------------------------------
            # 步骤3: 市场处理所有订单，更新价格
            # -----------------------------------------------------------------
            self.market.process_orders(orders)

            # -----------------------------------------------------------------
            # 步骤4: 更新所有Agent记忆中的PnL（基于最新价格）
            # -----------------------------------------------------------------
            for agent in self.agents:
                agent.update_memory_pnl(self.market.current_price)

        # 编译并返回结果
        return self._compile_results()

    async def _run_async(self, progress_bar: bool = True) -> dict:
        """
        异步执行仿真（并行调用LLM）

        此方法使用 asyncio.gather 并行执行所有Agent的LLM调用，
        显著提升仿真效率（约 N 倍加速，N 为Agent数量）。

        并行执行原理：
        =============
        ```
        Round N:
            ┌─────────────────────────────────────────────┐
            │     Agent 0        Agent 1       Agent N-1 │
            │       │              │              │       │
            │       ▼              ▼              ▼       │
            │   LLM Call       LLM Call       LLM Call    │  ← 并行
            │       │              │              │       │
            │       ▼              ▼              ▼       │
            │   Decision       Decision       Decision   │
            └─────────────────────────────────────────────┘
                              │
                              ▼
                    Market.process_orders()
        ```

        Args:
            progress_bar: 是否显示进度条

        Returns:
            仿真结果字典
        """
        # 创建轮次迭代器
        iterator = range(self.config.num_rounds)
        if progress_bar:
            iterator = tqdm(iterator, desc="Simulation", leave=False)

        # =====================================================================
        # 主仿真循环：逐轮执行
        # =====================================================================
        for round_num in iterator:
            # -----------------------------------------------------------------
            # 步骤1: 选择本轮的新闻事件
            # -----------------------------------------------------------------
            news_idx = (round_num + random.randint(0, 5)) % len(NEWS_EVENTS)
            news = NEWS_EVENTS[news_idx]

            # -----------------------------------------------------------------
            # 步骤2: 并行执行所有Agent的LLM调用
            # 使用 asyncio.gather 并发执行所有协程
            # -----------------------------------------------------------------
            tasks = [
                agent.act_async(
                    news=news,
                    current_price=self.market.current_price,
                    price_history=self.market.get_price_history(),
                    round_num=round_num,
                )
                for agent in self.agents
            ]

            # 等待所有Agent完成决策
            results = await asyncio.gather(*tasks)

            # -----------------------------------------------------------------
            # 步骤3: 收集订单和决策记录
            # -----------------------------------------------------------------
            orders = []
            for order, record in results:
                orders.append(order)
                self.decision_records.append(record)

            # -----------------------------------------------------------------
            # 步骤4: 市场处理所有订单，更新价格
            # -----------------------------------------------------------------
            self.market.process_orders(orders)

            # -----------------------------------------------------------------
            # 步骤5: 更新所有Agent记忆中的PnL
            # -----------------------------------------------------------------
            for agent in self.agents:
                agent.update_memory_pnl(self.market.current_price)

        # 编译并返回结果
        return self._compile_results()

    def _compile_results(self) -> dict:
        """
        编译仿真结果

        将仿真过程中收集的数据整合为结构化结果，包括：
        - 配置信息
        - 价格/成交量历史
        - 市场统计
        - 金融指标
        - Agent最终投资组合

        Returns:
            结构化的结果字典
        """
        # 将决策记录转换为 DataFrame
        df = pd.DataFrame([asdict(r) for r in self.decision_records])

        # 计算金融指标
        metrics = calculate_all_metrics(self.market.price_history, df)

        # 编译完整结果
        results = {
            # 配置信息
            "config": self.config.to_dict(),

            # 市场历史数据
            "price_history": self.market.price_history,
            "volume_history": self.market.volume_history,
            "buy_history": self.market.buy_history,
            "sell_history": self.market.sell_history,

            # 市场统计摘要
            "market_stats": self.market.get_stats(),

            # 金融指标
            "metrics": metrics,

            # 决策记录 DataFrame
            "decisions_df": df,

            # Agent最终投资组合价值
            "final_portfolios": {
                agent.id: agent.portfolio_value(self.market.current_price)
                for agent in self.agents
            },

            # Agent人格类型映射
            "agent_personalities": {agent.id: agent.personality for agent in self.agents},
        }

        return results


# =============================================================================
# 实验运行器
# =============================================================================


class ExperimentRunner:
    """
    完整实验运行器

    管理多条件、多次运行的完整实验，实现：
    - 2×2 实验设计（基线、仅记忆、仅社交、完整模型）
    - 蒙特卡洛仿真（多次独立运行）
    - 结果聚合和统计分析
    - 自动保存结果文件

    实验结构：
    =========
    ```
    Experiment
        │
        ├── Condition: baseline (无记忆、无社交)
        │   ├── Run 0 (seed=0)
        │   ├── Run 1 (seed=1)
        │   └── ... (共 num_runs 次)
        │
        ├── Condition: memory_only (有记忆、无社交)
        │   └── ...
        │
        ├── Condition: social_only (无记忆、有社交)
        │   └── ...
        │
        └── Condition: full (有记忆、有社交)
            └── ...
    ```

    输出文件：
    =========
    - {name}_results.json: 聚合统计结果
    - {name}_decisions.parquet: 所有决策记录
    - {name}_report.txt: 文本报告

    Attributes:
        config: 实验配置
        parallel: 是否并行执行LLM调用
        results: 按条件存储的运行结果
        output_dir: 输出目录路径

    Example:
        >>> config = ExperimentConfig(
        ...     name="2x2_design",
        ...     conditions=[...],
        ...     num_runs=15
        ... )
        >>> runner = ExperimentRunner(config)
        >>> aggregated = runner.run()
    """

    def __init__(self, config: ExperimentConfig, parallel: bool = True):
        """
        初始化实验运行器

        Args:
            config: 实验配置对象
            parallel: 是否使用并行LLM调用（默认True）
        """
        self.config = config
        self.parallel = parallel

        # 按条件存储运行结果: {condition_name: [run_results, ...]}
        self.results: dict[str, list[dict]] = {}

        # 输出目录
        self.output_dir = Path(config.output_dir)

    def run(self) -> dict:
        """
        执行完整实验

        执行流程：
        1. 创建输出目录
        2. 打印实验信息
        3. 遍历所有条件
           - 遍历所有运行
             - 执行单次仿真
             - 收集结果
        4. 聚合结果
        5. 保存文件

        Returns:
            聚合后的实验结果字典
        """
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # 打印实验信息
        # =====================================================================
        print("=" * 70)
        print(f"EXPERIMENT: {self.config.name}")
        print(f"Description: {self.config.description}")
        print("=" * 70)
        print(f"Conditions: {[c.value for c in self.config.conditions]}")
        print(f"Monte Carlo runs per condition: {self.config.num_runs}")
        print(f"Total simulations: {len(self.config.conditions) * self.config.num_runs}")
        print(f"Parallel mode: {'enabled' if self.parallel else 'disabled'}")
        print("=" * 70)

        start_time = time.time()

        # =====================================================================
        # 主循环：遍历所有实验条件
        # =====================================================================
        for condition in self.config.conditions:
            print(f"\n[Condition: {condition.value}]")
            self.results[condition.value] = []

            # 获取该条件对应的仿真配置
            sim_config = self.config.get_condition_config(condition)

            # -----------------------------------------------------------------
            # 内层循环：执行多次Monte Carlo运行
            # -----------------------------------------------------------------
            for run_idx in tqdm(range(self.config.num_runs), desc=f"  {condition.value}"):
                # 设置唯一的随机种子，确保每次运行独立且可重复
                # 种子计算：run_idx * 1000 + condition_hash
                sim_config.seed = run_idx * 1000 + hash(condition.value) % 1000

                # 创建并执行仿真
                runner = SimulationRunner(sim_config)
                result = runner.run(progress_bar=False, parallel=self.parallel)

                # 添加元数据
                result["run_idx"] = run_idx
                result["condition"] = condition.value

                # 收集结果
                self.results[condition.value].append(result)

        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.1f}s")

        # =====================================================================
        # 聚合结果并保存
        # =====================================================================
        aggregated = self._aggregate_results()
        self._save_results(aggregated)

        return aggregated

    def _aggregate_results(self) -> dict:
        """
        聚合所有运行的结果

        计算每个条件下各指标的均值和标准差，用于统计分析。

        Returns:
            聚合后的结果字典
        """
        aggregated = {
            "experiment_name": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_runs": self.config.num_runs,
                "base_config": self.config.base_config.to_dict(),
            },
            "conditions": {},
        }

        # 遍历每个条件，计算聚合统计
        for condition, runs in self.results.items():
            condition_stats = {
                "num_runs": len(runs),
                "metrics": self._aggregate_metrics(runs),
                "price_stats": self._aggregate_price_stats(runs),
                "portfolio_stats": self._aggregate_portfolio_stats(runs),
            }
            aggregated["conditions"][condition] = condition_stats

        return aggregated

    def _aggregate_metrics(self, runs: list[dict]) -> dict:
        """
        聚合金融指标

        计算各指标在多次运行中的均值、标准差、最小值、最大值。

        Args:
            runs: 单个条件下所有运行的结果列表

        Returns:
            聚合后的指标字典
        """
        # 要聚合的顶层指标
        metric_keys = [
            "hurst_exponent",     # Hurst指数
            "avg_volatility",     # 平均波动率
            "total_return",       # 总收益率
            "max_drawdown",       # 最大回撤
            "sharpe_ratio",       # 夏普比率
        ]

        aggregated = {}

        # 聚合顶层指标
        for key in metric_keys:
            values = [r["metrics"][key] for r in runs if key in r["metrics"]]
            if values:
                aggregated[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        # ---------------------------------------------------------------------
        # 聚合LSV羊群测度
        # ---------------------------------------------------------------------
        lsv_means = [r["metrics"]["lsv_herding"]["lsv_mean"] for r in runs]
        herding_ratios = [r["metrics"]["lsv_herding"]["herding_ratio"] for r in runs]
        aggregated["lsv_herding"] = {
            "lsv_mean": {"mean": np.mean(lsv_means), "std": np.std(lsv_means)},
            "herding_ratio": {"mean": np.mean(herding_ratios), "std": np.std(herding_ratios)},
        }

        # ---------------------------------------------------------------------
        # 聚合收益率分布统计
        # ---------------------------------------------------------------------
        kurtosis_values = [r["metrics"]["return_stats"]["kurtosis"] for r in runs]
        skewness_values = [r["metrics"]["return_stats"]["skewness"] for r in runs]
        aggregated["return_distribution"] = {
            "kurtosis": {"mean": np.mean(kurtosis_values), "std": np.std(kurtosis_values)},
            "skewness": {"mean": np.mean(skewness_values), "std": np.std(skewness_values)},
        }

        return aggregated

    def _aggregate_price_stats(self, runs: list[dict]) -> dict:
        """
        聚合价格统计

        Args:
            runs: 运行结果列表

        Returns:
            聚合后的价格统计字典
        """
        final_prices = [r["market_stats"]["final_price"] for r in runs]
        volatilities = [r["market_stats"]["volatility"] for r in runs]

        return {
            "final_price": {"mean": np.mean(final_prices), "std": np.std(final_prices)},
            "volatility": {"mean": np.mean(volatilities), "std": np.std(volatilities)},
        }

    def _aggregate_portfolio_stats(self, runs: list[dict]) -> dict:
        """
        按人格类型聚合投资组合统计

        分析不同人格类型的投资表现差异。

        Args:
            runs: 运行结果列表

        Returns:
            按人格类型分组的投资组合统计
        """
        # 按人格类型收集投资组合价值
        personality_portfolios: dict[str, list[float]] = {}

        for run in runs:
            portfolios = run["final_portfolios"]
            personalities = run["agent_personalities"]

            # 遍历每个Agent
            for agent_id, value in portfolios.items():
                personality = personalities[agent_id]
                if personality not in personality_portfolios:
                    personality_portfolios[personality] = []
                personality_portfolios[personality].append(value)

        # 计算每种人格的均值和标准差
        return {
            personality: {"mean": np.mean(values), "std": np.std(values)}
            for personality, values in personality_portfolios.items()
        }

    def _save_results(self, aggregated: dict) -> None:
        """
        保存实验结果到文件

        输出文件：
        1. JSON格式的聚合结果
        2. Parquet格式的决策记录
        3. 文本格式的实验报告

        Args:
            aggregated: 聚合后的结果字典
        """
        # =====================================================================
        # 保存JSON格式的聚合结果
        # =====================================================================
        json_path = self.output_dir / f"{self.config.name}_results.json"

        # NumPy类型转换函数（JSON不支持NumPy类型）
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, pd.DataFrame):
                return None  # 跳过DataFrame
            return obj

        json_data = json.loads(json.dumps(aggregated, default=convert))

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\n[Output] Results saved to: {json_path}")

        # =====================================================================
        # 保存Parquet格式的决策记录
        # =====================================================================
        all_decisions = []
        for condition, runs in self.results.items():
            for run in runs:
                df = run["decisions_df"].copy()
                df["condition"] = condition
                df["run_idx"] = run["run_idx"]
                all_decisions.append(df)

        if all_decisions:
            combined_df = pd.concat(all_decisions, ignore_index=True)
            parquet_path = self.output_dir / f"{self.config.name}_decisions.parquet"
            combined_df.to_parquet(parquet_path, index=False)
            print(f"[Output] Decisions saved to: {parquet_path}")
            print(f"         {len(combined_df)} total decision records")

        # =====================================================================
        # 生成并保存文本报告
        # =====================================================================
        report = self._generate_report(aggregated)
        report_path = self.output_dir / f"{self.config.name}_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"[Output] Report saved to: {report_path}")

    def _generate_report(self, aggregated: dict) -> str:
        """
        生成实验文本报告

        报告内容：
        1. 实验配置摘要
        2. 跨条件比较表
        3. 各人格类型投资表现

        Args:
            aggregated: 聚合后的结果字典

        Returns:
            格式化的文本报告
        """
        lines = [
            "=" * 70,
            f"EXPERIMENT REPORT: {aggregated['experiment_name']}",
            f"Generated: {aggregated['timestamp']}",
            "=" * 70,
            "",
            "[Configuration]",
            f"  Monte Carlo runs: {aggregated['config']['num_runs']}",
            f"  Agents: {aggregated['config']['base_config']['num_agents']}",
            f"  Rounds: {aggregated['config']['base_config']['num_rounds']}",
            "",
        ]

        # 条件对比表
        lines.extend(
            [
                "=" * 70,
                "CROSS-CONDITION COMPARISON",
                "=" * 70,
                "",
            ]
        )

        conditions = list(aggregated["conditions"].keys())

        # -----------------------------------------------------------------
        # Hurst指数对比
        # -----------------------------------------------------------------
        lines.append("[Hurst Exponent] (0.5 = random walk)")
        for cond in conditions:
            stats = aggregated["conditions"][cond]["metrics"]["hurst_exponent"]
            lines.append(f"  {cond:15s}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        lines.append("")

        # -----------------------------------------------------------------
        # 羊群效应比例对比
        # -----------------------------------------------------------------
        lines.append("[Herding Ratio] (proportion of rounds with significant herding)")
        for cond in conditions:
            stats = aggregated["conditions"][cond]["metrics"]["lsv_herding"]["herding_ratio"]
            lines.append(f"  {cond:15s}: {stats['mean']*100:.1f}% (±{stats['std']*100:.1f}%)")
        lines.append("")

        # -----------------------------------------------------------------
        # 峰度对比
        # -----------------------------------------------------------------
        lines.append("[Return Kurtosis] (>0 = fat tails)")
        for cond in conditions:
            stats = aggregated["conditions"][cond]["metrics"]["return_distribution"]["kurtosis"]
            lines.append(f"  {cond:15s}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        lines.append("")

        # -----------------------------------------------------------------
        # 波动率对比
        # -----------------------------------------------------------------
        lines.append("[Volatility]")
        for cond in conditions:
            stats = aggregated["conditions"][cond]["price_stats"]["volatility"]
            lines.append(f"  {cond:15s}: {stats['mean']*100:.4f}% (±{stats['std']*100:.4f}%)")
        lines.append("")

        # -----------------------------------------------------------------
        # 各人格类型投资表现
        # -----------------------------------------------------------------
        lines.extend(
            [
                "=" * 70,
                "PORTFOLIO PERFORMANCE BY PERSONALITY",
                "=" * 70,
                "",
            ]
        )

        for cond in conditions:
            lines.append(f"[{cond}]")
            portfolio_stats = aggregated["conditions"][cond]["portfolio_stats"]

            # 计算初始投资组合价值
            initial = (
                aggregated["config"]["base_config"]["initial_cash"]
                + aggregated["config"]["base_config"]["initial_holdings"]
                * aggregated["config"]["base_config"]["initial_price"]
            )

            # 按人格类型排序并输出
            for personality, stats in sorted(portfolio_stats.items()):
                ret = (stats["mean"] - initial) / initial * 100
                lines.append(
                    f"  {personality:15s}: ${stats['mean']:,.2f} ({ret:+.2f}%) ±${stats['std']:.2f}"
                )
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# 敏感性分析器
# =============================================================================


class SensitivityAnalyzer:
    """
    参数敏感性分析器

    系统性地研究单个参数变化对仿真结果的影响。

    支持的参数：
    ===========
    - price_impact: 价格影响系数（市场冲击）
    - num_agents: Agent数量（市场深度）
    - social_network_m: 社交网络连接数（信息传播）
    - memory_capacity: 记忆容量（学习能力）

    分析方法：
    =========
    对于参数的每个取值：
    1. 执行多次Monte Carlo仿真
    2. 计算关键指标的均值和标准差
    3. 分析参数-指标关系

    Attributes:
        config: 敏感性分析配置
        results: 分析结果

    Example:
        >>> config = SensitivityConfig(
        ...     price_impact_values=[0.001, 0.005, 0.01, 0.02],
        ...     num_runs=10
        ... )
        >>> analyzer = SensitivityAnalyzer(config)
        >>> results = analyzer.run("price_impact")
    """

    def __init__(self, config: SensitivityConfig):
        """
        初始化敏感性分析器

        Args:
            config: 敏感性分析配置
        """
        self.config = config
        self.results: dict[str, dict] = {}

    def run(self, parameter: str = "price_impact") -> dict:
        """
        执行敏感性分析

        Args:
            parameter: 要分析的参数名

        Returns:
            每个参数值对应的结果统计

        Raises:
            ValueError: 未知的参数名
        """
        # 参数-取值映射
        param_map = {
            "price_impact": self.config.price_impact_values,
            "num_agents": self.config.num_agents_values,
            "social_network_m": self.config.social_network_m_values,
            "memory_capacity": self.config.memory_capacity_values,
        }

        if parameter not in param_map:
            raise ValueError(f"Unknown parameter: {parameter}")

        values = param_map[parameter]
        results = {}

        print(f"\n[Sensitivity Analysis: {parameter}]")
        print(f"Values: {values}")
        print(f"Runs per value: {self.config.num_runs}")

        # =====================================================================
        # 对每个参数值执行分析
        # =====================================================================
        for value in tqdm(values, desc=parameter):
            # 创建基础配置
            sim_config = SimulationConfig(
                num_agents=self.config.base_config.num_agents,
                num_rounds=self.config.base_config.num_rounds,
                price_impact=self.config.base_config.price_impact,
                memory_capacity=self.config.base_config.memory_capacity,
                social_network_m=self.config.base_config.social_network_m,
            )

            # 设置目标参数值
            setattr(sim_config, parameter, value)

            # -----------------------------------------------------------------
            # 执行多次Monte Carlo运行
            # -----------------------------------------------------------------
            run_results = []
            for run_idx in range(self.config.num_runs):
                # 设置唯一种子
                sim_config.seed = run_idx * 100 + int(value * 1000)

                # 执行仿真
                runner = SimulationRunner(sim_config)
                result = runner.run(progress_bar=False)
                run_results.append(result)

            # -----------------------------------------------------------------
            # 聚合该参数值的结果
            # -----------------------------------------------------------------
            results[str(value)] = {
                "value": value,
                # Hurst指数
                "hurst_mean": np.mean([r["metrics"]["hurst_exponent"] for r in run_results]),
                "hurst_std": np.std([r["metrics"]["hurst_exponent"] for r in run_results]),
                # 波动率
                "volatility_mean": np.mean([r["metrics"]["avg_volatility"] for r in run_results]),
                "volatility_std": np.std([r["metrics"]["avg_volatility"] for r in run_results]),
                # 羊群效应
                "herding_mean": np.mean(
                    [r["metrics"]["lsv_herding"]["herding_ratio"] for r in run_results]
                ),
                "herding_std": np.std(
                    [r["metrics"]["lsv_herding"]["herding_ratio"] for r in run_results]
                ),
            }

        return results
