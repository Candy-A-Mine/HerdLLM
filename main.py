#!/usr/bin/env python3
"""
SynMarket-Gen 主程序入口

金融市场情绪演化仿真系统 - 基于LLM-Agent的噪声交易与羊群效应研究

本程序是整个仿真系统的入口点，提供以下运行模式：
    1. 单次仿真模式 (demo): 快速演示系统功能
    2. 快速测试模式 (quick): 用于开发调试
    3. 标准实验模式 (standard): 完整的对照实验
    4. 自定义模式 (custom): 灵活配置实验参数
    5. 图表生成模式 (figures): 基于已有结果生成全部14张图表

使用方法：
    # 演示模式（默认）
    python main.py

    # 快速测试
    python main.py --mode quick

    # 标准实验
    python main.py --mode standard

    # 自定义参数
    python main.py --mode custom --agents 50 --rounds 100 --runs 30

    # 生成全部14张图表（基于已有结果）
    python main.py --mode figures
    python main.py --mode figures --output results/custom_30a_100r

图表生成模式输出（14张）：
    Core: hurst_comparison, return_distributions, herding_analysis, portfolio_performance
    Extra: social_consensus, personality_behavior, price_timeseries, factor_decomposition,
           herding_heatmap, metrics_summary
    Supplementary: statistical_tests, network_topology
    Validation: stylized_facts, real_market_comparison

输出文件：
    - results/: 实验结果目录
        - *_results.json: 聚合统计结果
        - *_decisions.parquet: 完整决策数据
        - *_report.txt: 文本报告
        - figures/: 可视化图表 (14张)

作者: SuZX
日期: 2024
"""

# =============================================================================
# 标准库导入
# =============================================================================
import argparse      # 命令行参数解析
import asyncio       # 异步编程支持
import json          # JSON文件读写
import random        # 随机数生成
import sys           # 系统交互
from dataclasses import asdict  # 数据类转字典
from pathlib import Path        # 路径处理

# =============================================================================
# 第三方库导入
# =============================================================================
import matplotlib.pyplot as plt  # 绑图
import pandas as pd              # 数据处理
from tqdm import tqdm            # 进度条

# =============================================================================
# 项目模块导入
# =============================================================================
from src.llm_client import LLMClient
from src.memory import AgentMemory
from src.social_network import SocialNetwork
from src.agent import TraderAgent, Order, DecisionRecord
from src.market import Market
from src.config import (
    SimulationConfig,
    ExperimentConfig,
    ExperimentCondition,
    get_quick_test_config,
    get_standard_config,
)
from src.metrics import calculate_all_metrics, format_metrics_report
from src.visualization import generate_all_figures
from src.stylized_facts import analyze_stylized_facts  # Stylized Facts验证
from src.real_market_benchmark import compare_with_real_market  # 真实市场对比
from src.generate_figures import FigureGenerator  # 独立图表生成


# =============================================================================
# 常量定义
# =============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录
DATASET_DIR = PROJECT_ROOT / "dataset"

# 结果输出目录
RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# 数据加载函数
# =============================================================================

def load_news_events() -> list:
    """
    从数据文件加载新闻事件

    Returns:
        新闻事件列表，每个元素包含content, sentiment, category

    Raises:
        FileNotFoundError: 当新闻数据文件不存在时
    """
    news_file = DATASET_DIR / "news_events.json"

    if news_file.exists():
        with open(news_file, "r", encoding="utf-8") as f:
            news_data = json.load(f)
        return [item["content"] for item in news_data]
    else:
        # 使用内置默认新闻
        return [
            "Central Bank announces surprise interest rate hike of 0.5%",
            "Major tech company reports record-breaking quarterly earnings",
            "GDP growth falls below market expectations at 1.2%",
            "Inflation data comes in higher than forecasted at 4.5%",
            "Federal Reserve signals potential rate cuts in coming months",
            "Unemployment rate drops to historic low of 3.5%",
            "Trade tensions escalate between major economies",
            "Breakthrough in AI technology drives tech sector optimism",
            "Oil prices surge 10% due to geopolitical tensions",
            "Consumer confidence index reaches 5-year high",
        ]


# =============================================================================
# 仿真运行函数
# =============================================================================

def run_single_simulation(
    config: SimulationConfig,
    news_events: list,
    show_progress: bool = True,
    parallel: bool = False,
) -> dict:
    """
    执行单次仿真

    该函数完成一次完整的ABM仿真，包括：
    1. 初始化所有组件（LLM、市场、Agent、社交网络）
    2. 运行仿真主循环
    3. 收集并返回结果数据

    Args:
        config: 仿真配置对象
        news_events: 新闻事件列表
        show_progress: 是否显示进度条
        parallel: 是否并行执行Agent决策（显著提升速度）

    Returns:
        包含仿真结果的字典：
        - price_history: 价格历史
        - decisions_df: 决策DataFrame
        - metrics: 金融指标
        - market_stats: 市场统计
    """
    # -------------------------------------------------------------------------
    # 步骤1: 设置随机种子
    # -------------------------------------------------------------------------
    if config.seed is not None:
        random.seed(config.seed)

    # -------------------------------------------------------------------------
    # 步骤2: 初始化LLM客户端
    # -------------------------------------------------------------------------
    llm_client = LLMClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        seed=config.seed,
    )

    # -------------------------------------------------------------------------
    # 步骤3: 初始化社交网络（如果启用）
    # -------------------------------------------------------------------------
    social_network = None
    if config.enable_social_network:
        social_network = SocialNetwork(num_agents=config.num_agents)

    # -------------------------------------------------------------------------
    # 步骤4: 初始化市场
    # -------------------------------------------------------------------------
    market = Market(
        initial_price=config.initial_price,
        price_impact=config.price_impact,
        num_agents=config.num_agents,
    )

    # -------------------------------------------------------------------------
    # 步骤5: 创建Agent
    # -------------------------------------------------------------------------
    agents = create_agents(
        config=config,
        llm_client=llm_client,
        social_network=social_network,
    )

    # -------------------------------------------------------------------------
    # 步骤6: 运行仿真主循环
    # -------------------------------------------------------------------------
    all_decisions = []

    # 创建迭代器（带或不带进度条）
    rounds_iterator = range(config.num_rounds)
    if show_progress:
        rounds_iterator = tqdm(rounds_iterator, desc="Simulation", leave=False)

    # 定义异步执行单回合的辅助函数
    async def run_round_parallel(agents, news, current_price, price_history, round_num):
        """并行执行所有Agent的决策"""
        tasks = [
            agent.act_async(
                news=news,
                current_price=current_price,
                price_history=price_history,
                round_num=round_num,
            )
            for agent in agents
        ]
        return await asyncio.gather(*tasks)

    for round_num in rounds_iterator:
        # 选择本轮新闻
        news_idx = (round_num + random.randint(0, 3)) % len(news_events)
        news = news_events[news_idx]

        # 收集所有Agent的订单
        orders = []
        round_decisions = []

        if parallel:
            # 并行执行所有Agent决策
            results = asyncio.run(run_round_parallel(
                agents=agents,
                news=news,
                current_price=market.current_price,
                price_history=market.get_price_history(),
                round_num=round_num,
            ))
            for order, decision in results:
                orders.append(order)
                round_decisions.append(decision)
        else:
            # 串行执行（原有逻辑）
            for agent in agents:
                order, decision = agent.act(
                    news=news,
                    current_price=market.current_price,
                    price_history=market.get_price_history(),
                    round_num=round_num,
                )
                orders.append(order)
                round_decisions.append(decision)

        # 记录决策
        all_decisions.extend(round_decisions)

        # 处理订单，更新市场价格
        market.process_orders(orders)

        # 更新所有Agent的记忆PnL
        for agent in agents:
            agent.update_memory_pnl(market.current_price)

    # -------------------------------------------------------------------------
    # 步骤7: 整理结果
    # -------------------------------------------------------------------------
    # 转换决策记录为DataFrame
    decisions_df = pd.DataFrame([asdict(d) for d in all_decisions])

    # 计算金融指标
    metrics = calculate_all_metrics(market.price_history, decisions_df)

    return {
        "config": config.to_dict(),
        "price_history": market.price_history,
        "volume_history": market.volume_history,
        "market_stats": market.get_stats(),
        "metrics": metrics,
        "decisions_df": decisions_df,
        "agents": agents,
    }


def create_agents(
    config: SimulationConfig,
    llm_client: LLMClient,
    social_network: SocialNetwork | None,
) -> list:
    """
    根据配置创建Agent列表

    Args:
        config: 仿真配置
        llm_client: LLM客户端
        social_network: 社交网络（可选）

    Returns:
        TraderAgent列表
    """
    agents = []

    # 构建人格类型列表
    personalities = []
    for personality, count in config.personality_distribution.items():
        personalities.extend([personality] * count)

    # 调整数量以匹配配置
    while len(personalities) < config.num_agents:
        personalities.append(random.choice(list(config.personality_distribution.keys())))
    personalities = personalities[:config.num_agents]
    random.shuffle(personalities)

    # 创建Agent
    for i, personality in enumerate(personalities):
        # 创建记忆模块
        if config.enable_memory:
            memory = AgentMemory(max_memories=config.memory_capacity)
        else:
            memory = AgentMemory(max_memories=0)

        # 创建Agent
        agent = TraderAgent(
            id=i,
            cash=config.initial_cash,
            holdings=config.initial_holdings,
            personality=personality,
            llm_client=llm_client,
            trade_size=config.trade_size,
            memory=memory,
            social_network=social_network if config.enable_social_network else None,
        )
        agents.append(agent)

    return agents


# =============================================================================
# 演示模式
# =============================================================================

def run_demo_mode(parallel: bool = False, model: str = "qwen2.5:3b"):
    """
    演示模式：运行单次仿真并展示结果

    该模式用于快速演示系统功能，使用较少的Agent和回合数。

    Args:
        parallel: 是否启用并行执行
        model: LLM模型名称
    """
    print("=" * 70)
    print("SynMarket-Gen: 金融市场ABM仿真系统 - 演示模式")
    print("=" * 70)

    # 加载新闻
    news_events = load_news_events()
    print(f"\n[数据] 加载了 {len(news_events)} 条新闻事件")

    # 创建配置
    config = SimulationConfig(
        num_agents=20,
        num_rounds=15,
        enable_memory=True,
        enable_social_network=True,
        llm_model=model,
        seed=42,
    )

    print(f"[配置] Agent数量: {config.num_agents}")
    print(f"[配置] 仿真回合: {config.num_rounds}")
    print(f"[配置] 记忆模块: {'启用' if config.enable_memory else '禁用'}")
    print(f"[配置] 社交网络: {'启用' if config.enable_social_network else '禁用'}")
    print(f"[配置] 并行模式: {'启用' if parallel else '禁用'}")
    print(f"[配置] LLM模型: {model}")

    print("\n" + "-" * 70)
    print("开始仿真...")
    print("-" * 70)

    # 运行仿真
    results = run_single_simulation(config, news_events, show_progress=True, parallel=parallel)

    # 输出结果
    print("\n" + "=" * 70)
    print("仿真完成！")
    print("=" * 70)

    # 市场统计
    stats = results["market_stats"]
    print(f"\n[市场统计]")
    print(f"  初始价格: ${stats['initial_price']:.2f}")
    print(f"  最终价格: ${stats['final_price']:.2f}")
    print(f"  总收益率: {stats['total_return_pct']:+.2f}%")
    print(f"  波动率:   {stats['volatility']:.4f}")

    # 金融指标
    metrics = results["metrics"]
    print(f"\n[金融指标]")
    print(f"  Hurst指数: {metrics['hurst_exponent']:.4f}")
    print(f"  羊群比率:  {metrics['lsv_herding']['herding_ratio']*100:.1f}%")

    # 保存结果
    output_dir = RESULTS_DIR / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存决策数据
    results["decisions_df"].to_parquet(output_dir / "decisions.parquet")
    print(f"\n[输出] 决策数据已保存至: {output_dir / 'decisions.parquet'}")

    # 绘制简单图表
    plt.figure(figsize=(10, 4))
    plt.plot(results["price_history"], "b-", linewidth=2)
    plt.axhline(y=100, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Price ($)")
    plt.title("Price Evolution - Demo Run")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "price_chart.png", dpi=150, bbox_inches="tight")
    print(f"[输出] 价格图表已保存至: {output_dir / 'price_chart.png'}")
    plt.close()

    # -------------------------------------------------------------------------
    # Stylized Facts 金融典型事实验证
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Stylized Facts 金融典型事实验证")
    print("-" * 70)

    # 执行分析并生成可视化拼图
    stylized_facts_path = output_dir / "stylized_facts_analysis.png"
    stylized_results = analyze_stylized_facts(
        price_history=results["price_history"],
        output_path=str(stylized_facts_path),
        print_report=True,  # 打印详细报告到控制台
    )

    print(f"[输出] Stylized Facts分析图已保存至: {stylized_facts_path}")

    # -------------------------------------------------------------------------
    # 真实市场基准对比
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("真实市场基准对比 (Real Market Benchmark)")
    print("-" * 70)

    # 与SPY（标普500 ETF）进行对比
    benchmark_path = output_dir / "real_market_comparison.png"
    try:
        benchmark_results = compare_with_real_market(
            sim_prices=results["price_history"],
            ticker="SPY",           # 标普500 ETF
            period="1y",            # 最近1年数据
            output_path=str(benchmark_path),
            print_report=True,      # 打印详细对比报告
        )
        print(f"[输出] 真实市场对比图已保存至: {benchmark_path}")
    except Exception as e:
        print(f"[警告] 无法下载真实市场数据: {e}")
        print("[提示] 请检查网络连接，或稍后重试。")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_arguments():
    """
    解析命令行参数

    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="SynMarket-Gen: 金融市场ABM仿真系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 演示模式
  python main.py --mode quick             # 快速测试
  python main.py --mode standard          # 标准实验
  python main.py --mode custom --agents 50 --rounds 100
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "quick", "standard", "custom", "figures"],
        default="demo",
        help="运行模式 (默认: demo)"
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=50,
        help="Agent数量 (custom模式)"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="仿真回合数 (custom模式)"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Monte Carlo运行次数 (custom模式)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="输出目录"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并行执行Agent决策（显著提升速度）"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:3b",
        help="LLM模型名称 (默认: qwen2.5:3b, 可选: qwen2.5:1.5b/7b)"
    )

    return parser.parse_args()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """
    主函数入口

    根据命令行参数选择运行模式并执行仿真。
    """
    args = parse_arguments()

    if args.mode == "demo":
        # 演示模式
        run_demo_mode(parallel=args.parallel, model=args.model)

    elif args.mode == "quick":
        # 快速测试模式
        from src.experiment import ExperimentRunner
        config = get_quick_test_config()
        config.output_dir = args.output
        config.base_config.llm_model = args.model
        runner = ExperimentRunner(config, parallel=args.parallel)
        results = runner.run()
        generate_all_figures(results, Path(args.output) / "figures")

    elif args.mode == "standard":
        # 标准实验模式
        from src.experiment import ExperimentRunner
        config = get_standard_config()
        config.output_dir = args.output
        config.base_config.llm_model = args.model
        runner = ExperimentRunner(config, parallel=args.parallel)
        results = runner.run()
        generate_all_figures(results, Path(args.output) / "figures")

    elif args.mode == "custom":
        # 自定义模式
        from src.experiment import ExperimentRunner
        base_config = SimulationConfig(
            num_agents=args.agents,
            num_rounds=args.rounds,
            llm_model=args.model,
        )
        config = ExperimentConfig(
            name=f"custom_{args.agents}a_{args.rounds}r",
            num_runs=args.runs,
            base_config=base_config,
        )
        config.output_dir = args.output
        runner = ExperimentRunner(config, parallel=args.parallel)
        results = runner.run()
        generate_all_figures(results, Path(args.output) / "figures")

    elif args.mode == "figures":
        # 图表生成模式（基于已有结果）
        print("=" * 70)
        print("SynMarket-Gen: 图表生成模式")
        print("=" * 70)
        generator = FigureGenerator(args.output)
        generator.load_data()
        generator.generate_all()
        print(f"\n图表已保存到 {generator.output_dir}/")

    print("\n程序执行完毕！")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == "__main__":
    main()
