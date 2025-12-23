"""
实验运行器脚本 (Experiment Runner Script)

本脚本是运行ABM仿真实验的命令行入口。
支持多种预设模式和自定义配置，便于快速启动实验。

使用方法：
=========

1. 快速测试模式（验证代码是否正常工作）：
   ```bash
   poetry run python -m src.run_experiment --mode quick
   ```
   - 3次蒙特卡洛运行
   - 2个实验条件
   - 约2-5分钟完成

2. 标准实验模式（论文级别）：
   ```bash
   poetry run python -m src.run_experiment --mode standard
   ```
   - 30次蒙特卡洛运行
   - 4个实验条件（2×2设计）
   - 约30-60分钟完成

3. 大规模实验模式：
   ```bash
   poetry run python -m src.run_experiment --mode large
   ```
   - 更多Agent和更长轮次
   - 适合稳健性检验

4. 羊群效应专项研究：
   ```bash
   poetry run python -m src.run_experiment --mode herding
   ```
   - 参数针对羊群效应优化

5. 自定义实验：
   ```bash
   poetry run python -m src.run_experiment --mode custom \\
       --agents 50 --rounds 100 --runs 30 \\
       --conditions baseline social_only full \\
       --output results/my_experiment \\
       --name my_custom_experiment
   ```

命令行参数详解：
===============

--mode: 实验模式选择
    - quick: 快速测试，用于验证代码
    - standard: 标准实验，用于论文数据
    - large: 大规模实验，用于稳健性检验
    - herding: 羊群效应专项研究
    - custom: 自定义参数

--agents: Agent数量（仅custom模式）
    - 典型值: 30-100
    - 更多Agent = 更稳定的市场 + 更长的运行时间

--rounds: 仿真轮次（仅custom模式）
    - 典型值: 50-200
    - 更多轮次 = 更长的时间序列 + 更可靠的统计

--runs: 蒙特卡洛运行次数（仅custom模式）
    - 典型值: 15-30
    - 更多运行 = 更小的标准误差 + 更长的总时间

--conditions: 实验条件列表（仅custom模式）
    - baseline: 无记忆、无社交网络
    - memory_only: 有记忆、无社交网络
    - social_only: 无记忆、有社交网络
    - full: 有记忆、有社交网络

--output: 结果输出目录
    - 默认: results/

--name: 实验名称
    - 用于结果文件命名

--no-viz: 跳过可视化生成
    - 仅保存原始数据，不生成图表

输出文件：
=========
运行完成后，在输出目录下会生成：
- {name}_results.json: 聚合统计结果
- {name}_decisions.parquet: 所有决策记录
- {name}_report.txt: 文本报告
- figures/: 可视化图表目录

作者: SuZX
日期: 2024
"""

import argparse
import sys
from pathlib import Path

from src.config import (
    ExperimentCondition,
    ExperimentConfig,
    SimulationConfig,
    get_quick_test_config,
    get_standard_config,
    get_large_scale_config,
    get_herding_focus_config,
)
from src.experiment import ExperimentRunner
from src.visualization import generate_all_figures
from src.analysis import load_experiment_results


# =============================================================================
# 命令行参数解析
# =============================================================================


def parse_args():
    """
    解析命令行参数

    该函数定义了所有可用的命令行选项，并返回解析后的参数对象。

    参数分类：
    =========
    1. 模式选择参数：--mode
    2. 自定义配置参数：--agents, --rounds, --runs, --conditions
    3. 输出控制参数：--output, --name, --no-viz

    Returns:
        argparse.Namespace: 包含所有参数值的命名空间对象
    """
    # 创建解析器，设置程序描述
    parser = argparse.ArgumentParser(
        description="Run ABM experiments for noise trading and herding analysis"
    )

    # =========================================================================
    # 模式选择参数
    # =========================================================================
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "standard", "large", "herding", "custom"],
        default="quick",
        help="Experiment mode (default: quick)",
    )

    # =========================================================================
    # 自定义配置参数（仅在 --mode custom 时生效）
    # =========================================================================

    # Agent数量：影响市场深度和计算时间
    parser.add_argument(
        "--agents",
        type=int,
        default=50,
        help="Number of agents (for custom mode)",
    )

    # 仿真轮次：影响时间序列长度
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of simulation rounds (for custom mode)",
    )

    # 蒙特卡洛运行次数：影响统计可靠性
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of Monte Carlo runs per condition (for custom mode)",
    )

    # 实验条件：可选择多个条件进行对比
    # nargs="+" 表示接受一个或多个值
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        choices=["baseline", "memory_only", "social_only", "full"],
        default=["baseline", "memory_only", "social_only", "full"],
        help="Conditions to run (for custom mode)",
    )

    # =========================================================================
    # 输出控制参数
    # =========================================================================

    # 输出目录
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory",
    )

    # 实验名称（用于文件命名）
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name",
    )

    # 是否跳过可视化生成
    # action="store_true" 表示该参数是开关型，出现则为True
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    return parser.parse_args()


# =============================================================================
# 配置获取函数
# =============================================================================


def get_config(args) -> ExperimentConfig:
    """
    根据命令行参数获取实验配置

    该函数是参数到配置的映射器，支持预设模式和自定义模式。

    预设模式说明：
    =============
    - quick: 快速测试
        - 3次运行
        - 2个条件（baseline, full）
        - 30个Agent，50轮

    - standard: 标准实验
        - 30次运行
        - 4个条件
        - 50个Agent，100轮

    - large: 大规模实验
        - 30次运行
        - 4个条件
        - 100个Agent，200轮

    - herding: 羊群效应专项
        - 参数针对羊群效应优化
        - 更大的社交网络连接数

    Args:
        args: argparse解析后的参数对象

    Returns:
        ExperimentConfig: 完整的实验配置对象
    """
    # =========================================================================
    # 根据模式选择预设配置或创建自定义配置
    # =========================================================================
    if args.mode == "quick":
        # 快速测试配置：用于验证代码正确性
        config = get_quick_test_config()

    elif args.mode == "standard":
        # 标准实验配置：用于论文数据生成
        config = get_standard_config()

    elif args.mode == "large":
        # 大规模配置：用于稳健性检验
        config = get_large_scale_config()

    elif args.mode == "herding":
        # 羊群效应专项配置
        config = get_herding_focus_config()

    else:  # custom 模式
        # 创建基础仿真配置
        base = SimulationConfig(
            num_agents=args.agents,
            num_rounds=args.rounds,
        )

        # 将字符串条件名映射为枚举类型
        condition_map = {
            "baseline": ExperimentCondition.BASELINE,
            "memory_only": ExperimentCondition.MEMORY_ONLY,
            "social_only": ExperimentCondition.SOCIAL_ONLY,
            "full": ExperimentCondition.FULL,
        }
        conditions = [condition_map[c] for c in args.conditions]

        # 创建自定义实验配置
        config = ExperimentConfig(
            name=args.name or f"custom_{args.agents}a_{args.rounds}r",
            description=f"Custom experiment: {args.agents} agents, {args.rounds} rounds",
            num_runs=args.runs,
            base_config=base,
            conditions=conditions,
        )

    # =========================================================================
    # 覆盖输出目录和名称（如果命令行指定）
    # =========================================================================
    config.output_dir = args.output

    if args.name:
        config.name = args.name

    return config


# =============================================================================
# 主函数
# =============================================================================


def main():
    """
    主入口函数

    执行流程：
    =========
    1. 解析命令行参数
    2. 获取实验配置
    3. 打印实验信息
    4. 确认大规模运行（如果需要）
    5. 执行实验
    6. 生成可视化（如果未禁用）
    7. 输出完成信息
    """
    # =========================================================================
    # 步骤1: 解析命令行参数
    # =========================================================================
    args = parse_args()

    # =========================================================================
    # 步骤2: 打印标题
    # =========================================================================
    print("\n" + "=" * 70)
    print("SynMarket-Gen: Financial ABM Experiment Runner")
    print("=" * 70)

    # =========================================================================
    # 步骤3: 获取配置并打印实验信息
    # =========================================================================
    config = get_config(args)

    print(f"\nExperiment: {config.name}")
    print(f"Mode: {args.mode}")
    print(f"Agents: {config.base_config.num_agents}")
    print(f"Rounds: {config.base_config.num_rounds}")
    print(f"Monte Carlo runs: {config.num_runs}")
    print(f"Conditions: {[c.value for c in config.conditions]}")
    print(f"Output: {config.output_dir}")

    # =========================================================================
    # 步骤4: 大规模运行确认
    # 当总仿真次数超过20时，提示用户确认
    # =========================================================================
    total_sims = len(config.conditions) * config.num_runs
    if total_sims > 20:
        print(f"\nThis will run {total_sims} simulations. Continue? [y/N] ", end="")
        response = input().strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    # =========================================================================
    # 步骤5: 执行实验
    # =========================================================================
    runner = ExperimentRunner(config)
    results = runner.run()

    # =========================================================================
    # 步骤6: 生成可视化（除非指定 --no-viz）
    # =========================================================================
    if not args.no_viz:
        output_dir = Path(config.output_dir)
        generate_all_figures(results, output_dir / "figures")

    # =========================================================================
    # 步骤7: 打印完成信息和输出文件列表
    # =========================================================================
    print("\n" + "=" * 70)
    print("Experiment completed successfully!")
    print("=" * 70)

    # 列出所有输出文件
    print(f"\nOutput files:")
    output_dir = Path(config.output_dir)
    for f in output_dir.glob("*"):
        if f.is_file():
            print(f"  - {f}")
        elif f.is_dir():
            # 对于目录，显示包含的文件数量
            print(f"  - {f}/ ({len(list(f.glob('*')))} files)")


# =============================================================================
# 脚本入口
# =============================================================================

if __name__ == "__main__":
    main()
