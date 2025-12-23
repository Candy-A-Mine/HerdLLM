"""
实验结果可视化模块 (Visualization Module)

本模块为ABM仿真实验结果生成出版级别的可视化图表。
这些图表用于直观展示不同实验条件下的市场行为差异，支持论文发表和学术报告。

生成的图表类型：
================
1. 条件对比图 (Condition Comparison)
   - 柱状图展示各实验条件下关键指标的均值和标准差
   - 用于直观比较基线、记忆、社交、完整模型的差异

2. 价格轨迹图 (Price Trajectories)
   - 时间序列图展示价格随时间的演化
   - 多次运行的叠加展示仿真结果的稳定性

3. 收益率分布图 (Return Distributions)
   - 展示峰度和偏度的条件对比
   - 用于验证是否复现真实市场的"尖峰厚尾"特征

4. 羊群效应分析图 (Herding Analysis)
   - LSV测度、羊群比例、Hurst指数的三图联合展示
   - 核心分析图表，展示社交网络对羊群行为的影响

5. 投资组合绩效图 (Portfolio Performance)
   - 按人格类型分组的收益率对比
   - 展示不同策略在各实验条件下的表现差异

6. 敏感性分析图 (Sensitivity Analysis)
   - 参数变化对结果的影响曲线
   - 用于确定参数的最优取值范围

可视化设计原则：
===============
- 配色一致：各实验条件使用固定颜色，便于跨图比较
- 标注清晰：所有数值标注在柱顶，便于精确读取
- 学术风格：符合学术期刊的图表规范
- 高分辨率：savefig.dpi=300 确保打印质量

作者: SuZX
日期: 2024
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# 全局样式配置
# =============================================================================
# 使用 seaborn 的白色网格样式，清晰且专业
plt.style.use("seaborn-v0_8-whitegrid")

# Matplotlib 参数设置
# 这些参数确保图表在论文中显示效果一致
plt.rcParams.update(
    {
        # 字体大小设置
        "font.size": 10,           # 基础字体大小
        "axes.labelsize": 11,      # 坐标轴标签（xlabel, ylabel）
        "axes.titlesize": 12,      # 子图标题
        "legend.fontsize": 9,      # 图例字体
        "xtick.labelsize": 9,      # X轴刻度标签
        "ytick.labelsize": 9,      # Y轴刻度标签
        # 分辨率设置
        "figure.dpi": 150,         # 屏幕显示分辨率
        "savefig.dpi": 300,        # 保存图片分辨率（出版级别）
        "savefig.bbox": "tight",   # 自动裁剪空白边距
    }
)

# =============================================================================
# 颜色配置
# =============================================================================
# 实验条件颜色：保持跨图表一致性，便于读者识别
COLORS = {
    # 实验条件颜色
    "baseline": "#7f7f7f",      # 灰色 - 基线条件（中性，作为参照）
    "memory_only": "#2ca02c",   # 绿色 - 仅记忆条件（生长、学习的隐喻）
    "social_only": "#1f77b4",   # 蓝色 - 仅社交条件（连接、网络的隐喻）
    "full": "#d62728",          # 红色 - 完整模型（最复杂，最显眼）
    # 人格类型颜色
    "Conservative": "#2ecc71",   # 浅绿 - 保守型（稳健）
    "Aggressive": "#e74c3c",     # 红色 - 激进型（冲动）
    "Trend_Follower": "#3498db", # 蓝色 - 趋势跟随型（理性）
    "Herding": "#9b59b6",        # 紫色 - 羊群型（从众）
}


# =============================================================================
# 条件对比图
# =============================================================================


def plot_condition_comparison(
    results: dict,
    metric_path: list[str],
    metric_name: str,
    output_path: str | Path | None = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    创建跨实验条件的指标对比柱状图

    图表物理含义：
    =============
    该图展示某一金融指标在不同实验条件下的表现。
    每根柱子代表一个实验条件，柱高为多次蒙特卡洛运行的均值，
    误差棒表示标准差，反映结果的稳定性。

    使用场景：
    =========
    - 比较基线与处理组的差异
    - 展示记忆/社交网络对市场的边际效应
    - 验证假设：如"社交网络增加羊群效应"

    视觉设计：
    =========
    - 每个条件使用固定颜色（见 COLORS 字典）
    - 数值标注在柱顶，便于精确读取
    - 误差棒使用 capsize=5，端点清晰可见

    Args:
        results: 实验结果字典，结构为:
            {
                "conditions": {
                    "baseline": {"metrics": {...}},
                    "memory_only": {"metrics": {...}},
                    ...
                }
            }
        metric_path: 指标在结果字典中的路径，如 ["metrics", "hurst_exponent"]
        metric_name: 指标的显示名称，用于Y轴标签和标题
        output_path: 图片保存路径，None则不保存
        figsize: 图像尺寸 (宽, 高)，单位为英寸
            - (8, 5) 适合单栏论文
            - (12, 6) 适合双栏论文或幻灯片

    Returns:
        matplotlib.figure.Figure 对象，可进一步自定义

    Example:
        >>> plot_condition_comparison(
        ...     results,
        ...     ["metrics", "hurst_exponent"],
        ...     "Hurst Exponent",
        ...     "figures/hurst_comparison.png"
        ... )
    """
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=figsize)

    # 获取所有实验条件名称
    conditions = list(results["conditions"].keys())

    # =========================================================================
    # 提取各条件的均值和标准差
    # metric_path 指定了指标在嵌套字典中的位置
    # 例如 ["metrics", "hurst_exponent"] 表示 results["conditions"][cond]["metrics"]["hurst_exponent"]
    # =========================================================================
    means = []
    stds = []

    for cond in conditions:
        # 沿路径逐层访问嵌套字典
        val = results["conditions"][cond]
        for key in metric_path:
            val = val[key]
        # 最终 val 应该是 {"mean": x, "std": y} 形式
        means.append(val["mean"])
        stds.append(val["std"])

    # =========================================================================
    # 绑制柱状图
    # =========================================================================
    # X轴位置：0, 1, 2, 3... 对应各条件
    x = np.arange(len(conditions))

    # 获取各条件对应的颜色
    colors = [COLORS.get(c, "#333333") for c in conditions]

    # 绑制带误差棒的柱状图
    # yerr: 误差棒长度（标准差）
    # capsize: 误差棒端点的横线宽度（像素）
    # alpha: 透明度（0.8 略微透明，视觉更柔和）
    # edgecolor: 柱子边框颜色（黑色增加对比度）
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="black")

    # =========================================================================
    # 坐标轴和标题设置
    # =========================================================================
    ax.set_xlabel("Experimental Condition")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Condition")
    ax.set_xticks(x)
    # 将下划线替换为换行，使标签更易读
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions])

    # =========================================================================
    # 在柱顶添加数值标注
    # 这使读者无需查看Y轴刻度即可获取精确值
    # =========================================================================
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(
            f"{mean:.3f}",  # 保留3位小数
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),  # 标注位置：柱顶中央
            xytext=(0, 3),  # 向上偏移3点
            textcoords="offset points",
            ha="center",    # 水平居中
            va="bottom",    # 垂直底对齐
            fontsize=8,
        )

    # 自动调整布局，避免标签被裁剪
    plt.tight_layout()

    # 保存图片（如果指定了路径）
    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


# =============================================================================
# 价格轨迹图
# =============================================================================


def plot_price_trajectories(
    results: dict,
    n_samples: int = 5,
    output_path: str | Path | None = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    绑制各实验条件下的价格时间序列轨迹

    图表物理含义：
    =============
    该图展示仿真市场中价格随时间（轮次）的演化过程。
    通过叠加多次运行的轨迹，可以直观判断：
    - 价格波动的幅度（振幅）
    - 价格走势的稳定性（轨迹是否聚集）
    - 是否存在趋势或均值回归

    视觉设计：
    =========
    - 2×2子图布局，每个子图对应一个实验条件
    - 第一条轨迹高亮（alpha=0.7），其余轨迹半透明（alpha=0.3）
    - 灰色虚线标注初始价格（通常为100），作为参照

    使用场景：
    =========
    - 展示不同条件下价格动态的差异
    - 验证仿真是否产生合理的价格波动
    - 识别极端情况（如价格暴涨/暴跌）

    Args:
        results: 原始实验结果，结构为:
            {
                "baseline": [run1_dict, run2_dict, ...],
                "memory_only": [...],
                ...
            }
            每个 run_dict 包含 "price_history" 键
        n_samples: 每个条件显示的样本轨迹数量
            - 过少（<3）无法展示变异性
            - 过多（>10）会使图表杂乱
            - 推荐值: 5
        output_path: 图片保存路径
        figsize: 图像尺寸
            - (12, 8) 适合四个子图的展示

    Returns:
        matplotlib.figure.Figure 对象
    """
    # 获取所有实验条件
    conditions = list(results.keys())
    n_conditions = len(conditions)

    # 创建2×2子图布局
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()  # 将2D数组展平为1D，便于索引

    # =========================================================================
    # 为每个条件绑制子图
    # =========================================================================
    for idx, condition in enumerate(conditions):
        # 最多绑制4个子图
        if idx >= 4:
            break

        ax = axes[idx]
        # 取前 n_samples 次运行
        runs = results[condition][:n_samples]

        # 绑制每条价格轨迹
        for i, run in enumerate(runs):
            prices = run["price_history"]
            # 第一条轨迹高亮显示，其余半透明
            # alpha: 透明度，0完全透明，1完全不透明
            alpha = 0.7 if i == 0 else 0.3
            ax.plot(prices, color=COLORS.get(condition, "#333333"), alpha=alpha, linewidth=1)

        # 添加初始价格参照线
        # 虚线样式便于与实际轨迹区分
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Initial Price")

        # 设置坐标轴标签
        ax.set_xlabel("Round")
        ax.set_ylabel("Price ($)")
        ax.set_title(f"{condition.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)  # 浅色网格线

    # 总标题
    plt.suptitle("Sample Price Trajectories by Condition", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


# =============================================================================
# 收益率分布图
# =============================================================================


def plot_return_distributions(
    results: dict,
    output_path: str | Path | None = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    绑制收益率分布特征的跨条件对比图

    图表物理含义：
    =============
    该图通过峰度和偏度两个统计量，展示收益率分布的形态特征。

    1. 峰度 (Kurtosis) - 左图:
       - 衡量分布的"尾部厚度"
       - κ > 0: 尖峰厚尾（Leptokurtic），极端事件概率高于正态分布
       - κ = 0: 正态分布（Mesokurtic）
       - κ < 0: 扁峰薄尾（Platykurtic）
       - 真实金融市场通常 κ > 0（厚尾特征）

    2. 偏度 (Skewness) - 右图:
       - 衡量分布的"对称性"
       - S > 0: 右偏（正偏），右尾更长
       - S = 0: 对称分布
       - S < 0: 左偏（负偏），左尾更长
       - 股票市场通常轻微负偏（崩盘风险）

    使用场景：
    =========
    - 验证仿真是否复现"尖峰厚尾"的典型事实
    - 比较不同条件对收益率分布的影响
    - 为GARCH等波动率模型提供依据

    Args:
        results: 聚合后的实验结果，包含 return_distribution 统计量
        output_path: 图片保存路径
        figsize: 图像尺寸
            - (10, 6) 适合并排两图

    Returns:
        matplotlib.figure.Figure 对象
    """
    # 创建1行2列的子图布局
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    conditions = list(results["conditions"].keys())

    # =========================================================================
    # 左图：峰度对比
    # =========================================================================
    ax1 = axes[0]

    # 提取峰度数据
    kurtosis_data = []
    for cond in conditions:
        stats = results["conditions"][cond]["metrics"]["return_distribution"]["kurtosis"]
        kurtosis_data.append({"condition": cond, "mean": stats["mean"], "std": stats["std"]})

    kdf = pd.DataFrame(kurtosis_data)
    x = np.arange(len(conditions))
    colors = [COLORS.get(c, "#333333") for c in conditions]

    # 绑制柱状图
    ax1.bar(x, kdf["mean"], yerr=kdf["std"], capsize=5, color=colors, alpha=0.8)

    # 添加正态分布参照线（κ=0）
    # 红色虚线便于与柱子对比
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Normal (κ=0)")

    ax1.set_xlabel("Condition")
    ax1.set_ylabel("Excess Kurtosis")
    ax1.set_title("Return Distribution Kurtosis\n(>0 indicates fat tails)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n") for c in conditions])
    ax1.legend()

    # =========================================================================
    # 右图：偏度对比
    # =========================================================================
    ax2 = axes[1]

    # 提取偏度数据
    skewness_data = []
    for cond in conditions:
        stats = results["conditions"][cond]["metrics"]["return_distribution"]["skewness"]
        skewness_data.append({"condition": cond, "mean": stats["mean"], "std": stats["std"]})

    sdf = pd.DataFrame(skewness_data)

    # 绑制柱状图
    ax2.bar(x, sdf["mean"], yerr=sdf["std"], capsize=5, color=colors, alpha=0.8)

    # 添加对称分布参照线（S=0）
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="Symmetric")

    ax2.set_xlabel("Condition")
    ax2.set_ylabel("Skewness")
    ax2.set_title("Return Distribution Skewness\n(<0 indicates left tail)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in conditions])
    ax2.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


# =============================================================================
# 羊群效应分析图
# =============================================================================


def plot_herding_analysis(
    results: dict,
    output_path: str | Path | None = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    绑制羊群效应综合分析图

    图表物理含义：
    =============
    该图是本研究的核心分析图表，展示三个关键指标：

    1. LSV测度 (LSV Herding Measure) - 左图:
       - 基于 Lakonishok-Shleifer-Vishny (1992) 的经典测度
       - 衡量交易者在同一方向交易的倾向
       - H > 0: 存在羊群效应
       - 值越大，羊群效应越强

    2. 羊群比例 (Herding Ratio) - 中图:
       - 存在显著羊群效应的轮次占总轮次的比例
       - 以百分比表示，直观易懂
       - 例如 60% 表示有 60% 的交易轮次出现了显著羊群行为

    3. Hurst指数 (Hurst Exponent) - 右图:
       - 衡量市场效率和价格记忆性
       - H = 0.5: 随机游走（有效市场）
       - H > 0.5: 趋势持续（动量效应）
       - H < 0.5: 均值回归（反转效应）
       - 红色虚线标注 H=0.5 作为参照

    使用场景：
    =========
    - 验证假设：社交网络是否增加羊群效应
    - 分析记忆机制对市场效率的影响
    - 比较不同条件下的市场动态特征

    Args:
        results: 聚合后的实验结果
        output_path: 图片保存路径
        figsize: 图像尺寸
            - (12, 5) 适合三图并排展示

    Returns:
        matplotlib.figure.Figure 对象
    """
    # 创建1行3列的子图布局
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    conditions = list(results["conditions"].keys())

    # =========================================================================
    # 左图：LSV羊群测度
    # =========================================================================
    ax1 = axes[0]

    # 提取LSV均值数据
    lsv_data = []
    for cond in conditions:
        stats = results["conditions"][cond]["metrics"]["lsv_herding"]["lsv_mean"]
        lsv_data.append({"condition": cond, "mean": stats["mean"], "std": stats["std"]})

    ldf = pd.DataFrame(lsv_data)
    x = np.arange(len(conditions))
    colors = [COLORS.get(c, "#333333") for c in conditions]

    ax1.bar(x, ldf["mean"], yerr=ldf["std"], capsize=5, color=colors, alpha=0.8)
    # 零线参照：LSV=0 表示无羊群效应
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Condition")
    ax1.set_ylabel("LSV Measure")
    ax1.set_title("LSV Herding Measure\n(higher = more herding)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n") for c in conditions])

    # =========================================================================
    # 中图：羊群比例
    # =========================================================================
    ax2 = axes[1]

    # 提取羊群比例数据（转换为百分比）
    ratio_data = []
    for cond in conditions:
        stats = results["conditions"][cond]["metrics"]["lsv_herding"]["herding_ratio"]
        # 乘以100转换为百分比
        ratio_data.append({"condition": cond, "mean": stats["mean"] * 100, "std": stats["std"] * 100})

    rdf = pd.DataFrame(ratio_data)

    ax2.bar(x, rdf["mean"], yerr=rdf["std"], capsize=5, color=colors, alpha=0.8)
    ax2.set_xlabel("Condition")
    ax2.set_ylabel("Herding Ratio (%)")
    ax2.set_title("Proportion of Rounds\nwith Significant Herding")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in conditions])
    ax2.set_ylim(0, 100)  # Y轴范围固定为0-100%

    # =========================================================================
    # 右图：Hurst指数
    # =========================================================================
    ax3 = axes[2]

    # 提取Hurst指数数据
    hurst_data = []
    for cond in conditions:
        stats = results["conditions"][cond]["metrics"]["hurst_exponent"]
        hurst_data.append({"condition": cond, "mean": stats["mean"], "std": stats["std"]})

    hdf = pd.DataFrame(hurst_data)

    ax3.bar(x, hdf["mean"], yerr=hdf["std"], capsize=5, color=colors, alpha=0.8)
    # 随机游走参照线：H=0.5
    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random Walk (H=0.5)")
    ax3.set_xlabel("Condition")
    ax3.set_ylabel("Hurst Exponent")
    ax3.set_title("Market Efficiency\n(H>0.5 = trending)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace("_", "\n") for c in conditions])
    ax3.legend()
    ax3.set_ylim(0, 1)  # Hurst指数理论范围为[0,1]

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


# =============================================================================
# 投资组合绩效图
# =============================================================================


def plot_portfolio_performance(
    results: dict,
    output_path: str | Path | None = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    绑制按人格类型和实验条件分组的投资组合绩效图

    图表物理含义：
    =============
    该图展示不同人格类型的Agent在各实验条件下的投资收益率。
    通过对比可以分析：
    - 哪种人格类型在特定条件下表现最佳
    - 记忆/社交网络是否改变了人格优势
    - 投资策略的稳健性（通过标准差判断）

    人格类型说明：
    =============
    - Conservative（保守型）：倾向持有，交易频率低
    - Aggressive（激进型）：频繁交易，追求高收益
    - Trend_Follower（趋势跟随型）：追涨杀跌，跟随价格趋势
    - Herding（羊群型）：跟随社交信号，模仿邻居行为

    视觉设计：
    =========
    - 分组柱状图：每组对应一个实验条件
    - 每种人格使用固定颜色
    - 图例置于右侧，避免遮挡数据

    Args:
        results: 聚合后的实验结果，包含 portfolio_stats
        output_path: 图片保存路径
        figsize: 图像尺寸

    Returns:
        matplotlib.figure.Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    conditions = list(results["conditions"].keys())

    # =========================================================================
    # 计算初始投资组合价值
    # 初始价值 = 初始现金 + 初始持仓 × 初始价格
    # =========================================================================
    initial_value = (
        results["config"]["base_config"]["initial_cash"]
        + results["config"]["base_config"]["initial_holdings"]
        * results["config"]["base_config"]["initial_price"]
    )

    # =========================================================================
    # 准备数据：计算各人格在各条件下的收益率
    # =========================================================================
    data = []
    for cond in conditions:
        portfolio_stats = results["conditions"][cond]["portfolio_stats"]
        for personality, stats in portfolio_stats.items():
            # 收益率 = (最终价值 - 初始价值) / 初始价值 × 100%
            ret = (stats["mean"] - initial_value) / initial_value * 100
            data.append(
                {
                    "condition": cond,
                    "personality": personality,
                    "return": ret,
                    # 标准差也转换为百分比
                    "std": stats["std"] / initial_value * 100,
                }
            )

    df = pd.DataFrame(data)

    # =========================================================================
    # 绑制分组柱状图
    # =========================================================================
    personalities = df["personality"].unique()
    x = np.arange(len(conditions))
    width = 0.2  # 柱子宽度，需要根据人格数量调整

    for i, personality in enumerate(personalities):
        pers_data = df[df["personality"] == personality]
        # 计算每组柱子的偏移量，使柱子均匀分布
        offset = (i - len(personalities) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            pers_data["return"],
            width,
            yerr=pers_data["std"],
            capsize=3,  # 误差棒端点较小，避免重叠
            label=personality,
            color=COLORS.get(personality, f"C{i}"),
            alpha=0.8,
        )

    # 零收益参照线
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Experimental Condition")
    ax.set_ylabel("Return (%)")
    ax.set_title("Portfolio Performance by Personality Type")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions])
    # 图例置于图外右侧，避免遮挡数据
    ax.legend(title="Personality", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


# =============================================================================
# 敏感性分析图
# =============================================================================


def plot_sensitivity_analysis(
    sensitivity_results: dict,
    parameter_name: str,
    output_path: str | Path | None = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    绑制参数敏感性分析结果图

    图表物理含义：
    =============
    该图展示某个参数变化时，三个关键指标的响应曲线。
    用于确定参数的最优取值范围和模型的稳健性。

    三个子图含义：
    =============
    1. 市场效率 (Market Efficiency) - 左图:
       - Y轴: Hurst指数
       - 参照线: H=0.5（随机游走）
       - 分析: 参数如何影响市场效率

    2. 价格波动率 (Price Volatility) - 中图:
       - Y轴: 波动率（百分比）
       - 分析: 参数如何影响价格稳定性

    3. 羊群行为 (Herding Behavior) - 右图:
       - Y轴: 羊群比例（百分比）
       - 分析: 参数如何影响羊群效应强度

    视觉设计：
    =========
    - 误差棒曲线图（errorbar）：展示均值和不确定性
    - 不同形状的标记点便于区分
    - 浅色网格线辅助读数

    Args:
        sensitivity_results: 敏感性分析结果，结构为:
            {
                "0.001": {"value": 0.001, "hurst_mean": ..., "hurst_std": ..., ...},
                "0.005": {...},
                ...
            }
        parameter_name: 参数名称，用于X轴标签
        output_path: 图片保存路径
        figsize: 图像尺寸

    Returns:
        matplotlib.figure.Figure 对象
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # =========================================================================
    # 提取并整理数据
    # =========================================================================
    values = []
    hurst_means, hurst_stds = [], []
    vol_means, vol_stds = [], []
    herd_means, herd_stds = [], []

    for key, data in sensitivity_results.items():
        values.append(data["value"])
        hurst_means.append(data["hurst_mean"])
        hurst_stds.append(data["hurst_std"])
        # 波动率转换为百分比
        vol_means.append(data["volatility_mean"] * 100)
        vol_stds.append(data["volatility_std"] * 100)
        # 羊群比例转换为百分比
        herd_means.append(data["herding_mean"] * 100)
        herd_stds.append(data["herding_std"] * 100)

    # 按参数值排序（确保曲线单调）
    sort_idx = np.argsort(values)
    values = np.array(values)[sort_idx]
    hurst_means = np.array(hurst_means)[sort_idx]
    hurst_stds = np.array(hurst_stds)[sort_idx]
    vol_means = np.array(vol_means)[sort_idx]
    vol_stds = np.array(vol_stds)[sort_idx]
    herd_means = np.array(herd_means)[sort_idx]
    herd_stds = np.array(herd_stds)[sort_idx]

    # =========================================================================
    # 左图：Hurst指数敏感性
    # =========================================================================
    ax1 = axes[0]
    # marker='o': 圆形标记点
    # capsize=5: 误差棒端点宽度
    ax1.errorbar(values, hurst_means, yerr=hurst_stds, marker="o", capsize=5, color="#1f77b4")
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random Walk")
    ax1.set_xlabel(parameter_name)
    ax1.set_ylabel("Hurst Exponent")
    ax1.set_title("Market Efficiency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # 中图：波动率敏感性
    # =========================================================================
    ax2 = axes[1]
    # marker='s': 方形标记点
    ax2.errorbar(values, vol_means, yerr=vol_stds, marker="s", capsize=5, color="#2ca02c")
    ax2.set_xlabel(parameter_name)
    ax2.set_ylabel("Volatility (%)")
    ax2.set_title("Price Volatility")
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # 右图：羊群效应敏感性
    # =========================================================================
    ax3 = axes[2]
    # marker='^': 三角形标记点
    ax3.errorbar(values, herd_means, yerr=herd_stds, marker="^", capsize=5, color="#d62728")
    ax3.set_xlabel(parameter_name)
    ax3.set_ylabel("Herding Ratio (%)")
    ax3.set_title("Herding Behavior")
    ax3.grid(True, alpha=0.3)

    # 总标题
    plt.suptitle(f"Sensitivity Analysis: {parameter_name}", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        print(f"Saved: {output_path}")

    return fig


# =============================================================================
# 批量生成接口
# =============================================================================


def generate_all_figures(
    results: dict,
    output_dir: str | Path,
) -> None:
    """
    批量生成实验的所有标准图表

    该函数是可视化模块的主入口，一次性生成所有常用图表。
    适用于实验完成后的快速报告生成。

    生成的图表：
    ===========
    1. hurst_comparison.png - Hurst指数条件对比
    2. return_distributions.png - 收益率分布特征
    3. herding_analysis.png - 羊群效应综合分析
    4. portfolio_performance.png - 投资组合绩效

    Args:
        results: 聚合后的实验结果字典
        output_dir: 图表输出目录
            - 如果目录不存在，会自动创建

    Example:
        >>> from src.visualization import generate_all_figures
        >>> generate_all_figures(results, "results/figures")
        Generating figures...
        Saved: results/figures/hurst_comparison.png
        Saved: results/figures/return_distributions.png
        Saved: results/figures/herding_analysis.png
        Saved: results/figures/portfolio_performance.png
        All figures saved to: results/figures
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")

    # =========================================================================
    # 生成各类图表
    # =========================================================================

    # 1. Hurst指数对比图
    plot_condition_comparison(
        results,
        ["metrics", "hurst_exponent"],
        "Hurst Exponent",
        output_dir / "hurst_comparison.png",
    )

    # 2. 收益率分布图
    plot_return_distributions(results, output_dir / "return_distributions.png")

    # 3. 羊群效应分析图
    plot_herding_analysis(results, output_dir / "herding_analysis.png")

    # 4. 投资组合绩效图
    plot_portfolio_performance(results, output_dir / "portfolio_performance.png")

    print(f"\nAll figures saved to: {output_dir}")
