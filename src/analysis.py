"""
统计分析模块 (Statistical Analysis Module)

本模块提供实验结果的统计分析功能，包括假设检验、效应量计算、置信区间估计等。
这些工具是论文数据分析和结论支撑的基础。

模块功能概览：
=============

1. 假设检验 (Hypothesis Testing)
   ----------------------------------
   - Welch's t-test: 用于比较两组均值差异，不假设方差相等
   - Mann-Whitney U: 非参数检验，用于非正态分布数据

   **为什么选择 Welch's t-test 而非 Student's t-test？**

   在金融ABM仿真中，不同实验条件可能产生不同的方差：
   - 基线条件：价格波动可能较小
   - 社交网络条件：羊群效应可能放大波动

   Welch检验的优势：
   1. 不假设两组方差相等（更符合实际情况）
   2. 在方差不等时比Student's t更稳健
   3. 在方差相等时性能与Student's t相当
   4. 是更"安全"的默认选择

   参考：Ruxton (2006). The unequal variance t-test is an underused alternative
         to Student's t-test. Behavioral Ecology.

2. 效应量 (Effect Size)
   ----------------------
   - Cohen's d: 标准化均值差异，用于衡量实际显著性

   **为什么需要效应量？**

   p值只能告诉你"是否有差异"，不能告诉你"差异有多大"。
   在大样本量下，很小的差异也可能统计显著。
   效应量提供了与样本量无关的效应强度度量。

3. 置信区间 (Confidence Intervals)
   ---------------------------------
   - 参数化置信区间: 基于t分布，假设正态性
   - Bootstrap置信区间: 非参数方法，不假设分布形式

4. 条件对比 (Cross-Condition Comparison)
   ----------------------------------------
   - 自动比较所有条件与基线
   - 生成标准化的对比表格

5. 行为分析 (Behavioral Analysis)
   --------------------------------
   - 按人格类型分析投资绩效
   - 分析社交网络对决策的影响

作者: SuZX
日期: 2024
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 假设检验：Welch's t-test
# =============================================================================


def welch_ttest(group1: list | np.ndarray, group2: list | np.ndarray) -> dict:
    """
    Welch's t检验（韦尔奇t检验）

    该检验用于比较两个独立样本的均值是否存在显著差异。
    与Student's t检验不同，Welch检验不假设两组方差相等。

    数学原理：
    =========
    假设检验：
        H0（原假设）: μ1 = μ2（两组均值相等）
        H1（备择假设）: μ1 ≠ μ2（两组均值不等）

    Welch's t统计量计算公式：

        t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)

    其中：
        - x̄₁, x̄₂: 两组样本均值
        - s₁², s₂²: 两组样本方差
        - n₁, n₂: 两组样本量

    自由度使用 Welch-Satterthwaite 近似公式：

        df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]

    为什么选择 Welch 而非 Student's t？
    ===================================
    1. **更稳健**：不假设方差齐性，更符合ABM仿真的实际情况
    2. **无性能损失**：当方差确实相等时，统计效力与Student's t相当
    3. **学术推荐**：现代统计学推荐默认使用Welch检验

    判断标准：
    =========
        - p < 0.05: 拒绝原假设，认为两组均值显著不同
        - p < 0.01: 高度显著
        - p ≥ 0.05: 不拒绝原假设，无显著差异

    Args:
        group1: 第一组观测值（如基线条件的Hurst指数）
        group2: 第二组观测值（如处理条件的Hurst指数）

    Returns:
        包含以下键的字典：
        - t_statistic: t统计量
        - p_value: 双侧p值
        - significant_05: 是否在5%水平显著
        - significant_01: 是否在1%水平显著
        - mean_diff: 均值差（group1 - group2）

    Example:
        >>> baseline_hurst = [0.52, 0.48, 0.51, ...]
        >>> social_hurst = [0.58, 0.62, 0.55, ...]
        >>> result = welch_ttest(baseline_hurst, social_hurst)
        >>> if result["significant_05"]:
        ...     print("社交网络显著影响了Hurst指数")
    """
    # 确保输入为numpy数组
    group1 = np.array(group1)
    group2 = np.array(group2)

    # 调用scipy的ttest_ind，设置equal_var=False执行Welch检验
    # 默认为双侧检验（alternative='two-sided'）
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_05": p_value < 0.05,  # 5%显著性水平
        "significant_01": p_value < 0.01,  # 1%显著性水平
        "mean_diff": np.mean(group1) - np.mean(group2),  # 均值差
    }


# =============================================================================
# 假设检验：Mann-Whitney U
# =============================================================================


def mann_whitney_u(group1: list | np.ndarray, group2: list | np.ndarray) -> dict:
    """
    Mann-Whitney U检验（曼-惠特尼U检验）

    这是一种非参数检验，用于比较两个独立样本的分布是否相同。
    当数据不满足正态分布假设时，使用此检验代替t检验。

    数学原理：
    =========
    该检验基于秩次（ranks）而非原始值。

    计算步骤：
    1. 合并两组数据并按大小排序
    2. 计算每个观测值的秩次
    3. 计算每组的秩次和 R₁, R₂
    4. 计算U统计量：

        U₁ = R₁ - n₁(n₁+1)/2
        U₂ = R₂ - n₂(n₂+1)/2
        U = min(U₁, U₂)

    假设检验：
        H0: 两组数据来自相同分布
        H1: 两组数据来自不同分布

    何时使用？
    ==========
    - 数据明显非正态分布（如收益率有极端值）
    - 样本量较小（<30）
    - 数据为有序分类变量
    - 存在离群值

    与t检验的比较：
    ==============
    - t检验：假设正态分布，使用原始值，更敏感
    - U检验：不假设分布，使用秩次，更稳健

    Args:
        group1: 第一组观测值
        group2: 第二组观测值

    Returns:
        包含U统计量、p值和显著性判断的字典

    Example:
        >>> # 当收益率分布有厚尾时使用
        >>> result = mann_whitney_u(returns_baseline, returns_social)
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    # 执行双侧Mann-Whitney U检验
    # alternative='two-sided' 表示检验两组是否来自不同分布
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")

    return {
        "u_statistic": u_stat,
        "p_value": p_value,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


# =============================================================================
# 效应量：Cohen's d
# =============================================================================


def cohens_d(group1: list | np.ndarray, group2: list | np.ndarray) -> float:
    """
    计算Cohen's d效应量（标准化均值差异）

    效应量衡量两组之间差异的实际大小，与样本量无关。
    这是对p值的重要补充：p值告诉你"是否有差异"，
    Cohen's d告诉你"差异有多大"。

    数学定义：
    =========
    Cohen's d 计算公式：

        d = (x̄₁ - x̄₂) / s_pooled

    其中合并标准差（pooled standard deviation）为：

        s_pooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]

    效应量解释标准（Cohen, 1988）：
    ==============================
        - |d| < 0.2: 可忽略（negligible）
        - 0.2 ≤ |d| < 0.5: 小效应（small）
        - 0.5 ≤ |d| < 0.8: 中等效应（medium）
        - |d| ≥ 0.8: 大效应（large）

    为什么需要效应量？
    ==================
    考虑以下场景：
    - 实验A：n=1000，p=0.03，d=0.1（小样本量下不显著的微小差异）
    - 实验B：n=30，p=0.04，d=0.8（大效应但样本量小）

    仅看p值，两者都"显著"。但效应量告诉我们：
    - 实验A的差异实际上可以忽略
    - 实验B的差异在实践中有重要意义

    Args:
        group1: 第一组观测值
        group2: 第二组观测值

    Returns:
        Cohen's d 值（可为正或负，表示方向）

    Example:
        >>> d = cohens_d(hurst_baseline, hurst_full)
        >>> print(f"效应量: {d:.3f} ({effect_size_interpretation(d)})")
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    n1, n2 = len(group1), len(group2)
    # ddof=1 使用贝塞尔校正（无偏估计）
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # 计算合并标准差
    # 公式：sqrt[((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)]
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # 避免除以零
    if pooled_std == 0:
        return 0.0

    # Cohen's d = 均值差 / 合并标准差
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_interpretation(d: float) -> str:
    """
    解释Cohen's d效应量的大小

    根据Cohen (1988) 的经典标准进行解释。

    Args:
        d: Cohen's d 值

    Returns:
        效应量大小的文字描述

    Reference:
        Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    d_abs = abs(d)

    if d_abs < 0.2:
        return "negligible"  # 可忽略
    elif d_abs < 0.5:
        return "small"       # 小效应
    elif d_abs < 0.8:
        return "medium"      # 中等效应
    else:
        return "large"       # 大效应


# =============================================================================
# 置信区间：参数化方法
# =============================================================================


def confidence_interval(
    data: list | np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    计算均值的参数化置信区间

    置信区间提供了总体参数的估计范围。
    95%置信区间的含义：如果重复抽样无数次，
    约95%的置信区间会包含真实的总体均值。

    数学原理：
    =========
    基于t分布的置信区间公式：

        CI = x̄ ± t_{α/2, n-1} × SE

    其中：
        - x̄: 样本均值
        - t_{α/2, n-1}: t分布的临界值
        - SE: 标准误差 = s / √n
        - α = 1 - confidence（对于95%置信度，α=0.05）

    假设：
    =====
    - 数据来自正态分布（或样本量足够大，中心极限定理成立）
    - 观测值相互独立

    Args:
        data: 观测数据
        confidence: 置信水平（默认0.95，即95%）

    Returns:
        元组 (下界, 上界)

    Example:
        >>> ci = confidence_interval(hurst_values, 0.95)
        >>> print(f"Hurst指数 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)

    # 计算标准误差 SE = s / √n
    se = stats.sem(data)

    # 计算t分布的临界值
    # ppf: 百分点函数（分位数函数），返回给定概率对应的值
    # (1 + confidence) / 2 是因为双侧置信区间
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return (mean - h, mean + h)


# =============================================================================
# 置信区间：Bootstrap方法
# =============================================================================


def bootstrap_ci(
    data: list | np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    statistic: str = "mean",
) -> tuple[float, float]:
    """
    计算Bootstrap置信区间（非参数方法）

    Bootstrap是一种重采样技术，通过从原始数据有放回地
    抽取样本来估计统计量的分布。

    为什么使用Bootstrap？
    ====================
    1. **无分布假设**：不需要假设数据服从正态分布
    2. **灵活性**：可用于任何统计量（均值、中位数、标准差等）
    3. **小样本**：样本量较小时也能提供合理估计

    Bootstrap算法步骤：
    ===================
    1. 从原始数据（n个点）有放回地抽取n个点，形成一个bootstrap样本
    2. 计算该bootstrap样本的统计量（如均值）
    3. 重复步骤1-2共B次（如1000次）
    4. 得到B个统计量值的分布
    5. 取分布的α/2和1-α/2分位数作为置信区间

    置信区间类型：
    =============
    这里使用的是"百分位法"（percentile method），最简单直观。
    其他方法包括：BCa（偏差校正加速）、t-bootstrap等。

    Args:
        data: 原始观测数据
        n_bootstrap: Bootstrap重采样次数（通常1000-10000次）
        confidence: 置信水平（默认0.95）
        statistic: 要估计的统计量
            - "mean": 均值
            - "median": 中位数
            - "std": 标准差

    Returns:
        元组 (下界, 上界)

    Example:
        >>> # 对于厚尾分布，bootstrap比参数化方法更稳健
        >>> ci = bootstrap_ci(kurtosis_values, n_bootstrap=5000)
    """
    data = np.array(data)
    n = len(data)

    # 统计量函数映射
    stat_func = {"mean": np.mean, "median": np.median, "std": np.std}[statistic]

    # Bootstrap重采样
    boot_stats = []
    for _ in range(n_bootstrap):
        # 有放回抽样：从n个数据点中抽取n个点
        sample = np.random.choice(data, size=n, replace=True)
        # 计算该样本的统计量
        boot_stats.append(stat_func(sample))

    # 计算百分位置信区间
    alpha = 1 - confidence
    lower = np.percentile(boot_stats, alpha / 2 * 100)      # 下界（2.5%分位数）
    upper = np.percentile(boot_stats, (1 - alpha / 2) * 100)  # 上界（97.5%分位数）

    return (lower, upper)


# =============================================================================
# 条件对比分析
# =============================================================================


def compare_conditions(
    results: dict,
    metric_path: list[str],
    baseline: str = "baseline",
) -> pd.DataFrame:
    """
    将所有实验条件与基线进行对比

    这是2×2实验设计分析的核心函数。
    生成标准化的对比表格，便于论文展示。

    对比内容：
    =========
    - 各条件的均值和标准差
    - 与基线的绝对差异
    - 与基线的相对差异（百分比变化）

    使用场景：
    =========
    >>> comparison = compare_conditions(
    ...     results,
    ...     ["metrics", "lsv_herding", "herding_ratio"],
    ...     baseline="baseline"
    ... )
    >>> print(comparison)
       condition  vs_baseline  condition_mean  baseline_mean  diff  diff_pct
       memory_only  baseline      0.42           0.35         0.07   20.0%
       social_only  baseline      0.58           0.35         0.23   65.7%
       full         baseline      0.65           0.35         0.30   85.7%

    Args:
        results: 实验结果字典（聚合后的结果）
        metric_path: 指标在字典中的路径
            例如 ["metrics", "hurst_exponent"] 表示
            results["conditions"][cond]["metrics"]["hurst_exponent"]
        baseline: 基线条件名称（默认"baseline"）

    Returns:
        包含对比统计的DataFrame

    Raises:
        ValueError: 基线条件不存在于结果中
    """
    conditions = list(results["conditions"].keys())

    # 验证基线条件存在
    if baseline not in conditions:
        raise ValueError(f"Baseline '{baseline}' not in conditions")

    comparisons = []

    # 遍历每个非基线条件
    for condition in conditions:
        if condition == baseline:
            continue  # 跳过基线自身

        comparison = {
            "condition": condition,
            "vs_baseline": baseline,
        }

        # 获取两个条件的统计数据
        cond_stats = results["conditions"][condition]
        base_stats = results["conditions"][baseline]

        # 沿路径导航到目标指标
        cond_val = cond_stats
        base_val = base_stats
        for key in metric_path:
            cond_val = cond_val[key]
            base_val = base_val[key]

        # 填充对比数据
        comparison["condition_mean"] = cond_val["mean"]
        comparison["condition_std"] = cond_val["std"]
        comparison["baseline_mean"] = base_val["mean"]
        comparison["baseline_std"] = base_val["std"]

        # 计算差异
        comparison["diff"] = cond_val["mean"] - base_val["mean"]

        # 计算百分比变化（避免除以零）
        if base_val["mean"] != 0:
            comparison["diff_pct"] = (cond_val["mean"] - base_val["mean"]) / base_val["mean"] * 100
        else:
            comparison["diff_pct"] = 0

        comparisons.append(comparison)

    return pd.DataFrame(comparisons)


# =============================================================================
# 人格类型绩效分析
# =============================================================================


def analyze_personality_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    按人格类型分析交易绩效

    分析不同人格类型Agent的投资表现差异，
    用于验证人格设定的有效性和策略差异。

    分析指标：
    =========
    - 收益率统计（均值、标准差、最小、最大）
    - 交易行为（买入比例、卖出比例、交易频率）

    使用场景：
    =========
    - 验证人格设定是否产生了预期的行为差异
    - 分析哪种策略在特定市场条件下更优
    - 为论文的人格分析章节提供数据支持

    Args:
        df: 决策记录DataFrame，需包含以下列：
            - round_num: 轮次号
            - personality: 人格类型
            - action: 交易动作（BUY/SELL/HOLD）
            - cash_before, holdings_before: 交易前状态
            - portfolio_value_after: 交易后组合价值

    Returns:
        按人格类型分组的绩效统计DataFrame
    """
    # 获取最后一轮数据（用于计算最终收益）
    final_round = df["round_num"].max()
    final_df = df[df["round_num"] == final_round]

    # 计算初始投资组合价值（假设初始价格为100）
    initial_value = final_df["cash_before"].iloc[0] + final_df["holdings_before"].iloc[0] * 100

    results = []

    # 按人格类型分组分析
    for personality in df["personality"].unique():
        pers_df = df[df["personality"] == personality]
        final_pers = final_df[final_df["personality"] == personality]

        # ---------------------------------------------------------------------
        # 投资组合绩效
        # ---------------------------------------------------------------------
        final_values = final_pers["portfolio_value_after"]
        # 收益率 = (最终价值 - 初始价值) / 初始价值 × 100%
        returns = (final_values - initial_value) / initial_value * 100

        # ---------------------------------------------------------------------
        # 交易行为统计
        # ---------------------------------------------------------------------
        # 总交易次数（排除HOLD）
        total_trades = len(pers_df[pers_df["action"] != "HOLD"])
        # 买入比例
        buy_ratio = len(pers_df[pers_df["action"] == "BUY"]) / len(pers_df)
        # 卖出比例
        sell_ratio = len(pers_df[pers_df["action"] == "SELL"]) / len(pers_df)

        results.append(
            {
                "personality": personality,
                "n_agents": len(final_pers),  # Agent数量
                "mean_return": returns.mean(),     # 平均收益率
                "std_return": returns.std(),       # 收益率标准差
                "min_return": returns.min(),       # 最小收益率
                "max_return": returns.max(),       # 最大收益率
                "buy_ratio": buy_ratio,            # 买入比例
                "sell_ratio": sell_ratio,          # 卖出比例
                "avg_trades_per_agent": total_trades / len(final_pers),  # 人均交易次数
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# 社交网络影响分析
# =============================================================================


def analyze_social_influence(df: pd.DataFrame) -> dict:
    """
    分析社交网络对交易决策的影响

    衡量Agent是否跟随社交共识（邻居的多数意见）。
    这是验证社交网络机制有效性的关键分析。

    分析内容：
    =========
    1. **整体跟风率**：Agent选择与社交共识相同动作的比例
    2. **按人格类型的跟风率**：验证Herding人格是否更易跟风
    3. **按共识强度的跟风率**：共识越强，跟风率是否越高

    共识强度定义：
    =============
    - strong（强）: max(buy_pct, sell_pct, hold_pct) >= 70%
    - moderate（中）: 50% <= max < 70%
    - weak（弱）: max < 50%

    Args:
        df: 决策记录DataFrame，需包含社交信息列：
            - social_buy_pct: 邻居中买入的比例
            - social_sell_pct: 邻居中卖出的比例
            - social_hold_pct: 邻居中持有的比例
            - action: Agent的实际动作
            - personality: Agent的人格类型

    Returns:
        包含社交影响分析结果的字典
    """
    # 筛选有社交数据的记录（排除第一轮和基线条件）
    social_df = df[df["social_buy_pct"] + df["social_sell_pct"] > 0]

    if len(social_df) == 0:
        return {"has_social_data": False}

    results = {"has_social_data": True}

    # =========================================================================
    # 确定社交共识（邻居的多数意见）
    # =========================================================================
    def get_consensus(row):
        """根据邻居动作比例确定共识"""
        if row["social_buy_pct"] > row["social_sell_pct"]:
            return "BUY"
        elif row["social_sell_pct"] > row["social_buy_pct"]:
            return "SELL"
        return "HOLD"

    social_df = social_df.copy()
    social_df["consensus"] = social_df.apply(get_consensus, axis=1)
    # Agent是否跟随了共识
    social_df["followed_consensus"] = social_df["action"] == social_df["consensus"]

    # =========================================================================
    # 整体跟风率
    # =========================================================================
    results["overall_follow_rate"] = social_df["followed_consensus"].mean()

    # =========================================================================
    # 按人格类型的跟风率
    # =========================================================================
    follow_by_personality = social_df.groupby("personality")["followed_consensus"].mean()
    results["follow_rate_by_personality"] = follow_by_personality.to_dict()

    # =========================================================================
    # 按共识强度的跟风率
    # =========================================================================
    def consensus_strength(row):
        """计算共识强度"""
        max_pct = max(row["social_buy_pct"], row["social_sell_pct"], row["social_hold_pct"])
        if max_pct >= 70:
            return "strong"   # 强共识
        elif max_pct >= 50:
            return "moderate" # 中等共识
        return "weak"         # 弱共识

    social_df["consensus_strength"] = social_df.apply(consensus_strength, axis=1)
    follow_by_strength = social_df.groupby("consensus_strength")["followed_consensus"].mean()
    results["follow_rate_by_strength"] = follow_by_strength.to_dict()

    return results


# =============================================================================
# LaTeX表格生成
# =============================================================================


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    将DataFrame转换为LaTeX表格格式

    生成符合学术论文格式的LaTeX表格代码。

    格式化规则：
    ===========
    - 百分比列：显示为 xx.x%
    - 其他浮点数：保留4位小数

    Args:
        df: 要转换的DataFrame
        caption: 表格标题
        label: LaTeX标签（用于\\ref{label}引用）

    Returns:
        LaTeX表格代码字符串
    """
    # 设置数字格式化器
    formatters = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            # 百分比列特殊处理
            if "pct" in col.lower() or "ratio" in col.lower() or "rate" in col.lower():
                formatters[col] = lambda x: f"{x*100:.1f}\\%"
            else:
                formatters[col] = lambda x: f"{x:.4f}"

    # 生成LaTeX代码
    latex = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        formatters=formatters,
        escape=False,  # 允许LaTeX特殊字符
    )

    return latex


# =============================================================================
# 结果加载
# =============================================================================


def load_experiment_results(results_dir: str | Path) -> dict:
    """
    从目录加载实验结果

    自动查找并加载JSON结果文件和Parquet决策文件。

    Args:
        results_dir: 结果目录路径

    Returns:
        包含聚合结果和决策DataFrame的字典

    Raises:
        FileNotFoundError: 未找到结果文件
    """
    results_dir = Path(results_dir)

    # 查找JSON结果文件
    json_files = list(results_dir.glob("*_results.json"))
    if not json_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    # 加载JSON
    with open(json_files[0]) as f:
        results = json.load(f)

    # 尝试加载决策记录
    parquet_files = list(results_dir.glob("*_decisions.parquet"))
    if parquet_files:
        results["decisions_df"] = pd.read_parquet(parquet_files[0])

    return results


# =============================================================================
# 描述性统计
# =============================================================================


def summary_statistics(data: list | np.ndarray) -> dict:
    """
    计算全面的描述性统计量

    提供数据分布的完整画像，用于初步数据探索。

    计算的统计量：
    =============
    - 集中趋势：均值、中位数
    - 离散程度：标准差、标准误差、四分位距
    - 分布形态：偏度、峰度
    - 极值：最小值、最大值、四分位数
    - 置信区间：95%置信区间

    Args:
        data: 观测数据

    Returns:
        包含各统计量的字典

    Example:
        >>> stats = summary_statistics(hurst_values)
        >>> print(f"均值: {stats['mean']:.3f} (95% CI: {stats['ci_95']})")
    """
    data = np.array(data)

    return {
        "n": len(data),                              # 样本量
        "mean": np.mean(data),                       # 均值
        "std": np.std(data, ddof=1),                 # 标准差（无偏）
        "se": stats.sem(data),                       # 标准误差
        "median": np.median(data),                   # 中位数
        "min": np.min(data),                         # 最小值
        "max": np.max(data),                         # 最大值
        "q1": np.percentile(data, 25),               # 第一四分位数
        "q3": np.percentile(data, 75),               # 第三四分位数
        "iqr": np.percentile(data, 75) - np.percentile(data, 25),  # 四分位距
        "skewness": stats.skew(data),                # 偏度
        "kurtosis": stats.kurtosis(data),            # 峰度
        "ci_95": confidence_interval(data, 0.95),    # 95%置信区间
    }
