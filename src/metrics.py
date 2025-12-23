"""
金融市场指标计算模块 (Financial Metrics Module)

本模块实现了用于分析ABM仿真结果的核心金融统计指标。
这些指标是论文数据分析的基础，用于验证仿真市场是否复现真实市场特征。

实现的指标类别：

1. 市场效率指标 (Market Efficiency):
   - Hurst Exponent: 赫斯特指数，衡量市场的长期记忆性
   - Autocorrelation: 自相关函数，检测收益率的序列相关性
   - Ljung-Box Test: 联合检验多阶自相关是否显著

2. 羊群效应指标 (Herding Metrics):
   - LSV Measure: Lakonishok-Shleifer-Vishny (1992) 羊群测度
   - CSAD: Cross-Sectional Absolute Deviation，横截面绝对偏差

3. 收益率分布指标 (Return Distribution):
   - Kurtosis: 峰度，衡量分布尾部厚度
   - Skewness: 偏度，衡量分布对称性
   - Jarque-Bera Test: 正态性检验

4. 波动率指标 (Volatility Metrics):
   - Volatility Clustering: 波动聚集性检验（ARCH效应）
   - Realized Volatility: 已实现波动率

参考文献：
    - Hurst, H.E. (1951). Long-term storage capacity of reservoirs.
    - Lakonishok, J., Shleifer, A., & Vishny, R.W. (1992). The impact of
      institutional trading on stock prices. Journal of Financial Economics.
    - Cont, R. (2001). Empirical properties of asset returns: stylized facts
      and statistical issues. Quantitative Finance.

作者: SuZX
日期: 2024
"""

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 收益率计算函数
# =============================================================================


def calculate_returns(prices: list[float] | np.ndarray) -> np.ndarray:
    """
    计算对数收益率序列

    对数收益率（Log Returns）的计算公式：
        r_t = ln(P_t) - ln(P_{t-1}) = ln(P_t / P_{t-1})

    对数收益率的优点：
        1. 具有时间可加性：多期收益率 = 各期对数收益率之和
        2. 近似正态分布（在较短时间尺度上）
        3. 避免价格为负的问题

    Args:
        prices: 价格序列，长度为 n

    Returns:
        对数收益率序列，长度为 n-1

    Example:
        >>> prices = [100, 101, 99, 102]
        >>> returns = calculate_returns(prices)
        >>> # returns ≈ [0.00995, -0.02005, 0.02985]
    """
    prices = np.array(prices)

    # 对数收益率: r_t = ln(P_t / P_{t-1})
    returns = np.diff(np.log(prices))

    return returns


def calculate_simple_returns(prices: list[float] | np.ndarray) -> np.ndarray:
    """
    计算简单收益率序列

    简单收益率（Simple Returns）的计算公式：
        R_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1

    简单收益率的特点：
        1. 直观易理解，直接反映价格变化百分比
        2. 适用于计算组合收益率（加权平均）
        3. 在大收益率时与对数收益率有显著差异

    Args:
        prices: 价格序列，长度为 n

    Returns:
        简单收益率序列，长度为 n-1

    Example:
        >>> prices = [100, 105, 102]
        >>> returns = calculate_simple_returns(prices)
        >>> # returns = [0.05, -0.02857...]
    """
    prices = np.array(prices)

    # 简单收益率: R_t = (P_t - P_{t-1}) / P_{t-1}
    returns = np.diff(prices) / prices[:-1]

    return returns


# =============================================================================
# 市场效率指标
# =============================================================================


def hurst_exponent(prices: list[float] | np.ndarray, max_lag: int | None = None) -> float:
    """
    使用R/S分析法计算Hurst指数

    Hurst指数（H）是衡量时间序列长期记忆性的关键指标，
    由英国水文学家 Harold Edwin Hurst 于1951年提出。

    数学原理：
    ==========
    R/S分析法（Rescaled Range Analysis）步骤：

    1. 对于窗口大小 n，将序列分成 m = N/n 个子区间
    2. 对每个子区间计算：
       - 均值：μ = (1/n) Σ x_i
       - 累积离差：Y_t = Σ(x_i - μ)
       - 极差：R = max(Y_t) - min(Y_t)
       - 标准差：S = sqrt((1/n) Σ(x_i - μ)²)
       - R/S 比率：R/S
    3. 对不同窗口大小 n，计算平均 R/S
    4. 在 log-log 坐标下拟合：log(R/S) = H * log(n) + c
       斜率 H 即为 Hurst 指数

    Hurst指数的解释：
    ================
        - H < 0.5: 均值回归（Anti-persistent），价格趋势倾向于反转
        - H = 0.5: 随机游走（Random Walk），市场有效
        - H > 0.5: 趋势持续（Persistent），价格趋势倾向于延续

    Args:
        prices: 价格序列
        max_lag: R/S计算的最大滞后期，默认为 min(n/2, 100)

    Returns:
        Hurst指数，范围通常在 [0, 1]

    Note:
        - 可靠估计需要至少 100 个数据点
        - H > 0.5 表明存在动量效应，可能违反有效市场假说
    """
    prices = np.array(prices)
    n = len(prices)

    # 数据量不足时返回中性值
    if n < 20:
        return 0.5

    # 设置最大滞后期
    if max_lag is None:
        max_lag = min(n // 2, 100)

    # 计算对数收益率
    returns = calculate_returns(prices)

    # 存储不同窗口大小的 R/S 值
    lags = []
    rs_values = []

    # ==========================================================================
    # 核心计算：对每个窗口大小计算 R/S
    # ==========================================================================
    for lag in range(10, max_lag + 1):
        rs_list = []

        # 将序列分成不重叠的子区间
        for start in range(0, len(returns) - lag + 1, lag):
            subset = returns[start : start + lag]
            if len(subset) < lag:
                continue

            # 步骤1: 计算子区间均值
            mean = np.mean(subset)

            # 步骤2: 计算累积离差序列
            # cumdev_t = Σ(x_i - μ), i=1 to t
            cumdev = np.cumsum(subset - mean)

            # 步骤3: 计算极差 R = max(cumdev) - min(cumdev)
            r = np.max(cumdev) - np.min(cumdev)

            # 步骤4: 计算标准差 S（使用无偏估计，ddof=1）
            s = np.std(subset, ddof=1)

            # 步骤5: 计算 R/S 比率
            if s > 0:
                rs_list.append(r / s)

        # 计算该窗口大小下的平均 R/S
        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))

    # 数据点不足时返回中性值
    if len(lags) < 2:
        return 0.5

    # ==========================================================================
    # 线性回归估计 Hurst 指数
    # 在 log-log 坐标下：log(R/S) = H * log(n) + c
    # 斜率 H 即为 Hurst 指数
    # ==========================================================================
    log_lags = np.log(lags)
    log_rs = np.log(rs_values)

    # 使用最小二乘法拟合
    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)

    return slope


def autocorrelation(returns: np.ndarray, max_lag: int = 20) -> dict:
    """
    计算收益率的自相关函数 (ACF)

    自相关函数衡量时间序列与其滞后版本之间的相关性。

    数学定义：
    =========
    自相关系数 ρ(k) 定义为：

        ρ(k) = Cov(r_t, r_{t-k}) / Var(r_t)
             = E[(r_t - μ)(r_{t-k} - μ)] / σ²

    其中：
        - k: 滞后阶数
        - μ: 收益率均值
        - σ²: 收益率方差

    样本估计公式：
        ρ̂(k) = [Σ(r_t - μ̂)(r_{t-k} - μ̂)] / [(n-k) * σ̂²]

    解释：
    =====
        - ρ(k) > 0: 正自相关，过去的正收益预示未来的正收益（动量）
        - ρ(k) < 0: 负自相关，过去的正收益预示未来的负收益（反转）
        - ρ(k) ≈ 0: 无自相关，收益率独立（有效市场）

    Args:
        returns: 收益率序列
        max_lag: 计算的最大滞后阶数

    Returns:
        字典，键为滞后阶数，值为自相关系数

    Note:
        在有效市场假说下，收益率应该无显著自相关
    """
    n = len(returns)

    # 调整最大滞后期
    if n < max_lag + 1:
        max_lag = n - 1

    result = {}

    # 计算均值和方差
    mean = np.mean(returns)
    var = np.var(returns)

    # 方差为零时返回全零
    if var == 0:
        return {i: 0 for i in range(1, max_lag + 1)}

    # 计算各阶自相关系数
    for lag in range(1, max_lag + 1):
        # 计算滞后 k 阶的自协方差
        # Cov(r_t, r_{t-k}) = E[(r_t - μ)(r_{t-k} - μ)]
        cov = np.sum((returns[:-lag] - mean) * (returns[lag:] - mean)) / (n - lag)

        # 自相关 = 自协方差 / 方差
        result[lag] = cov / var

    return result


def ljung_box_test(returns: np.ndarray, lags: int = 10) -> dict:
    """
    Ljung-Box Q检验：联合检验多阶自相关是否显著

    Ljung-Box检验用于检验一组自相关系数是否全为零。

    数学原理：
    =========
    原假设 H0: ρ(1) = ρ(2) = ... = ρ(m) = 0（无自相关）
    备择假设 H1: 至少有一个 ρ(k) ≠ 0

    Q统计量计算公式（Ljung-Box修正版）：

        Q = n(n+2) * Σ[ρ̂(k)² / (n-k)]   , k=1 to m

    其中：
        - n: 样本量
        - m: 检验的滞后阶数
        - ρ̂(k): 样本自相关系数

    在原假设下，Q 统计量渐近服从 χ²(m) 分布。

    判断标准：
    =========
        - p-value < 0.05: 拒绝原假设，存在显著自相关
        - p-value ≥ 0.05: 不拒绝原假设，无显著自相关

    Args:
        returns: 收益率序列
        lags: 检验的滞后阶数 m

    Returns:
        包含 Q 统计量和 p 值的字典

    Note:
        该检验常用于验证市场有效性：有效市场的收益率应无显著自相关
    """
    n = len(returns)

    # 数据量不足时返回无显著结果
    if n < lags + 1:
        return {"Q": 0, "p_value": 1.0}

    # 计算自相关系数
    acf = autocorrelation(returns, lags)

    # ==========================================================================
    # 计算 Ljung-Box Q 统计量
    # Q = n(n+2) * Σ[ρ̂(k)² / (n-k)]
    # ==========================================================================
    q_stat = n * (n + 2) * sum((acf[k] ** 2) / (n - k) for k in range(1, lags + 1))

    # 计算 p 值（Q 统计量服从 χ²(lags) 分布）
    p_value = 1 - stats.chi2.cdf(q_stat, lags)

    return {"Q": q_stat, "p_value": p_value}


# =============================================================================
# 羊群效应指标
# =============================================================================


def lsv_herding_measure(
    df: pd.DataFrame,
    action_col: str = "action",
    round_col: str = "round_num",
) -> dict:
    """
    计算LSV羊群测度 (Lakonishok, Shleifer, Vishny 1992)

    LSV测度是最广泛使用的羊群效应指标之一，衡量交易者在同一方向
    交易的倾向是否超过随机预期。

    数学定义：
    =========
    对于第 t 轮：

        H_t = |p_t - E[p_t]| - E[|p_t - E[p_t]|]

    其中：
        - p_t = B_t / (B_t + S_t): 买方比例（买方数 / 活跃交易者数）
        - E[p_t] = 0.5: 在无羊群效应时的期望买方比例
        - E[|p_t - E[p_t]|]: 调整因子，消除小样本偏差

    解释：
    =====
        - H_t > 0: 存在羊群效应（交易者倾向于同方向交易）
        - H_t ≤ 0: 无显著羊群效应

    调整因子的作用：
        在小样本下，即使交易者随机交易，|p_t - 0.5| 的期望也不为零。
        调整因子消除了这种小样本偏差。

    Args:
        df: 包含交易决策记录的 DataFrame
        action_col: 交易动作列名（应包含 'BUY', 'SELL', 'HOLD'）
        round_col: 轮次列名

    Returns:
        包含以下键的字典：
        - lsv_mean: LSV测度均值
        - lsv_std: LSV测度标准差
        - lsv_max: LSV测度最大值
        - herding_rounds: 存在显著羊群效应的轮次数
        - total_rounds: 总轮次数
        - herding_ratio: 羊群效应比例
        - buy_ratio_mean: 平均买方比例
        - buy_ratio_std: 买方比例标准差

    Reference:
        Lakonishok, J., Shleifer, A., & Vishny, R.W. (1992).
        The impact of institutional trading on stock prices.
        Journal of Financial Economics, 32(1), 23-43.
    """
    results = []

    # 遍历每一轮计算 LSV 测度
    for round_num in df[round_col].unique():
        round_df = df[df[round_col] == round_num]

        total = len(round_df)
        if total == 0:
            continue

        # 统计买方和卖方数量
        buyers = len(round_df[round_df[action_col] == "BUY"])
        sellers = len(round_df[round_df[action_col] == "SELL"])

        # 活跃交易者 = 买方 + 卖方（排除持有者）
        active = buyers + sellers
        if active == 0:
            continue

        # 计算买方比例 p_t = B_t / (B_t + S_t)
        p_t = buyers / active

        results.append(
            {
                "round": round_num,
                "p_t": p_t,
                "buyers": buyers,
                "sellers": sellers,
                "active": active,
            }
        )

    # 无有效数据时返回零值
    if not results:
        return {"lsv_mean": 0, "lsv_std": 0, "herding_rounds": 0}

    results_df = pd.DataFrame(results)

    # ==========================================================================
    # 计算 LSV 测度
    # H_t = |p_t - E[p_t]| - E[|p_t - E[p_t]|]
    # ==========================================================================

    # 无羊群效应时的期望买方比例
    expected_p = 0.5

    # 计算 |p_t - E[p_t]|
    results_df["abs_dev"] = np.abs(results_df["p_t"] - expected_p)

    # 计算调整因子 E[|p_t - E[p_t]|]
    # 使用样本均值作为估计
    adjustment = results_df["abs_dev"].mean()

    # 计算每轮的 LSV 测度
    results_df["lsv"] = results_df["abs_dev"] - adjustment

    # 统计显著羊群效应的轮次（LSV > 0）
    herding_rounds = len(results_df[results_df["lsv"] > 0])

    return {
        "lsv_mean": results_df["lsv"].mean(),
        "lsv_std": results_df["lsv"].std(),
        "lsv_max": results_df["lsv"].max(),
        "herding_rounds": herding_rounds,
        "total_rounds": len(results_df),
        "herding_ratio": herding_rounds / len(results_df) if results_df.shape[0] > 0 else 0,
        "buy_ratio_mean": results_df["p_t"].mean(),
        "buy_ratio_std": results_df["p_t"].std(),
    }


def csad_measure(
    df: pd.DataFrame,
    portfolio_col: str = "portfolio_value_after",
    round_col: str = "round_num",
) -> dict:
    """
    计算CSAD (Cross-Sectional Absolute Deviation) 横截面绝对偏差

    CSAD衡量个体收益率相对于市场收益率的离散程度。
    低CSAD表示高度一致的收益（可能存在羊群效应）。

    数学定义：
    =========
    对于第 t 轮：

        CSAD_t = (1/N) * Σ|R_{i,t} - R_{m,t}|

    其中：
        - R_{i,t}: 第 i 个Agent在第 t 轮的收益率
        - R_{m,t}: 第 t 轮的市场平均收益率 = (1/N) Σ R_{i,t}
        - N: Agent数量

    解释：
    =====
        - 高 CSAD: 收益率分散，个体行为异质
        - 低 CSAD: 收益率集中，可能存在羊群效应

    与传统理论的关系：
    =================
    根据 CAPM，在市场极端波动时，CSAD 应该与 |R_m| 正相关
    （个体对市场信息的反应程度不同）。
    若 CSAD 与 |R_m| 或 R_m² 负相关，则表明存在羊群效应。

    Args:
        df: 包含投资组合价值的 DataFrame
        portfolio_col: 投资组合价值列名
        round_col: 轮次列名

    Returns:
        包含 CSAD 统计量的字典

    Reference:
        Chang, E.C., Cheng, J.W., & Khorana, A. (2000).
        An examination of herd behavior in equity markets.
        Journal of Banking & Finance.
    """
    rounds = sorted(df[round_col].unique())
    csad_values = []

    # 需要前后两轮数据来计算收益率
    for i, round_num in enumerate(rounds):
        if i == 0:
            continue

        prev_round = rounds[i - 1]

        # 获取当前轮和上一轮的投资组合价值
        current = df[df[round_col] == round_num].set_index("agent_id")[portfolio_col]
        previous = df[df[round_col] == prev_round].set_index("agent_id")[portfolio_col]

        # 找出两轮都存在的 Agent
        common_agents = current.index.intersection(previous.index)
        if len(common_agents) < 2:
            continue

        # =======================================================================
        # 计算个体收益率
        # R_{i,t} = (V_{i,t} - V_{i,t-1}) / V_{i,t-1}
        # =======================================================================
        returns = (current[common_agents] - previous[common_agents]) / previous[common_agents]

        # 计算市场收益率（所有Agent收益率的均值）
        # R_{m,t} = (1/N) Σ R_{i,t}
        market_return = returns.mean()

        # =======================================================================
        # 计算 CSAD
        # CSAD_t = (1/N) * Σ|R_{i,t} - R_{m,t}|
        # =======================================================================
        csad = np.abs(returns - market_return).mean()
        csad_values.append(csad)

    # 无有效数据时返回零值
    if not csad_values:
        return {"csad_mean": 0, "csad_std": 0}

    return {
        "csad_mean": np.mean(csad_values),
        "csad_std": np.std(csad_values),
        "csad_min": np.min(csad_values),
        "csad_max": np.max(csad_values),
    }


# =============================================================================
# 收益率分布指标
# =============================================================================


def return_distribution_stats(returns: np.ndarray) -> dict:
    """
    计算收益率分布的统计特征

    金融收益率分布通常具有"尖峰厚尾"特征，这是真实市场的典型事实之一。

    计算的统计量：
    =============

    1. 峰度 (Kurtosis):
       K = E[(X-μ)⁴] / σ⁴ - 3

       - K > 0: 尖峰厚尾（Leptokurtic），极端事件概率高于正态分布
       - K = 0: 正态分布（Mesokurtic）
       - K < 0: 扁峰薄尾（Platykurtic），极端事件概率低于正态分布

       scipy.stats.kurtosis 默认返回超额峰度（减去3后的值）

    2. 偏度 (Skewness):
       S = E[(X-μ)³] / σ³

       - S > 0: 右偏（正偏），右尾更长
       - S = 0: 对称分布
       - S < 0: 左偏（负偏），左尾更长

       金融收益率通常呈负偏（崩盘风险大于暴涨）

    Args:
        returns: 收益率序列

    Returns:
        包含分布统计量的字典

    Note:
        真实金融市场的典型特征：
        - 超额峰度 > 0（厚尾）
        - 轻微负偏度（非对称风险）
    """
    # 数据量不足时返回默认值
    if len(returns) < 4:
        return {
            "mean": 0,
            "std": 0,
            "skewness": 0,
            "kurtosis": 0,
            "is_fat_tail": False,
        }

    return {
        # 一阶矩：均值
        "mean": np.mean(returns),
        # 二阶矩：标准差（波动率）
        "std": np.std(returns),
        # 三阶矩：偏度（分布对称性）
        "skewness": stats.skew(returns),
        # 四阶矩：超额峰度（尾部厚度，正态分布=0）
        "kurtosis": stats.kurtosis(returns),
        # 是否厚尾（超额峰度>0表示尖峰厚尾）
        "is_fat_tail": stats.kurtosis(returns) > 0,
        # 其他描述性统计
        "min": np.min(returns),
        "max": np.max(returns),
        "median": np.median(returns),
        # 四分位距（IQR），衡量分布的离散程度
        "iqr": np.percentile(returns, 75) - np.percentile(returns, 25),
    }


def jarque_bera_test(returns: np.ndarray) -> dict:
    """
    Jarque-Bera 正态性检验

    JB检验通过偏度和峰度联合检验数据是否来自正态分布。

    数学原理：
    =========
    JB统计量计算公式：

        JB = (n/6) * [S² + (K²/4)]

    其中：
        - n: 样本量
        - S: 样本偏度
        - K: 样本超额峰度

    原假设 H0: 数据来自正态分布（S=0, K=0）

    在原假设下，JB 统计量渐近服从 χ²(2) 分布。

    判断标准：
    =========
        - p-value < 0.05: 拒绝正态性假设
        - p-value ≥ 0.05: 不拒绝正态性假设

    Args:
        returns: 收益率序列

    Returns:
        包含 JB 统计量、p 值和正态性判断的字典

    Note:
        金融收益率通常不服从正态分布（JB检验会拒绝正态性）
    """
    # 数据量不足时返回默认值
    if len(returns) < 4:
        return {"jb_stat": 0, "p_value": 1.0, "is_normal": True}

    # 执行 Jarque-Bera 检验
    jb_stat, p_value = stats.jarque_bera(returns)

    return {
        "jb_stat": jb_stat,
        "p_value": p_value,
        # 在 5% 显著性水平下判断是否正态
        "is_normal": p_value > 0.05,
    }


# =============================================================================
# 波动率指标
# =============================================================================


def volatility_clustering_test(returns: np.ndarray, lags: int = 5) -> dict:
    """
    波动聚集性检验（ARCH效应检验）

    波动聚集（Volatility Clustering）是金融市场的典型事实之一：
    大波动往往跟随大波动，小波动往往跟随小波动。

    数学原理：
    =========
    如果存在波动聚集，则平方收益率 r_t² 应该存在自相关。

    检验方法：对平方收益率序列进行 Ljung-Box Q 检验

        H0: 平方收益率无自相关（无 ARCH 效应）
        H1: 平方收益率存在自相关（存在 ARCH 效应）

    这等价于检验 ARCH(q) 效应：
        r_t = σ_t * ε_t
        σ_t² = α_0 + Σ α_i * r_{t-i}²

    Args:
        returns: 收益率序列
        lags: 检验的滞后阶数

    Returns:
        包含检验结果的字典

    Note:
        - has_clustering=True 表示存在波动聚集
        - 这是真实金融市场的典型特征
    """
    # 数据量不足时返回默认值
    if len(returns) < lags + 5:
        return {"has_clustering": False, "Q": 0, "p_value": 1.0}

    # 计算平方收益率
    # 如果存在波动聚集，r_t² 应该具有自相关性
    squared_returns = returns**2

    # 对平方收益率进行 Ljung-Box 检验
    result = ljung_box_test(squared_returns, lags)

    return {
        # p < 0.05 表示存在显著的波动聚集
        "has_clustering": result["p_value"] < 0.05,
        "Q": result["Q"],
        "p_value": result["p_value"],
    }


def realized_volatility(returns: np.ndarray, window: int = 5) -> np.ndarray:
    """
    计算滚动已实现波动率

    已实现波动率（Realized Volatility）是使用历史数据估计的波动率。

    数学定义：
    =========
    对于窗口大小 w，第 t 期的已实现波动率：

        RV_t = sqrt[(1/w) * Σ r_{t-i}²]  , i=0 to w-1

    或使用样本标准差：

        RV_t = std(r_{t-w+1}, ..., r_t)

    本函数使用样本标准差方法。

    Args:
        returns: 收益率序列
        window: 滚动窗口大小

    Returns:
        滚动已实现波动率序列

    Note:
        - 波动率通常随时间变化（波动率聚集）
        - 高波动率时期往往对应市场压力
    """
    # 数据量不足时返回整体标准差
    if len(returns) < window:
        return np.array([np.std(returns)])

    # 计算滚动标准差作为已实现波动率
    vol = []
    for i in range(window, len(returns) + 1):
        vol.append(np.std(returns[i - window : i]))

    return np.array(vol)


# =============================================================================
# 辅助性能指标
# =============================================================================


def calculate_max_drawdown(prices: list[float] | np.ndarray) -> float:
    """
    计算最大回撤 (Maximum Drawdown)

    最大回撤衡量从历史最高点到最低点的最大跌幅，
    是风险管理中的重要指标。

    数学定义：
    =========
    对于价格序列 P_t：

        Peak_t = max(P_1, P_2, ..., P_t)     # 历史最高价
        Drawdown_t = (Peak_t - P_t) / Peak_t  # 当前回撤
        MDD = max(Drawdown_t)                 # 最大回撤

    Args:
        prices: 价格序列

    Returns:
        最大回撤（0到1之间，1表示100%回撤）

    Note:
        - MDD 越大，表示投资风险越高
        - 专业投资者通常关注 MDD < 20%
    """
    prices = np.array(prices)

    # 计算历史最高价（累积最大值）
    peak = np.maximum.accumulate(prices)

    # 计算每个时点的回撤
    drawdown = (peak - prices) / peak

    # 返回最大回撤
    return np.max(drawdown)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free: float = 0) -> float:
    """
    计算夏普比率 (Sharpe Ratio)

    夏普比率衡量单位风险的超额收益，是最常用的风险调整收益指标。

    数学定义：
    =========
        SR = (E[R] - R_f) / σ(R)

    其中：
        - E[R]: 期望收益率
        - R_f: 无风险利率
        - σ(R): 收益率标准差

    年化夏普比率（假设日数据）：
        SR_annual = SR_daily * sqrt(252)

    解释：
    =====
        - SR > 1: 良好的风险调整收益
        - SR > 2: 优秀的风险调整收益
        - SR < 0: 收益低于无风险利率

    Args:
        returns: 收益率序列
        risk_free: 无风险利率（默认为0）

    Returns:
        年化夏普比率
    """
    # 处理边界情况
    if len(returns) == 0 or np.std(returns) == 0:
        return 0

    # 计算超额收益
    excess_returns = returns - risk_free

    # 计算夏普比率并年化（乘以 sqrt(252)，假设日频数据）
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


# =============================================================================
# 综合指标计算器
# =============================================================================


def calculate_all_metrics(
    prices: list[float],
    df: pd.DataFrame,
) -> dict:
    """
    计算仿真结果的全部金融指标

    该函数是指标计算的主入口，整合所有指标计算，
    为实验分析提供完整的数据支持。

    计算的指标类别：
    ===============
    1. 市场效率指标：
       - hurst_exponent: Hurst指数
       - autocorrelation: 自相关函数
       - ljung_box: Ljung-Box Q检验

    2. 羊群效应指标：
       - lsv_herding: LSV羊群测度
       - csad: 横截面绝对偏差

    3. 收益率分布：
       - return_stats: 均值、标准差、偏度、峰度
       - normality_test: Jarque-Bera检验

    4. 波动率指标：
       - volatility_clustering: 波动聚集检验
       - avg_volatility: 平均波动率

    5. 绩效指标：
       - total_return: 总收益率
       - max_drawdown: 最大回撤
       - sharpe_ratio: 夏普比率

    Args:
        prices: 价格历史序列
        df: 包含所有决策记录的 DataFrame

    Returns:
        包含全部指标的嵌套字典

    Example:
        >>> metrics = calculate_all_metrics(market.price_history, decisions_df)
        >>> print(f"Hurst: {metrics['hurst_exponent']:.3f}")
        >>> print(f"Kurtosis: {metrics['return_stats']['kurtosis']:.3f}")
    """
    # 计算对数收益率和简单收益率
    returns = calculate_returns(prices)
    simple_returns = calculate_simple_returns(prices)

    # 构建指标字典
    metrics = {
        # =====================================================================
        # 市场效率指标
        # =====================================================================
        "hurst_exponent": hurst_exponent(prices),
        "autocorrelation": autocorrelation(returns, min(10, len(returns) - 1)),
        "ljung_box": ljung_box_test(returns),

        # =====================================================================
        # 羊群效应指标
        # =====================================================================
        "lsv_herding": lsv_herding_measure(df),
        "csad": csad_measure(df),

        # =====================================================================
        # 收益率分布统计
        # =====================================================================
        "return_stats": return_distribution_stats(returns),
        "normality_test": jarque_bera_test(returns),

        # =====================================================================
        # 波动率指标
        # =====================================================================
        "volatility_clustering": volatility_clustering_test(returns),
        "avg_volatility": np.std(returns) if len(returns) > 0 else 0,

        # =====================================================================
        # 绩效指标
        # =====================================================================
        "total_return": (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0,
        "max_drawdown": calculate_max_drawdown(prices),
        "sharpe_ratio": calculate_sharpe_ratio(simple_returns),
    }

    return metrics


# =============================================================================
# 报告格式化
# =============================================================================


def format_metrics_report(metrics: dict) -> str:
    """
    将指标格式化为可读的文本报告

    生成结构化的文本报告，方便控制台输出和日志记录。

    报告结构：
    =========
    1. 市场效率分析
    2. 羊群效应分析
    3. 收益率分布分析
    4. 波动率分析
    5. 绩效总结

    Args:
        metrics: calculate_all_metrics 返回的指标字典

    Returns:
        格式化的文本报告字符串
    """
    lines = [
        "=" * 60,
        "FINANCIAL METRICS REPORT",
        "=" * 60,
        "",
        "[Market Efficiency]",
        f"  Hurst Exponent: {metrics['hurst_exponent']:.4f}",
    ]

    # 解释 Hurst 指数
    if metrics["hurst_exponent"] < 0.45:
        lines.append("    → Mean-reverting market (anti-persistent)")
    elif metrics["hurst_exponent"] > 0.55:
        lines.append("    → Trending market (persistent)")
    else:
        lines.append("    → Random walk (efficient market)")

    # Ljung-Box 检验结果
    lb = metrics["ljung_box"]
    lines.extend(
        [
            f"  Ljung-Box Q: {lb['Q']:.2f} (p={lb['p_value']:.4f})",
            f"    → {'Significant autocorrelation' if lb['p_value'] < 0.05 else 'No significant autocorrelation'}",
            "",
            "[Herding Behavior]",
        ]
    )

    # LSV 羊群测度
    lsv = metrics["lsv_herding"]
    lines.extend(
        [
            f"  LSV Measure: {lsv['lsv_mean']:.4f} (±{lsv['lsv_std']:.4f})",
            f"  Herding Rounds: {lsv['herding_rounds']}/{lsv['total_rounds']} ({lsv['herding_ratio']*100:.1f}%)",
            f"  Buy Ratio: {lsv['buy_ratio_mean']:.2f} (±{lsv['buy_ratio_std']:.2f})",
        ]
    )

    # CSAD
    csad = metrics["csad"]
    lines.extend(
        [
            f"  CSAD: {csad['csad_mean']:.4f} (±{csad['csad_std']:.4f})",
            "",
            "[Return Distribution]",
        ]
    )

    # 收益率分布统计
    rs = metrics["return_stats"]
    lines.extend(
        [
            f"  Mean: {rs['mean']*100:.4f}%",
            f"  Std Dev: {rs['std']*100:.4f}%",
            f"  Skewness: {rs['skewness']:.4f}",
            f"  Kurtosis: {rs['kurtosis']:.4f}",
            f"    → {'Fat-tailed (leptokurtic)' if rs['is_fat_tail'] else 'Normal/thin-tailed'}",
        ]
    )

    # Jarque-Bera 检验
    jb = metrics["normality_test"]
    lines.extend(
        [
            f"  Jarque-Bera: {jb['jb_stat']:.2f} (p={jb['p_value']:.4f})",
            f"    → {'Normal distribution' if jb['is_normal'] else 'Non-normal distribution'}",
            "",
            "[Volatility]",
        ]
    )

    # 波动率聚集检验
    vc = metrics["volatility_clustering"]
    lines.extend(
        [
            f"  Volatility Clustering: {'Yes' if vc['has_clustering'] else 'No'} (p={vc['p_value']:.4f})",
            f"  Average Volatility: {metrics['avg_volatility']*100:.4f}%",
            "",
            "[Performance]",
            f"  Total Return: {metrics['total_return']*100:.2f}%",
            f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%",
            f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}",
            "",
            "=" * 60,
        ]
    )

    return "\n".join(lines)
