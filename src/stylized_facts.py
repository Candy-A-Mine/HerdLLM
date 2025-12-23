"""
金融典型事实验证模块 (Stylized Facts Validation Module)

本模块用于验证ABM仿真生成的价格序列是否具有真实金融市场的典型统计特征。
这些特征被称为"Stylized Facts"，是金融市场实证研究中广泛认可的普遍现象。

Stylized Facts (典型事实) 背景：
    1992年，Cont等学者总结了金融市场收益率的若干统计规律，
    这些规律在不同市场、不同时间尺度上普遍存在，被称为"Stylized Facts"。
    一个好的金融市场模型应该能够复现这些典型特征。

本模块检验的三个核心特征：

1. 厚尾特性 (Fat Tails / Leptokurtosis):
   - 现象：金融收益率分布的尾部比正态分布更厚
   - 度量：超额峰度 (Excess Kurtosis) > 0，即峰度 > 3
   - 含义：极端事件（大涨大跌）发生的概率高于正态分布预期
   - 经济解释：市场恐慌、羊群效应、信息冲击等导致极端波动

2. 波动聚集 (Volatility Clustering):
   - 现象：大波动往往跟随大波动，小波动往往跟随小波动
   - 度量：绝对收益率的自相关系数显著不为零
   - 含义：波动率具有持续性，今天的高波动预示明天可能继续高波动
   - 经济解释：信息逐步扩散、投资者情绪传染、反馈交易

3. 长记忆性 (Long Memory / Persistence):
   - 现象：收益率序列存在长期依赖关系
   - 度量：Hurst指数 H > 0.5 表示趋势持续，H < 0.5 表示均值回归
   - 含义：过去的价格走势对未来有预测作用
   - 经济解释：趋势跟随交易、动量效应、市场非有效性

参考文献：
    - Cont, R. (2001). Empirical properties of asset returns: stylized facts
      and statistical issues. Quantitative Finance, 1(2), 223-236.
    - Mandelbrot, B. (1963). The variation of certain speculative prices.
      The Journal of Business, 36(4), 394-419.

类结构：
    StylizedFactsAnalyzer: 主分析类
    ├── analyze_fat_tails(): 厚尾特性分析
    ├── analyze_volatility_clustering(): 波动聚集分析
    ├── analyze_long_memory(): 长记忆性分析
    ├── generate_report(): 生成完整报告
    └── plot_stylized_facts(): 生成可视化拼图

作者: SuZX
日期: 2024
"""

# =============================================================================
# 导入依赖
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np                          # 数值计算
import matplotlib.pyplot as plt             # 绑图
from scipy import stats                     # 统计函数
from scipy.stats import norm, kurtosis, skew  # 正态分布、峰度、偏度


# =============================================================================
# 类型定义
# =============================================================================

# 分析结果字典类型
AnalysisResult = Dict[str, Any]


# =============================================================================
# Stylized Facts 分析器类
# =============================================================================

@dataclass
class StylizedFactsAnalyzer:
    """
    金融典型事实分析器

    该类对价格序列进行Stylized Facts检验，验证仿真数据
    是否具有真实金融市场的典型统计特征。

    分析流程：
        1. 从价格序列计算收益率序列
        2. 分析厚尾特性（峰度检验）
        3. 分析波动聚集（自相关检验）
        4. 分析长记忆性（Hurst指数）
        5. 生成综合报告和可视化

    Attributes:
        price_history: 价格历史序列
        returns: 计算得到的收益率序列
        results: 存储各项分析结果的字典

    Example:
        >>> analyzer = StylizedFactsAnalyzer(price_history=[100, 101, 99, 102, ...])
        >>> report = analyzer.generate_report()
        >>> analyzer.plot_stylized_facts("output/stylized_facts.png")
    """

    # -------------------------------------------------------------------------
    # 输入数据
    # -------------------------------------------------------------------------

    # 价格历史序列
    price_history: List[float]

    # -------------------------------------------------------------------------
    # 计算结果（延迟初始化）
    # -------------------------------------------------------------------------

    # 收益率序列（在__post_init__中计算）
    returns: np.ndarray = field(init=False, repr=False)

    # 分析结果存储字典
    results: Dict[str, AnalysisResult] = field(
        default_factory=dict,
        init=False,
        repr=False
    )

    # -------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------

    # ACF计算的最大滞后阶数
    MAX_LAG: int = 20

    # 统计显著性水平
    SIGNIFICANCE_LEVEL: float = 0.05

    def __post_init__(self) -> None:
        """
        构造后初始化

        计算收益率序列，为后续分析做准备。
        收益率计算公式：r_t = (P_t - P_{t-1}) / P_{t-1}
        """
        # 将价格列表转换为numpy数组
        prices = np.array(self.price_history)

        # 计算简单收益率
        # r_t = (P_t - P_{t-1}) / P_{t-1}
        self.returns = np.diff(prices) / prices[:-1]

    # =========================================================================
    # 厚尾特性分析
    # =========================================================================

    def analyze_fat_tails(self) -> AnalysisResult:
        """
        分析收益率分布的厚尾特性

        厚尾特性（Leptokurtosis）是金融收益率最显著的统计特征之一。
        正态分布的峰度为3，金融收益率的峰度通常远大于3。

        检验方法：
            1. 计算峰度（Kurtosis）：衡量分布尾部厚度
            2. 计算偏度（Skewness）：衡量分布对称性
            3. Jarque-Bera检验：联合检验正态性
            4. 与正态分布对比的可视化准备

        Returns:
            包含以下键的分析结果字典：
            - kurtosis: 峰度值
            - excess_kurtosis: 超额峰度（峰度-3）
            - skewness: 偏度值
            - jarque_bera_stat: JB检验统计量
            - jarque_bera_pvalue: JB检验p值
            - is_fat_tailed: 是否具有厚尾特性（布尔值）
            - interpretation: 结果解释文本

        Note:
            - 峰度 > 3（超额峰度 > 0）表示厚尾
            - JB检验p值 < 0.05 表示显著非正态
        """
        # ---------------------------------------------------------------------
        # 计算描述性统计量
        # ---------------------------------------------------------------------

        # 计算峰度（Fisher定义，即超额峰度）
        # scipy.stats.kurtosis 默认返回超额峰度（减去3后的值）
        excess_kurt = kurtosis(self.returns, fisher=True)

        # 计算原始峰度（加回3）
        raw_kurtosis = excess_kurt + 3

        # 计算偏度
        skewness_val = skew(self.returns)

        # ---------------------------------------------------------------------
        # Jarque-Bera 正态性检验
        # ---------------------------------------------------------------------

        # JB检验统计量：JB = (n/6) * (S^2 + (K-3)^2/4)
        # 其中 S 是偏度，K 是峰度
        # 原假设：数据来自正态分布
        jb_stat, jb_pvalue = stats.jarque_bera(self.returns)

        # ---------------------------------------------------------------------
        # 判断是否具有厚尾特性
        # ---------------------------------------------------------------------

        # 厚尾判断标准：超额峰度 > 0 且 JB检验拒绝正态假设
        is_fat_tailed = (excess_kurt > 0) and (jb_pvalue < self.SIGNIFICANCE_LEVEL)

        # ---------------------------------------------------------------------
        # 生成解释文本
        # ---------------------------------------------------------------------

        if is_fat_tailed:
            interpretation = (
                f"收益率分布呈现显著的厚尾特性（尖峰厚尾）。"
                f"峰度为{raw_kurtosis:.2f}（正态分布为3），"
                f"超额峰度为{excess_kurt:.2f}。"
                f"这表明极端收益事件的发生概率高于正态分布预期，"
                f"符合真实金融市场的典型特征。"
            )
        else:
            interpretation = (
                f"收益率分布的厚尾特性不显著。"
                f"峰度为{raw_kurtosis:.2f}，超额峰度为{excess_kurt:.2f}。"
                f"可能需要更长的仿真时间或调整模型参数。"
            )

        # ---------------------------------------------------------------------
        # 组装结果
        # ---------------------------------------------------------------------

        result = {
            "kurtosis": raw_kurtosis,
            "excess_kurtosis": excess_kurt,
            "skewness": skewness_val,
            "jarque_bera_stat": jb_stat,
            "jarque_bera_pvalue": jb_pvalue,
            "is_fat_tailed": is_fat_tailed,
            "interpretation": interpretation,
            # 用于绑图的数据
            "returns_for_plot": self.returns,
            "mean": np.mean(self.returns),
            "std": np.std(self.returns),
        }

        # 存储结果
        self.results["fat_tails"] = result

        return result

    # =========================================================================
    # 波动聚集分析
    # =========================================================================

    def analyze_volatility_clustering(self) -> AnalysisResult:
        """
        分析波动聚集特性

        波动聚集（Volatility Clustering）是指大波动往往跟随大波动，
        小波动往往跟随小波动。这一现象由Mandelbrot在1963年首次提出。

        检验方法：
            计算绝对收益率（|r_t|）的自相关函数（ACF）。
            如果ACF在多个滞后阶数上显著不为零，说明存在波动聚集。

        自相关函数定义：
            ACF(k) = Cov(|r_t|, |r_{t-k}|) / Var(|r_t|)

        Returns:
            包含以下键的分析结果字典：
            - acf_values: 各滞后阶数的ACF值列表
            - acf_lags: 滞后阶数列表
            - confidence_bound: 95%置信区间边界
            - significant_lags: 显著不为零的滞后阶数
            - has_clustering: 是否存在波动聚集（布尔值）
            - ljung_box_stat: Ljung-Box Q统计量
            - ljung_box_pvalue: Ljung-Box检验p值
            - interpretation: 结果解释文本

        Note:
            - 95%置信区间约为 ±1.96/√n
            - Ljung-Box检验用于联合检验多个滞后阶数的自相关
        """
        # ---------------------------------------------------------------------
        # 计算绝对收益率
        # ---------------------------------------------------------------------

        abs_returns = np.abs(self.returns)
        n = len(abs_returns)

        # ---------------------------------------------------------------------
        # 计算自相关函数 (ACF)
        # ---------------------------------------------------------------------

        # 计算均值和方差
        mean_abs = np.mean(abs_returns)
        var_abs = np.var(abs_returns)

        # 计算各滞后阶数的ACF
        acf_values = []
        lags = list(range(1, self.MAX_LAG + 1))

        for lag in lags:
            # 计算滞后lag阶的自协方差
            cov = np.mean(
                (abs_returns[lag:] - mean_abs) * (abs_returns[:-lag] - mean_abs)
            )
            # 自相关 = 自协方差 / 方差
            acf = cov / var_abs if var_abs > 0 else 0
            acf_values.append(acf)

        acf_values = np.array(acf_values)

        # ---------------------------------------------------------------------
        # 计算置信区间
        # ---------------------------------------------------------------------

        # 在原假设（无自相关）下，ACF渐近服从N(0, 1/n)
        # 95%置信区间为 ±1.96/√n
        confidence_bound = 1.96 / np.sqrt(n)

        # 找出显著不为零的滞后阶数
        significant_lags = [
            lag for lag, acf in zip(lags, acf_values)
            if abs(acf) > confidence_bound
        ]

        # ---------------------------------------------------------------------
        # Ljung-Box 检验
        # ---------------------------------------------------------------------

        # Ljung-Box Q统计量用于联合检验多个滞后阶数的自相关是否全为零
        # Q = n(n+2) * Σ(ACF(k)^2 / (n-k))
        # 原假设：所有滞后阶数的自相关都为零

        q_stat = 0
        for lag, acf in zip(lags, acf_values):
            q_stat += (acf ** 2) / (n - lag)
        q_stat *= n * (n + 2)

        # Q统计量渐近服从卡方分布，自由度为滞后阶数
        lb_pvalue = 1 - stats.chi2.cdf(q_stat, df=len(lags))

        # ---------------------------------------------------------------------
        # 判断是否存在波动聚集
        # ---------------------------------------------------------------------

        # 判断标准：至少有3个滞后阶数显著，且LB检验拒绝原假设
        has_clustering = (
            len(significant_lags) >= 3 or
            lb_pvalue < self.SIGNIFICANCE_LEVEL
        )

        # ---------------------------------------------------------------------
        # 生成解释文本
        # ---------------------------------------------------------------------

        if has_clustering:
            interpretation = (
                f"存在显著的波动聚集现象。"
                f"在滞后{significant_lags[:5]}等阶数上，"
                f"绝对收益率的自相关显著不为零。"
                f"Ljung-Box检验p值为{lb_pvalue:.4f}，"
                f"拒绝无自相关的原假设。"
                f"这表明大波动往往跟随大波动，符合真实市场特征。"
            )
        else:
            interpretation = (
                f"波动聚集现象不显著。"
                f"仅有{len(significant_lags)}个滞后阶数的自相关显著。"
                f"Ljung-Box检验p值为{lb_pvalue:.4f}。"
            )

        # ---------------------------------------------------------------------
        # 组装结果
        # ---------------------------------------------------------------------

        result = {
            "acf_values": acf_values.tolist(),
            "acf_lags": lags,
            "confidence_bound": confidence_bound,
            "significant_lags": significant_lags,
            "has_clustering": has_clustering,
            "ljung_box_stat": q_stat,
            "ljung_box_pvalue": lb_pvalue,
            "interpretation": interpretation,
        }

        # 存储结果
        self.results["volatility_clustering"] = result

        return result

    # =========================================================================
    # 长记忆性分析
    # =========================================================================

    def analyze_long_memory(self) -> AnalysisResult:
        """
        分析长记忆性特征

        长记忆性（Long Memory）是指时间序列存在长期依赖关系，
        即远距离的观测值之间仍存在相关性。

        检验方法：
            计算Hurst指数（H），使用R/S分析法（Rescaled Range Analysis）。

        Hurst指数解释：
            - H = 0.5: 随机游走，无记忆性（类似正态分布的独立增量）
            - H > 0.5: 趋势持续性（正长记忆），过去的趋势可能延续
            - H < 0.5: 均值回归（反持续性），过去的趋势可能反转

        R/S分析法步骤：
            1. 将序列分成若干子区间
            2. 对每个子区间计算累积离差的极差R和标准差S
            3. 计算R/S比率
            4. 通过log(R/S) vs log(n)的斜率估计H

        Returns:
            包含以下键的分析结果字典：
            - hurst_exponent: Hurst指数估计值
            - hurst_std_error: 估计的标准误差
            - is_trending: 是否存在趋势持续性（H > 0.5）
            - is_mean_reverting: 是否存在均值回归（H < 0.5）
            - memory_type: 记忆类型描述
            - interpretation: 结果解释文本

        Note:
            Hurst指数的可靠估计需要较长的时间序列（建议>100个观测值）
        """
        # ---------------------------------------------------------------------
        # R/S 分析法计算 Hurst 指数
        # ---------------------------------------------------------------------

        returns = self.returns
        n = len(returns)

        # 定义子区间长度列表
        # 从10开始，到n/4结束，取对数均匀分布的点
        min_window = 10
        max_window = n // 4

        if max_window < min_window:
            # 数据太短，无法可靠估计Hurst指数
            result = {
                "hurst_exponent": 0.5,
                "hurst_std_error": np.nan,
                "is_trending": False,
                "is_mean_reverting": False,
                "memory_type": "Insufficient Data",
                "interpretation": "数据序列太短，无法可靠估计Hurst指数。",
                "rs_data": {"log_n": [], "log_rs": []},
            }
            self.results["long_memory"] = result
            return result

        # 生成子区间长度序列
        window_sizes = np.unique(
            np.logspace(
                np.log10(min_window),
                np.log10(max_window),
                num=20
            ).astype(int)
        )

        log_n = []
        log_rs = []

        for window_size in window_sizes:
            rs_values = []

            # 将序列分成多个不重叠的子区间
            num_windows = n // window_size

            for i in range(num_windows):
                # 提取子区间
                start = i * window_size
                end = start + window_size
                window_data = returns[start:end]

                # 计算子区间的均值
                window_mean = np.mean(window_data)

                # 计算累积离差序列
                cumulative_deviations = np.cumsum(window_data - window_mean)

                # 计算极差 R
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

                # 计算标准差 S
                S = np.std(window_data, ddof=1)

                # 计算 R/S 比率
                if S > 0:
                    rs_values.append(R / S)

            if rs_values:
                # 计算该窗口大小下的平均 R/S
                mean_rs = np.mean(rs_values)
                log_n.append(np.log(window_size))
                log_rs.append(np.log(mean_rs))

        # ---------------------------------------------------------------------
        # 线性回归估计 Hurst 指数
        # ---------------------------------------------------------------------

        if len(log_n) >= 3:
            # 使用最小二乘法拟合 log(R/S) = H * log(n) + c
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_n, log_rs
            )
            hurst = slope
            hurst_std_error = std_err
        else:
            hurst = 0.5
            hurst_std_error = np.nan

        # ---------------------------------------------------------------------
        # 判断记忆类型
        # ---------------------------------------------------------------------

        # 使用置信区间判断是否显著偏离0.5
        # 假设H的估计近似正态，95%置信区间为 H ± 1.96 * std_err
        if not np.isnan(hurst_std_error):
            lower_bound = hurst - 1.96 * hurst_std_error
            upper_bound = hurst + 1.96 * hurst_std_error

            is_trending = lower_bound > 0.5
            is_mean_reverting = upper_bound < 0.5
        else:
            is_trending = hurst > 0.55  # 使用经验阈值
            is_mean_reverting = hurst < 0.45

        # 确定记忆类型
        if is_trending:
            memory_type = "Trending (Positive Long Memory)"
        elif is_mean_reverting:
            memory_type = "Mean Reverting (Anti-persistent)"
        else:
            memory_type = "Near Random Walk"

        # ---------------------------------------------------------------------
        # 生成解释文本
        # ---------------------------------------------------------------------

        interpretation = (
            f"Hurst指数估计值为 H = {hurst:.3f}。\n"
        )

        if hurst > 0.5:
            interpretation += (
                f"H > 0.5 表明序列存在趋势持续性（长记忆），"
                f"过去的价格趋势倾向于延续。"
                f"这可能反映了市场中趋势跟随交易者和动量效应的影响。"
            )
        elif hurst < 0.5:
            interpretation += (
                f"H < 0.5 表明序列存在均值回归特性，"
                f"过去的价格趋势倾向于反转。"
                f"这可能反映了市场中反向交易者和套利行为的影响。"
            )
        else:
            interpretation += (
                f"H ≈ 0.5 表明序列接近随机游走，"
                f"过去的价格走势对未来没有显著的预测作用。"
            )

        # ---------------------------------------------------------------------
        # 组装结果
        # ---------------------------------------------------------------------

        result = {
            "hurst_exponent": hurst,
            "hurst_std_error": hurst_std_error,
            "is_trending": is_trending,
            "is_mean_reverting": is_mean_reverting,
            "memory_type": memory_type,
            "interpretation": interpretation,
            # 用于绘图的数据
            "rs_data": {
                "log_n": log_n,
                "log_rs": log_rs,
            },
        }

        # 存储结果
        self.results["long_memory"] = result

        return result

    # =========================================================================
    # 综合报告生成
    # =========================================================================

    def generate_report(self) -> Dict[str, AnalysisResult]:
        """
        生成完整的Stylized Facts分析报告

        该方法依次执行所有分析，并汇总结果。

        Returns:
            包含所有分析结果的字典，键为分析类型，值为结果字典

        Example:
            >>> analyzer = StylizedFactsAnalyzer(price_history)
            >>> report = analyzer.generate_report()
            >>> print(report["fat_tails"]["interpretation"])
        """
        # 执行所有分析
        self.analyze_fat_tails()
        self.analyze_volatility_clustering()
        self.analyze_long_memory()

        # 返回汇总结果
        return self.results

    def print_report(self) -> None:
        """
        打印格式化的分析报告

        将分析结果以易读的格式输出到控制台。
        """
        # 确保已执行分析
        if not self.results:
            self.generate_report()

        print("\n" + "=" * 70)
        print("Stylized Facts 金融典型事实验证报告")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # 厚尾特性
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("1. 厚尾特性 (Fat Tails)")
        print("-" * 70)

        ft = self.results.get("fat_tails", {})
        print(f"  峰度 (Kurtosis):        {ft.get('kurtosis', 'N/A'):.4f}")
        print(f"  超额峰度:               {ft.get('excess_kurtosis', 'N/A'):.4f}")
        print(f"  偏度 (Skewness):        {ft.get('skewness', 'N/A'):.4f}")
        print(f"  Jarque-Bera p值:        {ft.get('jarque_bera_pvalue', 'N/A'):.4f}")
        print(f"  厚尾特性:               {'是' if ft.get('is_fat_tailed') else '否'}")
        print(f"\n  解释: {ft.get('interpretation', 'N/A')}")

        # ---------------------------------------------------------------------
        # 波动聚集
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("2. 波动聚集 (Volatility Clustering)")
        print("-" * 70)

        vc = self.results.get("volatility_clustering", {})
        print(f"  显著滞后阶数:           {vc.get('significant_lags', [])[:5]}")
        print(f"  Ljung-Box p值:          {vc.get('ljung_box_pvalue', 'N/A'):.4f}")
        print(f"  波动聚集:               {'是' if vc.get('has_clustering') else '否'}")
        print(f"\n  解释: {vc.get('interpretation', 'N/A')}")

        # ---------------------------------------------------------------------
        # 长记忆性
        # ---------------------------------------------------------------------
        print("\n" + "-" * 70)
        print("3. 长记忆性 (Long Memory)")
        print("-" * 70)

        lm = self.results.get("long_memory", {})
        print(f"  Hurst指数:              {lm.get('hurst_exponent', 'N/A'):.4f}")
        print(f"  记忆类型:               {lm.get('memory_type', 'N/A')}")
        print(f"\n  解释: {lm.get('interpretation', 'N/A')}")

        # ---------------------------------------------------------------------
        # 总结
        # ---------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("总结")
        print("=" * 70)

        # 统计符合的特征数量
        matched_features = 0
        total_features = 3

        if ft.get("is_fat_tailed"):
            matched_features += 1
        if vc.get("has_clustering"):
            matched_features += 1
        if lm.get("is_trending") or lm.get("is_mean_reverting"):
            matched_features += 1

        print(f"  符合的典型特征: {matched_features}/{total_features}")

        if matched_features >= 2:
            print("  结论: 仿真数据较好地复现了真实金融市场的典型特征。")
        elif matched_features == 1:
            print("  结论: 仿真数据部分复现了金融市场特征，建议调整参数。")
        else:
            print("  结论: 仿真数据与真实市场特征差异较大，需要改进模型。")

        print("=" * 70 + "\n")

    # =========================================================================
    # 可视化
    # =========================================================================

    def plot_stylized_facts(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 600,
    ) -> plt.Figure:
        """
        生成Stylized Facts可视化拼图

        创建一个2x2的子图布局，展示：
        1. 左上：收益率分布直方图（与正态分布对比）
        2. 右上：绝对收益率ACF图
        3. 左下：R/S分析图（Hurst指数拟合）
        4. 右下：汇总统计表

        Args:
            output_path: 图片保存路径，None则不保存
            figsize: 图片尺寸（英寸）
            dpi: 图片分辨率

        Returns:
            matplotlib Figure对象

        Example:
            >>> analyzer.plot_stylized_facts("results/stylized_facts.png")
        """
        # 确保已执行分析
        if not self.results:
            self.generate_report()

        # 创建图形和子图
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 设置字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # ---------------------------------------------------------------------
        # 子图1: 收益率分布直方图
        # ---------------------------------------------------------------------
        ax1 = axes[0, 0]
        self._plot_return_distribution(ax1)

        # ---------------------------------------------------------------------
        # 子图2: 波动聚集 ACF 图
        # ---------------------------------------------------------------------
        ax2 = axes[0, 1]
        self._plot_acf(ax2)

        # ---------------------------------------------------------------------
        # 子图3: R/S 分析图
        # ---------------------------------------------------------------------
        ax3 = axes[1, 0]
        self._plot_rs_analysis(ax3)

        # ---------------------------------------------------------------------
        # 子图4: 汇总统计表
        # ---------------------------------------------------------------------
        ax4 = axes[1, 1]
        self._plot_summary_table(ax4)

        # 调整布局
        plt.tight_layout()

        # 添加总标题
        fig.suptitle(
            "Stylized Facts Analysis",
            fontsize=14,
            fontweight="bold",
            y=1.02
        )

        # 保存图片
        if output_path:
            # 确保目录存在
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"[可视化] Stylized Facts图已保存至: {output_path}")

        return fig

    def _plot_return_distribution(self, ax: plt.Axes) -> None:
        """
        绘制收益率分布直方图（子图1）

        Args:
            ax: matplotlib Axes对象
        """
        ft = self.results.get("fat_tails", {})
        returns = ft.get("returns_for_plot", self.returns)
        mean = ft.get("mean", np.mean(returns))
        std = ft.get("std", np.std(returns))

        # 绘制直方图 - 使用统一配色
        n_bins = min(50, len(returns) // 5)
        counts, bins, patches = ax.hist(
            returns,
            bins=n_bins,
            density=True,
            alpha=0.7,
            color="#8da0cb",  # 淡紫蓝 - 与 generate_figures 一致
            edgecolor="white",
            label="Simulated Returns"
        )

        # 叠加正态分布曲线
        x = np.linspace(returns.min(), returns.max(), 100)
        normal_pdf = norm.pdf(x, mean, std)
        ax.plot(
            x, normal_pdf,
            "r-",
            linewidth=2,
            label=f"Normal(μ={mean:.4f}, σ={std:.4f})"
        )

        # 添加标注
        kurtosis_val = ft.get("kurtosis", 3)
        excess_kurt = ft.get("excess_kurtosis", 0)

        ax.axvline(x=mean, color="#66c2a5", linestyle="--", alpha=0.7, label="Mean")

        # 在图上标注峰度
        textstr = f"Kurtosis: {kurtosis_val:.2f}\nExcess: {excess_kurt:.2f}"
        ax.text(
            0.95, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.set_title("(a) Return Distribution vs Normal", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_acf(self, ax: plt.Axes) -> None:
        """
        绘制绝对收益率ACF图（子图2）

        Args:
            ax: matplotlib Axes对象
        """
        vc = self.results.get("volatility_clustering", {})
        lags = vc.get("acf_lags", list(range(1, 21)))
        acf_values = vc.get("acf_values", [0] * 20)
        conf_bound = vc.get("confidence_bound", 0.2)

        # 绘制ACF柱状图 - 使用统一配色
        ax.bar(
            lags, acf_values,
            color="#8da0cb",  # 淡紫蓝
            alpha=0.7,
            edgecolor="white"
        )

        # 绘制置信区间
        ax.axhline(y=conf_bound, color="#fc8d62", linestyle="--", label="95% CI")
        ax.axhline(y=-conf_bound, color="#fc8d62", linestyle="--")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # 标记显著的滞后阶数
        significant_lags = vc.get("significant_lags", [])
        for lag in significant_lags:
            if lag <= len(acf_values):
                ax.bar(
                    lag, acf_values[lag - 1],
                    color="#e78ac3",  # 粉紫色 - 高亮显著滞后
                    alpha=0.9
                )

        # 添加Ljung-Box p值标注
        lb_pvalue = vc.get("ljung_box_pvalue", 1.0)
        textstr = f"Ljung-Box p: {lb_pvalue:.4f}"
        ax.text(
            0.95, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title("(b) ACF of Absolute Returns", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_rs_analysis(self, ax: plt.Axes) -> None:
        """
        绘制R/S分析图（子图3）

        Args:
            ax: matplotlib Axes对象
        """
        lm = self.results.get("long_memory", {})
        rs_data = lm.get("rs_data", {})
        log_n = rs_data.get("log_n", [])
        log_rs = rs_data.get("log_rs", [])
        hurst = lm.get("hurst_exponent", 0.5)

        if log_n and log_rs:
            # 绘制散点
            ax.scatter(
                log_n, log_rs,
                color="#8da0cb",  # 淡紫蓝
                s=50,
                alpha=0.7,
                label="R/S Data"
            )

            # 绘制拟合线
            x_fit = np.array([min(log_n), max(log_n)])
            # 使用线性回归结果
            slope, intercept, _, _, _ = stats.linregress(log_n, log_rs)
            y_fit = slope * x_fit + intercept
            ax.plot(
                x_fit, y_fit,
                color="#e78ac3",  # 粉紫色
                linewidth=2,
                label=f"Fit (H={hurst:.3f})"
            )

            # 添加H=0.5参考线
            y_random = 0.5 * x_fit + intercept
            ax.plot(
                x_fit, y_random,
                color="#66c2a5",  # 青绿色
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label="H=0.5 (Random Walk)"
            )

        # 添加Hurst指数标注
        memory_type = lm.get("memory_type", "N/A")
        textstr = f"H = {hurst:.3f}\n{memory_type}"
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        ax.set_xlabel("log(n)")
        ax.set_ylabel("log(R/S)")
        ax.set_title("(c) R/S Analysis (Hurst Exponent)", fontweight="bold")
        # 只在有数据时显示图例，避免警告
        if log_n and log_rs:
            ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_summary_table(self, ax: plt.Axes) -> None:
        """
        绘制汇总统计表（子图4）

        Args:
            ax: matplotlib Axes对象
        """
        # 隐藏坐标轴
        ax.axis("off")

        # 准备表格数据
        ft = self.results.get("fat_tails", {})
        vc = self.results.get("volatility_clustering", {})
        lm = self.results.get("long_memory", {})

        # 构建表格内容
        table_data = [
            ["Stylized Fact", "Metric", "Value", "Result"],
            [
                "Fat Tails",
                "Kurtosis",
                f"{ft.get('kurtosis', 0):.2f}",
                "Yes" if ft.get("is_fat_tailed") else "No"
            ],
            [
                "",
                "Excess Kurtosis",
                f"{ft.get('excess_kurtosis', 0):.2f}",
                "(> 0 = Fat Tails)"
            ],
            [
                "",
                "JB p-value",
                f"{ft.get('jarque_bera_pvalue', 1):.4f}",
                ""
            ],
            [
                "Volatility",
                "Significant Lags",
                f"{len(vc.get('significant_lags', []))}",
                "Yes" if vc.get("has_clustering") else "No"
            ],
            [
                "Clustering",
                "LB p-value",
                f"{vc.get('ljung_box_pvalue', 1):.4f}",
                ""
            ],
            [
                "Long Memory",
                "Hurst Exponent",
                f"{lm.get('hurst_exponent', 0.5):.3f}",
                lm.get("memory_type", "N/A")
            ],
        ]

        # 绘制表格
        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="center",
            colColours=["#4472C4"] * 4,  # 与 generate_figures 一致
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # 设置表头文字颜色
        for i in range(4):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        # 设置标题
        ax.set_title(
            "(d) Summary Statistics",
            fontweight="bold",
            pad=20
        )


# =============================================================================
# 便捷函数
# =============================================================================

def analyze_stylized_facts(
    price_history: List[float],
    output_path: Optional[str] = None,
    print_report: bool = True,
) -> Dict[str, AnalysisResult]:
    """
    分析价格序列的Stylized Facts特征（便捷函数）

    该函数是StylizedFactsAnalyzer类的便捷封装，一次调用完成
    所有分析、打印报告、生成可视化。

    Args:
        price_history: 价格历史序列
        output_path: 图片保存路径（可选）
        print_report: 是否打印报告到控制台

    Returns:
        包含所有分析结果的字典

    Example:
        >>> results = analyze_stylized_facts(
        ...     price_history=market.price_history,
        ...     output_path="results/stylized_facts.png"
        ... )
    """
    # 创建分析器
    analyzer = StylizedFactsAnalyzer(price_history=price_history)

    # 生成报告
    results = analyzer.generate_report()

    # 打印报告
    if print_report:
        analyzer.print_report()

    # 生成可视化
    if output_path:
        analyzer.plot_stylized_facts(output_path)

    return results
