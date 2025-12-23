"""
真实市场基准对比模块 (Real Market Benchmark Module)

本模块用于将ABM仿真生成的价格序列与真实金融市场数据进行对比，
验证仿真数据的"逼真度"(Realism)。

核心问题：
    "我的仿真有多逼真？"

    回答这个问题需要将仿真数据与真实市场数据进行系统性对比，
    包括统计特性、分布形态、波动特征等多个维度。

对比维度：
    1. 收益率分布统计量：均值、标准差、偏度、峰度
    2. 风险指标：波动率、最大回撤、VaR
    3. 时间序列特性：自相关、Hurst指数
    4. 可视化对比：归一化价格走势

数据来源：
    使用yfinance库从Yahoo Finance下载真实市场数据。
    默认使用SPY（标普500 ETF）作为基准，因为：
    - 流动性高，价格发现效率高
    - 代表整体市场，而非个股特异性
    - 历史数据完整，便于研究

类结构：
    RealMarketBenchmark: 主类
    ├── download_market_data(): 下载真实市场数据
    ├── calculate_statistics(): 计算统计指标
    ├── compare_with_simulation(): 对比分析
    ├── generate_comparison_table(): 生成对比表格
    └── plot_comparison(): 生成对比图表

作者: SuZX
日期: 2024
"""

# =============================================================================
# 导入依赖
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import time

import numpy as np                          # 数值计算
import pandas as pd                         # 数据处理
import matplotlib.pyplot as plt             # 绑图
from scipy import stats                     # 统计函数

# Yahoo Finance 数据下载
import yfinance as yf


# =============================================================================
# 类型定义
# =============================================================================

# 统计结果字典类型
StatisticsDict = Dict[str, float]


# =============================================================================
# 真实市场基准类
# =============================================================================

@dataclass
class RealMarketBenchmark:
    """
    真实市场基准对比分析器

    该类从Yahoo Finance下载真实市场数据，并与仿真数据进行
    多维度对比分析，评估仿真的"逼真度"。

    对比分析包括：
        1. 统计量对比：均值、标准差、偏度、峰度等
        2. 风险指标对比：波动率、最大回撤、VaR等
        3. 时间序列对比：自相关、Hurst指数等
        4. 可视化对比：归一化价格走势、分布直方图

    Attributes:
        ticker: 股票/ETF代码，默认"SPY"（标普500 ETF）
        period: 数据周期，默认"1y"（1年）
        market_data: 下载的市场数据DataFrame
        market_prices: 市场收盘价序列
        market_returns: 市场收益率序列
        market_stats: 市场数据统计量

    Example:
        >>> benchmark = RealMarketBenchmark(ticker="SPY", period="1y")
        >>> benchmark.download_market_data()
        >>> comparison = benchmark.compare_with_simulation(sim_prices)
        >>> benchmark.plot_comparison(sim_prices, "output/comparison.png")
    """

    # -------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------

    # 股票/ETF代码
    # SPY: 标普500 ETF，代表美国大盘股市场
    # QQQ: 纳斯达克100 ETF，代表科技股
    # IWM: 罗素2000 ETF，代表小盘股
    ticker: str = "SPY"

    # 数据周期
    # 支持的格式：1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    period: str = "1y"

    # -------------------------------------------------------------------------
    # 数据存储（延迟初始化）
    # -------------------------------------------------------------------------

    # 原始市场数据DataFrame
    market_data: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    # 收盘价序列
    market_prices: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # 收益率序列
    market_returns: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # 市场统计量
    market_stats: Optional[StatisticsDict] = field(default=None, init=False, repr=False)

    # -------------------------------------------------------------------------
    # 数据下载状态
    # -------------------------------------------------------------------------

    # 数据是否已下载
    _data_loaded: bool = field(default=False, init=False, repr=False)

    # =========================================================================
    # 数据下载
    # =========================================================================

    def download_market_data(self, force: bool = False, max_retries: int = 3) -> bool:
        """
        从Yahoo Finance下载市场数据

        该方法使用yfinance库下载指定股票/ETF的历史价格数据。
        下载完成后，自动计算收益率序列。

        Args:
            force: 是否强制重新下载（即使已有数据）
            max_retries: 最大重试次数（默认3次，用于处理限流）

        Returns:
            下载是否成功

        Raises:
            无（失败时返回False并打印警告）

        Example:
            >>> benchmark = RealMarketBenchmark(ticker="SPY")
            >>> success = benchmark.download_market_data()
            >>> if success:
            ...     print(f"下载了{len(benchmark.market_prices)}个数据点")
        """
        # 检查是否需要下载
        if self._data_loaded and not force:
            print(f"[市场数据] 已有{self.ticker}数据，跳过下载。使用force=True强制刷新。")
            return True

        print(f"[市场数据] 正在从Yahoo Finance下载 {self.ticker} 数据...")

        # 重试循环，处理限流等临时错误
        for attempt in range(max_retries):
            try:
                # 使用yfinance下载数据
                # progress=False 禁用进度条，避免输出干扰
                ticker_obj = yf.Ticker(self.ticker)
                self.market_data = ticker_obj.history(period=self.period)

                # 检查数据是否有效
                if self.market_data is None or len(self.market_data) == 0:
                    warnings.warn(f"未能获取{self.ticker}的数据，请检查网络连接或股票代码。")
                    return False

                # 提取收盘价
                self.market_prices = self.market_data["Close"].values

                # 计算日收益率
                # r_t = (P_t - P_{t-1}) / P_{t-1}
                self.market_returns = np.diff(self.market_prices) / self.market_prices[:-1]

                # 标记数据已加载
                self._data_loaded = True

                # 计算统计量
                self.market_stats = self.calculate_statistics(self.market_returns)

                print(f"[市场数据] 成功下载 {len(self.market_prices)} 个交易日数据")
                print(f"[市场数据] 日期范围: {self.market_data.index[0].date()} 至 {self.market_data.index[-1].date()}")

                return True

            except Exception as e:
                error_msg = str(e)
                # 检查是否为限流错误
                if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)  # 指数退避: 2, 4, 8 秒
                        print(f"[市场数据] 请求被限流，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                # 其他错误或最后一次重试失败
                warnings.warn(f"下载市场数据时出错: {e}")
                return False

        return False

    # =========================================================================
    # 统计计算
    # =========================================================================

    def calculate_statistics(
        self,
        returns: np.ndarray,
        annualize: bool = True
    ) -> StatisticsDict:
        """
        计算收益率序列的统计指标

        该方法计算一组全面的统计指标，用于描述收益率分布的特征。

        计算的指标包括：
            1. 基本统计量：均值、标准差、偏度、峰度
            2. 风险指标：波动率、VaR、最大回撤
            3. 分布特征：最小值、最大值、中位数

        Args:
            returns: 收益率序列
            annualize: 是否年化波动率（假设252个交易日）

        Returns:
            包含各项统计指标的字典

        Note:
            - 年化波动率 = 日波动率 × √252
            - VaR使用5%分位数
        """
        # 基本检查
        if len(returns) == 0:
            return {}

        # ---------------------------------------------------------------------
        # 基本统计量
        # ---------------------------------------------------------------------

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)  # 样本标准差
        skewness = stats.skew(returns)
        kurtosis_val = stats.kurtosis(returns, fisher=False)  # 原始峰度
        excess_kurtosis = kurtosis_val - 3  # 超额峰度

        # ---------------------------------------------------------------------
        # 年化处理
        # ---------------------------------------------------------------------

        # 假设一年252个交易日
        trading_days = 252

        if annualize:
            # 年化收益率 = 日均收益率 × 252
            annual_return = mean_return * trading_days
            # 年化波动率 = 日波动率 × √252
            annual_volatility = std_return * np.sqrt(trading_days)
        else:
            annual_return = mean_return
            annual_volatility = std_return

        # ---------------------------------------------------------------------
        # 风险指标
        # ---------------------------------------------------------------------

        # VaR (Value at Risk) - 5%分位数
        # 表示在95%置信水平下，最大可能损失
        var_5pct = np.percentile(returns, 5)

        # 最大回撤
        # 计算累积收益率序列的最大回撤
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # ---------------------------------------------------------------------
        # 分布范围
        # ---------------------------------------------------------------------

        min_return = np.min(returns)
        max_return = np.max(returns)
        median_return = np.median(returns)

        # ---------------------------------------------------------------------
        # 组装结果
        # ---------------------------------------------------------------------

        return {
            # 基本统计量
            "mean": mean_return,
            "std": std_return,
            "skewness": skewness,
            "kurtosis": kurtosis_val,
            "excess_kurtosis": excess_kurtosis,

            # 年化指标
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,

            # 风险指标
            "var_5pct": var_5pct,
            "max_drawdown": max_drawdown,

            # 分布范围
            "min": min_return,
            "max": max_return,
            "median": median_return,

            # 数据量
            "n_observations": len(returns),
        }

    # =========================================================================
    # 对比分析
    # =========================================================================

    def compare_with_simulation(
        self,
        sim_prices: List[float],
        sim_name: str = "Simulation"
    ) -> Dict[str, Any]:
        """
        将仿真数据与真实市场数据进行对比分析

        该方法是本模块的核心功能，执行全面的对比分析。

        对比内容：
            1. 收益率分布统计量对比
            2. 风险指标对比
            3. 差异显著性检验（K-S检验）
            4. 相似度评分

        Args:
            sim_prices: 仿真价格序列
            sim_name: 仿真数据的名称（用于显示）

        Returns:
            包含对比结果的字典：
            - sim_stats: 仿真数据统计量
            - market_stats: 市场数据统计量
            - comparison_table: 对比表格DataFrame
            - ks_test: K-S检验结果
            - similarity_score: 相似度评分

        Example:
            >>> comparison = benchmark.compare_with_simulation(sim_prices)
            >>> print(comparison["comparison_table"])
        """
        # 确保市场数据已下载
        if not self._data_loaded:
            success = self.download_market_data()
            if not success:
                return {"error": "无法下载市场数据"}

        # ---------------------------------------------------------------------
        # 计算仿真数据的统计量
        # ---------------------------------------------------------------------

        sim_prices_arr = np.array(sim_prices)
        sim_returns = np.diff(sim_prices_arr) / sim_prices_arr[:-1]
        sim_stats = self.calculate_statistics(sim_returns, annualize=False)

        # ---------------------------------------------------------------------
        # 生成对比表格
        # ---------------------------------------------------------------------

        comparison_table = self._generate_comparison_table(
            sim_stats,
            self.market_stats,
            sim_name
        )

        # ---------------------------------------------------------------------
        # Kolmogorov-Smirnov 检验
        # ---------------------------------------------------------------------

        # K-S检验用于检验两个样本是否来自同一分布
        # 原假设：两个样本来自同一分布
        ks_stat, ks_pvalue = stats.ks_2samp(sim_returns, self.market_returns)

        ks_result = {
            "statistic": ks_stat,
            "pvalue": ks_pvalue,
            "same_distribution": ks_pvalue > 0.05,
            "interpretation": (
                "仿真收益率分布与真实市场相似（未能拒绝同分布假设）"
                if ks_pvalue > 0.05
                else "仿真收益率分布与真实市场存在显著差异"
            )
        }

        # ---------------------------------------------------------------------
        # 相似度评分
        # ---------------------------------------------------------------------

        similarity_score = self._calculate_similarity_score(sim_stats, self.market_stats)

        # ---------------------------------------------------------------------
        # 组装结果
        # ---------------------------------------------------------------------

        return {
            "sim_stats": sim_stats,
            "market_stats": self.market_stats,
            "comparison_table": comparison_table,
            "ks_test": ks_result,
            "similarity_score": similarity_score,
            "sim_returns": sim_returns,
            "market_returns": self.market_returns,
        }

    def _generate_comparison_table(
        self,
        sim_stats: StatisticsDict,
        market_stats: StatisticsDict,
        sim_name: str
    ) -> pd.DataFrame:
        """
        生成统计量对比表格

        Args:
            sim_stats: 仿真数据统计量
            market_stats: 市场数据统计量
            sim_name: 仿真数据名称

        Returns:
            对比表格DataFrame
        """
        # 定义要对比的指标
        metrics = [
            ("Mean Return", "mean", "{:.6f}"),
            ("Std Deviation", "std", "{:.6f}"),
            ("Skewness", "skewness", "{:.4f}"),
            ("Kurtosis", "kurtosis", "{:.4f}"),
            ("Excess Kurtosis", "excess_kurtosis", "{:.4f}"),
            ("VaR (5%)", "var_5pct", "{:.6f}"),
            ("Max Drawdown", "max_drawdown", "{:.4f}"),
            ("Min Return", "min", "{:.6f}"),
            ("Max Return", "max", "{:.6f}"),
            ("Observations", "n_observations", "{:.0f}"),
        ]

        # 构建表格数据
        rows = []
        for display_name, key, fmt in metrics:
            sim_val = sim_stats.get(key, np.nan)
            mkt_val = market_stats.get(key, np.nan)

            # 计算差异
            if not np.isnan(sim_val) and not np.isnan(mkt_val) and mkt_val != 0:
                diff_pct = (sim_val - mkt_val) / abs(mkt_val) * 100
                diff_str = f"{diff_pct:+.1f}%"
            else:
                diff_str = "N/A"

            rows.append({
                "Metric": display_name,
                sim_name: fmt.format(sim_val) if not np.isnan(sim_val) else "N/A",
                f"{self.ticker} (Real)": fmt.format(mkt_val) if not np.isnan(mkt_val) else "N/A",
                "Difference": diff_str,
            })

        return pd.DataFrame(rows)

    def _calculate_similarity_score(
        self,
        sim_stats: StatisticsDict,
        market_stats: StatisticsDict
    ) -> Dict[str, Any]:
        """
        计算仿真与真实市场的相似度评分

        评分标准：
            - 波动率相似度（权重30%）
            - 峰度相似度（权重25%）
            - 偏度相似度（权重20%）
            - VaR相似度（权重25%）

        Args:
            sim_stats: 仿真数据统计量
            market_stats: 市场数据统计量

        Returns:
            相似度评分结果字典
        """
        scores = {}

        # 波动率相似度
        sim_std = sim_stats.get("std", 0)
        mkt_std = market_stats.get("std", 1)
        if mkt_std > 0:
            std_ratio = min(sim_std, mkt_std) / max(sim_std, mkt_std)
            scores["volatility"] = std_ratio * 100
        else:
            scores["volatility"] = 0

        # 峰度相似度（厚尾特性）
        sim_kurt = sim_stats.get("kurtosis", 3)
        mkt_kurt = market_stats.get("kurtosis", 3)
        # 真实市场通常峰度>3，仿真也应该>3才好
        if mkt_kurt > 3:
            if sim_kurt > 3:
                kurt_score = min(sim_kurt - 3, mkt_kurt - 3) / max(sim_kurt - 3, mkt_kurt - 3)
                scores["kurtosis"] = kurt_score * 100
            else:
                scores["kurtosis"] = 0
        else:
            scores["kurtosis"] = 50  # 基准分

        # 偏度相似度
        sim_skew = sim_stats.get("skewness", 0)
        mkt_skew = market_stats.get("skewness", 0)
        # 偏度差异越小越好
        skew_diff = abs(sim_skew - mkt_skew)
        scores["skewness"] = max(0, 100 - skew_diff * 100)

        # VaR相似度
        sim_var = abs(sim_stats.get("var_5pct", 0))
        mkt_var = abs(market_stats.get("var_5pct", 1))
        if mkt_var > 0:
            var_ratio = min(sim_var, mkt_var) / max(sim_var, mkt_var)
            scores["var"] = var_ratio * 100
        else:
            scores["var"] = 0

        # 加权总分
        weights = {
            "volatility": 0.30,
            "kurtosis": 0.25,
            "skewness": 0.20,
            "var": 0.25,
        }

        total_score = sum(scores[k] * weights[k] for k in weights)

        # 评级
        if total_score >= 80:
            grade = "A (Excellent)"
        elif total_score >= 60:
            grade = "B (Good)"
        elif total_score >= 40:
            grade = "C (Fair)"
        else:
            grade = "D (Poor)"

        return {
            "component_scores": scores,
            "weights": weights,
            "total_score": total_score,
            "grade": grade,
        }

    # =========================================================================
    # 可视化
    # =========================================================================

    def plot_comparison(
        self,
        sim_prices: List[float],
        output_path: Optional[str] = None,
        sim_name: str = "Simulation",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 600,
    ) -> plt.Figure:
        """
        生成仿真与真实市场的对比可视化

        创建一个2x2的子图布局：
            1. 左上：归一化价格走势对比
            2. 右上：收益率分布直方图对比
            3. 左下：QQ图（分位数对比）
            4. 右下：统计量对比表格

        Args:
            sim_prices: 仿真价格序列
            output_path: 图片保存路径（可选）
            sim_name: 仿真数据名称
            figsize: 图片尺寸
            dpi: 图片分辨率

        Returns:
            matplotlib Figure对象

        Example:
            >>> benchmark.plot_comparison(
            ...     sim_prices=simulation_result["price_history"],
            ...     output_path="results/market_comparison.png"
            ... )
        """
        # 确保市场数据已下载
        if not self._data_loaded:
            self.download_market_data()

        # 执行对比分析
        comparison = self.compare_with_simulation(sim_prices, sim_name)

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 设置字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']

        # ---------------------------------------------------------------------
        # 子图1: 归一化价格走势对比
        # ---------------------------------------------------------------------
        ax1 = axes[0, 0]
        self._plot_normalized_prices(ax1, sim_prices, sim_name)

        # ---------------------------------------------------------------------
        # 子图2: 收益率分布直方图对比
        # ---------------------------------------------------------------------
        ax2 = axes[0, 1]
        self._plot_return_distributions(
            ax2,
            comparison["sim_returns"],
            comparison["market_returns"],
            sim_name
        )

        # ---------------------------------------------------------------------
        # 子图3: QQ图
        # ---------------------------------------------------------------------
        ax3 = axes[1, 0]
        self._plot_qq(ax3, comparison["sim_returns"], comparison["market_returns"])

        # ---------------------------------------------------------------------
        # 子图4: 统计量对比表格
        # ---------------------------------------------------------------------
        ax4 = axes[1, 1]
        self._plot_stats_table(ax4, comparison, sim_name)

        # 调整布局
        plt.tight_layout()

        # 添加总标题 - 简洁版，不显示评分
        fig.suptitle(
            f"Simulation vs Real Market ({self.ticker}) Comparison",
            fontsize=12,
            fontweight="bold",
            y=1.02
        )

        # 保存图片
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            print(f"[可视化] 市场对比图已保存至: {output_path}")

        return fig

    def _plot_normalized_prices(
        self,
        ax: plt.Axes,
        sim_prices: List[float],
        sim_name: str
    ) -> None:
        """
        绘制归一化价格走势对比（子图1）

        将两个价格序列都归一化到起点100，便于直接比较走势。

        Args:
            ax: matplotlib Axes对象
            sim_prices: 仿真价格序列
            sim_name: 仿真数据名称
        """
        # 归一化仿真价格（起点设为100）
        sim_prices_arr = np.array(sim_prices)
        sim_normalized = sim_prices_arr / sim_prices_arr[0] * 100

        # 归一化市场价格（起点设为100）
        market_normalized = self.market_prices / self.market_prices[0] * 100

        # 创建x轴（按比例对齐）
        sim_x = np.linspace(0, 100, len(sim_normalized))
        market_x = np.linspace(0, 100, len(market_normalized))

        # 绘制 - 使用统一配色
        ax.plot(sim_x, sim_normalized, color="#8da0cb", linewidth=1.5, label=sim_name, alpha=0.8)
        ax.plot(market_x, market_normalized, color="#fc8d62", linewidth=1.5, label=f"{self.ticker} (Real)", alpha=0.8)

        # 添加起始参考线
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Start (100)")

        ax.set_xlabel("Time Progress (%)")
        ax.set_ylabel("Normalized Price (Start=100)")
        ax.set_title("(a) Normalized Price Comparison", fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_return_distributions(
        self,
        ax: plt.Axes,
        sim_returns: np.ndarray,
        market_returns: np.ndarray,
        sim_name: str
    ) -> None:
        """
        绘制收益率分布直方图对比（子图2）

        Args:
            ax: matplotlib Axes对象
            sim_returns: 仿真收益率
            market_returns: 市场收益率
            sim_name: 仿真数据名称
        """
        # 确定共同的bin范围
        all_returns = np.concatenate([sim_returns, market_returns])
        bins = np.linspace(
            np.percentile(all_returns, 1),
            np.percentile(all_returns, 99),
            40
        )

        # 绘制直方图 - 使用统一配色
        ax.hist(
            sim_returns, bins=bins, density=True, alpha=0.6,
            color="#8da0cb", label=sim_name, edgecolor="white"
        )
        ax.hist(
            market_returns, bins=bins, density=True, alpha=0.6,
            color="#fc8d62", label=f"{self.ticker} (Real)", edgecolor="white"
        )

        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.set_title("(b) Return Distribution Comparison", fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_qq(
        self,
        ax: plt.Axes,
        sim_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> None:
        """
        绘制QQ图（子图3）

        QQ图（分位数-分位数图）用于直观比较两个分布。
        如果两个分布相同，点会落在45度对角线上。

        Args:
            ax: matplotlib Axes对象
            sim_returns: 仿真收益率
            market_returns: 市场收益率
        """
        # 计算分位数
        quantiles = np.linspace(0.01, 0.99, 50)
        sim_quantiles = np.percentile(sim_returns, quantiles * 100)
        market_quantiles = np.percentile(market_returns, quantiles * 100)

        # 绘制散点
        ax.scatter(
            market_quantiles, sim_quantiles,
            c="#8da0cb", alpha=0.7, s=30
        )

        # 绘制45度参考线
        lims = [
            min(market_quantiles.min(), sim_quantiles.min()),
            max(market_quantiles.max(), sim_quantiles.max())
        ]
        ax.plot(lims, lims, color="#fc8d62", linestyle="--", linewidth=1.5, label="Perfect Match")

        ax.set_xlabel(f"{self.ticker} Quantiles")
        ax.set_ylabel("Simulation Quantiles")
        ax.set_title("(c) Q-Q Plot (Return Distribution)", fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    def _plot_stats_table(
        self,
        ax: plt.Axes,
        comparison: Dict,
        sim_name: str
    ) -> None:
        """
        绘制统计量对比表格（子图4）

        Args:
            ax: matplotlib Axes对象
            comparison: 对比分析结果
            sim_name: 仿真数据名称
        """
        ax.axis("off")

        # 准备关键统计量
        sim_stats = comparison["sim_stats"]
        mkt_stats = comparison["market_stats"]
        similarity = comparison["similarity_score"]
        ks_test = comparison["ks_test"]

        # 构建表格数据
        table_data = [
            ["Statistic", sim_name, f"{self.ticker}", "Match"],
            [
                "Std Dev",
                f"{sim_stats['std']:.5f}",
                f"{mkt_stats['std']:.5f}",
                f"{similarity['component_scores']['volatility']:.0f}%"
            ],
            [
                "Kurtosis",
                f"{sim_stats['kurtosis']:.2f}",
                f"{mkt_stats['kurtosis']:.2f}",
                f"{similarity['component_scores']['kurtosis']:.0f}%"
            ],
            [
                "Skewness",
                f"{sim_stats['skewness']:.3f}",
                f"{mkt_stats['skewness']:.3f}",
                f"{similarity['component_scores']['skewness']:.0f}%"
            ],
            [
                "VaR (5%)",
                f"{sim_stats['var_5pct']:.4f}",
                f"{mkt_stats['var_5pct']:.4f}",
                f"{similarity['component_scores']['var']:.0f}%"
            ],
            ["", "", "", ""],
            [
                "K-S Test",
                f"p={ks_test['pvalue']:.4f}",
                "Same Dist?" if ks_test['same_distribution'] else "Diff Dist",
                ""
            ],
            [
                "Total Score",
                f"{similarity['total_score']:.1f}/100",
                similarity['grade'],
                ""
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

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # 设置表头文字颜色
        for i in range(4):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax.set_title("(d) Statistical Comparison", fontweight="bold", pad=20)

    # =========================================================================
    # 报告打印
    # =========================================================================

    def print_comparison_report(
        self,
        sim_prices: List[float],
        sim_name: str = "Simulation"
    ) -> None:
        """
        打印格式化的对比分析报告

        Args:
            sim_prices: 仿真价格序列
            sim_name: 仿真数据名称
        """
        # 执行对比分析
        comparison = self.compare_with_simulation(sim_prices, sim_name)

        if "error" in comparison:
            print(f"[错误] {comparison['error']}")
            return

        print("\n" + "=" * 70)
        print(f"仿真 vs 真实市场 ({self.ticker}) 对比分析报告")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # 统计量对比表
        # ---------------------------------------------------------------------
        print("\n[统计量对比]")
        print(comparison["comparison_table"].to_string(index=False))

        # ---------------------------------------------------------------------
        # K-S 检验结果
        # ---------------------------------------------------------------------
        ks = comparison["ks_test"]
        print("\n[分布检验 - Kolmogorov-Smirnov Test]")
        print(f"  K-S统计量: {ks['statistic']:.4f}")
        print(f"  p值: {ks['pvalue']:.4f}")
        print(f"  结论: {ks['interpretation']}")

        # ---------------------------------------------------------------------
        # 相似度评分
        # ---------------------------------------------------------------------
        sim = comparison["similarity_score"]
        print("\n[相似度评分]")
        print(f"  波动率相似度:    {sim['component_scores']['volatility']:.1f}% (权重30%)")
        print(f"  峰度相似度:      {sim['component_scores']['kurtosis']:.1f}% (权重25%)")
        print(f"  偏度相似度:      {sim['component_scores']['skewness']:.1f}% (权重20%)")
        print(f"  VaR相似度:       {sim['component_scores']['var']:.1f}% (权重25%)")
        print(f"  ────────────────────────────")
        print(f"  总分: {sim['total_score']:.1f}/100")
        print(f"  评级: {sim['grade']}")

        print("\n" + "=" * 70)


# =============================================================================
# 便捷函数
# =============================================================================

def compare_with_real_market(
    sim_prices: List[float],
    ticker: str = "SPY",
    period: str = "1y",
    output_path: Optional[str] = None,
    print_report: bool = True,
) -> Dict[str, Any]:
    """
    将仿真数据与真实市场进行对比（便捷函数）

    一次调用完成数据下载、对比分析、报告打印和图表生成。

    Args:
        sim_prices: 仿真价格序列
        ticker: 股票/ETF代码
        period: 数据周期
        output_path: 图片保存路径（可选）
        print_report: 是否打印报告

    Returns:
        对比分析结果字典

    Example:
        >>> results = compare_with_real_market(
        ...     sim_prices=market.price_history,
        ...     output_path="results/comparison.png"
        ... )
    """
    # 创建基准对象
    benchmark = RealMarketBenchmark(ticker=ticker, period=period)

    # 下载数据
    success = benchmark.download_market_data()
    if not success:
        return {"error": "无法下载市场数据"}

    # 打印报告
    if print_report:
        benchmark.print_comparison_report(sim_prices)

    # 生成图表
    if output_path:
        benchmark.plot_comparison(sim_prices, output_path)

    # 返回对比结果
    return benchmark.compare_with_simulation(sim_prices)
