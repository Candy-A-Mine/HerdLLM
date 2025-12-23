"""
独立运行的图表生成模块 (Figure Generation Module)

本模块基于已有实验结果生成所有分析图表，无需重新运行实验。
支持生成全部14张图表：

    Core Figures (核心图表，原 visualization.py):
        1. hurst_comparison: Hurst指数条件对比
        2. return_distributions: 收益率分布特征（峰度/偏度）
        3. herding_analysis: 羊群效应综合分析
        4. portfolio_performance: 投资组合绩效

    Extra Figures (额外分析图表):
        5. social_consensus_effect: 社交共识影响分析
        6. personality_behavior: 人格类型行为差异
        7. price_timeseries: 价格时间序列
        8. factor_decomposition: 因子分解分析
        9. herding_heatmap: 羊群效应热力图
        10. metrics_summary: 关键指标汇总

    Supplementary Figures (补充图表):
        11. statistical_tests: 统计检验表格
        12. network_topology: 网络拓扑图

    Validation Figures (验证图表):
        13. stylized_facts: 典型事实验证
        14. real_market_comparison: 真实市场对比

使用方法:
    poetry run python -m src.generate_figures
    poetry run python -m src.generate_figures --results results/custom_30a_100r
    poetry run python -m src.generate_figures --only core
    poetry run python -m src.generate_figures --only extra
    poetry run python -m src.generate_figures --only supplementary
    poetry run python -m src.generate_figures --only validation

作者: SuZX
日期: 2024
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

# 导入验证模块
from src.stylized_facts import StylizedFactsAnalyzer
from src.real_market_benchmark import RealMarketBenchmark

# =============================================================================
# 全局样式配置 (符合学术期刊要求)
# =============================================================================

# 使用 seaborn 白色网格样式
plt.style.use("seaborn-v0_8-whitegrid")

# Matplotlib 参数设置 - 符合期刊出版标准
plt.rcParams.update({
    # 字体设置 - 使用期刊常用字体
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    # 分辨率设置 - 出版级别
    'figure.dpi': 150,
    'savefig.dpi': 600,  # 期刊要求 ≥600 DPI
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    # 线宽设置
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8,
    # 其他
    'axes.unicode_minus': False,
    'figure.constrained_layout.use': False,
    # 图例设置
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
})

# =============================================================================
# 实验条件配置 - 色盲友好配色方案 (Okabe-Ito palette)
# =============================================================================

# 四种实验条件：基线、仅记忆、仅社交、完整模型
CONDITIONS = ['baseline', 'memory_only', 'social_only', 'full']

# 色盲友好配色 (Okabe-Ito palette - 经 Nature/Science 验证的色盲友好配色)
# 该配色方案在所有类型的色盲下均可区分，且灰度打印时有足够对比度
COLORS = {
    'baseline': '#8da0cb',      # 淡紫蓝 - 基线条件
    'memory_only': '#66c2a5',   # 青绿色 - 仅记忆条件
    'social_only': '#fc8d62',   # 橙色 - 仅社交条件
    'full': '#e78ac3',          # 粉紫色 - 完整模型
}

# 线型配置 - 用于灰度打印区分
LINESTYLES = {
    'baseline': '-',        # 实线
    'memory_only': '--',    # 虚线
    'social_only': '-.',    # 点划线
    'full': ':',            # 点线
}

# 标记配置
MARKERS = {
    'baseline': 'o',
    'memory_only': 's',
    'social_only': '^',
    'full': 'D',
}

# Hatching 图案配置 - 用于黑白打印区分
HATCHES = {
    'baseline': '',         # 无图案
    'memory_only': '///',   # 斜线
    'social_only': '...',   # 点阵
    'full': 'xxx',          # 交叉
}

# 各条件的显示标签
LABELS = {
    'baseline': 'Baseline',
    'memory_only': 'Memory Only',
    'social_only': 'Social Only',
    'full': 'Full Model'
}

# 各条件的缩写标签 (用于空间紧凑的图表)
LABELS_SHORT = {
    'baseline': 'BL',
    'memory_only': 'MO',
    'social_only': 'SO',
    'full': 'FM'
}

# 人格类型配色 (与主配色方案协调)
PERSONALITY_COLORS = {
    'Conservative': '#66c2a5',  # 青绿色 - 保守型
    'Aggressive': '#e78ac3',    # 粉紫色 - 激进型
}

# 人格类型 Hatching 图案
PERSONALITY_HATCHES = {
    'Conservative': '///',      # 斜线
    'Aggressive': 'xxx',        # 交叉
}

# 图表尺寸标准 (单位: 英寸)
# 单栏图: 3.5 inch, 双栏图: 7.0 inch
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.0


def add_significance_brackets(
    ax: plt.Axes,
    x1: int, x2: int,
    y: float, h: float,
    significance: str,
    fontsize: int = 8
) -> None:
    """
    在柱状图上添加显著性标记括号

    Args:
        ax: matplotlib Axes 对象
        x1: 左侧柱子的 x 位置
        x2: 右侧柱子的 x 位置
        y: 括号底部的 y 坐标
        h: 括号高度
        significance: 显著性标记 ('*', '**', '***', 'ns')
        fontsize: 字体大小
    """
    # 绘制括号
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color='black', linewidth=0.8, clip_on=False)
    # 添加显著性文字
    ax.text((x1 + x2) / 2, y + h, significance,
            ha='center', va='bottom', fontsize=fontsize)


def get_significance_symbol(p_value: float) -> str:
    """
    根据 p 值返回显著性符号

    Args:
        p_value: 统计检验的 p 值

    Returns:
        显著性符号: '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), 'ns' (not significant)
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def _save_figure(fig: plt.Figure, output_path: Path, formats: list = None) -> None:
    """
    保存图表为多种格式

    Args:
        fig: matplotlib Figure 对象
        output_path: 输出路径 (不含扩展名或含 .png)
        formats: 输出格式列表，默认 ['pdf', 'png']
    """
    if formats is None:
        formats = ['pdf', 'png']

    output_path = Path(output_path)
    stem = output_path.stem
    parent = output_path.parent

    for fmt in formats:
        save_path = parent / f"{stem}.{fmt}"
        fig.savefig(
            save_path,
            format=fmt,
            dpi=600 if fmt == 'png' else None,  # PDF 是矢量格式
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
        )


class FigureGenerator:
    """
    实验结果图表生成器

    该类基于已保存的实验结果文件，生成全部12张分析图表。

    Attributes:
        results_dir: 实验结果目录
        output_dir: 图表输出目录
        results_file: JSON格式的聚合结果文件路径
        decisions_file: Parquet格式的决策记录文件路径
        results: 加载的聚合结果数据
        df: 加载的决策记录DataFrame

    Example:
        >>> generator = FigureGenerator("results/experiment_01")
        >>> generator.load_data()
        >>> generator.generate_all()
    """

    def __init__(self, results_dir: str | Path = "results"):
        """
        初始化图表生成器

        Args:
            results_dir: 实验结果目录路径
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "figures"
        self.output_dir.mkdir(exist_ok=True)

        # 自动查找结果文件
        self.results_file = self._find_results_file()
        self.decisions_file = self._find_decisions_file()

        # 数据容器（延迟加载）
        self.results = None
        self.df = None

    def _find_results_file(self) -> Path | None:
        """查找 JSON 格式的聚合结果文件"""
        patterns = ["*_results.json", "results.json"]
        for pattern in patterns:
            files = list(self.results_dir.glob(pattern))
            if files:
                return files[0]
        return None

    def _find_decisions_file(self) -> Path | None:
        """查找 Parquet 格式的决策记录文件"""
        patterns = ["*_decisions.parquet", "decisions.parquet"]
        for pattern in patterns:
            files = list(self.results_dir.glob(pattern))
            if files:
                return files[0]
        return None

    def load_data(self) -> None:
        """加载实验数据"""
        if self.results_file and self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            print(f"✓ 加载结果: {self.results_file}")

        if self.decisions_file and self.decisions_file.exists():
            self.df = pd.read_parquet(self.decisions_file)
            print(f"✓ 加载决策: {self.decisions_file}")

    def _get_available_conditions(self) -> list:
        """获取数据中实际存在的条件"""
        if self.results:
            return [c for c in CONDITIONS if c in self.results.get('conditions', {})]
        if self.df is not None:
            return [c for c in CONDITIONS if c in self.df['condition'].unique()]
        return []

    # =========================================================================
    # Core Figures (核心图表 - 原 visualization.py)
    # =========================================================================

    def plot_hurst_comparison(self) -> None:
        """
        绘制 Hurst 指数条件对比图

        输出文件: hurst_comparison.pdf, hurst_comparison.png
        """
        if self.results is None:
            print("✗ 缺少结果数据")
            return

        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 0.6, 4))
        available_conditions = self._get_available_conditions()

        # 提取数据
        means, stds = [], []
        for cond in available_conditions:
            stats_data = self.results['conditions'][cond]['metrics']['hurst_exponent']
            means.append(stats_data['mean'])
            stds.append(stats_data['std'])

        x = np.arange(len(available_conditions))
        colors = [COLORS.get(c, '#333') for c in available_conditions]
        hatches = [HATCHES.get(c, '') for c in available_conditions]

        # 绑制柱状图 - 添加边框和hatching增加可读性
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      alpha=0.85, edgecolor='black', linewidth=0.8,
                      error_kw={'linewidth': 1.0, 'capthick': 1.0})

        # 添加 hatching 图案
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # 柱顶数值标注 - 使用正体（非斜体）
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9,
                        fontweight='normal', fontstyle='normal')

        ax.set_xlabel('Experimental Condition')
        ax.set_ylabel('Hurst Exponent')
        ax.set_title('Hurst Exponent by Condition')
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(c, c) for c in available_conditions])
        ax.set_ylim(0, 1.05)

        # 添加 H=0.5 参考线
        ax.axhline(y=0.5, color='#666666', linestyle='--', linewidth=1.0,
                   alpha=0.7, label='Random Walk (H=0.5)')
        ax.legend(loc='lower right', fontsize=8)

        # 添加显著性标记 (baseline vs full)
        if 'baseline' in available_conditions and 'full' in available_conditions:
            baseline_idx = available_conditions.index('baseline')
            full_idx = available_conditions.index('full')
            # 计算 Welch's t-test
            n = self.results.get('config', {}).get('num_runs', 15)
            mean_bl = self.results['conditions']['baseline']['metrics']['hurst_exponent']['mean']
            std_bl = self.results['conditions']['baseline']['metrics']['hurst_exponent']['std']
            mean_full = self.results['conditions']['full']['metrics']['hurst_exponent']['mean']
            std_full = self.results['conditions']['full']['metrics']['hurst_exponent']['std']
            _, p_val, _ = self._welch_ttest(mean_bl, std_bl, n, mean_full, std_full, n)
            sig = get_significance_symbol(p_val)
            if sig != 'ns':
                y_max = max(means) + max(stds) + 0.05
                add_significance_brackets(ax, baseline_idx, full_idx, y_max, 0.03, sig)

        plt.tight_layout()
        _save_figure(fig, self.output_dir / 'hurst_comparison.png')
        plt.close()
        print("✓ hurst_comparison.pdf/png")

    def plot_return_distributions(self) -> None:
        """
        绘制收益率分布特征图（峰度和偏度）

        输出文件: return_distributions.pdf, return_distributions.png
        """
        if self.results is None:
            print("✗ 缺少结果数据")
            return

        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.2))
        available_conditions = self._get_available_conditions()

        # 提取数据
        kurtosis_data, skewness_data = [], []
        for cond in available_conditions:
            k_stats = self.results['conditions'][cond]['metrics']['return_distribution']['kurtosis']
            s_stats = self.results['conditions'][cond]['metrics']['return_distribution']['skewness']
            kurtosis_data.append({'mean': k_stats['mean'], 'std': k_stats['std']})
            skewness_data.append({'mean': s_stats['mean'], 'std': s_stats['std']})

        x = np.arange(len(available_conditions))
        colors = [COLORS.get(c, '#333') for c in available_conditions]
        hatches = [HATCHES.get(c, '') for c in available_conditions]

        # 左图：峰度
        ax1 = axes[0]
        bars1 = ax1.bar(x, [d['mean'] for d in kurtosis_data],
                yerr=[d['std'] for d in kurtosis_data],
                capsize=4, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8,
                error_kw={'linewidth': 1.0, 'capthick': 1.0})
        for bar, hatch in zip(bars1, hatches):
            bar.set_hatch(hatch)
        ax1.axhline(y=0, color='#666666', linestyle='--', alpha=0.7,
                    label='Normal (κ=0)', linewidth=1.0)
        ax1.set_xlabel('Condition')
        ax1.set_ylabel('Excess Kurtosis')
        ax1.set_title('(a) Return Kurtosis')
        ax1.set_xticks(x)
        ax1.set_xticklabels([LABELS.get(c, c).replace(' ', '\n') for c in available_conditions],
                           fontsize=8)
        ax1.legend(loc='upper right', fontsize=7)

        # 右图：偏度
        ax2 = axes[1]
        bars2 = ax2.bar(x, [d['mean'] for d in skewness_data],
                yerr=[d['std'] for d in skewness_data],
                capsize=4, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8,
                error_kw={'linewidth': 1.0, 'capthick': 1.0})
        for bar, hatch in zip(bars2, hatches):
            bar.set_hatch(hatch)
        ax2.axhline(y=0, color='#666666', linestyle='--', alpha=0.7,
                    label='Symmetric', linewidth=1.0)
        ax2.set_xlabel('Condition')
        ax2.set_ylabel('Skewness')
        ax2.set_title('(b) Return Skewness')
        ax2.set_xticks(x)
        ax2.set_xticklabels([LABELS.get(c, c).replace(' ', '\n') for c in available_conditions],
                           fontsize=8)
        ax2.legend(loc='upper right', fontsize=7)

        plt.tight_layout()
        _save_figure(fig, self.output_dir / 'return_distributions.png')
        plt.close()
        print("✓ return_distributions.pdf/png")

    def plot_herding_analysis(self) -> None:
        """
        绘制羊群效应综合分析图（三图联合）

        输出文件: herding_analysis.pdf, herding_analysis.png
        """
        if self.results is None:
            print("✗ 缺少结果数据")
            return

        # 增大图表尺寸以容纳缩写说明
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 3.2))
        available_conditions = self._get_available_conditions()

        # 提取数据
        lsv_data, ratio_data, hurst_data = [], [], []
        for cond in available_conditions:
            metrics = self.results['conditions'][cond]['metrics']
            lsv_mean = metrics['lsv_herding']['lsv_mean']['mean']
            lsv_std = metrics['lsv_herding']['lsv_mean']['std']
            # 处理接近0的值，避免科学计数法显示问题
            if abs(lsv_mean) < 1e-10:
                lsv_mean = 0.0
            if abs(lsv_std) < 1e-10:
                lsv_std = 0.0
            lsv_data.append({'mean': lsv_mean, 'std': lsv_std})
            ratio_data.append({
                'mean': metrics['lsv_herding']['herding_ratio']['mean'] * 100,
                'std': metrics['lsv_herding']['herding_ratio']['std'] * 100
            })
            hurst_data.append({
                'mean': metrics['hurst_exponent']['mean'],
                'std': metrics['hurst_exponent']['std']
            })

        x = np.arange(len(available_conditions))
        colors = [COLORS.get(c, '#333') for c in available_conditions]
        hatches = [HATCHES.get(c, '') for c in available_conditions]
        # 使用缩写标签避免换行问题
        xlabels = [LABELS_SHORT.get(c, c) for c in available_conditions]

        # 左图：LSV测度
        ax1 = axes[0]
        bars1 = ax1.bar(x, [d['mean'] for d in lsv_data], yerr=[d['std'] for d in lsv_data],
                capsize=3, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8,
                error_kw={'linewidth': 0.8, 'capthick': 0.8})
        for bar, hatch in zip(bars1, hatches):
            bar.set_hatch(hatch)
        ax1.axhline(y=0, color='#666666', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.set_xlabel('Condition', fontsize=9)
        ax1.set_ylabel('LSV Measure', fontsize=9)
        ax1.set_title('(a) LSV Herding', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(xlabels, fontsize=8)
        # 使用 ScalarFormatter 避免科学计数法
        ax1.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
        ax1.ticklabel_format(style='plain', axis='y')

        # 中图：羊群比例
        ax2 = axes[1]
        bars2 = ax2.bar(x, [d['mean'] for d in ratio_data], yerr=[d['std'] for d in ratio_data],
                capsize=3, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8,
                error_kw={'linewidth': 0.8, 'capthick': 0.8})
        for bar, hatch in zip(bars2, hatches):
            bar.set_hatch(hatch)
        ax2.set_xlabel('Condition', fontsize=9)
        ax2.set_ylabel('Herding Ratio (%)', fontsize=9)
        ax2.set_title('(b) Herding Proportion', fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(xlabels, fontsize=8)
        ax2.set_ylim(0, 100)

        # 右图：Hurst指数
        ax3 = axes[2]
        bars3 = ax3.bar(x, [d['mean'] for d in hurst_data], yerr=[d['std'] for d in hurst_data],
                capsize=3, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8,
                error_kw={'linewidth': 0.8, 'capthick': 0.8})
        for bar, hatch in zip(bars3, hatches):
            bar.set_hatch(hatch)
        ax3.axhline(y=0.5, color='#666666', linestyle='--', alpha=0.7,
                    label='H=0.5', linewidth=1.0)
        ax3.set_xlabel('Condition', fontsize=9)
        ax3.set_ylabel('Hurst Exponent', fontsize=9)
        ax3.set_title('(c) Market Efficiency', fontsize=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(xlabels, fontsize=8)
        # 将图例移到图表下方，避免遮挡数据
        ax3.legend(loc='upper right', fontsize=7, framealpha=0.9)
        ax3.set_ylim(0, 1)

        # 添加缩写说明到图表底部
        abbr_text = "BL: Baseline, MO: Memory Only, SO: Social Only, FM: Full Model"
        fig.text(0.5, 0.01, abbr_text, ha='center', va='bottom', fontsize=7,
                 style='italic', color='#555555')

        plt.tight_layout(rect=[0, 0.04, 1, 1])  # 为底部缩写留出空间
        _save_figure(fig, self.output_dir / 'herding_analysis.png')
        plt.close()
        print("✓ herding_analysis.pdf/png")

    def plot_portfolio_performance(self) -> None:
        """
        绘制按人格类型和实验条件分组的投资组合绩效图

        输出文件: portfolio_performance.pdf, portfolio_performance.png
        """
        if self.results is None:
            print("✗ 缺少结果数据")
            return

        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.5))
        available_conditions = self._get_available_conditions()

        # 计算初始投资组合价值
        config = self.results.get('config', {})
        base_config = config.get('base_config', {})
        initial_value = (
            base_config.get('initial_cash', 10000) +
            base_config.get('initial_holdings', 10) *
            base_config.get('initial_price', 100)
        )

        # 准备数据
        data = []
        for cond in available_conditions:
            portfolio_stats = self.results['conditions'][cond].get('portfolio_stats', {})
            for personality, stats_data in portfolio_stats.items():
                ret = (stats_data['mean'] - initial_value) / initial_value * 100
                data.append({
                    'condition': cond,
                    'personality': personality,
                    'return': ret,
                    'std': stats_data['std'] / initial_value * 100,
                })

        if not data:
            print("✗ 缺少投资组合数据")
            plt.close()
            return

        df = pd.DataFrame(data)
        personalities = df['personality'].unique()
        x = np.arange(len(available_conditions))
        width = 0.35

        for i, personality in enumerate(personalities):
            pers_data = df[df['personality'] == personality]
            offset = (i - len(personalities) / 2 + 0.5) * width
            bars = ax.bar(x + offset, pers_data['return'], width,
                   yerr=pers_data['std'], capsize=3,
                   label=personality,
                   color=PERSONALITY_COLORS.get(personality, f'C{i}'),
                   hatch=PERSONALITY_HATCHES.get(personality, ''),
                   alpha=0.85, edgecolor='black', linewidth=0.8,
                   error_kw={'linewidth': 0.8, 'capthick': 0.8})

        ax.axhline(y=0, color='#666666', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xlabel('Experimental Condition')
        ax.set_ylabel('Return (%)')
        ax.set_title('Portfolio Performance by Personality Type')
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(c, c) for c in available_conditions])
        # 图例放在图内右上角
        ax.legend(title='Personality', loc='upper right', fontsize=8)

        plt.tight_layout()
        _save_figure(fig, self.output_dir / 'portfolio_performance.png')
        plt.close()
        print("✓ portfolio_performance.pdf/png")

    # =========================================================================
    # Extra Figures (额外分析图表)
    # =========================================================================

    def plot_social_consensus_effect(self) -> None:
        """
        分析社交共识对Agent决策的影响

        输出文件: social_consensus_effect.pdf, social_consensus_effect.png
        """
        if self.df is None:
            print("✗ 缺少决策数据")
            return

        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.2))

        # 筛选包含社交网络的条件
        social_df = self.df[self.df['condition'].isin(['social_only', 'full'])].copy()

        if social_df.empty:
            print("✗ 缺少社交条件数据")
            plt.close()
            return

        # 计算社交共识强度
        social_df['consensus_strength'] = social_df[
            ['social_buy_pct', 'social_sell_pct', 'social_hold_pct']
        ].max(axis=1)

        social_df['dominant_social'] = social_df[
            ['social_buy_pct', 'social_sell_pct', 'social_hold_pct']
        ].idxmax(axis=1).map({
            'social_buy_pct': 'BUY',
            'social_sell_pct': 'SELL',
            'social_hold_pct': 'HOLD'
        })

        social_df = social_df[social_df['round_num'] > 0]

        # 左图：共识强度分布 - 使用阶梯直方图避免重叠
        ax1 = axes[0]
        bins = np.linspace(30, 100, 15)
        for cond in ['social_only', 'full']:
            data = social_df[social_df['condition'] == cond]['consensus_strength']
            ax1.hist(data, bins=bins, alpha=0.7, label=LABELS[cond],
                     color=COLORS[cond], edgecolor=COLORS[cond],
                     histtype='stepfilled', linewidth=1.5)
        ax1.set_xlabel('Social Consensus Strength (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('(a) Distribution of Social Consensus')
        ax1.legend(fontsize=8, framealpha=0.9)
        ax1.axvline(x=70, color='#666666', linestyle='--', alpha=0.7, linewidth=1.0)

        # 右图：跟风率
        ax2 = axes[1]
        social_df['followed'] = social_df['action'] == social_df['dominant_social']

        bins_rate = [0, 40, 50, 60, 70, 80, 90, 100]
        social_df['consensus_bin'] = pd.cut(social_df['consensus_strength'], bins=bins_rate)
        follow_rate = social_df.groupby(['consensus_bin', 'condition'], observed=True)['followed'].mean().unstack()

        x = np.arange(len(bins_rate) - 1)
        width = 0.35
        if 'social_only' in follow_rate.columns:
            ax2.bar(x - width/2, follow_rate['social_only']*100, width,
                    label=LABELS['social_only'], color=COLORS['social_only'],
                    hatch=HATCHES['social_only'],
                    alpha=0.85, edgecolor='black', linewidth=0.8)
        if 'full' in follow_rate.columns:
            ax2.bar(x + width/2, follow_rate['full']*100, width,
                    label=LABELS['full'], color=COLORS['full'],
                    hatch=HATCHES['full'],
                    alpha=0.85, edgecolor='black', linewidth=0.8)
        ax2.set_xlabel('Social Consensus Strength (%)')
        ax2.set_ylabel('Follow Rate (%)')
        ax2.set_title('(b) Consensus Following Behavior')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['<40', '40-50', '50-60', '60-70', '70-80', '80-90', '>90'],
                           fontsize=8)
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        _save_figure(fig, self.output_dir / 'social_consensus_effect.png')
        plt.close()
        print("✓ social_consensus_effect.pdf/png")

    def plot_personality_behavior(self) -> None:
        """
        分析不同人格类型的交易行为差异

        输出文件: personality_behavior.pdf, personality_behavior.png
        """
        if self.df is None:
            print("✗ 缺少决策数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.5))
        available_conditions = [c for c in CONDITIONS if c in self.df['condition'].unique()]

        # 定义动作颜色 - 使用易区分的配色
        action_colors = {'BUY': '#648FFF', 'HOLD': '#999999', 'SELL': '#DC267F'}

        for idx, cond in enumerate(available_conditions[:4]):
            ax = axes[idx // 2, idx % 2]
            cond_df = self.df[self.df['condition'] == cond]

            action_by_pers = cond_df.groupby(['personality', 'action']).size().unstack(fill_value=0)
            action_by_pers = action_by_pers.div(action_by_pers.sum(axis=1), axis=0) * 100

            cols = [c for c in ['BUY', 'HOLD', 'SELL'] if c in action_by_pers.columns]
            colors_list = [action_colors.get(c, '#666666') for c in cols]

            action_by_pers[cols].plot(
                kind='bar', stacked=True, ax=ax,
                color=colors_list,
                edgecolor='white', linewidth=0.5,
                legend=False  # 禁用所有子图图例
            )
            ax.set_title(f'({chr(97+idx)}) {LABELS.get(cond, cond)}', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Action Distribution (%)' if idx % 2 == 0 else '')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)
            ax.set_ylim(0, 100)

        # 添加统一的公共图例
        handles = [plt.Rectangle((0,0),1,1, facecolor=action_colors[a], edgecolor='white')
                   for a in ['BUY', 'HOLD', 'SELL']]
        fig.legend(handles, ['BUY', 'HOLD', 'SELL'], title='Action',
                   loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=8)

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        _save_figure(fig, self.output_dir / 'personality_behavior.png')
        plt.close()
        print("✓ personality_behavior.pdf/png")

    def plot_price_timeseries(self) -> None:
        """
        绘制四个实验条件下的价格时间序列对比图

        输出文件: price_timeseries.pdf, price_timeseries.png
        """
        if self.df is None:
            print("✗ 缺少决策数据")
            return

        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 3.5))
        available_conditions = [c for c in CONDITIONS if c in self.df['condition'].unique()]

        for cond in available_conditions:
            cond_df = self.df[(self.df['condition'] == cond) & (self.df['run_idx'] == 0)]
            price_series = cond_df.groupby('round_num')['current_price'].first()
            ax.plot(price_series.index, price_series.values,
                    label=LABELS.get(cond, cond),
                    color=COLORS.get(cond, '#333'),
                    linestyle=LINESTYLES.get(cond, '-'),
                    marker=MARKERS.get(cond, 'o'),
                    markevery=10,  # 每10个点标记一次
                    markersize=4,
                    linewidth=1.5, alpha=0.9)

        ax.set_xlabel('Round')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price Evolution Across Conditions')
        ax.legend(loc='best', framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # 添加初始价格参考线
        ax.axhline(y=100, color='#666666', linestyle=':', alpha=0.5,
                   linewidth=0.8, label='_nolegend_')

        plt.tight_layout()
        _save_figure(fig, self.output_dir / 'price_timeseries.png')
        plt.close()
        print("✓ price_timeseries.pdf/png")

    def plot_factor_decomposition(self) -> None:
        """
        因子分解分析：展示记忆和社交网络对收益率峰度的边际效应

        输出文件: factor_decomposition.pdf, factor_decomposition.png
        """
        if self.results is None:
            print("✗ 缺少结果数据")
            return

        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.2))
        available_conditions = self._get_available_conditions()

        # 提取峰度数据
        kurtosis = {}
        kurtosis_std = {}
        for cond in available_conditions:
            kurtosis[cond] = self.results['conditions'][cond]['metrics']['return_distribution']['kurtosis']['mean']
            kurtosis_std[cond] = self.results['conditions'][cond]['metrics']['return_distribution']['kurtosis']['std']

        # 左图：条件对比
        ax1 = axes[0]
        x = np.arange(len(available_conditions))
        hatches = [HATCHES.get(c, '') for c in available_conditions]
        bars = ax1.bar(x, [kurtosis[c] for c in available_conditions],
                       yerr=[kurtosis_std[c] for c in available_conditions],
                       color=[COLORS.get(c, '#333') for c in available_conditions],
                       capsize=4, alpha=0.85, edgecolor='black', linewidth=0.8,
                       error_kw={'linewidth': 0.8, 'capthick': 0.8})
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax1.set_xticks(x)
        ax1.set_xticklabels([LABELS.get(c, c) for c in available_conditions],
                           rotation=15, fontsize=8, ha='right')
        ax1.set_ylabel('Return Kurtosis')
        ax1.set_title('(a) Kurtosis by Condition')
        ax1.axhline(y=0, color='#666666', linestyle='-', alpha=0.3, linewidth=0.8)

        # 柱顶数值标注 - 统一不带正号
        for bar, cond in zip(bars, available_conditions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        # 右图：因子分解
        ax2 = axes[1]
        if all(c in kurtosis for c in CONDITIONS):
            baseline = kurtosis['baseline']
            memory_effect = kurtosis['memory_only'] - baseline
            social_effect = kurtosis['social_only'] - baseline
            interaction = kurtosis['full'] - kurtosis['social_only'] - kurtosis['memory_only'] + baseline

            effects = ['Baseline', 'Memory\nEffect', 'Social\nEffect', 'Interaction']
            values = [baseline, memory_effect, social_effect, interaction]
            colors_eff = [COLORS['baseline'], COLORS['memory_only'],
                         COLORS['social_only'], COLORS['full']]
            hatches_eff = [HATCHES['baseline'], HATCHES['memory_only'],
                          HATCHES['social_only'], HATCHES['full']]

            bars2 = ax2.bar(effects, values, color=colors_eff, alpha=0.85,
                            edgecolor='black', linewidth=0.8)
            for bar, hatch in zip(bars2, hatches_eff):
                bar.set_hatch(hatch)
            ax2.set_ylabel('Kurtosis Contribution')
            ax2.set_title('(b) Factor Decomposition')
            ax2.axhline(y=0, color='#666666', linestyle='-', alpha=0.5, linewidth=0.8)

            # 柱顶数值标注 - 统一不带正号
            for bar, val in zip(bars2, values):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 0.1 if height >= 0 else -0.1
                ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                         f'{val:.2f}', ha='center', va=va, fontsize=8)

        plt.tight_layout()
        _save_figure(fig, self.output_dir / 'factor_decomposition.png')
        plt.close()
        print("✓ factor_decomposition.pdf/png")

    def plot_herding_heatmap(self) -> None:
        """
        绘制羊群效应热力图

        输出文件: herding_heatmap.pdf, herding_heatmap.png
        """
        if self.df is None:
            print("✗ 缺少决策数据")
            return

        available_conditions = [c for c in CONDITIONS if c in self.df['condition'].unique()]
        n_conds = len(available_conditions)
        if n_conds == 0:
            return

        rows = (n_conds + 1) // 2
        # 增加右侧空间用于共享 colorbar
        fig, axes = plt.subplots(rows, 2, figsize=(DOUBLE_COL_WIDTH + 0.5, 2.5 * rows))
        if n_conds == 1:
            axes = np.array([[axes, None]])
        elif n_conds == 2:
            axes = np.array([axes])
        axes = axes.flatten()

        # 使用蓝-白-红色调，色盲友好
        cmap = plt.cm.RdBu_r

        # 先计算所有条件的买入率范围，用于统一色标
        all_buy_pcts = []
        for cond in available_conditions:
            cond_df = self.df[(self.df['condition'] == cond) & (self.df['run_idx'] == 0)]
            action_counts = cond_df.groupby(['round_num', 'action']).size().unstack(fill_value=0)
            action_pct = action_counts.div(action_counts.sum(axis=1), axis=0) * 100
            buy_pct = action_pct.get('BUY', pd.Series([0] * len(action_pct))).values
            all_buy_pcts.extend(buy_pct)

        # 使用实际数据范围（添加一些边距）
        vmin = max(0, np.percentile(all_buy_pcts, 2) - 5)
        vmax = min(100, np.percentile(all_buy_pcts, 98) + 5)

        # 保存最后一个 imshow 对象用于共享 colorbar
        im = None
        for idx, cond in enumerate(available_conditions):
            ax = axes[idx]
            cond_df = self.df[(self.df['condition'] == cond) & (self.df['run_idx'] == 0)]

            action_counts = cond_df.groupby(['round_num', 'action']).size().unstack(fill_value=0)
            action_pct = action_counts.div(action_counts.sum(axis=1), axis=0) * 100
            buy_pct = action_pct.get('BUY', pd.Series([0] * len(action_pct))).values

            im = ax.imshow(buy_pct.reshape(1, -1), aspect='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax, extent=[0, len(buy_pct), 0, 1])
            ax.set_title(f'({chr(97+idx)}) {LABELS.get(cond, cond)}', fontsize=10)
            ax.set_xlabel('Round', fontsize=9)
            ax.set_yticks([])

        for idx in range(len(available_conditions), len(axes)):
            if axes[idx] is not None:
                axes[idx].axis('off')

        # 先调用 tight_layout
        plt.tight_layout()

        # 然后添加共享 colorbar（位于右侧）
        if im is not None:
            # 调整子图位置为 colorbar 腾出空间
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Buy Rate (%)', fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        _save_figure(fig, self.output_dir / 'herding_heatmap.png')
        plt.close()
        print("✓ herding_heatmap.pdf/png")

    def plot_metrics_summary(self) -> None:
        """
        绘制关键指标汇总对比图 - 拆分为4个独立子图

        输出文件: metrics_summary.pdf, metrics_summary.png
        """
        if self.results is None:
            print("✗ 缺少结果数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5))
        available_conditions = self._get_available_conditions()

        # 提取数据
        metrics_data = {
            'Kurtosis': [],
            'Herding (%)': [],
            'Hurst': [],
            'Volatility (%)': []
        }
        for cond in available_conditions:
            cond_results = self.results['conditions'][cond]['metrics']
            metrics_data['Kurtosis'].append({
                'mean': cond_results['return_distribution']['kurtosis']['mean'],
                'std': cond_results['return_distribution']['kurtosis']['std']
            })
            metrics_data['Herding (%)'].append({
                'mean': cond_results['lsv_herding']['herding_ratio']['mean'] * 100,
                'std': cond_results['lsv_herding']['herding_ratio']['std'] * 100
            })
            metrics_data['Hurst'].append({
                'mean': cond_results['hurst_exponent']['mean'],
                'std': cond_results['hurst_exponent']['std']
            })
            metrics_data['Volatility (%)'].append({
                'mean': cond_results['avg_volatility']['mean'] * 100,
                'std': cond_results['avg_volatility']['std'] * 100
            })

        metric_names = ['Kurtosis', 'Herding (%)', 'Hurst', 'Volatility (%)']
        colors = [COLORS.get(c, '#333') for c in available_conditions]
        hatches = [HATCHES.get(c, '') for c in available_conditions]

        for idx, metric in enumerate(metric_names):
            ax = axes[idx // 2, idx % 2]
            x = np.arange(len(available_conditions))

            means = [d['mean'] for d in metrics_data[metric]]
            stds = [d['std'] for d in metrics_data[metric]]

            bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                         alpha=0.85, edgecolor='black', linewidth=0.8,
                         error_kw={'linewidth': 0.8, 'capthick': 0.8})
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

            ax.set_title(f'({chr(97+idx)}) {metric}', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([LABELS.get(c, c) for c in available_conditions],
                             fontsize=8, rotation=15, ha='right')
            ax.set_ylabel(metric, fontsize=9)

            # 为每个指标添加适当的参考线
            if metric == 'Hurst':
                ax.axhline(y=0.5, color='#666666', linestyle='--', alpha=0.7, linewidth=0.8)
                ax.set_ylim(0, 1)
            elif metric == 'Herding (%)':
                ax.set_ylim(0, 100)

        plt.tight_layout()
        _save_figure(fig, self.output_dir / 'metrics_summary.png')
        plt.close()
        print("✓ metrics_summary.pdf/png")

    # =========================================================================
    # Supplementary Figures (补充图表)
    # =========================================================================

    @staticmethod
    def _welch_ttest(
        mean1: float, std1: float, n1: int,
        mean2: float, std2: float, n2: int
    ) -> Tuple[float, float, float]:
        """Welch's t-test（韦尔奇t检验）"""
        se1 = std1**2 / n1
        se2 = std2**2 / n2
        if se1 + se2 == 0:
            return 0.0, 1.0, 1.0
        t_stat = (mean1 - mean2) / np.sqrt(se1 + se2)
        df = (se1 + se2)**2 / (se1**2/(n1-1) + se2**2/(n2-1)) if (se1**2/(n1-1) + se2**2/(n2-1)) > 0 else 1
        p_value = 2 * stats.t.sf(abs(t_stat), df)
        return t_stat, p_value, df

    def plot_statistical_table(self) -> None:
        """
        生成统计显著性检验表格图

        输出文件: statistical_tests.pdf, statistical_tests.png, statistical_tests.tex
        """
        if self.results is None:
            print("✗ 缺少结果数据")
            return

        conditions = self.results.get("conditions", {})
        n = self.results.get("config", {}).get("num_runs", 15)

        metrics = [
            ("Kurtosis", "return_distribution", "kurtosis"),
            ("Herding Ratio", "lsv_herding", "herding_ratio"),
            ("Hurst Exponent", "hurst_exponent", None),
            ("Volatility", "avg_volatility", None),
        ]

        comparisons = [
            ("baseline", "social_only", "Social Network Effect"),
            ("baseline", "memory_only", "Memory Effect"),
            ("baseline", "full", "Full vs Baseline"),
            ("social_only", "full", "Memory | Social"),
        ]

        fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH * 1.4, 6))
        ax.axis('off')

        headers = ["Metric", "Comparison", "Group1", "Group2", "t-stat", "p-value", "Sig."]
        table_data = []

        for metric_name, metric_key1, metric_key2 in metrics:
            for cond1, cond2, comp_name in comparisons:
                if cond1 not in conditions or cond2 not in conditions:
                    continue

                if metric_key2:
                    d1 = conditions[cond1]["metrics"][metric_key1][metric_key2]
                    d2 = conditions[cond2]["metrics"][metric_key1][metric_key2]
                else:
                    d1 = conditions[cond1]["metrics"][metric_key1]
                    d2 = conditions[cond2]["metrics"][metric_key1]

                mean1, std1 = d1["mean"], d1["std"]
                mean2, std2 = d2["mean"], d2["std"]

                t_stat, p_val, df = self._welch_ttest(mean1, std1, n, mean2, std2, n)

                if p_val < 0.001:
                    sig = "***"
                elif p_val < 0.01:
                    sig = "**"
                elif p_val < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

                if "Volatility" in metric_name:
                    m1_str = f"{mean1*100:.2f}±{std1*100:.2f}%"
                    m2_str = f"{mean2*100:.2f}±{std2*100:.2f}%"
                elif "Herding" in metric_name:
                    m1_str = f"{mean1*100:.1f}±{std1*100:.1f}%"
                    m2_str = f"{mean2*100:.1f}±{std2*100:.1f}%"
                else:
                    m1_str = f"{mean1:.3f}±{std1:.3f}"
                    m2_str = f"{mean2:.3f}±{std2:.3f}"

                table_data.append([
                    metric_name,
                    f"{LABELS.get(cond1, cond1)} vs {LABELS.get(cond2, cond2)}",
                    m1_str, m2_str,
                    f"{t_stat:.2f}",
                    f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001",
                    sig
                ])

        if not table_data:
            print("✗ 无法生成统计表格")
            plt.close()
            return

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=['#4472C4'] * len(headers),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.1, 1.6)

        for i in range(len(headers)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        for i, row in enumerate(table_data):
            sig = row[-1]
            cell = table[(i+1, 6)]
            # 使用色盲友好配色 (蓝-青-橙 渐变，避免红绿)
            if sig == "***":
                cell.set_facecolor('#B3CDE3')  # 淡蓝色 - 高度显著
            elif sig == "**":
                cell.set_facecolor('#CCEBC5')  # 淡青绿色 - 中度显著
            elif sig == "*":
                cell.set_facecolor('#FED9A6')  # 淡橙色 - 显著

        plt.title("Statistical Significance Tests (Welch's t-test)\n*** p<0.001, ** p<0.01, * p<0.05, ns: not significant",
                  fontsize=10, pad=15)

        plt.tight_layout()
        _save_figure(fig, self.output_dir / "statistical_tests.png")
        plt.close()
        print("✓ statistical_tests.pdf/png")

        self._save_latex_table(table_data, headers)

    def _save_latex_table(self, table_data: list, headers: list) -> None:
        """将统计表格保存为LaTeX格式"""
        latex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Statistical Significance Tests (Welch's t-test)}",
            r"\label{tab:statistical_tests}",
            r"\small",
            r"\begin{tabular}{lllllll}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]

        current_metric = None
        for row in table_data:
            if row[0] != current_metric:
                if current_metric is not None:
                    latex_lines.append(r"\midrule")
                current_metric = row[0]
            # 转义 LaTeX 特殊字符
            escaped_row = [cell.replace('%', r'\%').replace('_', r'\_') for cell in row]
            latex_lines.append(" & ".join(escaped_row) + r" \\")

        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item Note: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant",
            r"\end{tablenotes}",
            r"\end{table}",
        ])

        with open(self.output_dir / "statistical_tests.tex", "w") as f:
            f.write("\n".join(latex_lines))
        print("✓ statistical_tests.tex")

    def plot_network_topology(self, num_agents: int = 30, m: int = 3, seed: int = 42) -> None:
        """
        生成社交网络拓扑可视化

        输出文件: network_topology.pdf, network_topology.png
        """
        G = nx.barabasi_albert_graph(n=num_agents, m=m, seed=seed)
        degrees = dict(G.degree())

        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.8))

        # 左图：网络拓扑
        ax1 = axes[0]
        pos = nx.spring_layout(G, seed=42, k=2)

        node_sizes = [200 + degrees[n] * 80 for n in G.nodes()]
        node_colors = [degrees[n] for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, edge_color='gray', width=0.5)
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax1,
                                       node_size=node_sizes,
                                       node_color=node_colors,
                                       cmap=plt.cm.YlOrRd,
                                       alpha=0.9,
                                       edgecolors='black',
                                       linewidths=0.5)

        # 只标注 Hub 节点 - 将标签偏移到节点外部
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        labels = {n: f"k={d}" for n, d in top_nodes}
        # 创建偏移后的标签位置 (增加偏移量避免重叠)
        label_pos = {n: (pos[n][0], pos[n][1] + 0.15) for n in labels.keys()}
        nx.draw_networkx_labels(G, label_pos, labels, ax=ax1, font_size=8, font_weight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                         edgecolor='none', alpha=0.8))

        cbar = plt.colorbar(nodes, ax=ax1, shrink=0.7, aspect=15)
        cbar.set_label('Node Degree', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax1.set_title(f"(a) Barabási-Albert Network (n={num_agents}, m={m})", fontsize=10)
        ax1.axis('off')

        # 右图：度分布
        ax2 = axes[1]
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)

        degree_count = {}
        for d in degree_sequence:
            degree_count[d] = degree_count.get(d, 0) + 1

        degrees_list = list(degree_count.keys())
        counts = list(degree_count.values())

        # 使用与主配色方案一致的颜色
        ax2.bar(degrees_list, counts, color='#8da0cb', edgecolor='black',
                alpha=0.85, linewidth=0.8)
        ax2.set_xlabel('Node Degree (k)', fontsize=9)
        ax2.set_ylabel('Number of Nodes', fontsize=9)
        ax2.set_title('(b) Degree Distribution', fontsize=10)

        ax2.set_xticks(range(min(degrees_list), max(degrees_list)+1, 2))
        ax2.grid(axis='y', alpha=0.3, linewidth=0.5)

        # 将网络统计信息移到图表底部作为简洁注释
        stats_text = (
            f"N={G.number_of_nodes()}, E={G.number_of_edges()}, "
            f"⟨k⟩={np.mean(degree_sequence):.1f}, "
            f"C={nx.average_clustering(G):.3f}, "
            f"ρ={nx.density(G):.3f}"
        )
        fig.text(0.5, 0.01, stats_text, ha='center', va='bottom', fontsize=8,
                 style='italic', color='#555555')

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        _save_figure(fig, self.output_dir / "network_topology.png")
        plt.close()
        print("✓ network_topology.pdf/png")

    # =========================================================================
    # Validation Figures (验证图表)
    # =========================================================================

    def _extract_price_series(self, condition: str = "full", run_idx: int = 0) -> list:
        """
        从决策数据中提取价格序列

        Args:
            condition: 实验条件
            run_idx: 运行索引

        Returns:
            价格序列列表
        """
        if self.df is None:
            return []

        price_df = self.df[(self.df["condition"] == condition) & (self.df["run_idx"] == run_idx)]
        if price_df.empty:
            # 尝试使用可用的条件
            available = self.df["condition"].unique()
            if len(available) > 0:
                condition = available[-1]  # 使用最后一个条件（通常是 full）
                price_df = self.df[(self.df["condition"] == condition) & (self.df["run_idx"] == run_idx)]

        if price_df.empty:
            return []

        price_series = price_df.groupby("round_num")["current_price"].first()
        return price_series.tolist()

    def plot_stylized_facts(self, condition: str = "full") -> None:
        """
        生成 Stylized Facts（典型事实）验证图

        输出文件: stylized_facts.pdf, stylized_facts.png
        """
        prices = self._extract_price_series(condition)
        if not prices:
            print("✗ 缺少价格数据，无法生成 stylized_facts")
            return

        analyzer = StylizedFactsAnalyzer(price_history=prices)
        analyzer.generate_report()
        fig = analyzer.plot_stylized_facts(
            output_path=str(self.output_dir / "stylized_facts.png"),
            dpi=600
        )
        # 同时保存 PDF
        fig.savefig(self.output_dir / "stylized_facts.pdf", format='pdf', bbox_inches='tight')
        plt.close()
        print("✓ stylized_facts.pdf/png")

    def plot_real_market_comparison(self, condition: str = "full", ticker: str = "SPY") -> None:
        """
        生成与真实市场对比图

        输出文件: real_market_comparison.pdf, real_market_comparison.png
        """
        prices = self._extract_price_series(condition)
        if not prices:
            print("✗ 缺少价格数据，无法生成 real_market_comparison")
            return

        try:
            benchmark = RealMarketBenchmark(ticker=ticker, period="1y")
            benchmark.download_market_data()
            fig = benchmark.plot_comparison(
                sim_prices=prices,
                output_path=str(self.output_dir / "real_market_comparison.png"),
                dpi=600
            )
            # 同时保存 PDF
            fig.savefig(self.output_dir / "real_market_comparison.pdf", format='pdf', bbox_inches='tight')
            plt.close()
            print("✓ real_market_comparison.pdf/png")
        except Exception as e:
            print(f"✗ real_market_comparison 生成失败: {e}")

    # =========================================================================
    # 批量生成接口
    # =========================================================================

    def generate_core_figures(self) -> None:
        """生成核心图表（原 visualization.py 的4张图）"""
        print("\n[Core Figures]")
        self.plot_hurst_comparison()
        self.plot_return_distributions()
        self.plot_herding_analysis()
        self.plot_portfolio_performance()

    def generate_extra_figures(self) -> None:
        """生成额外分析图表"""
        print("\n[Extra Figures]")
        self.plot_social_consensus_effect()
        self.plot_personality_behavior()
        self.plot_price_timeseries()
        self.plot_factor_decomposition()
        self.plot_herding_heatmap()
        self.plot_metrics_summary()

    def generate_supplementary_figures(self) -> None:
        """生成补充图表"""
        print("\n[Supplementary Figures]")
        self.plot_statistical_table()
        self.plot_network_topology()

    def generate_validation_figures(self) -> None:
        """生成验证图表（典型事实 + 真实市场对比）"""
        print("\n[Validation Figures]")
        self.plot_stylized_facts()
        self.plot_real_market_comparison()

    def generate_all(self) -> None:
        """生成全部14张图表"""
        self.generate_core_figures()
        self.generate_extra_figures()
        self.generate_supplementary_figures()
        self.generate_validation_figures()


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="生成实验结果图表（全部14张）")
    parser.add_argument("--results", type=str, default="results",
                        help="结果目录路径 (默认: results)")
    parser.add_argument("--only", type=str,
                        choices=["core", "extra", "supplementary", "validation"],
                        help="只生成指定类型的图表")
    args = parser.parse_args()

    print("=" * 50)
    print("图表生成器 (Full 14 Figures)")
    print("输出格式: PDF (矢量) + PNG (600 DPI)")
    print("=" * 50)

    generator = FigureGenerator(args.results)
    generator.load_data()

    if args.only == "core":
        generator.generate_core_figures()
    elif args.only == "extra":
        generator.generate_extra_figures()
    elif args.only == "supplementary":
        generator.generate_supplementary_figures()
    elif args.only == "validation":
        generator.generate_validation_figures()
    else:
        generator.generate_all()

    print("\n" + "=" * 50)
    print(f"完成！图表已保存到 {generator.output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
