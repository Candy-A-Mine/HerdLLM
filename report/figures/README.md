# 实验图表说明文档

本文档描述了实验生成的所有图表，供论文写作时引用参考。

---

## 实验基本信息

| 参数 | 值 |
|------|-----|
| Agent数量 | 30 |
| 交易轮次 | 100 |
| 蒙特卡洛重复 | 15次 |
| 实验条件 | baseline, memory_only, social_only, full |
| 生成时间 | 2025-12-15 |

---

## 图表索引

| 编号 | 文件名 | 类型 | 推荐章节 |
|------|--------|------|----------|
| 1 | return_distributions.png | 核心证据 | 实验结果 |
| 2 | factor_decomposition.png | 核心证据 | 实验结果 |
| 3 | social_consensus_effect.png | 机制分析 | 结果分析 |
| 4 | herding_analysis.png | 核心证据 | 实验结果 |
| 5 | hurst_comparison.png | 辅助证据 | 实验结果 |
| 6 | price_timeseries.png | 直观展示 | 实验设置/结果 |
| 7 | personality_behavior.png | 模型验证 | 模型描述 |
| 8 | herding_heatmap.png | 动态展示 | 结果分析 |
| 9 | portfolio_performance.png | 辅助分析 | 附录 |
| 10 | metrics_summary.png | 汇总对比 | 实验结果 |
| 11 | statistical_tests.png | 统计检验 | 实验结果 |
| 12 | network_topology.png | 模型描述 | 实验设置 |
| 13 | stylized_facts_analysis.png | 模型验证 | 实验结果 |
| 14 | real_market_comparison.png | 模型验证 | 实验结果 |

---

## 详细说明

### 1. return_distributions.png

**收益率分布对比图**

- **内容**: 四个实验条件下收益率的概率密度分布
- **关键发现**:
  - baseline/memory_only 呈现接近正态的分布
  - social_only/full 呈现明显的尖峰厚尾特征
- **统计数据**:
  - baseline 峰度: 0.81
  - social_only 峰度: 5.51 (增加580%)
- **论文引用建议**:
  > 如图X所示，引入社交网络后，收益率分布呈现显著的尖峰厚尾特征，峰度从基准条件的0.81上升至5.51，增幅达580%。

---

### 2. factor_decomposition.png

**2×2因子效应分解图**

- **内容**:
  - 左图: 四个条件的峰度对比（带误差棒）
  - 右图: 因子边际效应分解
- **关键发现**:
  - 社交网络边际效应: +4.70
  - 记忆机制边际效应: +0.25
  - 交互效应: +0.11
- **论文引用建议**:
  > 因子分解分析表明（图X），社交网络对峰度的边际贡献为+4.70，而记忆机制仅贡献+0.25，说明社交网络是导致肥尾效应的主要因素。

---

### 3. social_consensus_effect.png

**社交共识与跟风行为关系图**

- **内容**:
  - 左图: 社交共识强度的分布
  - 右图: 不同共识强度下Agent的跟风率
- **关键发现**:
  - 当社交共识>70%时，Agent跟风率显著上升
  - 证明了信息级联的微观机制
- **论文引用建议**:
  > 图X展示了社交共识强度与Agent跟风行为的关系。当网络中超过70%的邻居采取相同行动时，Agent的跟风概率显著提升，形成正反馈循环。

---

### 4. herding_analysis.png

**羊群效应分析图**

- **内容**: 四个条件下的羊群行为比例对比
- **关键发现**:
  - baseline: 74.7%
  - social_only: 84.6% (+10个百分点)
  - full: 86.7% (+12个百分点)
- **论文引用建议**:
  > 羊群效应分析（图X）显示，社交网络使市场的羊群行为比例从74.7%上升至84.6%，增加了约10个百分点。

---

### 5. hurst_comparison.png

**Hurst指数对比图**

- **内容**: 四个条件下的Hurst指数分布
- **关键发现**:
  - baseline: 0.92 (强趋势性)
  - social_only: 0.83 (更接近随机游走)
  - Hurst=0.5表示随机游走
- **论文引用建议**:
  > Hurst指数分析（图X）表明，社交网络的引入使市场的Hurst指数从0.92下降至0.83，市场行为更接近随机游走。

---

### 6. price_timeseries.png

**价格时间序列对比图**

- **内容**: 单次运行中四个条件的价格走势
- **关键发现**:
  - baseline/memory_only: 价格稳步上涨
  - social_only/full: 价格波动更大，有明显的涨跌周期
- **论文引用建议**:
  > 图X展示了典型运行中四个条件下的价格演化过程。可以观察到，引入社交网络后，价格呈现更剧烈的波动模式。

---

### 7. personality_behavior.png

**人格类型行为差异图**

- **内容**: 四个条件下不同人格类型的交易行为分布（BUY/HOLD/SELL）
- **关键发现**:
  - Conservative（保守型）: 更倾向HOLD
  - Aggressive（激进型）: BUY/SELL比例更高
  - 社交网络条件下，行为趋于同质化
- **论文引用建议**:
  > 图X展示了不同人格类型Agent的行为特征。在引入社交网络后，原本异质的交易风格趋于同质化，这是羊群效应的微观表现。

---

### 8. herding_heatmap.png

**决策同质性热力图**

- **内容**: 每轮交易中买入比例的时间演化（颜色越绿=买入越多）
- **关键发现**:
  - baseline: 颜色分布相对均匀
  - social_only/full: 出现明显的绿色/红色条带，表示集体买入/卖出
- **论文引用建议**:
  > 热力图（图X）直观展示了羊群效应的时间演化。在社交网络条件下，可观察到明显的"颜色条带"，表明Agent在特定时刻集体采取相同行动。

---

### 9. portfolio_performance.png

**投资组合绩效图**

- **内容**: 不同人格类型在各条件下的最终资产价值
- **关键发现**:
  - baseline: Aggressive +7.02%, Conservative +3.33%
  - full: Aggressive +0.08%, Conservative -0.53%
  - 社交网络降低了整体收益
- **论文引用建议**:
  > 投资组合分析（图X）显示，社交网络的引入虽然增加了市场的肥尾特征，但降低了Agent的整体投资回报，这可能源于羊群行为导致的集体非理性。

---

### 10. metrics_summary.png

**关键指标汇总对比图**

- **内容**: 四个条件下Kurtosis、Herding、Hurst、Volatility的分组对比
- **用途**: 提供所有关键指标的全景视图
- **论文引用建议**:
  > 图X汇总了四个实验条件下的关键指标。可以清晰看到，社交网络条件（social_only和full）在峰度和羊群比例上显著高于基准条件。

---

### 11. statistical_tests.png

**统计显著性检验表格**

- **内容**: Welch's t检验结果，比较各实验条件间的指标差异
- **检验指标**:
  - Kurtosis（峰度）
  - Herding Ratio（羊群比例）
  - Hurst Exponent（Hurst指数）
  - Volatility（波动率）
- **比较组合**:
  - baseline vs social_only（社交网络效应）
  - baseline vs memory_only（记忆效应）
  - baseline vs full（完整模型效应）
  - social_only vs full（记忆在社交条件下的增量效应）
- **显著性标记**: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著
- **配套文件**: `statistical_tests.tex`（LaTeX格式，可直接插入论文）
- **论文引用建议**:
  > 表X展示了各实验条件间关键指标的统计检验结果。Welch's t检验表明，社交网络对峰度的影响在p<0.001水平上显著，而记忆机制的边际效应未达到统计显著性。

---

### 12. network_topology.png

**社交网络拓扑结构图**

- **内容**:
  - 左图: BA无标度网络的可视化，节点大小和颜色按度数编码
  - 右图: 网络度分布直方图及统计信息
- **网络参数**:
  - 节点数: 30
  - 新节点连接数 m=3
  - 网络类型: Barabási-Albert无标度网络
- **关键特征**:
  - 存在少数高度数Hub节点（意见领袖）
  - 度分布呈幂律分布（无标度特性）
  - 聚类系数和网络密度信息
- **论文引用建议**:
  > 图X展示了实验中使用的社交网络拓扑结构。采用Barabási-Albert模型生成的无标度网络具有少数高连接度的"意见领袖"节点，这种结构有利于信息级联的形成和传播。

---

### 13. stylized_facts_analysis.png

**Stylized Facts 金融典型事实验证图**

- **内容**:
  - 收益率分布与正态分布对比
  - 波动率聚集性（绝对收益率自相关函数 ACF）
  - R/S 分析法计算的 Hurst 指数
  - 验证指标汇总表
- **验证的三个典型事实**:
  - **厚尾特性 (Fat Tails)**: 峰度 > 3，收益率分布尾部比正态分布更厚
  - **波动聚集 (Volatility Clustering)**: 大波动后跟随大波动，ACF 显著不为零
  - **长记忆性 (Long Memory)**: Hurst 指数 H > 0.5 表示趋势持续性
- **关键发现**:
  - 仿真数据成功复现真实市场的厚尾特性
  - 波动聚集性在社交网络条件下更明显
- **论文引用建议**:
  > 图X展示了 Stylized Facts 验证结果。仿真数据的收益率分布呈现显著的厚尾特性（峰度=X.XX），波动率自相关在多个滞后阶上显著，Hurst指数为X.XX，表明市场具有趋势持续性。这些特征与真实金融市场的实证发现一致（Cont, 2001）。

---

### 14. real_market_comparison.png

**真实市场对比分析图**

- **内容**:
  - 仿真价格与真实市场（SPY）价格走势对比（归一化）
  - 收益率分布对比（直方图 + 核密度估计）
  - Q-Q 图对比
  - 统计指标对比表格
- **对比维度**:
  - 基本统计: 均值、标准差、偏度、峰度
  - 风险指标: VaR (5%)、最大回撤
  - 分布检验: Kolmogorov-Smirnov 检验
- **相似度评分体系**:
  - 波动率相似度 (30%)
  - 峰度相似度 (25%)
  - 偏度相似度 (20%)
  - VaR 相似度 (25%)
- **关键发现**:
  - 相似度总分与评级（A/B/C/D）
  - 仿真数据与真实市场数据的统计特征对比
- **论文引用建议**:
  > 图X展示了仿真数据与真实市场（SPY）的对比分析。仿真数据的波动率为X.XX%，与SPY的X.XX%相近；峰度为X.XX，略高于SPY的X.XX。K-S检验p值为X.XX，综合相似度评分为XX分（评级X），表明仿真模型具有较好的真实市场拟合能力。

---

## 推荐的论文图表组合

### 方案A：精简版（4张图）

适用于篇幅受限的情况：

1. **factor_decomposition.png** - 核心结论：因子贡献分解
2. **return_distributions.png** - 直接证据：肥尾分布
3. **social_consensus_effect.png** - 机制解释：跟风行为
4. **herding_analysis.png** - 中间变量：羊群效应

### 方案B：完整版（8张图）

在方案A基础上增加：

5. **statistical_tests.png** - 统计检验支撑
6. **network_topology.png** - 模型架构说明
7. **price_timeseries.png** - 直观展示价格动态
8. **stylized_facts_analysis.png** - 金融典型事实验证

### 方案C：附录补充

可放入附录的图表：

- **hurst_comparison.png** - 市场效率分析
- **portfolio_performance.png** - 投资绩效分析
- **herding_heatmap.png** - 动态可视化
- **metrics_summary.png** - 指标汇总
- **personality_behavior.png** - 模型异质性验证
- **real_market_comparison.png** - 真实市场对比

---

## 数据来源

所有图表基于以下数据文件生成：

- `results/custom_30a_100r_results.json` - 聚合统计数据
- `results/custom_30a_100r_decisions.parquet` - 原始决策记录（180,000条）

**生成命令**:

```bash
# 方式1: 通过 main.py
poetry run python main.py --mode figures

# 方式2: 直接运行模块
poetry run python -m src.generate_figures

# 指定结果目录
poetry run python -m src.generate_figures --results results/custom_30a_100r

# 只生成特定类型图表
poetry run python -m src.generate_figures --only extra        # 额外分析图
poetry run python -m src.generate_figures --only supplementary # 补充图（统计表+网络图）
```

**生成脚本**: `src/generate_figures.py`
