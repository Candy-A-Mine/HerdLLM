"""
实验配置模块 (Experiment Configuration Module)

本模块定义了ABM仿真实验的各种配置类和预设配置函数。
通过灵活的配置系统，支持以下实验场景：

1. 对照实验：比较不同实验条件（记忆、社交网络）的影响
2. 参数敏感性分析：研究关键参数对结果的影响
3. 蒙特卡洛模拟：多次运行以获取统计稳健的结果

实验条件说明：
    为了研究记忆和社交网络对市场行为的影响，本模块定义了四种实验条件：

    1. BASELINE（基线）：
       - 记忆模块: 禁用
       - 社交网络: 禁用
       - 用途: 作为对照组，Agent仅根据当前信息决策

    2. MEMORY_ONLY（仅记忆）：
       - 记忆模块: 启用
       - 社交网络: 禁用
       - 用途: 研究记忆对个体决策的影响

    3. SOCIAL_ONLY（仅社交）：
       - 记忆模块: 禁用
       - 社交网络: 启用
       - 用途: 研究社交网络对羊群效应的影响

    4. FULL（完整）：
       - 记忆模块: 启用
       - 社交网络: 启用
       - 用途: 研究两种机制的协同效应

类结构：
    ExperimentCondition: 实验条件枚举
    SimulationConfig: 单次仿真配置
    ExperimentConfig: 多条件实验配置
    SensitivityConfig: 参数敏感性分析配置

预设配置函数：
    get_quick_test_config(): 快速测试配置
    get_standard_config(): 标准实验配置
    get_large_scale_config(): 大规模实验配置
    get_herding_focus_config(): 羊群行为研究配置

作者: SuZX
日期: 2024
"""

# =============================================================================
# 导入依赖
# =============================================================================

from dataclasses import dataclass, field  # 数据类装饰器
from enum import Enum                      # 枚举类型
from typing import Any, Dict, List         # 类型提示


# =============================================================================
# 实验条件枚举
# =============================================================================

class ExperimentCondition(Enum):
    """
    实验条件枚举类

    该枚举定义了ABM仿真实验中的四种对照条件，
    用于系统性地研究记忆和社交网络对市场行为的影响。

    通过组合"记忆启用/禁用"和"社交网络启用/禁用"，
    形成2x2的实验设计矩阵。

    Attributes:
        BASELINE: 基线条件（无记忆，无社交网络）
        MEMORY_ONLY: 仅启用记忆模块
        SOCIAL_ONLY: 仅启用社交网络
        FULL: 同时启用记忆和社交网络

    Example:
        >>> condition = ExperimentCondition.FULL
        >>> print(condition.value)  # 输出: 'full'
    """

    # 基线条件：Agent不使用历史记忆，也不受社交网络影响
    # 作为对照组，用于评估其他条件的影响
    BASELINE = "baseline"

    # 仅记忆条件：Agent能从历史经验中学习，但不受同伴影响
    # 用于研究个体学习对决策的影响
    MEMORY_ONLY = "memory_only"

    # 仅社交条件：Agent不使用历史记忆，但受社交网络中同伴的影响
    # 用于研究羊群效应的纯粹影响
    SOCIAL_ONLY = "social_only"

    # 完整条件：Agent同时具有记忆能力和社交网络连接
    # 最接近真实市场中交易者的行为模式
    FULL = "full"


# =============================================================================
# 单次仿真配置类
# =============================================================================

@dataclass
class SimulationConfig:
    """
    单次仿真配置类

    该数据类包含运行一次ABM仿真所需的所有参数配置。
    参数分为以下几类：

    1. 基础参数：Agent数量、仿真回合数、初始价格等
    2. 市场参数：价格影响系数等
    3. Agent分布：不同人格类型的Agent数量
    4. 功能开关：记忆、社交网络的启用/禁用
    5. LLM参数：模型连接配置

    Attributes:
        num_agents: Agent总数，决定市场参与者数量
        num_rounds: 仿真回合数，即交易时段数
        initial_price: 资产初始价格
        initial_cash: 每个Agent的初始现金
        initial_holdings: 每个Agent的初始持股数
        trade_size: 每次交易的最大股数
        price_impact: 价格影响参数λ
        personality_distribution: 各人格类型的Agent数量分布
        enable_memory: 是否启用记忆模块
        enable_social_network: 是否启用社交网络
        memory_capacity: 记忆容量（最大记忆条数）
        social_network_m: BA网络模型的m参数
        llm_base_url: LLM服务地址
        llm_api_key: LLM API密钥
        llm_model: LLM模型名称
        llm_temperature: LLM采样温度
        seed: 随机种子（None表示随机）

    Example:
        >>> config = SimulationConfig(
        ...     num_agents=50,
        ...     num_rounds=100,
        ...     enable_memory=True,
        ...     enable_social_network=True,
        ... )
        >>> print(f"Agent数: {config.num_agents}, 回合数: {config.num_rounds}")
    """

    # -------------------------------------------------------------------------
    # 基础参数
    # -------------------------------------------------------------------------

    # Agent总数
    # 影响市场规模和订单不平衡的计算
    # 推荐值：20-200
    num_agents: int = 50

    # 仿真回合数
    # 每个回合所有Agent做一次决策
    # 推荐值：50-500
    num_rounds: int = 100

    # 资产初始价格
    # 选择100便于计算收益率百分比
    initial_price: float = 100.0

    # 每个Agent的初始现金
    # 决定Agent的购买力
    initial_cash: float = 10000.0

    # 每个Agent的初始持股数
    # 决定Agent的卖出能力
    initial_holdings: int = 50

    # 每次交易的最大股数
    # 控制单次交易对市场的影响
    trade_size: int = 10

    # -------------------------------------------------------------------------
    # 市场参数
    # -------------------------------------------------------------------------

    # 价格影响参数 (λ)
    # 公式: P_{t+1} = P_t × (1 + λ × OrderImbalance)
    # - 较大的值(如0.05)表示市场流动性差，价格波动大，更容易产生厚尾
    # - 较小的值(如0.01)表示市场流动性好，价格波动小
    price_impact: float = 0.02

    # -------------------------------------------------------------------------
    # Agent人格分布
    # -------------------------------------------------------------------------

    # 各人格类型的Agent数量
    # 总数应接近num_agents，不足部分随机补充
    personality_distribution: Dict[str, int] = field(
        default_factory=lambda: {
            "Conservative": 15,     # 保守型：风险厌恶
            "Aggressive": 15,       # 激进型：风险偏好
            "Trend_Follower": 10,   # 趋势跟随型：追涨杀跌
            "Herding": 10,          # 羊群型：跟随同伴
        }
    )

    # -------------------------------------------------------------------------
    # 功能开关
    # -------------------------------------------------------------------------

    # 是否启用记忆模块
    # 启用后Agent能从历史决策中学习
    enable_memory: bool = True

    # 是否启用社交网络
    # 启用后Agent能感知同伴行为
    enable_social_network: bool = True

    # 记忆容量
    # 每个Agent最多记住的历史决策数
    memory_capacity: int = 20

    # -------------------------------------------------------------------------
    # 社交网络参数
    # -------------------------------------------------------------------------

    # BA模型的m参数
    # 每个新加入的节点连接到m个现有节点
    # 值越大，网络越密集
    social_network_m: int = 3

    # -------------------------------------------------------------------------
    # LLM参数
    # -------------------------------------------------------------------------

    # LLM服务的基础URL
    # 默认指向本地Ollama服务
    llm_base_url: str = "http://localhost:11434/v1"

    # LLM API密钥
    # Ollama不需要真实密钥，使用占位符即可
    llm_api_key: str = "ollama"

    # LLM模型名称
    # 必须是Ollama中已安装的模型
    # 推荐: qwen2.5:3b (8GB显存) 或 qwen2.5:1.5b (更快)
    llm_model: str = "qwen2.5:3b"

    # LLM采样温度
    # 控制输出的随机性，0表示确定性，1表示高随机性
    llm_temperature: float = 0.7

    # -------------------------------------------------------------------------
    # 随机种子
    # -------------------------------------------------------------------------

    # 随机种子
    # 设置后可确保实验可重复
    # None表示每次运行使用不同的随机种子
    seed: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典格式

        该方法用于序列化配置，便于保存到JSON文件或日志记录。

        Returns:
            包含所有配置参数的字典

        Note:
            不包含LLM连接参数（敏感信息）
        """
        return {
            # 基础参数
            "num_agents": self.num_agents,
            "num_rounds": self.num_rounds,
            "initial_price": self.initial_price,
            "initial_cash": self.initial_cash,
            "initial_holdings": self.initial_holdings,
            "trade_size": self.trade_size,

            # 市场参数
            "price_impact": self.price_impact,

            # Agent分布
            "personality_distribution": self.personality_distribution,

            # 功能开关
            "enable_memory": self.enable_memory,
            "enable_social_network": self.enable_social_network,
            "memory_capacity": self.memory_capacity,

            # 网络参数
            "social_network_m": self.social_network_m,

            # 随机种子
            "seed": self.seed,
        }


# =============================================================================
# 实验配置类
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    多条件实验配置类

    该类用于配置一个完整的对照实验，包括：
    - 多个实验条件的定义
    - 每个条件的蒙特卡洛运行次数
    - 基础仿真配置
    - 输出目录设置

    实验设计说明：
        为了获得统计稳健的结果，每个实验条件需要进行
        多次蒙特卡洛模拟。推荐的运行次数为30-50次，
        以便进行假设检验和置信区间估计。

    Attributes:
        name: 实验名称，用于标识输出文件
        description: 实验描述文本
        num_runs: 每个条件的蒙特卡洛运行次数
        base_config: 基础仿真配置（将根据条件修改）
        conditions: 要运行的实验条件列表
        output_dir: 结果输出目录

    Example:
        >>> exp_config = ExperimentConfig(
        ...     name="herding_study",
        ...     num_runs=30,
        ...     conditions=[ExperimentCondition.BASELINE, ExperimentCondition.FULL],
        ... )
        >>> for condition in exp_config.conditions:
        ...     sim_config = exp_config.get_condition_config(condition)
        ...     print(f"{condition.value}: memory={sim_config.enable_memory}")
    """

    # -------------------------------------------------------------------------
    # 实验元数据
    # -------------------------------------------------------------------------

    # 实验名称
    # 用于命名输出文件和目录
    name: str = "default_experiment"

    # 实验描述
    # 记录实验目的和设置说明
    description: str = ""

    # -------------------------------------------------------------------------
    # 运行参数
    # -------------------------------------------------------------------------

    # 每个条件的蒙特卡洛运行次数
    # 推荐值：30-50次（满足中心极限定理）
    num_runs: int = 30

    # -------------------------------------------------------------------------
    # 基础配置
    # -------------------------------------------------------------------------

    # 基础仿真配置
    # 该配置会根据每个实验条件进行修改
    base_config: SimulationConfig = field(default_factory=SimulationConfig)

    # -------------------------------------------------------------------------
    # 实验条件
    # -------------------------------------------------------------------------

    # 要运行的实验条件列表
    # 默认运行全部四种条件
    conditions: List[ExperimentCondition] = field(
        default_factory=lambda: [
            ExperimentCondition.BASELINE,
            ExperimentCondition.MEMORY_ONLY,
            ExperimentCondition.SOCIAL_ONLY,
            ExperimentCondition.FULL,
        ]
    )

    # -------------------------------------------------------------------------
    # 输出设置
    # -------------------------------------------------------------------------

    # 结果输出目录
    output_dir: str = "results"

    def get_condition_config(
        self,
        condition: ExperimentCondition
    ) -> SimulationConfig:
        """
        获取特定实验条件的仿真配置

        该方法根据指定的实验条件，创建一个修改后的
        SimulationConfig对象。主要修改enable_memory
        和enable_social_network两个参数。

        Args:
            condition: 实验条件枚举值

        Returns:
            针对该条件的SimulationConfig对象

        Example:
            >>> config = exp_config.get_condition_config(ExperimentCondition.MEMORY_ONLY)
            >>> print(config.enable_memory)  # True
            >>> print(config.enable_social_network)  # False
        """
        # 复制基础配置的所有参数
        config = SimulationConfig(
            # 基础参数
            num_agents=self.base_config.num_agents,
            num_rounds=self.base_config.num_rounds,
            initial_price=self.base_config.initial_price,
            initial_cash=self.base_config.initial_cash,
            initial_holdings=self.base_config.initial_holdings,
            trade_size=self.base_config.trade_size,

            # 市场参数
            price_impact=self.base_config.price_impact,

            # Agent分布（需要复制字典以避免引用问题）
            personality_distribution=self.base_config.personality_distribution.copy(),

            # 功能参数
            memory_capacity=self.base_config.memory_capacity,
            social_network_m=self.base_config.social_network_m,

            # LLM参数
            llm_base_url=self.base_config.llm_base_url,
            llm_api_key=self.base_config.llm_api_key,
            llm_model=self.base_config.llm_model,
            llm_temperature=self.base_config.llm_temperature,

            # 随机种子
            seed=self.base_config.seed,
        )

        # 根据实验条件设置功能开关
        if condition == ExperimentCondition.BASELINE:
            # 基线条件：禁用所有增强功能
            config.enable_memory = False
            config.enable_social_network = False

        elif condition == ExperimentCondition.MEMORY_ONLY:
            # 仅记忆条件：只启用记忆模块
            config.enable_memory = True
            config.enable_social_network = False

        elif condition == ExperimentCondition.SOCIAL_ONLY:
            # 仅社交条件：只启用社交网络
            config.enable_memory = False
            config.enable_social_network = True

        elif condition == ExperimentCondition.FULL:
            # 完整条件：启用所有功能
            config.enable_memory = True
            config.enable_social_network = True

        return config


# =============================================================================
# 敏感性分析配置类
# =============================================================================

@dataclass
class SensitivityConfig:
    """
    参数敏感性分析配置类

    该类用于配置参数敏感性分析实验，研究关键参数
    变化对仿真结果的影响。

    敏感性分析的目的：
        1. 识别对结果影响最大的关键参数
        2. 确定参数的合理取值范围
        3. 验证模型对参数变化的稳健性

    可分析的参数：
        - price_impact: 价格影响系数
        - num_agents: Agent数量
        - social_network_m: 社交网络密度
        - memory_capacity: 记忆容量

    Attributes:
        name: 分析任务名称
        base_config: 基础仿真配置
        num_runs: 每个参数值的运行次数
        price_impact_values: 价格影响系数的测试值列表
        num_agents_values: Agent数量的测试值列表
        social_network_m_values: 网络m参数的测试值列表
        memory_capacity_values: 记忆容量的测试值列表
        output_dir: 输出目录

    Example:
        >>> sensitivity = SensitivityConfig(
        ...     num_runs=10,
        ...     price_impact_values=[0.01, 0.02, 0.04],
        ... )
    """

    # 分析任务名称
    name: str = "sensitivity_analysis"

    # 基础仿真配置
    base_config: SimulationConfig = field(default_factory=SimulationConfig)

    # 每个参数值的蒙特卡洛运行次数
    # 由于敏感性分析需要测试多个参数值，
    # 每个值的运行次数可以适当减少
    num_runs: int = 10

    # -------------------------------------------------------------------------
    # 待测试的参数值列表
    # -------------------------------------------------------------------------

    # 价格影响系数的测试值
    # 范围：从低流动性到高流动性
    price_impact_values: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.02, 0.04, 0.08]
    )

    # Agent数量的测试值
    # 研究市场规模对结果的影响
    num_agents_values: List[int] = field(
        default_factory=lambda: [20, 50, 100, 200]
    )

    # 社交网络m参数的测试值
    # 研究网络密度对羊群效应的影响
    social_network_m_values: List[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 8]
    )

    # 记忆容量的测试值
    # 研究记忆深度对学习效果的影响
    memory_capacity_values: List[int] = field(
        default_factory=lambda: [5, 10, 20, 50]
    )

    # 输出目录
    output_dir: str = "results/sensitivity"


# =============================================================================
# 预设配置函数
# =============================================================================


def get_quick_test_config() -> ExperimentConfig:
    """
    获取快速测试配置

    该配置用于快速验证代码是否正常运行，
    使用较少的Agent、回合数和运行次数。

    配置特点：
        - Agent数量: 20
        - 回合数: 15
        - 运行次数: 3
        - 条件: 仅基线和完整条件
        - 固定随机种子: 42
        - price_impact: 0.05（便于观察厚尾效应）

    Returns:
        快速测试的ExperimentConfig对象

    Example:
        >>> config = get_quick_test_config()
        >>> print(config.num_runs)  # 输出: 3
    """
    # 创建精简的基础配置
    base = SimulationConfig(
        num_agents=20,      # 少量Agent
        num_rounds=15,      # 少量回合
        price_impact=0.05,  # 高价格影响，便于观察厚尾
        seed=42,            # 固定种子以确保可重复
    )

    return ExperimentConfig(
        name="quick_test",
        description="快速测试配置，用于调试和验证代码",
        num_runs=3,         # 少量运行
        base_config=base,
        # 只测试两种极端条件
        conditions=[
            ExperimentCondition.BASELINE,
            ExperimentCondition.FULL
        ],
    )


def get_standard_config() -> ExperimentConfig:
    """
    获取标准实验配置

    该配置用于正式的学术实验，参数设置符合
    统计学要求（足够的样本量和运行次数）。

    配置特点：
        - Agent数量: 50
        - 回合数: 100
        - 运行次数: 30
        - 条件: 全部四种条件

    Returns:
        标准实验的ExperimentConfig对象

    Note:
        30次运行是满足中心极限定理的最小样本量
    """
    base = SimulationConfig(
        num_agents=50,
        num_rounds=100,
    )

    return ExperimentConfig(
        name="standard_experiment",
        description="标准4条件对照实验，30次蒙特卡洛运行",
        num_runs=30,
        base_config=base,
    )


def get_large_scale_config() -> ExperimentConfig:
    """
    获取大规模实验配置

    该配置用于需要高统计精度的重要实验，
    使用更多的Agent、回合数和运行次数。

    配置特点：
        - Agent数量: 100
        - 回合数: 200
        - 运行次数: 50

    Returns:
        大规模实验的ExperimentConfig对象

    Warning:
        该配置运行时间较长，请确保有足够的计算资源
    """
    base = SimulationConfig(
        num_agents=100,
        num_rounds=200,
    )

    return ExperimentConfig(
        name="large_scale_experiment",
        description="大规模实验，100个Agent，200回合，50次运行",
        num_runs=50,
        base_config=base,
    )


def get_herding_focus_config() -> ExperimentConfig:
    """
    获取羊群行为研究配置

    该配置专门用于研究羊群效应，通过增加
    羊群型Agent的比例和社交网络密度来
    放大羊群效应的影响。

    配置特点：
        - 羊群型Agent占比: 40%（20/50）
        - 社交网络m参数: 5（更密集的网络）
        - 其他Agent平均分配

    Returns:
        羊群行为研究的ExperimentConfig对象

    Research Focus:
        该配置适合研究：
        - 羊群效应对价格波动的影响
        - 信息级联的形成条件
        - 社交网络结构与羊群行为的关系
    """
    base = SimulationConfig(
        num_agents=50,
        num_rounds=100,
        # 调整人格分布，增加羊群型Agent
        personality_distribution={
            "Conservative": 10,     # 减少保守型
            "Aggressive": 10,       # 减少激进型
            "Trend_Follower": 10,   # 保持趋势跟随型
            "Herding": 20,          # 增加羊群型（40%）
        },
        # 增加网络密度以加强社交影响
        social_network_m=5,
    )

    return ExperimentConfig(
        name="herding_focus",
        description="羊群行为研究配置，增加羊群型Agent比例和网络密度",
        num_runs=30,
        base_config=base,
    )
