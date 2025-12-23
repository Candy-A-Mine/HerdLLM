"""
市场模块 (Market Module)

本模块实现了一个简化的金融市场模型，是ABM仿真系统的核心组件之一。
市场负责接收所有Agent的订单，根据订单不平衡计算价格变化，
并维护完整的市场历史数据。

价格动态机制：
    本模块使用一个简化但经济学上合理的价格影响公式：
    P_{t+1} = P_t × (1 + λ × (BuyOrders - SellOrders) / TotalAgents)

    其中：
    - P_t: 当前价格
    - P_{t+1}: 下一期价格
    - λ (lambda/price_impact): 价格敏感度参数，控制订单不平衡对价格的影响程度
    - BuyOrders: 买入订单数量
    - SellOrders: 卖出订单数量
    - TotalAgents: Agent总数（用于标准化）

    这个公式的经济学含义：
    - 当买入 > 卖出时，需求大于供给，价格上涨
    - 当卖出 > 买入时，供给大于需求，价格下跌
    - λ参数越大，市场对订单不平衡越敏感

市场功能：
    1. 订单处理：接收并处理所有Agent的订单
    2. 价格形成：根据订单不平衡计算新价格
    3. 历史记录：维护价格、成交量等历史数据
    4. 统计分析：计算收益率、波动率等市场指标

数据结构：
    Market类使用以下列表记录历史数据：
    - price_history: 价格历史
    - volume_history: 成交量历史
    - buy_history: 买入订单数历史
    - sell_history: 卖出订单数历史

作者: SuZX
日期: 2024
"""

# =============================================================================
# 导入依赖
# =============================================================================

from dataclasses import dataclass, field  # 数据类装饰器
from typing import List, Dict             # 类型提示

# 项目内部模块
from src.agent import Order               # 订单数据类


# =============================================================================
# 市场类
# =============================================================================

@dataclass
class Market:
    """
    金融市场模拟类

    该类模拟一个简化的金融市场，实现以下功能：
    1. 接收所有Agent的交易订单
    2. 根据订单不平衡计算价格变化
    3. 维护市场历史数据
    4. 提供市场统计信息

    价格形成机制说明：
        本模型使用线性价格影响函数，这是市场微观结构理论中的
        常用简化假设。公式为：

        P_{t+1} = P_t × (1 + λ × OrderImbalance)

        其中 OrderImbalance = (BuyOrders - SellOrders) / TotalAgents

        该公式的特点：
        - 订单不平衡被Agent总数标准化，使得影响程度与市场规模无关
        - λ参数控制市场的流动性：λ越小，流动性越好
        - 价格变化是乘法形式，保证价格始终为正（除非订单严重失衡）

    Attributes:
        initial_price: 资产的初始价格，默认100.0
        price_impact: 价格影响参数λ，控制市场对订单不平衡的敏感度，默认0.02
        num_agents: Agent总数，用于计算订单不平衡比例
        current_price: 当前市场价格（运行时更新）
        price_history: 价格历史列表，包含每个回合的收盘价
        volume_history: 成交量历史列表
        buy_history: 每回合买入订单数历史
        sell_history: 每回合卖出订单数历史

    Example:
        >>> market = Market(initial_price=100.0, price_impact=0.02, num_agents=20)
        >>> orders = [Order(agent_id=0, action="BUY", quantity=10, reason="看好")]
        >>> result = market.process_orders(orders)
        >>> print(f"新价格: ${market.current_price:.2f}")

    Note:
        - 价格有下限保护（0.01），防止价格变为负数或零
        - 所有历史数据在process_orders()调用后自动更新
    """

    # -------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------

    # 资产初始价格
    # 选择100.0作为默认值便于计算收益率百分比
    initial_price: float = 100.0

    # 价格影响参数 (λ)
    # 该参数决定了市场对订单不平衡的敏感程度
    # - 较大的值(如0.05)表示市场流动性较差，价格波动大
    # - 较小的值(如0.01)表示市场流动性较好，价格波动小
    # 默认值0.02是一个适中的选择
    price_impact: float = 0.02

    # Agent总数
    # 用于标准化订单不平衡，确保价格影响与市场规模无关
    num_agents: int = 20

    # -------------------------------------------------------------------------
    # 运行时状态（延迟初始化）
    # -------------------------------------------------------------------------

    # 当前市场价格
    # 在__post_init__中初始化为initial_price
    current_price: float = field(init=False)

    # 价格历史记录
    # 包含初始价格及每个回合后的价格
    # 长度 = 回合数 + 1
    price_history: List[float] = field(default_factory=list)

    # 成交量历史记录
    # 记录每个回合的总成交量（买入 + 卖出的股数）
    volume_history: List[int] = field(default_factory=list)

    # 买入订单数历史
    # 记录每个回合的买入订单数量（非成交量）
    buy_history: List[int] = field(default_factory=list)

    # 卖出订单数历史
    # 记录每个回合的卖出订单数量（非成交量）
    sell_history: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        构造后初始化

        在dataclass创建实例后自动调用，用于：
        1. 设置初始价格
        2. 将初始价格添加到历史记录

        Note:
            使用__post_init__而非__init__是dataclass的标准做法，
            允许在所有字段初始化后执行额外的设置逻辑。
        """
        # 设置当前价格为初始价格
        self.current_price = self.initial_price

        # 将初始价格作为第一个历史点
        # 这样price_history[0]始终是初始价格
        self.price_history.append(self.current_price)

    def process_orders(self, orders: List[Order]) -> Dict:
        """
        处理所有订单并更新市场价格

        这是Market类的核心方法，完成以下步骤：
        1. 统计买入、卖出、持有订单的数量
        2. 计算总成交量
        3. 根据订单不平衡计算新价格
        4. 更新历史记录
        5. 返回本回合的市场统计

        价格计算公式：
            order_imbalance = (buy_orders - sell_orders) / num_agents
            price_change_factor = 1 + price_impact × order_imbalance
            new_price = current_price × price_change_factor

        Args:
            orders: Order对象列表，包含所有Agent的订单

        Returns:
            包含以下键的字典：
            - buy_orders: 买入订单数
            - sell_orders: 卖出订单数
            - hold_orders: 持有订单数
            - total_volume: 总成交量（股数）
            - order_imbalance: 订单不平衡度 (-1到1之间)
            - price_change_pct: 价格变化百分比
            - new_price: 更新后的价格

        Example:
            >>> orders = [
            ...     Order(0, "BUY", 10, "买入"),
            ...     Order(1, "SELL", 5, "卖出"),
            ...     Order(2, "HOLD", 0, "观望"),
            ... ]
            >>> result = market.process_orders(orders)
            >>> print(f"买入: {result['buy_orders']}, 卖出: {result['sell_orders']}")
        """
        # ---------------------------------------------------------------------
        # 步骤1: 统计订单分布
        # ---------------------------------------------------------------------

        # 统计有效的买入订单数（动作为BUY且数量>0）
        buy_orders = sum(
            1 for order in orders
            if order.action == "BUY" and order.quantity > 0
        )

        # 统计有效的卖出订单数（动作为SELL且数量>0）
        sell_orders = sum(
            1 for order in orders
            if order.action == "SELL" and order.quantity > 0
        )

        # 持有订单数 = 总订单数 - 买入 - 卖出
        hold_orders = len(orders) - buy_orders - sell_orders

        # ---------------------------------------------------------------------
        # 步骤2: 计算成交量
        # ---------------------------------------------------------------------

        # 总成交量 = 所有买入和卖出订单的股数之和
        # HOLD订单的quantity为0，不计入成交量
        total_volume = sum(
            order.quantity for order in orders
            if order.action in ("BUY", "SELL")
        )

        # ---------------------------------------------------------------------
        # 步骤3: 计算价格变化
        # ---------------------------------------------------------------------

        # 计算订单不平衡度
        # 范围: [-1, 1]
        # - 正值表示买方压力大
        # - 负值表示卖方压力大
        # - 0表示买卖平衡
        order_imbalance = (buy_orders - sell_orders) / self.num_agents

        # 计算价格变化因子
        # 当order_imbalance为正时，factor > 1，价格上涨
        # 当order_imbalance为负时，factor < 1，价格下跌
        price_change_factor = 1 + self.price_impact * order_imbalance

        # 更新当前价格
        self.current_price = self.current_price * price_change_factor

        # ---------------------------------------------------------------------
        # 步骤4: 价格下限保护
        # ---------------------------------------------------------------------

        # 确保价格不会变为负数或接近零
        # 这是一个技术保护措施，在极端市场条件下防止模型崩溃
        # 最小价格设为0.01，这在实际市场中也是合理的（许多市场有最小价格变动单位）
        self.current_price = max(self.current_price, 0.01)

        # ---------------------------------------------------------------------
        # 步骤5: 记录历史数据
        # ---------------------------------------------------------------------

        # 将新价格添加到价格历史
        self.price_history.append(self.current_price)

        # 记录成交量
        self.volume_history.append(total_volume)

        # 记录买卖订单数
        self.buy_history.append(buy_orders)
        self.sell_history.append(sell_orders)

        # ---------------------------------------------------------------------
        # 步骤6: 返回回合统计
        # ---------------------------------------------------------------------

        return {
            "buy_orders": buy_orders,           # 买入订单数
            "sell_orders": sell_orders,         # 卖出订单数
            "hold_orders": hold_orders,         # 持有订单数
            "total_volume": total_volume,       # 总成交量
            "order_imbalance": order_imbalance, # 订单不平衡度
            "price_change_pct": (price_change_factor - 1) * 100,  # 价格变化%
            "new_price": self.current_price,    # 新价格
        }

    def get_price_history(self) -> List[float]:
        """
        获取完整的价格历史

        该方法返回从仿真开始到当前的所有价格数据的副本。
        返回副本是为了防止外部代码意外修改内部状态。

        Returns:
            价格历史列表的副本

        Note:
            返回列表的第一个元素是初始价格，
            后续元素是每个回合结束后的价格。
            因此 len(price_history) = num_rounds + 1
        """
        # 返回副本以保护内部数据
        return self.price_history.copy()

    def get_stats(self) -> Dict:
        """
        获取市场统计摘要

        该方法计算并返回整个仿真期间的市场汇总统计信息，
        包括收益率、波动率、最高/最低价等关键指标。

        Returns:
            包含以下统计量的字典：
            - initial_price: 初始价格
            - final_price: 最终价格
            - total_return_pct: 总收益率（百分比）
            - max_price: 期间最高价
            - min_price: 期间最低价
            - volatility: 价格波动率（收益率标准差）
            - total_rounds: 总回合数
            - total_volume: 总成交量

        Example:
            >>> stats = market.get_stats()
            >>> print(f"总收益率: {stats['total_return_pct']:+.2f}%")
            >>> print(f"波动率: {stats['volatility']:.4f}")
        """
        # 获取价格历史引用（内部使用，无需复制）
        prices = self.price_history

        # 计算总收益率
        # 公式: (最终价格 - 初始价格) / 初始价格 × 100%
        total_return_pct = (
            (self.current_price - self.initial_price)
            / self.initial_price
            * 100
        )

        return {
            # 价格信息
            "initial_price": self.initial_price,    # 初始价格
            "final_price": self.current_price,      # 最终价格
            "total_return_pct": total_return_pct,   # 总收益率%

            # 价格范围
            "max_price": max(prices),               # 最高价
            "min_price": min(prices),               # 最低价

            # 波动性
            "volatility": self._calculate_volatility(),  # 波动率

            # 交易统计
            "total_rounds": len(prices) - 1,        # 总回合数
            "total_volume": sum(self.volume_history),  # 总成交量
        }

    def _calculate_volatility(self) -> float:
        """
        计算价格波动率

        波动率是金融市场中衡量风险的核心指标，定义为
        对数收益率（或简单收益率）的标准差。

        本方法使用简单收益率计算波动率：
            r_t = (P_t - P_{t-1}) / P_{t-1}
            volatility = std(r_1, r_2, ..., r_n)

        计算步骤：
            1. 计算每期的简单收益率
            2. 计算收益率的均值
            3. 计算收益率的方差
            4. 返回方差的平方根（标准差）

        Returns:
            价格波动率（收益率的标准差）
            如果历史数据不足（少于2个数据点），返回0.0

        Note:
            - 这里使用的是总体标准差（除以n），而非样本标准差（除以n-1）
            - 在金融实践中，波动率通常会年化处理，但本方法返回原始波动率
        """
        # 检查是否有足够的数据计算收益率
        # 至少需要2个价格点才能计算1个收益率
        if len(self.price_history) < 2:
            return 0.0

        # 计算简单收益率序列
        # r_t = (P_t - P_{t-1}) / P_{t-1}
        returns = [
            (self.price_history[i] - self.price_history[i - 1])
            / self.price_history[i - 1]
            for i in range(1, len(self.price_history))
        ]

        # 计算收益率均值
        mean_return = sum(returns) / len(returns)

        # 计算方差 (使用总体方差公式)
        # variance = (1/n) × Σ(r_i - mean)²
        variance = sum(
            (r - mean_return) ** 2 for r in returns
        ) / len(returns)

        # 返回标准差（方差的平方根）
        return variance ** 0.5

    def get_order_history(self) -> Dict:
        """
        获取订单历史记录

        该方法返回每个回合的买入和卖出订单数量，
        用于分析市场的买卖力量对比。

        Returns:
            包含以下键的字典：
            - buy_history: 每回合买入订单数列表
            - sell_history: 每回合卖出订单数列表
            - imbalance_history: 每回合订单不平衡度列表

        Example:
            >>> history = market.get_order_history()
            >>> avg_imbalance = sum(history['imbalance_history']) / len(history['imbalance_history'])
            >>> print(f"平均订单不平衡度: {avg_imbalance:.3f}")
        """
        # 计算每回合的订单不平衡度
        imbalance_history = [
            (buy - sell) / self.num_agents
            for buy, sell in zip(self.buy_history, self.sell_history)
        ]

        return {
            "buy_history": self.buy_history.copy(),
            "sell_history": self.sell_history.copy(),
            "imbalance_history": imbalance_history,
        }

    def reset(self) -> None:
        """
        重置市场状态

        该方法将市场恢复到初始状态，清除所有历史数据。
        用于在多次仿真运行之间重置市场。

        重置内容：
        - 当前价格重置为初始价格
        - 清空所有历史记录
        - 将初始价格添加到价格历史

        Note:
            调用此方法后，市场状态与新创建的Market对象相同
        """
        # 重置当前价格
        self.current_price = self.initial_price

        # 清空所有历史记录
        self.price_history.clear()
        self.volume_history.clear()
        self.buy_history.clear()
        self.sell_history.clear()

        # 将初始价格作为第一个历史点
        self.price_history.append(self.current_price)
