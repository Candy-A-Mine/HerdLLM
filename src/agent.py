"""
交易Agent模块 (Trader Agent Module)

本模块实现了金融市场中的交易Agent，是ABM仿真系统的核心组件。
每个Agent代表一个独立的交易者，具有以下特性：
    1. 持有现金和股票资产
    2. 具有特定的交易人格（保守、激进、跟风、羊群）
    3. 通过LLM进行智能决策
    4. 具有记忆能力，能从历史经验中学习
    5. 能感知社交网络中其他Agent的行为

核心类：
    Order: 交易订单数据类
    DecisionRecord: 决策记录数据类（用于数据导出）
    TraderAgent: 交易Agent主类

设计说明：
    - 使用dataclass简化数据结构定义
    - Agent通过组合模式整合LLM、记忆、社交网络功能
    - 决策过程是确定性的流程，但LLM输出引入随机性

作者: SuZX
日期: 2024
"""

# =============================================================================
# 导入依赖
# =============================================================================

from dataclasses import dataclass, field  # 数据类装饰器
from typing import TYPE_CHECKING, Literal, Dict, List, Tuple  # 类型提示

# 项目内部模块
from src.llm_client import LLMClient    # LLM通信客户端
from src.memory import AgentMemory       # 记忆管理模块

# 类型检查时导入（避免循环导入）
if TYPE_CHECKING:
    from src.social_network import SocialNetwork


# =============================================================================
# 类型定义
# =============================================================================

# 交易动作类型：买入、卖出、持有
ActionType = Literal["BUY", "SELL", "HOLD"]


# =============================================================================
# 订单数据类
# =============================================================================

@dataclass
class Order:
    """
    交易订单数据类

    表示一个交易Agent发出的订单，包含订单的基本信息。
    订单将被Market类处理，用于计算价格影响。

    Attributes:
        agent_id: 发出订单的Agent的唯一标识符
        action: 交易动作类型
            - "BUY": 买入
            - "SELL": 卖出
            - "HOLD": 持有（不交易）
        quantity: 交易数量（股数）
            - 对于BUY：实际买入的股数
            - 对于SELL：实际卖出的股数
            - 对于HOLD：始终为0
        reason: LLM给出的决策理由

    Example:
        >>> order = Order(
        ...     agent_id=0,
        ...     action="BUY",
        ...     quantity=10,
        ...     reason="看好后市，买入"
        ... )
    """

    agent_id: int                  # Agent标识符
    action: ActionType             # 交易动作
    quantity: int                  # 交易数量
    reason: str                    # 决策理由


# =============================================================================
# 决策记录数据类
# =============================================================================

@dataclass
class DecisionRecord:
    """
    完整决策记录数据类

    该类用于记录一次交易决策的完整上下文信息，包括：
    - 决策前的Agent状态
    - 市场环境信息
    - 决策结果
    - 决策后的Agent状态

    主要用途：
    1. 数据导出：转换为DataFrame后保存为Parquet文件
    2. 事后分析：研究Agent行为模式
    3. 可视化：展示决策过程

    Attributes:
        round_num: 仿真回合编号
        agent_id: Agent唯一标识符
        personality: Agent人格类型
        cash_before: 决策前的现金余额
        holdings_before: 决策前的持股数量
        portfolio_value_before: 决策前的总资产价值
        news: 当前回合的新闻内容
        current_price: 当前市场价格
        action: 执行的交易动作
        quantity: 交易数量
        reason: LLM给出的决策理由
        memory_reflection: 记忆模块生成的历史反思文本
        social_buy_pct: 社交网络中买入的比例
        social_sell_pct: 社交网络中卖出的比例
        social_hold_pct: 社交网络中持有的比例
        cash_after: 决策后的现金余额
        holdings_after: 决策后的持股数量
        portfolio_value_after: 决策后的总资产价值
    """

    # -------------------------------------------------------------------------
    # 基本信息
    # -------------------------------------------------------------------------
    round_num: int           # 回合编号
    agent_id: int            # Agent ID
    personality: str         # 人格类型

    # -------------------------------------------------------------------------
    # 决策前状态
    # -------------------------------------------------------------------------
    cash_before: float           # 决策前现金
    holdings_before: int         # 决策前持股
    portfolio_value_before: float  # 决策前总资产

    # -------------------------------------------------------------------------
    # 市场环境
    # -------------------------------------------------------------------------
    news: str                # 新闻内容
    current_price: float     # 当前价格

    # -------------------------------------------------------------------------
    # 决策结果
    # -------------------------------------------------------------------------
    action: str              # 交易动作
    quantity: int            # 交易数量
    reason: str              # 决策理由

    # -------------------------------------------------------------------------
    # 辅助决策信息
    # -------------------------------------------------------------------------
    memory_reflection: str   # 历史反思文本
    social_buy_pct: float    # 社交网络买入比例
    social_sell_pct: float   # 社交网络卖出比例
    social_hold_pct: float   # 社交网络持有比例

    # -------------------------------------------------------------------------
    # 决策后状态
    # -------------------------------------------------------------------------
    cash_after: float            # 决策后现金
    holdings_after: int          # 决策后持股
    portfolio_value_after: float # 决策后总资产


# =============================================================================
# 交易Agent主类
# =============================================================================

@dataclass
class TraderAgent:
    """
    交易Agent类

    这是ABM仿真系统的核心类，代表一个具有AI决策能力的交易者。
    每个Agent通过以下方式做出交易决策：

    1. 感知阶段：
       - 接收市场价格和新闻信息
       - 查询社交网络获取同行行为
       - 回顾历史记忆中的相似经验

    2. 决策阶段：
       - 将所有信息组织成提示词
       - 调用LLM获取交易建议

    3. 执行阶段：
       - 验证交易可行性（资金/持仓是否充足）
       - 执行交易，更新资产状态
       - 记录决策到记忆模块

    Attributes:
        id: Agent的唯一标识符
        cash: 当前持有的现金金额
        holdings: 当前持有的股票数量
        personality: 人格类型，影响LLM的决策风格
            - "Conservative": 保守型，风险厌恶
            - "Aggressive": 激进型，风险偏好
            - "Trend_Follower": 趋势跟随型
            - "Herding": 羊群型，跟随同行
        llm_client: LLM通信客户端实例
        trade_size: 每次交易的最大股数
        memory: 记忆管理模块实例
        social_network: 社交网络引用（可选）

    Example:
        >>> agent = TraderAgent(
        ...     id=0,
        ...     cash=10000.0,
        ...     holdings=50,
        ...     personality="Conservative",
        ...     llm_client=llm_client
        ... )
        >>> order, record = agent.act(
        ...     news="央行加息",
        ...     current_price=100.0,
        ...     price_history=[98, 99, 100],
        ...     round_num=1
        ... )
    """

    # -------------------------------------------------------------------------
    # 必需属性
    # -------------------------------------------------------------------------

    id: int              # Agent唯一标识符
    cash: float          # 持有现金
    holdings: int        # 持有股票数量
    personality: str     # 人格类型

    # -------------------------------------------------------------------------
    # 依赖组件（不参与repr输出）
    # -------------------------------------------------------------------------

    # LLM客户端：用于智能决策
    llm_client: LLMClient = field(repr=False)

    # 每次交易的最大股数
    trade_size: int = field(default=10, repr=False)

    # 记忆模块：存储历史决策和结果
    memory: AgentMemory = field(default_factory=AgentMemory, repr=False)

    # 社交网络引用（可选，用于获取同行行为）
    social_network: "SocialNetwork | None" = field(default=None, repr=False)

    def get_profile(self, current_price: float) -> Dict:
        """
        获取Agent的当前状态概要

        该方法生成一个字典，包含Agent的关键状态信息，
        用于构建LLM的提示词。

        Args:
            current_price: 当前市场价格，用于计算总资产

        Returns:
            包含以下键的字典：
            - id: Agent标识符
            - personality: 人格类型
            - cash: 当前现金
            - holdings: 当前持股
            - portfolio_value: 总资产价值
        """
        return {
            "id": self.id,
            "personality": self.personality,
            "cash": self.cash,
            "holdings": self.holdings,
            "portfolio_value": self.portfolio_value(current_price),
        }

    def act(
        self,
        news: str,
        current_price: float,
        price_history: List[float],
        round_num: int = 0,
    ) -> Tuple[Order, DecisionRecord]:
        """
        执行一次交易决策

        这是TraderAgent的核心方法，完成从感知到执行的完整决策流程。

        决策流程：
        1. 记录决策前状态
        2. 获取记忆反思（如果启用）
        3. 获取社交情绪（如果启用）
        4. 调用LLM获取决策
        5. 验证并执行交易
        6. 更新记忆和社交网络
        7. 生成决策记录

        Args:
            news: 当前回合的新闻标题
            current_price: 当前市场价格
            price_history: 历史价格列表
            round_num: 当前回合编号

        Returns:
            元组 (Order, DecisionRecord)：
            - Order: 交易订单，用于Market处理
            - DecisionRecord: 完整决策记录，用于数据导出

        Note:
            即使LLM建议买入/卖出，如果资金/持仓不足，
            实际执行的动作会变为HOLD
        """
        # ---------------------------------------------------------------------
        # 步骤1: 记录决策前状态
        # ---------------------------------------------------------------------
        cash_before = self.cash
        holdings_before = self.holdings
        portfolio_before = self.portfolio_value(current_price)

        # ---------------------------------------------------------------------
        # 步骤2: 获取记忆反思
        # ---------------------------------------------------------------------
        # 记忆模块会检索相似的历史经验，生成反思文本
        memory_reflection = self.memory.generate_reflection_prompt(news)

        # ---------------------------------------------------------------------
        # 步骤3: 获取社交情绪
        # ---------------------------------------------------------------------
        social_sentiment = ""
        social_stats = {"buy_pct": 0, "sell_pct": 0, "hold_pct": 0}

        if self.social_network is not None:
            # 查询社交网络，获取邻居的行为统计
            social_data = self.social_network.get_social_sentiment(self.id)
            social_sentiment = social_data.get("prompt_text", "")
            social_stats = {
                "buy_pct": social_data.get("buy_pct", 0),
                "sell_pct": social_data.get("sell_pct", 0),
                "hold_pct": social_data.get("hold_pct", 0),
            }

        # ---------------------------------------------------------------------
        # 步骤4: 调用LLM获取决策
        # ---------------------------------------------------------------------
        decision = self.llm_client.get_decision(
            agent_profile=self.get_profile(current_price),
            news=news,
            current_price=current_price,
            history=price_history,
            memory_reflection=memory_reflection,
            social_sentiment=social_sentiment,
        )

        # 提取决策结果
        action = decision["action"]
        reason = decision["reason"]
        quantity = 0

        # ---------------------------------------------------------------------
        # 步骤5: 验证并执行交易
        # ---------------------------------------------------------------------
        if action == "BUY":
            # 计算最大可买入数量
            max_affordable = int(self.cash / current_price) if current_price > 0 else 0
            quantity = min(self.trade_size, max_affordable)

            if quantity > 0:
                # 执行买入
                cost = quantity * current_price
                self.cash -= cost
                self.holdings += quantity
            else:
                # 资金不足，改为持有
                action = "HOLD"
                reason = f"资金不足无法买入。原因: {reason}"

        elif action == "SELL":
            # 计算最大可卖出数量
            quantity = min(self.trade_size, self.holdings)

            if quantity > 0:
                # 执行卖出
                revenue = quantity * current_price
                self.cash += revenue
                self.holdings -= quantity
            else:
                # 持仓不足，改为持有
                action = "HOLD"
                reason = f"持仓不足无法卖出。原因: {reason}"

        # ---------------------------------------------------------------------
        # 步骤6: 更新记忆和社交网络
        # ---------------------------------------------------------------------
        # 将本次决策添加到记忆
        self.memory.add_record(
            round_num=round_num,
            news=news,
            action=action,
            price_at_decision=current_price,
            quantity=quantity,
            reason=reason,
        )

        # 将本次动作广播到社交网络
        if self.social_network is not None:
            self.social_network.update_action(self.id, action)

        # ---------------------------------------------------------------------
        # 步骤7: 生成决策记录
        # ---------------------------------------------------------------------
        record = DecisionRecord(
            round_num=round_num,
            agent_id=self.id,
            personality=self.personality,
            cash_before=cash_before,
            holdings_before=holdings_before,
            portfolio_value_before=portfolio_before,
            news=news,
            current_price=current_price,
            action=action,
            quantity=quantity,
            reason=reason,
            # 截断过长的反思文本
            memory_reflection=memory_reflection[:500] if memory_reflection else "",
            social_buy_pct=social_stats["buy_pct"],
            social_sell_pct=social_stats["sell_pct"],
            social_hold_pct=social_stats["hold_pct"],
            cash_after=self.cash,
            holdings_after=self.holdings,
            portfolio_value_after=self.portfolio_value(current_price),
        )

        # 返回订单和决策记录
        order = Order(
            agent_id=self.id,
            action=action,
            quantity=quantity,
            reason=reason
        )

        return order, record

    def update_memory_pnl(self, new_price: float) -> None:
        """
        更新最近一次决策的盈亏

        在市场价格更新后调用此方法，用新价格计算
        最近一次决策的盈亏(PnL)。

        Args:
            new_price: 最新的市场价格
        """
        self.memory.update_last_pnl(new_price)

    def portfolio_value(self, current_price: float) -> float:
        """
        计算当前总资产价值

        总资产 = 现金 + 持股数量 × 当前价格

        Args:
            current_price: 当前市场价格

        Returns:
            总资产价值
        """
        return self.cash + self.holdings * current_price

    def get_memory_stats(self) -> Dict:
        """
        获取记忆模块的绩效统计

        返回Agent历史交易的聚合指标。

        Returns:
            包含以下键的字典：
            - total_pnl: 总盈亏
            - win_rate: 胜率
            - avg_pnl: 平均盈亏
            - total_trades: 总交易次数
        """
        return self.memory.get_performance_summary()

    async def act_async(
        self,
        news: str,
        current_price: float,
        price_history: List[float],
        round_num: int = 0,
    ) -> Tuple[Order, DecisionRecord]:
        """
        异步执行交易决策（用于并行处理多个Agent）

        与 act 方法功能相同，但使用异步LLM调用，
        可以并行处理多个Agent的决策，显著提升性能。

        Args:
            news: 当前回合的新闻标题
            current_price: 当前市场价格
            price_history: 历史价格列表
            round_num: 当前回合编号

        Returns:
            元组 (Order, DecisionRecord)
        """
        # 记录决策前状态
        cash_before = self.cash
        holdings_before = self.holdings
        portfolio_before = self.portfolio_value(current_price)

        # 获取记忆反思
        memory_reflection = self.memory.generate_reflection_prompt(news)

        # 获取社交情绪
        social_sentiment = ""
        social_stats = {"buy_pct": 0, "sell_pct": 0, "hold_pct": 0}

        if self.social_network is not None:
            social_data = self.social_network.get_social_sentiment(self.id)
            social_sentiment = social_data.get("prompt_text", "")
            social_stats = {
                "buy_pct": social_data.get("buy_pct", 0),
                "sell_pct": social_data.get("sell_pct", 0),
                "hold_pct": social_data.get("hold_pct", 0),
            }

        # 异步调用LLM获取决策
        decision = await self.llm_client.get_decision_async(
            agent_profile=self.get_profile(current_price),
            news=news,
            current_price=current_price,
            history=price_history,
            memory_reflection=memory_reflection,
            social_sentiment=social_sentiment,
        )

        # 提取决策结果
        action = decision["action"]
        reason = decision["reason"]
        quantity = 0

        # 验证并执行交易
        if action == "BUY":
            max_affordable = int(self.cash / current_price) if current_price > 0 else 0
            quantity = min(self.trade_size, max_affordable)

            if quantity > 0:
                cost = quantity * current_price
                self.cash -= cost
                self.holdings += quantity
            else:
                action = "HOLD"
                reason = f"资金不足无法买入。原因: {reason}"

        elif action == "SELL":
            quantity = min(self.trade_size, self.holdings)

            if quantity > 0:
                revenue = quantity * current_price
                self.cash += revenue
                self.holdings -= quantity
            else:
                action = "HOLD"
                reason = f"持仓不足无法卖出。原因: {reason}"

        # 更新记忆和社交网络
        self.memory.add_record(
            round_num=round_num,
            news=news,
            action=action,
            price_at_decision=current_price,
            quantity=quantity,
            reason=reason,
        )

        if self.social_network is not None:
            self.social_network.update_action(self.id, action)

        # 生成决策记录
        record = DecisionRecord(
            round_num=round_num,
            agent_id=self.id,
            personality=self.personality,
            cash_before=cash_before,
            holdings_before=holdings_before,
            portfolio_value_before=portfolio_before,
            news=news,
            current_price=current_price,
            action=action,
            quantity=quantity,
            reason=reason,
            memory_reflection=memory_reflection[:500] if memory_reflection else "",
            social_buy_pct=social_stats["buy_pct"],
            social_sell_pct=social_stats["sell_pct"],
            social_hold_pct=social_stats["hold_pct"],
            cash_after=self.cash,
            holdings_after=self.holdings,
            portfolio_value_after=self.portfolio_value(current_price),
        )

        order = Order(
            agent_id=self.id,
            action=action,
            quantity=quantity,
            reason=reason
        )

        return order, record
