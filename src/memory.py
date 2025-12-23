"""
Agent记忆模块 (Agent Memory Module)

本模块实现了交易Agent的记忆系统，使Agent能够从历史经验中学习。
核心思想是记录每次决策的上下文（新闻、市场状况）和结果（盈亏），
在面临相似情境时，通过回顾历史来辅助决策。

主要功能：
    1. 记录决策历史：保存每次交易的完整上下文
    2. 计算盈亏(PnL)：基于后续价格变动评估决策效果
    3. 检索相似经验：根据新闻情绪查找类似的历史决策
    4. 生成反思文本：为LLM提供历史经验总结

类结构：
    MemoryRecord (数据类)
    └── 存储单条决策记录的所有信息

    AgentMemory (管理类)
    ├── add_record(): 添加新的决策记录
    ├── update_last_pnl(): 更新最近决策的盈亏
    ├── get_similar_experiences(): 检索相似历史经验
    ├── generate_reflection_prompt(): 生成历史反思提示词
    └── _classify_news_sentiment(): 新闻情绪分类

设计模式：
    - 使用dataclass简化数据结构定义
    - 记忆容量限制防止内存无限增长
    - 基于关键词的简单情绪分类（可扩展为NLP模型）

作者: SuZX
日期: 2024
"""

from dataclasses import dataclass, field
from typing import List, Literal


# =============================================================================
# 类型定义
# =============================================================================

# 新闻情绪类型：看涨(bullish)、看跌(bearish)、中性(neutral)
SentimentType = Literal["bullish", "bearish", "neutral"]

# 交易动作类型
ActionType = Literal["BUY", "SELL", "HOLD"]


# =============================================================================
# 记忆记录数据类
# =============================================================================

@dataclass
class MemoryRecord:
    """
    单条决策记忆记录

    该数据类存储一次交易决策的完整信息，包括：
    - 决策时的市场环境（新闻、价格）
    - 做出的决策（动作、数量、理由）
    - 决策的结果（盈亏）

    Attributes:
        round_num: 决策发生的回合编号
        news: 当时的新闻标题
        news_sentiment: 新闻的情绪分类
        action: 执行的交易动作
        price_at_decision: 决策时的市场价格
        quantity: 交易数量
        reason: LLM给出的决策理由
        pnl: 盈亏金额（在下一回合计算）
        price_after: 下一回合的价格（用于计算PnL）

    Example:
        >>> record = MemoryRecord(
        ...     round_num=5,
        ...     news="央行加息",
        ...     news_sentiment="bearish",
        ...     action="SELL",
        ...     price_at_decision=100.0,
        ...     quantity=10,
        ...     reason="利空消息，减仓"
        ... )
    """

    # -------------------------------------------------------------------------
    # 基本信息字段
    # -------------------------------------------------------------------------

    round_num: int          # 回合编号，用于时序追踪
    news: str               # 新闻内容，用于上下文记录
    news_sentiment: SentimentType  # 新闻情绪，用于相似性匹配

    # -------------------------------------------------------------------------
    # 决策字段
    # -------------------------------------------------------------------------

    action: ActionType      # 交易动作：BUY/SELL/HOLD
    price_at_decision: float  # 决策时的价格
    quantity: int           # 交易数量（股数）
    reason: str             # LLM给出的决策理由

    # -------------------------------------------------------------------------
    # 结果字段（延迟填充）
    # -------------------------------------------------------------------------

    pnl: float = 0.0        # 盈亏金额，在市场更新后计算
    price_after: float = 0.0  # 下一回合价格，用于PnL计算

    def outcome_description(self) -> str:
        """
        生成决策结果的人类可读描述

        该方法将记录中的数据转换为自然语言描述，
        用于生成反思提示词。

        Returns:
            描述决策及其结果的字符串

        Example:
            >>> record.outcome_description()
            "Round 5: On bearish news, sold 10 shares at $100.00,
             resulting in +$50.00 (profit)"
        """
        # 如果是持有动作，描述市场变动
        if self.action == "HOLD":
            return f"Held position, market moved to ${self.price_after:.2f}"

        # 确定动作的过去式
        action_verb = "bought" if self.action == "BUY" else "sold"

        # 确定结果描述词
        if self.pnl > 0:
            outcome = "profit"
        elif self.pnl < 0:
            outcome = "loss"
        else:
            outcome = "break-even"

        # 组装完整描述
        description = (
            f"Round {self.round_num}: On {self.news_sentiment} news, "
            f"{action_verb} {self.quantity} shares at ${self.price_at_decision:.2f}, "
            f"resulting in ${self.pnl:+.2f} ({outcome})"
        )

        return description


# =============================================================================
# Agent记忆管理类
# =============================================================================

@dataclass
class AgentMemory:
    """
    Agent记忆管理系统

    该类管理Agent的决策历史记录，提供以下功能：
    1. 添加和存储决策记录
    2. 更新决策的盈亏结果
    3. 基于新闻情绪检索相似历史经验
    4. 生成用于LLM的历史反思提示词

    记忆系统的设计理念：
    - 有限容量：只保留最近的N条记录，防止内存无限增长
    - 情绪匹配：根据新闻情绪（而非具体内容）查找相似经验
    - 延迟评估：PnL在市场价格更新后才能计算

    Attributes:
        max_memories: 最大记忆容量，超过时删除最旧记录
        records: 存储MemoryRecord的列表

    Example:
        >>> memory = AgentMemory(max_memories=20)
        >>> memory.add_record(
        ...     round_num=1,
        ...     news="GDP超预期",
        ...     action="BUY",
        ...     price_at_decision=100.0,
        ...     quantity=10,
        ...     reason="利好消息"
        ... )
        >>> memory.update_last_pnl(new_price=105.0)
        >>> print(memory.records[-1].pnl)  # 输出: 50.0 (5 * 10)
    """

    # -------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------

    max_memories: int = 20  # 最大记忆容量

    # -------------------------------------------------------------------------
    # 内部状态
    # -------------------------------------------------------------------------

    records: List[MemoryRecord] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # 新闻情绪分类的关键词列表
    # -------------------------------------------------------------------------

    # 看涨关键词：表示积极的市场信号
    BULLISH_KEYWORDS: List[str] = field(default_factory=lambda: [
        "record", "beats", "strong", "surge", "high", "growth",
        "optimism", "breakthrough", "cuts", "stimulus", "boom",
        "rally", "gains", "positive", "exceeds", "rises",
    ])

    # 看跌关键词：表示消极的市场信号
    BEARISH_KEYWORDS: List[str] = field(default_factory=lambda: [
        "falls", "below", "contracts", "tension", "concerns",
        "volatility", "down", "cooling", "invert", "recession",
        "hike", "inflation", "drops", "decline", "weak", "fears",
    ])

    def add_record(
        self,
        round_num: int,
        news: str,
        action: str,
        price_at_decision: float,
        quantity: int,
        reason: str,
    ) -> MemoryRecord:
        """
        添加新的决策记录

        该方法创建一个新的MemoryRecord并添加到记忆列表中。
        如果记忆数量超过容量限制，会自动删除最旧的记录。

        Args:
            round_num: 当前回合编号
            news: 新闻内容
            action: 交易动作
            price_at_decision: 决策时价格
            quantity: 交易数量
            reason: 决策理由

        Returns:
            新创建的MemoryRecord对象

        Note:
            PnL字段此时为0，需要在下一回合调用update_last_pnl()更新
        """
        # 对新闻进行情绪分类
        sentiment = self._classify_news_sentiment(news)

        # 创建记录对象
        record = MemoryRecord(
            round_num=round_num,
            news=news,
            news_sentiment=sentiment,
            action=action,
            price_at_decision=price_at_decision,
            quantity=quantity,
            reason=reason,
        )

        # 添加到记录列表
        self.records.append(record)

        # 如果超过容量限制，删除最旧的记录
        if len(self.records) > self.max_memories:
            self.records = self.records[-self.max_memories:]

        return record

    def update_last_pnl(self, new_price: float) -> None:
        """
        更新最近一条记录的盈亏(PnL)

        该方法根据新的市场价格计算最近一次决策的盈亏。
        计算逻辑：
        - BUY: PnL = (new_price - buy_price) * quantity
        - SELL: PnL = (sell_price - new_price) * quantity
        - HOLD: PnL = 0

        Args:
            new_price: 最新的市场价格

        Note:
            对于SELL，PnL表示"卖出的机会收益"：
            如果卖后价格下跌，说明卖对了，PnL为正；
            如果卖后价格上涨，说明卖错了，PnL为负。
        """
        # 检查是否有记录
        if not self.records:
            return

        # 获取最近一条记录
        record = self.records[-1]

        # 记录新价格
        record.price_after = new_price

        # 根据动作类型计算PnL
        if record.action == "BUY":
            # 买入的盈亏 = (当前价 - 买入价) × 数量
            record.pnl = (new_price - record.price_at_decision) * record.quantity

        elif record.action == "SELL":
            # 卖出的机会盈亏 = (卖出价 - 当前价) × 数量
            # 正值表示卖出后价格下跌（卖对了）
            # 负值表示卖出后价格上涨（卖错了）
            record.pnl = (record.price_at_decision - new_price) * record.quantity

        else:  # HOLD
            record.pnl = 0.0

    def get_similar_experiences(
        self,
        current_news: str,
        limit: int = 3
    ) -> List[MemoryRecord]:
        """
        检索与当前新闻情绪相似的历史经验

        该方法首先对当前新闻进行情绪分类，然后在记忆中
        查找具有相同情绪的历史记录。

        Args:
            current_news: 当前的新闻内容
            limit: 返回的最大记录数

        Returns:
            相似经验的记录列表（最近的优先）

        Example:
            >>> similar = memory.get_similar_experiences("央行加息")
            >>> for exp in similar:
            ...     print(exp.outcome_description())
        """
        # 如果没有记录，返回空列表
        if not self.records:
            return []

        # 对当前新闻进行情绪分类
        current_sentiment = self._classify_news_sentiment(current_news)

        # 筛选出相同情绪的记录
        similar_records = [
            record for record in self.records
            if record.news_sentiment == current_sentiment
        ]

        # 返回最近的N条记录
        return similar_records[-limit:]

    def generate_reflection_prompt(self, current_news: str) -> str:
        """
        生成历史反思提示词

        该方法是记忆模块的核心输出接口。它检索相似的历史经验，
        并将其转换为LLM可以理解的反思文本，注入到决策提示词中。

        提示词格式：
            [Historical Reflection]
            You recall similar past experiences:
            - In Round X, on bullish news, you chose to BUY and profited (+$50)
            - In Round Y, on bullish news, you chose to SELL and lost (-$30)
            Consider whether your past decisions...

        Args:
            current_news: 当前的新闻内容

        Returns:
            格式化的反思提示词字符串，如果没有相似经验则返回空字符串

        Example:
            >>> reflection = memory.generate_reflection_prompt("GDP增长")
            >>> print(reflection)
            [Historical Reflection]
            You recall similar past experiences:
            ...
        """
        # 获取相似的历史经验
        similar_experiences = self.get_similar_experiences(current_news, limit=3)

        # 如果没有相似经验，返回空字符串
        if not similar_experiences:
            return ""

        # 构建反思内容
        reflections = []

        for record in similar_experiences:
            # 确定盈亏描述词
            if record.pnl > 0:
                outcome = "profited"
            elif record.pnl < 0:
                outcome = "lost money"
            else:
                outcome = "broke even"

            # 格式化单条反思
            reflection_line = (
                f"- In Round {record.round_num}, on similar {record.news_sentiment} news, "
                f"you chose to {record.action} and {outcome} (${record.pnl:+.2f})"
            )
            reflections.append(reflection_line)

        # 组装完整的反思提示词
        reflection_text = "\n".join(reflections)

        prompt = f"""
[Historical Reflection]
You recall similar past experiences:
{reflection_text}

Consider whether your past decisions in similar situations were successful before deciding.
"""

        return prompt

    def get_performance_summary(self) -> dict:
        """
        获取记忆中的绩效摘要统计

        该方法计算Agent历史交易的聚合指标，用于评估Agent的整体表现。

        Returns:
            包含以下指标的字典：
            - total_pnl: 总盈亏
            - win_rate: 胜率（盈利交易的比例）
            - avg_pnl: 平均每笔交易盈亏
            - total_trades: 总交易次数（不含HOLD）

        Example:
            >>> stats = memory.get_performance_summary()
            >>> print(f"胜率: {stats['win_rate']:.1f}%")
        """
        # 空记录的情况
        if not self.records:
            return {
                "total_pnl": 0,
                "win_rate": 0,
                "avg_pnl": 0,
                "total_trades": 0,
            }

        # 计算总盈亏
        total_pnl = sum(record.pnl for record in self.records)

        # 筛选出实际交易（排除HOLD）
        trades = [record for record in self.records if record.action != "HOLD"]

        # 如果没有实际交易
        if not trades:
            return {
                "total_pnl": total_pnl,
                "win_rate": 0,
                "avg_pnl": 0,
                "total_trades": 0,
            }

        # 计算胜率（盈利交易数 / 总交易数）
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        win_rate = winning_trades / len(trades) * 100

        # 计算平均盈亏
        avg_pnl = total_pnl / len(trades)

        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "total_trades": len(trades),
        }

    def _classify_news_sentiment(self, news: str) -> SentimentType:
        """
        对新闻进行情绪分类

        该方法使用基于关键词的简单规则对新闻进行情绪分类。
        分类逻辑：
        1. 统计新闻中包含的看涨关键词数量
        2. 统计新闻中包含的看跌关键词数量
        3. 比较两者，多数决定最终分类

        Args:
            news: 新闻文本

        Returns:
            情绪分类：'bullish', 'bearish', 或 'neutral'

        Note:
            这是一个简化的实现，实际应用中可以使用
            更复杂的NLP模型（如FinBERT）进行情绪分析
        """
        # 转换为小写以进行大小写不敏感的匹配
        news_lower = news.lower()

        # 统计看涨关键词出现次数
        bullish_score = sum(
            1 for keyword in self.BULLISH_KEYWORDS
            if keyword in news_lower
        )

        # 统计看跌关键词出现次数
        bearish_score = sum(
            1 for keyword in self.BEARISH_KEYWORDS
            if keyword in news_lower
        )

        # 根据得分确定情绪
        if bullish_score > bearish_score:
            return "bullish"
        elif bearish_score > bullish_score:
            return "bearish"
        else:
            return "neutral"
