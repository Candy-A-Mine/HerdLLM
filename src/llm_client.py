"""
LLM客户端模块 (LLM Client Module)

本模块提供与本地Ollama大语言模型的通信接口，封装了OpenAI兼容的API调用。
主要功能包括：
    1. 建立与Ollama服务的连接
    2. 构建交易决策的提示词(Prompt)
    3. 解析LLM返回的JSON格式决策
    4. 处理各种异常情况，确保系统稳定性

类关系图：
    LLMClient
    ├── __init__(): 初始化OpenAI客户端连接
    ├── get_decision(): 获取交易决策（核心方法）
    ├── _build_system_prompt(): 构建系统提示词
    ├── _build_user_prompt(): 构建用户提示词
    └── _parse_response(): 解析LLM响应

作者: SuZX
日期: 2024
"""

import json  # JSON解析库
import re    # 正则表达式库，用于从非标准响应中提取JSON
from typing import Any, Dict, List  # 类型提示

from openai import OpenAI, AsyncOpenAI  # OpenAI SDK，用于与Ollama通信


class LLMClient:
    """
    大语言模型客户端类

    该类封装了与本地Ollama实例的通信逻辑，使用OpenAI兼容的API。
    主要职责是将市场信息和Agent状态转换为提示词，获取LLM的交易决策。

    Attributes:
        client (OpenAI): OpenAI客户端实例，配置为连接本地Ollama
        model (str): 使用的模型名称，默认为'qwen2.5:7b'

    Example:
        >>> client = LLMClient()
        >>> decision = client.get_decision(
        ...     agent_profile={'personality': 'Aggressive', 'cash': 10000},
        ...     news='GDP增长超预期',
        ...     current_price=100.0,
        ...     history=[98.0, 99.0, 100.0]
        ... )
        >>> print(decision)
        {'action': 'BUY', 'reason': '利好消息，看涨'}
    """

    # =========================================================================
    # 类常量定义
    # =========================================================================

    # 默认Ollama服务地址
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    # 默认API密钥（Ollama使用虚拟密钥）
    DEFAULT_API_KEY = "ollama"

    # 默认模型名称
    DEFAULT_MODEL = "qwen2.5:7b"

    # 人格类型与交易风格的映射关系
    # 这个映射定义了不同人格类型的Agent在决策时的行为倾向
    PERSONALITY_GUIDES = {
        "Conservative": (
            "You are risk-averse. "  # 风险厌恶型
            "You prefer to hold during uncertainty and only trade on strong signals. "
            "You prioritize capital preservation over potential gains."
        ),
        "Aggressive": (
            "You are risk-seeking. "  # 风险偏好型
            "You are quick to act on opportunities and willing to make bold moves. "
            "You accept higher volatility for potentially higher returns."
        ),
        "Trend_Follower": (
            "You follow market momentum and price trends. "  # 趋势跟随型
            "You buy in uptrends and sell in downtrends. "
            "Technical signals are more important to you than fundamental news."
        ),
        "Herding": (
            "You strongly value social proof. "  # 羊群效应型
            "You tend to follow what the majority of your peers are doing. "
            "You feel safer when your actions align with the crowd."
        ),
    }

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        seed: int | None = None,
    ) -> None:
        """
        初始化LLM客户端

        创建与Ollama服务的连接，配置API端点和认证信息。

        Args:
            base_url: Ollama服务的API端点URL
                      默认值: "http://localhost:11434/v1"
            api_key: API认证密钥，Ollama使用虚拟密钥
                     默认值: "ollama"
            model: 使用的模型标识符
                   默认值: "qwen2.5:7b"
            seed: 随机种子，用于Monte Carlo实验的可变性
                  不同的seed会产生不同的LLM输出

        Raises:
            ConnectionError: 当无法连接到Ollama服务时抛出

        Note:
            确保在调用此构造函数前，Ollama服务已经启动
        """
        # 创建OpenAI客户端，配置为连接本地Ollama
        self.client = OpenAI(
            base_url=base_url,  # API基础URL
            api_key=api_key     # 认证密钥
        )

        # 异步客户端使用懒加载（避免事件循环关闭时的错误）
        self._async_client = None
        self._base_url = base_url
        self._api_key = api_key

        # 保存模型名称，用于后续API调用
        self.model = model

        # 保存基础种子
        self.base_seed = seed

        # 调用计数器（用于生成唯一种子）
        self._call_counter = 0

    @property
    def async_client(self) -> AsyncOpenAI:
        """懒加载异步客户端，只在需要时创建"""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key
            )
        return self._async_client

    def reset_async_client(self) -> None:
        """
        重置异步客户端（在事件循环关闭后调用）

        当 asyncio.run() 结束后，旧的事件循环会被关闭，
        绑定到该循环的 AsyncOpenAI 客户端会变得不可用。
        调用此方法清理旧客户端，下次使用时会自动创建新的。
        """
        self._async_client = None

    def get_decision(
        self,
        agent_profile: Dict[str, Any],
        news: str,
        current_price: float,
        history: List[float],
        memory_reflection: str = "",
        social_sentiment: str = "",
    ) -> Dict[str, str]:
        """
        获取LLM的交易决策

        这是LLMClient的核心方法。它将所有输入信息（Agent状态、市场数据、
        新闻、历史反思、社交情绪）组合成提示词，发送给LLM，并解析返回的
        交易决策。

        Args:
            agent_profile: Agent的当前状态字典，包含：
                - personality (str): 人格类型
                - cash (float): 可用现金
                - holdings (int): 持有股数
                - portfolio_value (float): 总资产价值
            news: 当前回合的新闻标题
            current_price: 当前市场价格
            history: 历史价格列表
            memory_reflection: 来自记忆模块的历史反思文本（可选）
            social_sentiment: 来自社交网络的情绪信息（可选）

        Returns:
            包含交易决策的字典：
                - action (str): "BUY", "SELL", 或 "HOLD"
                - reason (str): 决策理由的简短解释

        Example:
            >>> decision = client.get_decision(
            ...     agent_profile={
            ...         'personality': 'Conservative',
            ...         'cash': 10000.0,
            ...         'holdings': 50,
            ...         'portfolio_value': 15000.0
            ...     },
            ...     news='央行宣布加息0.5%',
            ...     current_price=100.0,
            ...     history=[98.0, 99.0, 100.0],
            ...     memory_reflection='上次加息时卖出后亏损...',
            ...     social_sentiment='80%的交易者在买入'
            ... )

        Note:
            当LLM调用失败时，默认返回HOLD操作以确保系统稳定性
        """
        # ---------------------------------------------------------------------
        # 步骤1: 计算价格趋势
        # ---------------------------------------------------------------------
        # 获取最近5个价格点用于趋势分析
        recent_prices = history[-5:] if len(history) >= 5 else history

        # 计算价格变化百分比
        price_trend = ""
        if len(recent_prices) >= 2:
            # 计算从最早价格到最新价格的变化率
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            price_trend = f"Recent trend: {price_change:+.2f}% over last {len(recent_prices)} rounds."

        # ---------------------------------------------------------------------
        # 步骤2: 构建提示词
        # ---------------------------------------------------------------------
        # 获取Agent人格类型
        personality = agent_profile.get("personality", "Neutral")

        # 构建系统提示词（定义LLM的角色和输出格式）
        system_prompt = self._build_system_prompt(personality)

        # 构建用户提示词（包含所有上下文信息）
        user_prompt = self._build_user_prompt(
            agent_profile=agent_profile,
            current_price=current_price,
            price_trend=price_trend,
            recent_prices=recent_prices,
            news=news,
            memory_reflection=memory_reflection,
            social_sentiment=social_sentiment,
        )

        # ---------------------------------------------------------------------
        # 步骤3: 调用LLM API
        # ---------------------------------------------------------------------
        # 计算本次调用的唯一种子
        call_seed = None
        if self.base_seed is not None:
            call_seed = self.base_seed + self._call_counter
            self._call_counter += 1

        try:
            # 首先尝试使用JSON模式（更可靠的输出格式）
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # 控制输出的随机性
                max_tokens=200,   # 限制输出长度
                response_format={"type": "json_object"},  # 强制JSON输出
                seed=call_seed,   # 使用唯一种子确保Monte Carlo变异性
            )

            # 提取响应内容
            content = response.choices[0].message.content.strip()

            # 解析响应
            return self._parse_response(content)

        except Exception:
            # 如果JSON模式失败，尝试普通模式
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                    seed=call_seed,
                )
                content = response.choices[0].message.content.strip()
                return self._parse_response(content)

            except Exception as error:
                # 记录错误并返回安全的默认值
                print(f"[LLM Error] {error}")
                return {
                    "action": "HOLD",
                    "reason": "LLM通信错误，默认持有"
                }

    def _build_system_prompt(self, personality: str) -> str:
        """
        构建系统提示词

        系统提示词定义了LLM的角色、行为准则和输出格式要求。
        根据Agent的人格类型，注入相应的交易风格指导。

        Args:
            personality: Agent的人格类型，如'Conservative', 'Aggressive'等

        Returns:
            完整的系统提示词字符串

        Note:
            系统提示词强制要求JSON格式输出，这对于后续解析至关重要
        """
        # 获取人格类型对应的交易风格描述
        # 如果人格类型未定义，使用默认的中性描述
        personality_description = self.PERSONALITY_GUIDES.get(
            personality,
            "You make balanced decisions based on available information."
        )

        # 组装完整的系统提示词
        system_prompt = f"""You are a financial trader AI with the following personality:
{personality_description}

IMPORTANT: You MUST respond with valid JSON only.
Your response format MUST be exactly: {{"action": "BUY" or "SELL" or "HOLD", "reason": "brief explanation (1-2 sentences)"}}
Do not include any text outside the JSON object. No markdown, no explanations before or after."""

        return system_prompt

    def _build_user_prompt(
        self,
        agent_profile: Dict[str, Any],
        current_price: float,
        price_trend: str,
        recent_prices: List[float],
        news: str,
        memory_reflection: str,
        social_sentiment: str,
    ) -> str:
        """
        构建用户提示词

        用户提示词包含所有的上下文信息，按照结构化的方式组织，
        使LLM能够全面了解当前的市场状况和Agent状态。

        提示词结构：
            1. [Your Portfolio] - Agent的持仓信息
            2. [Market Information] - 市场价格和趋势
            3. [Latest News] - 当前新闻
            4. [Historical Reflection] - 历史经验反思（可选）
            5. [Social Sentiment] - 社交网络情绪（可选）
            6. [Decision Required] - 决策指令

        Args:
            agent_profile: Agent状态字典
            current_price: 当前价格
            price_trend: 价格趋势描述
            recent_prices: 近期价格列表
            news: 新闻内容
            memory_reflection: 记忆反思文本
            social_sentiment: 社交情绪文本

        Returns:
            完整的用户提示词字符串
        """
        # 使用列表收集各个部分，最后用换行符连接
        sections = []

        # -----------------------------------------------------------------
        # 部分1: 投资组合信息
        # -----------------------------------------------------------------
        portfolio_section = f"""[Your Portfolio]
- Personality: {agent_profile.get('personality', 'Neutral')}
- Cash: ${agent_profile.get('cash', 0):,.2f}
- Holdings: {agent_profile.get('holdings', 0)} shares
- Portfolio Value: ${agent_profile.get('portfolio_value', 0):,.2f}"""
        sections.append(portfolio_section)

        # -----------------------------------------------------------------
        # 部分2: 市场信息
        # -----------------------------------------------------------------
        # 格式化价格列表为美元格式
        formatted_prices = [f"${p:.2f}" for p in recent_prices]

        market_section = f"""[Market Information]
- Current Price: ${current_price:.2f}
- {price_trend}
- Recent Prices: {formatted_prices}"""
        sections.append(market_section)

        # -----------------------------------------------------------------
        # 部分3: 新闻
        # -----------------------------------------------------------------
        news_section = f"""[Latest News]
"{news}" """
        sections.append(news_section)

        # -----------------------------------------------------------------
        # 部分4: 历史反思（如果有）
        # -----------------------------------------------------------------
        if memory_reflection:
            sections.append(memory_reflection)

        # -----------------------------------------------------------------
        # 部分5: 社交情绪（如果有）
        # -----------------------------------------------------------------
        if social_sentiment:
            sections.append(social_sentiment)

        # -----------------------------------------------------------------
        # 部分6: 决策指令
        # -----------------------------------------------------------------
        decision_section = """[Decision Required]
Based on all the above information, your personality, past experiences, and social signals, decide your action.
Respond with JSON: {"action": "BUY/SELL/HOLD", "reason": "your reasoning"}"""
        sections.append(decision_section)

        # 用双换行符连接各部分
        return "\n\n".join(sections)

    def _parse_response(self, content: str) -> Dict[str, str]:
        """
        解析LLM响应

        该方法实现了多层次的解析策略，以处理LLM可能产生的各种输出格式：
            1. 首先尝试直接JSON解析
            2. 如果失败，尝试用正则表达式提取JSON
            3. 最后使用关键词匹配作为兜底方案

        Args:
            content: LLM返回的原始文本内容

        Returns:
            标准化的决策字典：
                - action: "BUY", "SELL", 或 "HOLD"
                - reason: 决策理由

        Note:
            这个方法的健壮性对系统稳定性至关重要，
            因为LLM的输出格式不总是完全符合预期
        """
        # ---------------------------------------------------------------------
        # 策略1: 直接JSON解析
        # ---------------------------------------------------------------------
        try:
            result = json.loads(content)

            # 提取并标准化action字段
            action = result.get("action", "HOLD").upper()

            # 验证action是否为有效值
            if action not in ("BUY", "SELL", "HOLD"):
                action = "HOLD"  # 无效值默认为HOLD

            return {
                "action": action,
                "reason": result.get("reason", "No reason provided")
            }

        except json.JSONDecodeError:
            # JSON解析失败，继续尝试其他策略
            pass

        # ---------------------------------------------------------------------
        # 策略2: 正则表达式提取JSON
        # ---------------------------------------------------------------------
        # 尝试匹配包含"action"字段的JSON对象
        json_pattern = r'\{[^{}]*"action"[^{}]*\}'
        json_match = re.search(json_pattern, content, re.IGNORECASE)

        if json_match:
            try:
                result = json.loads(json_match.group())
                action = result.get("action", "HOLD").upper()

                if action not in ("BUY", "SELL", "HOLD"):
                    action = "HOLD"

                return {
                    "action": action,
                    "reason": result.get("reason", "从非标准响应中提取")
                }

            except json.JSONDecodeError:
                pass

        # ---------------------------------------------------------------------
        # 策略3: 关键词匹配（兜底方案）
        # ---------------------------------------------------------------------
        content_upper = content.upper()

        if "BUY" in content_upper:
            return {
                "action": "BUY",
                "reason": "从关键词提取的决策"
            }
        elif "SELL" in content_upper:
            return {
                "action": "SELL",
                "reason": "从关键词提取的决策"
            }
        else:
            return {
                "action": "HOLD",
                "reason": "无法解析LLM响应，默认持有"
            }

    async def get_decision_async(
        self,
        agent_profile: Dict[str, Any],
        news: str,
        current_price: float,
        history: List[float],
        memory_reflection: str = "",
        social_sentiment: str = "",
    ) -> Dict[str, str]:
        """
        异步获取LLM的交易决策（用于并行请求）

        与 get_decision 功能相同，但使用异步API，
        可以并行处理多个Agent的决策请求，显著提升性能。

        Args:
            agent_profile: Agent的当前状态字典
            news: 当前回合的新闻标题
            current_price: 当前市场价格
            history: 历史价格列表
            memory_reflection: 来自记忆模块的历史反思文本（可选）
            social_sentiment: 来自社交网络的情绪信息（可选）

        Returns:
            包含交易决策的字典：
                - action (str): "BUY", "SELL", 或 "HOLD"
                - reason (str): 决策理由的简短解释
        """
        # 计算价格趋势
        recent_prices = history[-5:] if len(history) >= 5 else history
        price_trend = ""
        if len(recent_prices) >= 2:
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            price_trend = f"Recent trend: {price_change:+.2f}% over last {len(recent_prices)} rounds."

        # 构建提示词
        personality = agent_profile.get("personality", "Neutral")
        system_prompt = self._build_system_prompt(personality)
        user_prompt = self._build_user_prompt(
            agent_profile=agent_profile,
            current_price=current_price,
            price_trend=price_trend,
            recent_prices=recent_prices,
            news=news,
            memory_reflection=memory_reflection,
            social_sentiment=social_sentiment,
        )

        # 计算本次调用的唯一种子
        call_seed = None
        if self.base_seed is not None:
            call_seed = self.base_seed + self._call_counter
            self._call_counter += 1

        # 异步调用LLM API
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"},
                seed=call_seed,
            )
            content = response.choices[0].message.content.strip()
            return self._parse_response(content)

        except Exception:
            # 如果JSON模式失败，尝试普通模式
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=200,
                    seed=call_seed,
                )
                content = response.choices[0].message.content.strip()
                return self._parse_response(content)

            except Exception as error:
                return {
                    "action": "HOLD",
                    "reason": "LLM通信错误，默认持有"
                }
