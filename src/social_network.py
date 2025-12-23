"""
社交网络模块 (Social Network Module)

本模块实现了Agent之间的社交网络结构，用于模拟金融市场中的
信息传播和羊群效应(Herding Effect)。

核心思想：
    在真实金融市场中，交易者的决策不仅受到市场信息影响，
    还会受到同行行为的影响。当看到"朋友们都在买入"时，
    交易者更倾向于跟随买入，这就是羊群效应的微观基础。

网络模型：
    本模块使用Barabási-Albert(BA)模型生成无标度网络，
    该模型的特点是：
    - 少数节点拥有大量连接（意见领袖/Hub）
    - 大多数节点只有少量连接（普通交易者）
    - 符合真实社交网络的幂律分布特征

类结构：
    SocialNetwork
    ├── __init__(): 初始化并创建网络
    ├── _create_network(): 使用BA模型创建网络
    ├── get_neighbors(): 获取某节点的邻居列表
    ├── update_action(): 记录节点的交易动作
    ├── get_social_sentiment(): 计算社交情绪统计
    ├── get_network_stats(): 获取网络拓扑统计
    └── get_influencers(): 获取最具影响力的节点

作者: SuZX
日期: 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import networkx as nx  # 网络分析库


# =============================================================================
# 社交网络类
# =============================================================================

@dataclass
class SocialNetwork:
    """
    社交网络管理类

    该类管理Agent之间的社交关系和信息流动，主要功能包括：
    1. 创建和维护社交网络图结构
    2. 记录每个Agent的交易动作
    3. 计算某Agent视角下的社交情绪（邻居们在做什么）
    4. 提供网络分析统计信息

    网络模型说明：
        使用Barabási-Albert优先连接模型：
        - 新节点倾向于连接到已有很多连接的节点
        - 产生"富者愈富"效应，形成意见领袖
        - 网络度分布呈幂律分布

    Attributes:
        num_agents: Agent总数，即网络节点数
        graph: NetworkX图对象，存储网络结构
        last_actions: 字典，记录上一轮每个Agent的动作

    Example:
        >>> network = SocialNetwork(num_agents=20)
        >>> network.update_action(agent_id=0, action="BUY")
        >>> sentiment = network.get_social_sentiment(agent_id=1)
        >>> print(f"邻居中{sentiment['buy_pct']:.0f}%在买入")
    """

    # -------------------------------------------------------------------------
    # 配置参数
    # -------------------------------------------------------------------------

    num_agents: int  # 网络中的Agent数量

    # -------------------------------------------------------------------------
    # 网络结构（延迟初始化）
    # -------------------------------------------------------------------------

    graph: nx.Graph = field(init=False)  # NetworkX图对象

    # -------------------------------------------------------------------------
    # 运行时状态
    # -------------------------------------------------------------------------

    # 记录上一轮各Agent的交易动作
    # 键: agent_id, 值: "BUY"/"SELL"/"HOLD"
    last_actions: Dict[int, str] = field(default_factory=dict)

    # 缓存的邻居列表（网络结构是静态的，无需每次重新计算）
    _neighbors_cache: Dict[int, List[int]] = field(default_factory=dict, repr=False)

    # -------------------------------------------------------------------------
    # 网络参数
    # -------------------------------------------------------------------------

    # BA模型中每个新节点连接到的现有节点数
    # 值越大，网络越密集
    BA_MODEL_M: int = 3

    def __post_init__(self) -> None:
        """
        构造后初始化

        在dataclass创建实例后自动调用，用于初始化网络图。
        使用__post_init__而非__init__是dataclass的标准做法。
        """
        self.graph = self._create_network()

    def _create_network(self) -> nx.Graph:
        """
        创建社交网络图

        使用Barabási-Albert模型生成无标度网络。该模型的关键参数m
        决定了每个新加入的节点会连接到多少个现有节点。

        网络特性：
        - 节点数 = num_agents
        - 边数 ≈ m * (num_agents - m)
        - 存在少数高度节点（意见领袖）
        - 度分布服从幂律 P(k) ~ k^(-3)

        Returns:
            NetworkX Graph对象

        Note:
            当Agent数量很少时，使用完全图代替BA模型
        """
        # 确定BA模型的m参数（每个新节点的连接数）
        m = min(self.BA_MODEL_M, self.num_agents - 1)

        # 如果Agent数量太少，创建完全图
        if self.num_agents <= m:
            return nx.complete_graph(self.num_agents)

        # 使用BA模型创建网络
        # seed=42 确保可重复性
        return nx.barabasi_albert_graph(
            n=self.num_agents,  # 节点数
            m=m,                # 每个新节点的连接数
            seed=42             # 随机种子
        )

    def get_neighbors(self, agent_id: int) -> List[int]:
        """
        获取某Agent的邻居列表（带缓存）

        邻居是指在社交网络中与该Agent直接相连的其他Agent。
        在模拟中，这些邻居的行为会影响该Agent的决策。

        Args:
            agent_id: 目标Agent的ID

        Returns:
            邻居Agent的ID列表

        Example:
            >>> neighbors = network.get_neighbors(agent_id=0)
            >>> print(f"Agent 0有{len(neighbors)}个邻居")
        """
        # 检查缓存
        if agent_id in self._neighbors_cache:
            return self._neighbors_cache[agent_id]

        # 检查节点是否存在
        if agent_id not in self.graph:
            return []

        # 计算并缓存邻居列表
        neighbors = list(self.graph.neighbors(agent_id))
        self._neighbors_cache[agent_id] = neighbors
        return neighbors

    def update_action(self, agent_id: int, action: str) -> None:
        """
        记录Agent的交易动作

        该方法在每个Agent完成决策后调用，将其动作记录到
        last_actions字典中。这些信息将在下一轮被其邻居查询。

        Args:
            agent_id: 执行动作的Agent ID
            action: 交易动作 ("BUY", "SELL", 或 "HOLD")

        Note:
            动作记录在当前轮次结束时会被清空（通过clear_round_actions）
        """
        self.last_actions[agent_id] = action

    def clear_round_actions(self) -> None:
        """
        清空当前轮次的动作记录

        在每个新回合开始时调用，清除上一轮的动作记录。
        这确保了社交情绪计算总是基于最新的一轮数据。
        """
        self.last_actions.clear()

    def get_social_sentiment(self, agent_id: int) -> Dict:
        """
        计算某Agent视角下的社交情绪

        该方法是社交网络模块的核心功能。它统计目标Agent的
        邻居们在上一轮的交易动作分布，生成社交情绪报告。

        社交情绪的作用：
        - 为Agent提供"同行在做什么"的信息
        - Herding类型的Agent会强烈受此影响
        - 可以产生信息级联和羊群效应

        Args:
            agent_id: 查询社交情绪的Agent ID

        Returns:
            包含以下键的字典：
            - has_connections: 是否有社交连接
            - buy_pct: 邻居中买入的比例 (0-100)
            - sell_pct: 邻居中卖出的比例 (0-100)
            - hold_pct: 邻居中持有的比例 (0-100)
            - dominant_action: 主导动作
            - connection_count: 总连接数
            - prompt_text: 格式化的提示词文本

        Example:
            >>> sentiment = network.get_social_sentiment(0)
            >>> if sentiment['buy_pct'] > 70:
            ...     print("强烈的买入信号！")
        """
        # 获取邻居列表
        neighbors = self.get_neighbors(agent_id)

        # 没有邻居的情况
        if not neighbors:
            return {
                "has_connections": False,
                "buy_pct": 0,
                "sell_pct": 0,
                "hold_pct": 0,
                "dominant_action": "UNKNOWN",
                "connection_count": 0,
                "prompt_text": "",
            }

        # 统计邻居的动作分布
        buy_count = 0
        sell_count = 0
        hold_count = 0
        active_neighbors = 0  # 有动作记录的邻居数

        for neighbor_id in neighbors:
            action = self.last_actions.get(neighbor_id)
            if action:
                active_neighbors += 1
                if action == "BUY":
                    buy_count += 1
                elif action == "SELL":
                    sell_count += 1
                else:
                    hold_count += 1

        # 如果没有活跃邻居（第一轮）
        if active_neighbors == 0:
            return {
                "has_connections": True,
                "buy_pct": 0,
                "sell_pct": 0,
                "hold_pct": 0,
                "dominant_action": "NO_DATA",
                "connection_count": len(neighbors),
                "prompt_text": f"You have {len(neighbors)} connections, but no data from last round.",
            }

        # 计算百分比
        buy_pct = buy_count / active_neighbors * 100
        sell_pct = sell_count / active_neighbors * 100
        hold_pct = hold_count / active_neighbors * 100

        # 确定主导动作
        if buy_pct >= sell_pct and buy_pct >= hold_pct:
            dominant = "BUY"
        elif sell_pct >= buy_pct and sell_pct >= hold_pct:
            dominant = "SELL"
        else:
            dominant = "HOLD"

        # 生成提示词文本
        prompt_text = self._generate_social_prompt(
            buy_pct=buy_pct,
            sell_pct=sell_pct,
            hold_pct=hold_pct,
            active_count=active_neighbors,
            dominant=dominant
        )

        return {
            "has_connections": True,
            "buy_pct": buy_pct,
            "sell_pct": sell_pct,
            "hold_pct": hold_pct,
            "dominant_action": dominant,
            "connection_count": len(neighbors),
            "active_connections": active_neighbors,
            "prompt_text": prompt_text,
        }

    def _generate_social_prompt(
        self,
        buy_pct: float,
        sell_pct: float,
        hold_pct: float,
        active_count: int,
        dominant: str,
    ) -> str:
        """
        生成社交情绪的提示词文本

        该方法将统计数据转换为LLM可理解的自然语言描述，
        注入到Agent的决策提示词中。

        Args:
            buy_pct: 买入比例
            sell_pct: 卖出比例
            hold_pct: 持有比例
            active_count: 活跃邻居数
            dominant: 主导动作

        Returns:
            格式化的提示词字符串
        """
        # 构建基本信息
        lines = [
            "[Social Sentiment from Your Network]",
            f"You observe {active_count} traders in your network:",
            f"  - {buy_pct:.0f}% are BUYING",
            f"  - {sell_pct:.0f}% are SELLING",
            f"  - {hold_pct:.0f}% are HOLDING",
        ]

        # 根据共识强度添加解释
        max_pct = max(buy_pct, sell_pct, hold_pct)

        if max_pct >= 70:
            # 强烈共识
            lines.append(f"\nStrong consensus: Most of your peers are choosing to {dominant}.")
        elif max_pct >= 50:
            # 中等共识
            lines.append(f"\nModerate consensus: The majority leans toward {dominant}.")
        else:
            # 分歧
            lines.append("\nMixed signals: Your network is divided on the best action.")

        return "\n".join(lines)

    def get_network_stats(self) -> Dict:
        """
        获取网络拓扑统计信息

        该方法计算网络结构的各种统计指标，用于分析
        社交网络的特性和验证BA模型的正确性。

        Returns:
            包含以下统计量的字典：
            - num_nodes: 节点数
            - num_edges: 边数
            - avg_degree: 平均度
            - density: 网络密度
            - clustering_coeff: 平均聚类系数

        Note:
            这些指标可用于论文中描述网络结构
        """
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": (
                sum(dict(self.graph.degree()).values()) /
                self.graph.number_of_nodes()
            ),
            "density": nx.density(self.graph),
            "clustering_coeff": nx.average_clustering(self.graph),
        }

    def get_influencers(self, top_n: int = 5) -> List[Tuple[int, int]]:
        """
        获取网络中最具影响力的节点（意见领袖）

        影响力以节点的度（连接数）衡量。在BA网络中，
        高度节点是信息传播的关键枢纽。

        Args:
            top_n: 返回的节点数量

        Returns:
            (节点ID, 度)元组的列表，按度降序排列

        Example:
            >>> influencers = network.get_influencers(top_n=3)
            >>> for agent_id, degree in influencers:
            ...     print(f"Agent {agent_id} has {degree} connections")
        """
        # 获取所有节点的度
        degrees = self.graph.degree()

        # 按度降序排序
        sorted_degrees = sorted(degrees, key=lambda x: x[1], reverse=True)

        # 返回前N个
        return sorted_degrees[:top_n]
