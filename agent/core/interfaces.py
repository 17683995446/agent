"""
核心接口定义 - 架构重构基础
定义所有模块的抽象接口，实现依赖倒置
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRY = "retry"


@dataclass
class Task:
    """任务数据对象 - 与具体实现解耦"""
    id: str
    name: str
    task_type: str
    priority: int
    status: TaskStatus
    created_at: float
    started_at: float = 0.0
    finished_at: float = 0.0
    duration: float = 0.0
    result: Any = None
    error: str = ""
    retries: int = 0
    max_retries: int = 3
    timeout: int = 300
    metadata: Dict[str, Any] = None


# ========== 执行引擎接口 ==========

class ITaskEngine(ABC):
    """任务执行引擎抽象接口"""

    @abstractmethod
    def submit(self, name: str, handler: Callable, **kwargs) -> Task:
        """提交任务"""
        pass

    @abstractmethod
    def submit_many(self, tasks: List[Dict]) -> List[Task]:
        """批量提交任务"""
        pass

    @abstractmethod
    def run_next(self) -> Optional[Task]:
        """执行下一个任务"""
        pass

    @abstractmethod
    def run_batch(self, count: int) -> List[Task]:
        """批量执行任务"""
        pass

    @abstractmethod
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        pass

    @abstractmethod
    def on_complete(self, callback: Callable[[Task], None]):
        """注册完成回调"""
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True, cancel_pending: bool = False):
        """关闭引擎"""
        pass

    @property
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass

    @property
    @abstractmethod
    def pending_count(self) -> int:
        """待处理任务数"""
        pass

    @property
    @abstractmethod
    def running_count(self) -> int:
        """运行中任务数"""
        pass


# ========== 知识图谱接口 ==========

class IKnowledgeGraph(ABC):
    """知识图谱抽象接口"""

    @abstractmethod
    def add_node(self, category: str, title: str, content: str = "",
                 tags: str = "", importance: float = 0.5) -> int:
        """添加知识节点"""
        pass

    @abstractmethod
    def get_node(self, node_id: int) -> Optional[Dict]:
        """获取知识节点"""
        pass

    @abstractmethod
    def search_nodes(self, query: str, category: str = "", limit: int = 10) -> List[Dict]:
        """搜索知识节点"""
        pass

    @abstractmethod
    def add_edge(self, source_id: int, target_id: int,
                 relation: str = "related_to", weight: float = 1.0) -> int:
        """添加知识关系"""
        pass

    @abstractmethod
    def log_task(self, task_id: str, task_type: str, status: str,
                 result: str = "", error: str = "", duration: float = 0,
                 lessons: str = ""):
        """记录任务日志"""
        pass

    @abstractmethod
    def get_task_stats(self, task_type: str = "") -> Dict[str, Any]:
        """获取任务统计"""
        pass

    @abstractmethod
    def add_strategy(self, name: str, description: str = "",
                     context: str = "") -> int:
        """添加优化策略"""
        pass

    @abstractmethod
    def get_active_strategies(self, limit: int = 10) -> List[Dict]:
        """获取活跃策略"""
        pass

    @abstractmethod
    def store_experience(self, title: str, content: str,
                         category: str = "experience", tags: str = "",
                         importance: float = 0.5) -> int:
        """存储经验"""
        pass

    @abstractmethod
    def query_experience(self, query: str, limit: int = 5) -> List[Dict]:
        """查询经验"""
        pass

    @abstractmethod
    def get_all_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识摘要"""
        pass

    @abstractmethod
    def vacuum(self):
        """数据库优化"""
        pass

    @abstractmethod
    def close(self):
        """关闭连接"""
        pass


# ========== LLM客户端接口 ==========

class ILLMClient(ABC):
    """LLM客户端抽象接口"""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]],
             system_prompt: str = "", **kwargs) -> str:
        """发送对话请求"""
        pass

    @abstractmethod
    def analyze_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """发送请求并解析JSON"""
        pass

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """检查健康状态"""
        pass


# ========== 优化器接口 ==========

class IOptimizer(ABC):
    """优化器抽象接口"""

    @abstractmethod
    def reflect_on_tasks(self, tasks: List[Task]) -> Dict[str, Any]:
        """反思任务执行结果"""
        pass

    @abstractmethod
    def get_optimization_advice(self, task_type: str = "") -> Dict[str, Any]:
        """获取优化建议"""
        pass

    @abstractmethod
    def analyze_failure(self, task: Task) -> Dict[str, Any]:
        """分析失败原因"""
        pass

    @abstractmethod
    def optimize_knowledge(self) -> Dict[str, Any]:
        """优化知识图谱"""
        pass

    @property
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        pass


# ========== Agent核心接口 ==========

class IAgent(ABC):
    """Agent主接口"""

    @abstractmethod
    def submit_task(self, name: str, handler: Callable, **kwargs) -> Task:
        """提交任务"""
        pass

    @abstractmethod
    def set_task_generator(self, generator: Callable):
        """设置任务生成器"""
        pass

    @abstractmethod
    def run(self, max_cycles: int = 0, cycle_interval: float = 2.0):
        """运行Agent"""
        pass

    @abstractmethod
    def stop(self):
        """停止Agent"""
        pass

    @property
    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """获取状态"""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """是否运行中"""
        pass
