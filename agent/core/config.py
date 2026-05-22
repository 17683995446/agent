"""
极简全能持续运行Agent - 配置模块
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM API配置"""
    provider: str = "openai"           # openai / ollama / custom
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60

    # S8.1修复：防止API密钥通过日志/打印泄露
    def __repr__(self):
        return f"LLMConfig(provider={self.provider}, model={self.model}, base_url={self.base_url})"

    def __str__(self):
        return self.__repr__()


@dataclass
class KnowledgeConfig:
    """知识图谱配置"""
    db_path: str = "data/knowledge.db"
    auto_save_interval: int = 30       # 每N次操作自动持久化


@dataclass
class AgentConfig:
    """Agent全局配置"""
    # 任务
    task_queue_size: int = 100
    max_concurrent_tasks: int = 3
    task_retry_limit: int = 3
    task_timeout: int = 300            # 单任务超时(秒)

    # 优化
    optimize_interval: int = 5         # 每完成N个任务触发一次优化
    reflection_depth: str = "deep"     # light / deep

    # 保活
    heartbeat_interval: int = 10       # 心跳间隔(秒)
    restart_on_error: bool = True
    max_restart_attempts: int = 10
    restart_cooldown: int = 5          # 重启冷却(秒)

    # 日志
    log_level: str = "INFO"
    log_dir: str = "logs"

    # 子配置
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """从环境变量加载配置"""
        cfg = cls()
        # LLM
        cfg.llm.api_key = os.getenv("LLM_API_KEY", cfg.llm.api_key)
        cfg.llm.base_url = os.getenv("LLM_BASE_URL", cfg.llm.base_url)
        cfg.llm.model = os.getenv("LLM_MODEL", cfg.llm.model)
        cfg.llm.provider = os.getenv("LLM_PROVIDER", cfg.llm.provider)
        # 知识库
        cfg.knowledge.db_path = os.getenv("KNOWLEDGE_DB_PATH", cfg.knowledge.db_path)
        return cfg
