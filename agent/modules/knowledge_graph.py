"""
知识图谱模块 - 基于SQLite的轻量级知识图谱（线程安全版本）
支持：经验存储、知识检索、自动归类、关系管理、并发安全
"""
import sqlite3
import json
import time
import os
import logging
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """知识节点"""
    id: Optional[int] = None
    category: str = ""          # experience / method / fact / error / project
    title: str = ""
    content: str = ""
    tags: str = ""              # 逗号分隔
    importance: float = 0.5     # 0-1
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class KnowledgeEdge:
    """知识关系边"""
    id: Optional[int] = None
    source_id: int = 0
    target_id: int = 0
    relation: str = ""          # related_to / derived_from / improves / replaces
    weight: float = 1.0


class KnowledgeGraph:
    """知识图谱 - SQLite实现（线程安全 + 连接池）"""

    def __init__(self, db_path: str = "data/knowledge.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._lock = threading.RLock()  # 可重入锁
        self._local = threading.local()  # 线程本地存储
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程专属连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,  # 等待锁超时
                isolation_level=None,  # 自动提交模式，避免长期持有锁
            )
            self._local.conn.row_factory = sqlite3.Row
            # 启用WAL模式提高并发性能
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """事务上下文管理器"""
        conn = self._get_connection()
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        with self._lock:
            c = conn.cursor()

            # 知识节点表
            c.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    importance REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_at REAL DEFAULT 0,
                    updated_at REAL DEFAULT 0
                )
            """)

            # 知识关系表
            c.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relation TEXT DEFAULT '',
                    weight REAL DEFAULT 1.0,
                    FOREIGN KEY (source_id) REFERENCES nodes(id),
                    FOREIGN KEY (target_id) REFERENCES nodes(id)
                )
            """)

            # 任务执行记录表
            c.execute("""
                CREATE TABLE IF NOT EXISTS task_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    task_type TEXT,
                    status TEXT,
                    result TEXT DEFAULT '',
                    error TEXT DEFAULT '',
                    duration REAL DEFAULT 0,
                    lessons TEXT DEFAULT '',
                    created_at REAL DEFAULT 0
                )
            """)

            # 优化策略表
            c.execute("""
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    context TEXT DEFAULT '',
                    success_count INTEGER DEFAULT 0,
                    fail_count INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    created_at REAL DEFAULT 0,
                    updated_at REAL DEFAULT 0
                )
            """)

            # 索引
            c.execute("CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(category)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_nodes_tags ON nodes(tags)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_nodes_title ON nodes(title)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_task_logs_type ON task_logs(task_type)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_task_logs_status ON task_logs(status)")

            conn.commit()
            logger.info(f"知识图谱初始化完成: {self.db_path}")

    # ========== 节点操作 ==========

    def add_node(self, node: KnowledgeNode) -> int:
        """添加知识节点（线程安全）"""
        now = time.time()
        node.created_at = now
        node.updated_at = now

        with self._transaction() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO nodes (category, title, content, tags, importance, usage_count, success_rate, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (node.category, node.title, node.content, node.tags,
                  node.importance, node.usage_count, node.success_rate,
                  node.created_at, node.updated_at))
            node.id = c.lastrowid

        logger.debug(f"添加知识节点: [{node.category}] {node.title} (id={node.id})")
        return node.id

    def get_node(self, node_id: int) -> Optional[KnowledgeNode]:
        """获取知识节点"""
        conn = self._get_connection()
        with self._lock:
            c = conn.cursor()
            c.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = c.fetchone()
            if row:
                return KnowledgeNode(**dict(row))
            return None

    def search_nodes(self, query: str, category: str = "", limit: int = 10) -> List[KnowledgeNode]:
        """搜索知识节点（关键词匹配）"""
        conn = self._get_connection()
        with self._lock:
            c = conn.cursor()
            conditions = []
            params = []

            if query:
                conditions.append("(title LIKE ? OR content LIKE ? OR tags LIKE ?)")
                q = f"%{query}%"
                params.extend([q, q, q])

            if category:
                conditions.append("category = ?")
                params.append(category)

            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"SELECT * FROM nodes{where} ORDER BY importance DESC, usage_count DESC LIMIT ?"
            params.append(limit)

            c.execute(sql, params)
            return [KnowledgeNode(**dict(row)) for row in c.fetchall()]

    def get_related_nodes(self, node_id: int, relation: str = "", depth: int = 1) -> List[KnowledgeNode]:
        """获取关联节点"""
        conn = self._get_connection()
        with self._lock:
            c = conn.cursor()
            if relation:
                c.execute("""
                    SELECT DISTINCT n.* FROM nodes n
                    JOIN edges e ON (e.source_id = ? AND e.target_id = n.id) OR (e.target_id = ? AND e.source_id = n.id)
                    WHERE e.relation = ?
                """, (node_id, node_id, relation))
            else:
                c.execute("""
                    SELECT DISTINCT n.* FROM nodes n
                    JOIN edges e ON (e.source_id = ? AND e.target_id = n.id) OR (e.target_id = ? AND e.source_id = n.id)
                """, (node_id, node_id))
            return [KnowledgeNode(**dict(row)) for row in c.fetchall()]

    def update_node_usage(self, node_id: int, success: bool = True):
        """更新节点使用统计"""
        with self._transaction() as conn:
            c = conn.cursor()
            c.execute("""
                UPDATE nodes SET
                    usage_count = usage_count + 1,
                    success_rate = CASE
                        WHEN usage_count = 0 THEN ? ELSE (success_rate * usage_count + ?) / (usage_count + 1)
                    END,
                    updated_at = ?
                WHERE id = ?
            """, (1.0 if success else 0.0, 1.0 if success else 0.0, time.time(), node_id))

    def delete_node(self, node_id: int):
        """删除节点及其关系"""
        with self._transaction() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (node_id, node_id))
            c.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

    # ========== 关系操作 ==========

    def add_edge(self, source_id: int, target_id: int, relation: str = "related_to", weight: float = 1.0) -> int:
        """添加知识关系"""
        with self._transaction() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO edges (source_id, target_id, relation, weight)
                VALUES (?, ?, ?, ?)
            """, (source_id, target_id, relation, weight))
            return c.lastrowid

    # ========== 任务日志 ==========

    def log_task(self, task_id: str, task_type: str, status: str,
                 result: str = "", error: str = "", duration: float = 0, lessons: str = ""):
        """记录任务执行日志（批量优化版本）"""
        # 使用批量插入优化
        with self._transaction() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO task_logs (task_id, task_type, status, result, error, duration, lessons, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, task_type, status, result, error, duration, lessons, time.time()))

    def get_task_stats(self, task_type: str = "") -> Dict[str, Any]:
        """获取任务统计"""
        conn = self._get_connection()
        with self._lock:
            c = conn.cursor()
            if task_type:
                c.execute("""
                    SELECT status, COUNT(*) as cnt, AVG(duration) as avg_duration
                    FROM task_logs WHERE task_type = ? GROUP BY status
                """, (task_type,))
            else:
                c.execute("""
                    SELECT status, COUNT(*) as cnt, AVG(duration) as avg_duration
                    FROM task_logs GROUP BY status
                """)
            stats = {}
            for row in c.fetchall():
                stats[row["status"]] = {"count": row["cnt"], "avg_duration": round(row["avg_duration"] or 0, 2)}
            return stats

    def get_recent_failures(self, limit: int = 5) -> List[Dict]:
        """获取最近的失败记录"""
        conn = self._get_connection()
        with self._lock:
            c = conn.cursor()
            c.execute("""
                SELECT * FROM task_logs WHERE status = 'failed'
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in c.fetchall()]

    # ========== 策略管理 ==========

    def add_strategy(self, name: str, description: str = "", context: str = "") -> int:
        """添加优化策略"""
        now = time.time()
        with self._transaction() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO strategies (name, description, context, success_count, fail_count, is_active, created_at, updated_at)
                VALUES (?, ?, ?, 0, 0, 1, ?, ?)
            """, (name, description, context, now, now))
            return c.lastrowid

    def get_active_strategies(self, limit: int = 10) -> List[Dict]:
        """获取活跃策略"""
        conn = self._get_connection()
        with self._lock:
            c = conn.cursor()
            c.execute("""
                SELECT * FROM strategies WHERE is_active = 1
                ORDER BY (CAST(success_count AS REAL) / MAX(CAST(success_count + fail_count AS REAL), 1)) DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in c.fetchall()]

    def update_strategy_result(self, strategy_id: int, success: bool):
        """更新策略执行结果"""
        with self._transaction() as conn:
            c = conn.cursor()
            if success:
                c.execute("UPDATE strategies SET success_count = success_count + 1, updated_at = ? WHERE id = ?",
                          (time.time(), strategy_id))
            else:
                c.execute("UPDATE strategies SET fail_count = fail_count + 1, updated_at = ? WHERE id = ?",
                          (time.time(), strategy_id))

    # ========== 经验自动提取与存储 ==========

    def store_experience(self, title: str, content: str, category: str = "experience",
                         tags: str = "", importance: float = 0.5, related_to: Optional[int] = None) -> int:
        """存储一条经验，自动关联相关节点"""
        node = KnowledgeNode(
            category=category,
            title=title,
            content=content,
            tags=tags,
            importance=importance,
        )
        node_id = self.add_node(node)

        if related_to:
            self.add_edge(related_to, node_id, "derived_from")

        return node_id

    def query_experience(self, query: str, limit: int = 5) -> List[KnowledgeNode]:
        """查询相关经验"""
        return self.search_nodes(query, category="experience", limit=limit)

    def get_all_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识库全局摘要"""
        conn = self._get_connection()
        with self._lock:
            c = conn.cursor()
            c.execute("SELECT category, COUNT(*) as cnt FROM nodes GROUP BY category")
            category_counts = {row["category"]: row["cnt"] for row in c.fetchall()}

            c.execute("SELECT COUNT(*) as cnt FROM edges")
            edge_count = c.fetchone()["cnt"]

            c.execute("SELECT COUNT(*) as cnt FROM task_logs")
            task_count = c.fetchone()["cnt"]

            c.execute("SELECT COUNT(*) as cnt FROM strategies WHERE is_active = 1")
            strategy_count = c.fetchone()["cnt"]

            return {
                "nodes_by_category": category_counts,
                "total_edges": edge_count,
                "total_tasks": task_count,
                "active_strategies": strategy_count,
            }

    def vacuum(self):
        """数据库优化（清理碎片）"""
        conn = self._get_connection()
        with self._lock:
            conn.execute("VACUUM")
            logger.info("数据库VACUUM完成")

    def close(self):
        """关闭数据库连接"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
            logger.debug("知识图谱连接已关闭")
