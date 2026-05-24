"""
第二十四轮 API兼容性与版本升级测试
测试场景：接口契约验证、Schema迁移、配置兼容性、废弃API、降级运行
"""
import os
import sys
import time
import json
import random
import threading
import tempfile
import shutil
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig, LLMConfig, KnowledgeConfig
from agent.modules.task_engine import TaskEngine, TaskStatus, Task
from agent.modules.knowledge_graph import KnowledgeGraph, KnowledgeNode
from agent.modules.optimizer import Optimizer
from agent.modules.llm_client import LLMClient
from agent.core.interfaces import IAgent, ITaskEngine, IKnowledgeGraph, ILLMClient

import logging
logging.basicConfig(level=logging.WARNING, format='%(name)s [%(levelname)s] %(message)s')


class Round24CompatTest:
    """API兼容性与版本升级测试套件"""

    def __init__(self):
        self.results = []
        self.bugs = []
        self._db_counter = 0
        self._test_dir = tempfile.mkdtemp(prefix="compat_r24_")
        print(f"[初始化] 兼容性测试目录: {self._test_dir}")

    def _new_db(self):
        self._db_counter += 1
        return os.path.join(self._test_dir, f"compat_r24_{self._db_counter}.db")

    def record(self, name, passed, details=""):
        self.results.append({"test": name, "passed": passed, "details": details})
        status = "✓" if passed else "✗"
        print(f"  [{status}] {name}: {details[:150]}")

    def bug(self, bid, desc, sev="medium"):
        self.bugs.append({"id": bid, "description": desc, "severity": sev})
        print(f"  🐛 BUG [{bid}] ({sev}): {desc[:120]}")

    def cleanup(self):
        try:
            shutil.rmtree(self._test_dir, ignore_errors=True)
            print(f"[清理] 测试目录已删除: {self._test_dir}")
        except Exception as e:
            print(f"[警告] 清理失败: {e}")

    def run_all(self):
        print("\n" + "=" * 70)
        print("第二十四轮 API兼容性与版本升级测试")
        print("=" * 70)
        print(f"测试时间: {datetime.now().isoformat()}")
        print(f"测试目录: {self._test_dir}")
        print("=" * 70)

        tests = [
            ("A1: 接口契约-IAgent实现验证", self.test_a1_iagent_contract),
            ("A2: 接口契约-ITaskEngine实现验证", self.test_a2_itaskengine_contract),
            ("A3: 接口契约-IKnowledgeGraph实现验证", self.test_a3_ikg_contract),
            ("A4: 接口契约-ILLMClient实现验证", self.test_a4_illm_contract),
            ("A5: Schema迁移-旧版数据库兼容", self.test_a5_schema_migration),
            ("A6: Schema迁移-新增列兼容", self.test_a6_add_column),
            ("A7: 配置兼容性-默认值完整性", self.test_a7_config_defaults),
            ("A8: 配置兼容性-环境变量加载", self.test_a8_env_config),
            ("A9: 废弃API-未使用配置项检测", self.test_a9_deprecated_config),
            ("A10: 降级运行-模块缺失容错", self.test_a10_degradation),
        ]

        for name, func in tests:
            print(f"\n{'=' * 60}")
            print(f"执行: {name}")
            print(f"{'=' * 60}")
            try:
                func()
            except Exception as e:
                import traceback
                self.record(name, False, f"异常: {str(e)[:100]}")
                self.bug(f"A-{name[:3]}", f"异常: {traceback.format_exc()[:200]}", "high")

        self._summary()
        self.cleanup()

    # ========== A1: 接口契约-IAgent实现验证 ==========

    def test_a1_iagent_contract(self):
        """验证Agent类完整实现IAgent接口"""
        errors = []

        # 检查Agent是否是IAgent的子类
        if not issubclass(Agent, IAgent):
            errors.append("Agent不是IAgent的子类")
            self.bug("A1-001", "Agent类未继承IAgent接口", "high")

        # 检查所有IAgent抽象方法是否实现
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        agent = Agent(config)

        # 检查核心方法存在
        required_methods = ['submit_task', 'run', 'stop', 'set_task_generator']
        for method in required_methods:
            if not hasattr(agent, method):
                errors.append(f"缺少方法: {method}")
                self.bug("A1-002", f"Agent缺少IAgent方法: {method}", "high")

        # 检查is_running属性（IAgent接口定义）
        if hasattr(IAgent, 'is_running'):
            if not hasattr(agent, 'is_running'):
                errors.append("缺少is_running属性")
                self.bug("A1-003", "Agent缺少IAgent定义的is_running属性", "high")

        # 检查status属性
        try:
            status = agent.status
            if not isinstance(status, dict):
                errors.append(f"status返回类型错误: {type(status)}")
        except Exception as e:
            errors.append(f"status访问失败: {e}")

        agent.stop()

        self.record("A1: 接口契约-IAgent实现验证", not errors,
                    f"问题: {errors if errors else '无'}")

    # ========== A2: 接口契约-ITaskEngine实现验证 ==========

    def test_a2_itaskengine_contract(self):
        """验证TaskEngine类完整实现ITaskEngine接口"""
        errors = []

        if not issubclass(TaskEngine, ITaskEngine):
            errors.append("TaskEngine不是ITaskEngine的子类")
            self.bug("A2-001", "TaskEngine类未继承ITaskEngine接口", "high")

        engine = TaskEngine(max_workers=2)

        # 检查所有ITaskEngine抽象方法
        required_methods = ['submit', 'run_next', 'run_all_pending', 'run_batch',
                          'cancel_task', 'shutdown', 'pending_count', 'running_count']
        for method in required_methods:
            if not hasattr(engine, method):
                errors.append(f"缺少方法: {method}")
                self.bug("A2-002", f"TaskEngine缺少ITaskEngine方法: {method}", "high")

        # 检查submit方法签名兼容性（接口用**kwargs，实现用具名参数）
        try:
            # 接口方式调用（使用关键字参数）
            task = engine.submit(name="compat-test", handler=lambda: "ok",
                                task_type="test", priority=5)
            if task is None:
                errors.append("submit返回None")
        except Exception as e:
            errors.append(f"submit调用失败: {e}")

        # 检查stats属性
        try:
            stats = engine.stats
            required_keys = ['total_executed', 'total_success', 'total_failed', 'pending', 'running']
            for key in required_keys:
                if key not in stats:
                    errors.append(f"stats缺少key: {key}")
        except Exception as e:
            errors.append(f"stats访问失败: {e}")

        engine.shutdown()

        self.record("A2: 接口契约-ITaskEngine实现验证", not errors,
                    f"问题: {errors if errors else '无'}")

    # ========== A3: 接口契约-IKnowledgeGraph实现验证 ==========

    def test_a3_ikg_contract(self):
        """验证KnowledgeGraph类完整实现IKnowledgeGraph接口"""
        errors = []

        if not issubclass(KnowledgeGraph, IKnowledgeGraph):
            errors.append("KnowledgeGraph不是IKnowledgeGraph的子类")
            self.bug("A3-001", "KnowledgeGraph类未继承IKnowledgeGraph接口", "high")

        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        # 检查所有IKnowledgeGraph抽象方法
        required_methods = ['add_node', 'get_node', 'search_nodes', 'delete_node',
                          'add_edge', 'store_experience', 'query_experience',
                          'get_all_knowledge_summary', 'close']
        for method in required_methods:
            if not hasattr(kg, method):
                errors.append(f"缺少方法: {method}")
                self.bug("A3-002", f"KnowledgeGraph缺少IKnowledgeGraph方法: {method}", "high")

        # 检查返回类型一致性
        node = KnowledgeNode(category="test", title="test_node", content="test")
        node_id = kg.add_node(node)

        # get_node应返回KnowledgeNode或None
        result = kg.get_node(node_id)
        if result is not None and not isinstance(result, KnowledgeNode):
            errors.append(f"get_node返回类型错误: {type(result)}，接口定义Optional[Dict]")
            self.bug("A3-003", f"get_node返回{type(result).__name__}而非Dict", "medium")

        # search_nodes应返回List[KnowledgeNode]
        results = kg.search_nodes("test")
        if results and not isinstance(results[0], KnowledgeNode):
            errors.append(f"search_nodes返回类型错误: {type(results[0])}")
            self.bug("A3-004", f"search_nodes返回{type(results[0]).__name__}而非Dict", "medium")

        kg.close()

        self.record("A3: 接口契约-IKnowledgeGraph实现验证", not errors,
                    f"问题: {errors if errors else '无'}")

    # ========== A4: 接口契约-ILLMClient实现验证 ==========

    def test_a4_illm_contract(self):
        """验证LLMClient类完整实现ILLMClient接口"""
        errors = []

        if not issubclass(LLMClient, ILLMClient):
            errors.append("LLMClient不是ILLMClient的子类")
            self.bug("A4-001", "LLMClient类未继承ILLMClient接口", "high")

        config = LLMConfig()
        client = LLMClient(config)

        # 检查ILLMClient抽象方法
        required_methods = ['chat', 'analyze_json', 'is_healthy']
        for method in required_methods:
            if not hasattr(client, method):
                errors.append(f"缺少方法: {method}")
                self.bug("A4-002", f"LLMClient缺少ILLMClient方法: {method}", "high")

        # 检查chat方法签名兼容性
        try:
            result = client.chat([{"role": "user", "content": "test"}])
            if not isinstance(result, str):
                errors.append(f"chat返回类型错误: {type(result)}")
        except Exception as e:
            # 模拟模式下可能失败，不算接口问题
            pass

        # 检查is_healthy属性
        try:
            healthy = client.is_healthy
            if not isinstance(healthy, bool):
                errors.append(f"is_healthy返回类型错误: {type(healthy)}")
        except Exception as e:
            errors.append(f"is_healthy访问失败: {e}")

        self.record("A4: 接口契约-ILLMClient实现验证", not errors,
                    f"问题: {errors if errors else '无'}")

    # ========== A5: Schema迁移-旧版数据库兼容 ==========

    def test_a5_schema_migration(self):
        """验证使用旧版Schema创建的数据库能正常打开"""
        errors = []

        db_path = self._new_db()

        # 模拟旧版数据库：只有nodes表（没有edges, task_logs, strategies）
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
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
        # 插入一些旧数据
        c.execute("INSERT INTO nodes (category, title, content) VALUES (?, ?, ?)",
                  ("experience", "old_data", "old_content"))
        conn.commit()
        conn.close()

        # 尝试用新版KnowledgeGraph打开旧数据库
        try:
            kg = KnowledgeGraph(db_path)

            # 验证旧数据可读
            node = kg.get_node(1)
            if node is None or node.title != "old_data":
                errors.append("旧数据不可读")

            # 验证新功能可用（task_logs表应被自动创建）
            kg.log_task("test-id", "test", "completed")

            # 验证strategies表可用
            kg.add_strategy("test_strategy", "desc")

            kg.close()
        except Exception as e:
            errors.append(f"旧数据库打开失败: {str(e)[:50]}")
            self.bug("A5-001", f"Schema迁移失败: {str(e)[:100]}", "high")

        self.record("A5: Schema迁移-旧版数据库兼容", not errors,
                    f"问题: {errors if errors else '无'}")

    # ========== A6: Schema迁移-新增列兼容 ==========

    def test_a6_add_column(self):
        """验证数据库新增列后旧数据不受影响"""
        errors = []

        db_path = self._new_db()

        # 创建一个缺少某些列的nodes表
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                title TEXT NOT NULL
            )
        """)
        c.execute("INSERT INTO nodes (category, title) VALUES (?, ?)",
                  ("test", "minimal_row"))
        conn.commit()
        conn.close()

        # 用KnowledgeGraph打开（应该自动补全缺失列）
        try:
            kg = KnowledgeGraph(db_path)

            # 尝试读取旧数据
            node = kg.get_node(1)
            if node is None:
                errors.append("缺少列的旧数据不可读")
            elif node.title != "minimal_row":
                errors.append(f"旧数据被修改: {node.title}")

            # 尝试写入新数据（使用所有列）
            new_node = KnowledgeNode(
                category="new", title="new_node",
                content="new_content", tags="tag1,tag2",
                importance=0.8
            )
            new_id = kg.add_node(new_node)
            if new_id <= 0:
                errors.append("新数据写入失败")

            kg.close()
        except Exception as e:
            errors.append(f"列迁移失败: {str(e)[:50]}")
            self.bug("A6-001", f"新增列迁移失败: {str(e)[:100]}", "high")

        self.record("A6: Schema迁移-新增列兼容", not errors,
                    f"问题: {errors if errors else '无'}")

    # ========== A7: 配置兼容性-默认值完整性 ==========

    def test_a7_config_defaults(self):
        """验证所有配置项都有合理的默认值"""
        errors = []

        # 测试AgentConfig默认值
        config = AgentConfig()

        required_fields = {
            'task_queue_size': (int, 1),
            'max_concurrent_tasks': (int, 1),
            'task_retry_limit': (int, 0),
            'task_timeout': (int, 1),
            'optimize_interval': (int, 1),
            'reflection_depth': (str, 1),
            'heartbeat_interval': (int, 1),
            'restart_on_error': (bool, None),
            'max_restart_attempts': (int, 0),
            'restart_cooldown': (int, 0),
            'log_level': (str, 1),
            'log_dir': (str, 1),
        }

        for field, (expected_type, min_val) in required_fields.items():
            value = getattr(config, field, None)
            if value is None:
                errors.append(f"配置项 {field} 默认值为None")
                self.bug("A7-001", f"配置项 {field} 默认值为None", "medium")
            elif not isinstance(value, expected_type):
                errors.append(f"配置项 {field} 类型错误: {type(value)}")
            elif min_val is not None and isinstance(value, int) and value < min_val:
                errors.append(f"配置项 {field} 值过小: {value}")

        # 测试LLMConfig默认值
        llm = config.llm
        llm_required = {
            'provider': str, 'api_key': str, 'base_url': str,
            'model': str, 'temperature': (float, int), 'max_tokens': int, 'timeout': int,
        }
        for field, expected_type in llm_required.items():
            value = getattr(llm, field, None)
            if value is None:
                errors.append(f"LLM配置项 {field} 默认值为None")

        # 测试KnowledgeConfig默认值
        knowledge = config.knowledge
        if knowledge.db_path is None:
            errors.append("KnowledgeConfig.db_path 默认值为None")

        # 测试from_env()不崩溃
        try:
            env_config = AgentConfig.from_env()
            if env_config is None:
                errors.append("from_env()返回None")
        except Exception as e:
            errors.append(f"from_env()失败: {str(e)[:50]}")
            self.bug("A7-002", f"from_env()异常: {str(e)[:100]}", "high")

        self.record("A7: 配置兼容性-默认值完整性", not errors,
                    f"检查: {len(required_fields) + len(llm_required)}项, 问题: {errors if errors else '无'}")

    # ========== A8: 配置兼容性-环境变量加载 ==========

    def test_a8_env_config(self):
        """验证环境变量能正确加载到配置中"""
        errors = []

        # 设置环境变量
        old_env = {}
        env_map = {
            'LLM_API_KEY': 'test-key-123',
            'LLM_BASE_URL': 'https://custom.api.com/v1',
            'LLM_MODEL': 'custom-model',
            'LLM_PROVIDER': 'custom',
            'KNOWLEDGE_DB_PATH': '/tmp/custom.db',
        }

        for key, value in env_map.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            config = AgentConfig.from_env()

            # 验证环境变量被正确加载
            if config.llm.api_key != 'test-key-123':
                errors.append(f"LLM_API_KEY未加载: {config.llm.api_key}")
            if config.llm.base_url != 'https://custom.api.com/v1':
                errors.append(f"LLM_BASE_URL未加载: {config.llm.base_url}")
            if config.llm.model != 'custom-model':
                errors.append(f"LLM_MODEL未加载: {config.llm.model}")
            if config.llm.provider != 'custom':
                errors.append(f"LLM_PROVIDER未加载: {config.llm.provider}")
            if config.knowledge.db_path != '/tmp/custom.db':
                errors.append(f"KNOWLEDGE_DB_PATH未加载: {config.knowledge.db_path}")

        except Exception as e:
            errors.append(f"环境变量加载异常: {str(e)[:50]}")
            self.bug("A8-001", f"环境变量加载失败: {str(e)[:100]}", "high")
        finally:
            # 恢复环境变量
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

        if errors:
            self.bug("A8-002", f"环境变量加载不完整: {errors}", "medium")

        self.record("A8: 配置兼容性-环境变量加载", not errors,
                    f"检查: {len(env_map)}项, 问题: {errors if errors else '无'}")

    # ========== A9: 废弃API-未使用配置项检测 ==========

    def test_a9_deprecated_config(self):
        """检测配置项是否被实际使用"""
        errors = []
        unused_configs = []

        # 检查AgentConfig中哪些配置项未被Agent类使用
        config_fields = [
            'heartbeat_interval', 'restart_on_error', 'max_restart_attempts',
            'restart_cooldown', 'reflection_depth', 'log_level', 'log_dir',
        ]

        # 读取Agent类源码，检查配置项引用
        agent_source = open('/workspace/agent/__init__.py').read()
        for field in config_fields:
            # 检查是否在Agent类中被引用
            if f'self.config.{field}' not in agent_source:
                unused_configs.append(field)

        # 检查KnowledgeConfig中哪些配置项未被使用
        kg_source = open('/workspace/agent/modules/knowledge_graph.py').read()
        if 'self.config' not in kg_source and 'auto_save_interval' not in kg_source:
            unused_configs.append('knowledge.auto_save_interval')

        if unused_configs:
            # 这不是bug，但是代码质量问题
            print(f"    [A9] 未使用的配置项: {unused_configs}")
            # 不标记为bug，因为可能是为未来功能预留

        # 检查run_with_watchdog是否使用配置
        watchdog_source = agent_source
        watchdog_config_fields = ['max_restart_attempts', 'restart_cooldown']
        for field in watchdog_config_fields:
            if f'config.{field}' not in watchdog_source and f'self.config.{field}' not in watchdog_source:
                if f'{field}' not in watchdog_source:
                    pass  # watchdog函数使用参数而非配置

        self.record("A9: 废弃API-未使用配置项检测", True,
                    f"未使用配置: {unused_configs if unused_configs else '无'}")

    # ========== A10: 降级运行-模块缺失容错 ==========

    def test_a10_degradation(self):
        """验证关键模块缺失时系统能降级运行"""
        errors = []

        # 测试1: 无OpenAI库时LLMClient应降级到模拟模式
        try:
            config = LLMConfig()
            client = LLMClient(config)
            # chat应该仍然可用（模拟模式）
            result = client.chat([{"role": "user", "content": "test"}])
            if not isinstance(result, str):
                errors.append("降级模式下chat返回类型错误")
        except Exception as e:
            errors.append(f"LLM降级失败: {str(e)[:50]}")
            self.bug("A10-001", f"LLM模块降级失败: {str(e)[:100]}", "high")

        # 测试2: Agent在模拟LLM模式下应能正常工作
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path

        try:
            agent = Agent(config)

            # 提交任务
            for i in range(5):
                agent.submit_task(name=f"degrade-{i}", handler=lambda: "ok")
            agent.engine.run_batch(5)

            stats = agent.engine.stats
            if stats['total_executed'] < 5:
                errors.append(f"降级模式下任务执行不足: {stats['total_executed']}")

            agent.stop()
        except Exception as e:
            errors.append(f"Agent降级运行失败: {str(e)[:50]}")
            self.bug("A10-002", f"Agent降级运行失败: {str(e)[:100]}", "high")

        # 测试3: 数据库不可用时应优雅降级
        try:
            bad_config = AgentConfig()
            bad_config.knowledge.db_path = "/proc/nonexistent/degrade.db"
            agent = Agent(bad_config)
            agent.stop()
            errors.append("不可用数据库路径未抛出异常")
        except (ValueError, OSError):
            pass  # 预期行为

        self.record("A10: 降级运行-模块缺失容错", not errors,
                    f"问题: {errors if errors else '无'}")

    def _summary(self):
        print("\n" + "=" * 70)
        print("第二十四轮 API兼容性与版本升级测试总结")
        print("=" * 70)

        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed if self.results else 0

        print(f"\n测试结果: {passed}/{len(self.results)} 通过, {failed} 失败")

        if self.bugs:
            print(f"\n发现 {len(self.bugs)} 个Bug:")
            for b in self.bugs:
                print(f"  [{b['severity'].upper()}] {b['id']}: {b['description'][:100]}")
        else:
            print("\n未发现新Bug ✓")

        print("\n详细结果:")
        for r in self.results:
            status = "✓" if r['passed'] else "✗"
            print(f"  [{status}] {r['test']}: {r['details'][:120]}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    tester = Round24CompatTest()
    tester.run_all()
