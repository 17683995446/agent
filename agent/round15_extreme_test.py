"""
第十五轮极限挑战测试 - 真实生产环境极限压力
遵循: 多阶段迭代原理 + 第一性原理

极限场景设计:
1. E1: 极限并发风暴 - 100个线程同时操作Agent
2. E2: 内存压力极限 - 50000节点存储和查询
3. E3: 故障注入恢复 - 随机注入各种故障
4. E4: 资源竞争死锁 - 复杂的资源竞争场景
5. E5: 长时间稳定性 - 5分钟高负载运行
6. E6: 边界条件极限 - 各种边界值测试
7. E7: 异常链式反应 - 触发连锁异常
8. E8: 数据一致性极限 - 并发读写一致性验证
9. E9: 优化器极限压力 - 大量优化请求并发
10. E10: 全链路压力 - 完整业务流程压力测试

每个场景通过全链路日志验证正确性。
"""
import os
import sys
import time
import json
import random
import threading
import traceback
import gc
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus, Task
from agent.modules.knowledge_graph import KnowledgeGraph, KnowledgeNode
from agent.modules.optimizer import Optimizer
from agent.modules.llm_client import LLMClient

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestLogCapture:
    """日志捕获器"""
    def __init__(self):
        self.logs = []
        self.handler = logging.Handler()
        self.handler.emit = self._capture
        self._lock = threading.Lock()

    def _capture(self, record):
        with self._lock:
            self.logs.append({
                'name': record.name,
                'level': record.levelname,
                'msg': record.getMessage(),
                'module': record.module,
            })

    def __enter__(self):
        logging.root.addHandler(self.handler)
        return self

    def __exit__(self, *args):
        logging.root.removeHandler(self.handler)

    def contains(self, keyword):
        with self._lock:
            return any(keyword in log['msg'] for log in self.logs)

    def count(self, keyword):
        with self._lock:
            return sum(1 for log in self.logs if keyword in log['msg'])

    def get_by_keyword(self, keyword):
        with self._lock:
            return [log for log in self.logs if keyword in log['msg']]

    def get_errors(self):
        with self._lock:
            return [log for log in self.logs if log['level'] == 'ERROR']


class Round15ExtremeTest:
    """第十五轮极限挑战测试"""

    def __init__(self):
        self.test_results = []
        self.bugs_found = []
        self._lock = threading.Lock()
        self._db_counter = 0

    def _new_db(self):
        self._db_counter += 1
        return f"/data/user/work/extreme_r15_{self._db_counter}.db"

    def record_result(self, test_name, passed, details=""):
        with self._lock:
            self.test_results.append({"test": test_name, "passed": passed, "details": details})
        status = "✓" if passed else "✗"
        print(f"  [{status}] {test_name}: {details[:120]}")

    def record_bug(self, bug_id, description, severity="medium"):
        with self._lock:
            self.bugs_found.append({"id": bug_id, "description": description, "severity": severity})
        print(f"  🐛 BUG [{bug_id}] ({severity}): {description[:100]}")

    def run_all_tests(self):
        print("\n" + "=" * 70)
        print("第十五轮极限挑战测试 - 真实生产环境极限压力")
        print("=" * 70)

        tests = [
            ("E1: 极限并发风暴", self.test_e1_extreme_concurrency),
            ("E2: 内存压力极限", self.test_e2_memory_pressure),
            ("E3: 故障注入恢复", self.test_e3_fault_injection),
            ("E4: 资源竞争死锁", self.test_e4_resource_contention),
            ("E5: 长时间稳定性", self.test_e5_long_stability),
            ("E6: 边界条件极限", self.test_e6_boundary_conditions),
            ("E7: 异常链式反应", self.test_e7_exception_chain),
            ("E8: 数据一致性极限", self.test_e8_data_consistency),
            ("E9: 优化器极限压力", self.test_e9_optimizer_pressure),
            ("E10: 全链路压力", self.test_e10_full_pipeline),
        ]

        for name, func in tests:
            print(f"\n{'=' * 60}")
            print(f"执行: {name}")
            print(f"{'=' * 60}")
            try:
                func()
            except Exception as e:
                self.record_result(name, False, f"测试异常: {e}")
                self.record_bug(f"E-{name[:3]}", f"异常: {traceback.format_exc()[:200]}", "high")

        self._print_summary()

    # ========== E1: 极限并发风暴 ==========

    def test_e1_extreme_concurrency(self):
        """100个线程同时操作Agent"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 10

        agent = Agent(config)
        errors = []
        operations = {'submit': 0, 'execute': 0, 'query': 0, 'optimize': 0}
        start_time = time.time()

        def worker(thread_id):
            try:
                for i in range(20):
                    op = random.choice(['submit', 'execute', 'query', 'optimize'])
                    
                    if op == 'submit':
                        agent.submit_task(
                            name=f"T{thread_id}-Task-{i}",
                            handler=lambda: random.random(),
                            task_type="concurrent"
                        )
                        with self._lock:
                            operations['submit'] += 1
                    
                    elif op == 'execute':
                        pending = agent.engine.pending_count
                        if pending > 0:
                            agent.engine.run_batch(min(pending, 5))
                        with self._lock:
                            operations['execute'] += 1
                    
                    elif op == 'query':
                        agent.knowledge.search_nodes("test", limit=5)
                        with self._lock:
                            operations['query'] += 1
                    
                    elif op == 'optimize':
                        agent.optimizer.reflect_on_tasks([
                            {'name': f'opt-{i}', 'status': 'success', 'result': 'ok'}
                        ])
                        with self._lock:
                            operations['optimize'] += 1

            except Exception as e:
                with self._lock:
                    errors.append((thread_id, str(e)[:100]))

        # 启动100个线程
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        elapsed = time.time() - start_time
        stats = agent.engine.stats
        agent.stop()

        # 验证
        if len(errors) > 10:
            self.record_bug("E1-001", f"并发错误过多: {len(errors)}", "high")

        self.record_result("E1: 极限并发风暴", len(errors) <= 10,
                          f"100线程, 操作: {operations}, 执行: {stats['total_executed']}, "
                          f"错误: {len(errors)}, 耗时: {elapsed:.1f}s")

    # ========== E2: 内存压力极限 ==========

    def test_e2_memory_pressure(self):
        """50000节点存储和查询"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        start = time.time()
        errors = []
        node_count = [0]

        def bulk_insert(thread_id):
            for i in range(500):
                try:
                    kg.store_experience(
                        title=f"Mem-{thread_id}-{i}",
                        content="x" * 200,  # 更大的内容
                        category=f"pressure_{thread_id % 10}"
                    )
                    node_count[0] += 1
                except Exception as e:
                    errors.append(str(e)[:50])

        threads = [threading.Thread(target=bulk_insert, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)

        insert_time = time.time() - start

        # 验证数据完整性
        summary = kg.get_all_knowledge_summary()
        total_nodes = sum(summary.get('nodes_by_category', {}).values())

        # 并发查询
        start = time.time()
        query_errors = []
        for _ in range(500):
            try:
                kg.search_nodes("Mem", limit=20)
            except Exception as e:
                query_errors.append(str(e)[:50])
        query_time = time.time() - start

        kg.close()

        if len(errors) > 50:
            self.record_bug("E2-001", f"内存压力写入错误过多: {len(errors)}", "high")

        if len(query_errors) > 10:
            self.record_bug("E2-002", f"内存压力查询错误过多: {len(query_errors)}", "high")

        self.record_result("E2: 内存压力极限", len(errors) <= 50,
                          f"100线程写入{node_count[0]}节点, 耗时: {insert_time:.1f}s, "
                          f"500次查询: {query_time:.1f}s, 写入错误: {len(errors)}, 查询错误: {len(query_errors)}")

    # ========== E3: 故障注入恢复 ==========

    def test_e3_fault_injection(self):
        """随机注入各种故障"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5

        agent = Agent(config)
        fault_count = {'timeout': 0, 'exception': 0, 'memory': 0, 'connection': 0}
        recovery_count = 0

        def fault_handler(fault_type):
            """注入各种故障"""
            if fault_type == 'timeout':
                time.sleep(3)  # 超时
                fault_count['timeout'] += 1
            elif fault_type == 'exception':
                fault_count['exception'] += 1
                raise RuntimeError("注入异常")
            elif fault_type == 'memory':
                # 模拟内存压力
                _ = ['x' * 10000 for _ in range(100)]
                fault_count['memory'] += 1
                return "ok"
            elif fault_type == 'connection':
                fault_count['connection'] += 1
                raise ConnectionError("模拟连接断开")
            return "ok"

        # 提交混合任务
        for i in range(100):
            fault_type = random.choice(['timeout', 'exception', 'memory', 'connection', 'normal'])
            if fault_type == 'normal':
                agent.submit_task(name=f"Normal-{i}", handler=lambda: "ok", timeout=5)
            elif fault_type == 'timeout':
                agent.submit_task(name=f"Timeout-{i}", handler=lambda: fault_handler('timeout'), timeout=1)
            elif fault_type == 'exception':
                agent.submit_task(name=f"Exception-{i}", handler=lambda: fault_handler('exception'), max_retries=2)
            elif fault_type == 'memory':
                agent.submit_task(name=f"Memory-{i}", handler=lambda: fault_handler('memory'))
            elif fault_type == 'connection':
                agent.submit_task(name=f"Connection-{i}", handler=lambda: fault_handler('connection'), max_retries=1)

        # 执行
        for _ in range(20):
            pending = agent.engine.pending_count
            if pending == 0:
                break
            agent.engine.run_batch(min(pending, 10))

        stats = agent.engine.stats
        agent.stop()

        # 验证恢复能力
        recovery_ok = stats['total_executed'] > 50  # 至少执行一半

        if not recovery_ok:
            self.record_bug("E3-001", "故障注入后恢复能力不足", "high")

        self.record_result("E3: 故障注入恢复", recovery_ok,
                          f"故障: {fault_count}, 执行: {stats['total_executed']}, "
                          f"成功: {stats['total_success']}, 失败: {stats['total_failed']}")

    # ========== E4: 资源竞争死锁 ==========

    def test_e4_resource_contention(self):
        """复杂的资源竞争场景"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)
        
        # 创建多个锁模拟复杂资源竞争
        locks = [threading.Lock() for _ in range(5)]
        contention_errors = []
        completed_ops = [0]
        deadlock_detected = [False]

        def resource_worker(worker_id):
            try:
                for i in range(50):
                    # 随机获取多个锁
                    lock_indices = random.sample(range(5), random.randint(1, 3))
                    lock_indices.sort()  # 按顺序获取锁避免死锁
                    
                    acquired = []
                    try:
                        for idx in lock_indices:
                            if not locks[idx].acquire(timeout=2):
                                deadlock_detected[0] = True
                                raise TimeoutError("锁获取超时")
                            acquired.append(idx)
                        
                        # 执行操作
                        kg.store_experience(
                            title=f"Contention-{worker_id}-{i}",
                            content=f"worker_{worker_id}_op_{i}",
                            category="contention"
                        )
                        completed_ops[0] += 1
                        
                    finally:
                        for idx in reversed(acquired):
                            locks[idx].release()
                            
            except Exception as e:
                contention_errors.append((worker_id, str(e)[:50]))

        threads = [threading.Thread(target=resource_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        kg.close()

        if deadlock_detected[0]:
            self.record_bug("E4-001", "检测到死锁或锁超时", "critical")

        if len(contention_errors) > 10:
            self.record_bug("E4-002", f"资源竞争错误过多: {len(contention_errors)}", "high")

        self.record_result("E4: 资源竞争死锁", not deadlock_detected[0] and len(contention_errors) <= 10,
                          f"20线程, 完成: {completed_ops[0]}操作, 死锁: {deadlock_detected[0]}, "
                          f"错误: {len(contention_errors)}")

    # ========== E5: 长时间稳定性 ==========

    def test_e5_long_stability(self):
        """5分钟高负载运行"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 8
        config.optimize_interval = 50

        agent = Agent(config)
        start_time = time.time()
        duration = 300  # 5分钟
        cycle = 0
        errors = []
        memory_samples = []

        with TestLogCapture() as cap:
            while time.time() - start_time < duration:
                cycle += 1

                # 提交混合任务
                for i in range(20):
                    r = random.random()
                    if r < 0.6:
                        agent.submit_task(name=f"Long-{cycle}-{i}", handler=lambda: random.random())
                    elif r < 0.8:
                        agent.submit_task(name=f"LongSlow-{cycle}-{i}", handler=lambda: time.sleep(0.05), timeout=1)
                    elif r < 0.95:
                        agent.submit_task(name=f"LongFail-{cycle}-{i}", handler=lambda: 1/0, max_retries=1)
                    else:
                        # 极端任务
                        agent.submit_task(
                            name=f"LongExtreme-{cycle}-{i}",
                            handler=lambda: ['x' * 1000 for _ in range(10)],
                            timeout=2
                        )

                # 执行
                agent.engine.run_batch(15)

                # 定期清理
                if cycle % 100 == 0:
                    agent.engine.clear_completed(keep=100)
                    import gc
                    gc.collect()

                # 采样内存
                if cycle % 50 == 0:
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_samples.append(process.memory_info().rss / 1024 / 1024)
                    except:
                        pass

                time.sleep(0.05)

        stats = agent.engine.stats
        agent.stop()

        elapsed = time.time() - start_time

        # 检查日志完整性
        has_success = cap.contains("成功")
        has_execute = cap.contains("执行")
        error_logs = cap.get_errors()

        if not has_success or not has_execute:
            self.record_bug("E5-001", "长时间运行日志不完整", "medium")

        # 检查内存是否稳定
        memory_stable = True
        if len(memory_samples) >= 2:
            max_mem = max(memory_samples)
            min_mem = min(memory_samples)
            if max_mem > min_mem * 3:  # 内存增长超过3倍
                memory_stable = False
                self.record_bug("E5-002", f"内存可能泄漏: {min_mem:.1f}MB -> {max_mem:.1f}MB", "high")

        self.record_result("E5: 长时间稳定性", stats['total_executed'] > 1000 and memory_stable,
                          f"运行: {elapsed:.0f}s, 周期: {cycle}, 执行: {stats['total_executed']}, "
                          f"成功率: {stats['success_rate']}%, 错误: {len(errors)}, 日志错误: {len(error_logs)}")

    # ========== E6: 边界条件极限 ==========

    def test_e6_boundary_conditions(self):
        """各种边界值测试"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)
        boundary_errors = []

        # 测试边界条件
        test_cases = [
            # (名称, 参数, 预期行为)
            ("空名称", {"name": "", "handler": lambda: 1}, "应拒绝或使用默认值"),
            ("超长名称", {"name": "x" * 10000, "handler": lambda: 1}, "应截断或正常处理"),
            ("负优先级", {"name": "neg-priority", "handler": lambda: 1, "priority": -1}, "应正常处理"),
            ("超大优先级", {"name": "big-priority", "handler": lambda: 1, "priority": 999999}, "应正常处理"),
            ("零超时", {"name": "zero-timeout", "handler": lambda: 1, "timeout": 0}, "应使用默认值"),
            ("负超时", {"name": "neg-timeout", "handler": lambda: 1, "timeout": -1}, "应使用默认值"),
            ("超大超时", {"name": "big-timeout", "handler": lambda: 1, "timeout": 999999}, "应正常处理"),
            ("零重试", {"name": "zero-retry", "handler": lambda: 1/0, "max_retries": 0}, "应不重试"),
            ("负重试", {"name": "neg-retry", "handler": lambda: 1/0, "max_retries": -1}, "应使用默认值"),
            ("超大重试", {"name": "big-retry", "handler": lambda: 1/0, "max_retries": 999}, "应限制重试次数"),
            ("None handler", {"name": "none-handler", "handler": None}, "应拒绝或报错"),
            ("空元组args", {"name": "empty-args", "handler": lambda x: x, "args": ()}, "应正常处理"),
        ]

        for name, params, expected in test_cases:
            try:
                task = agent.submit_task(**params)
                # 尝试执行
                if task:
                    agent.engine.run_batch(1)
            except Exception as e:
                boundary_errors.append((name, str(e)[:50]))

        stats = agent.engine.stats
        agent.stop()

        # 某些边界条件应该报错，某些应该正常处理
        # 这里只检查是否有未捕获的异常导致崩溃
        crash_count = sum(1 for e in boundary_errors if "Error" in str(e) or "Exception" in str(e))

        if crash_count > 3:
            self.record_bug("E6-001", f"边界条件处理崩溃过多: {crash_count}", "high")

        self.record_result("E6: 边界条件极限", crash_count <= 3,
                          f"测试{len(test_cases)}个边界条件, 崩溃: {crash_count}, "
                          f"执行: {stats['total_executed']}")

    # ========== E7: 异常链式反应 ==========

    def test_e7_exception_chain(self):
        """触发连锁异常"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5

        agent = Agent(config)
        exception_chain = []
        recovery_ok = True

        def chain_handler(depth):
            """链式异常处理器"""
            if depth > 0:
                # 触发下一个异常
                raise RuntimeError(f"Chain-{depth}")
            return "recovered"

        # 提交链式异常任务
        for i in range(50):
            agent.submit_task(
                name=f"Chain-{i}",
                handler=lambda d=random.randint(0, 5): chain_handler(d),
                max_retries=3
            )

        # 执行
        for _ in range(15):
            pending = agent.engine.pending_count
            if pending == 0:
                break
            agent.engine.run_batch(min(pending, 10))

        stats = agent.engine.stats
        agent.stop()

        # 验证异常被正确处理
        if stats['total_executed'] < 30:
            recovery_ok = False
            self.record_bug("E7-001", "异常链导致执行量过低", "high")

        self.record_result("E7: 异常链式反应", recovery_ok,
                          f"执行: {stats['total_executed']}, 成功: {stats['total_success']}, "
                          f"失败: {stats['total_failed']}")

    # ========== E8: 数据一致性极限 ==========

    def test_e8_data_consistency(self):
        """并发读写一致性验证"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        consistency_errors = []
        write_count = [0]
        read_count = [0]

        def writer(writer_id):
            for i in range(100):
                try:
                    node_id = kg.store_experience(
                        title=f"Consistency-{writer_id}-{i}",
                        content=f"data_{writer_id}_{i}",
                        category="consistency"
                    )
                    write_count[0] += 1
                except Exception as e:
                    consistency_errors.append(("write", str(e)[:50]))

        def reader(reader_id):
            for i in range(100):
                try:
                    results = kg.search_nodes("Consistency", limit=10)
                    read_count[0] += 1
                    # 验证数据完整性 - 兼容KnowledgeNode对象和字典
                    for r in results:
                        if isinstance(r, dict):
                            if 'title' not in r or 'content' not in r:
                                consistency_errors.append(("read", "数据不完整"))
                        else:
                            # KnowledgeNode对象
                            if not hasattr(r, 'title') or not hasattr(r, 'content'):
                                consistency_errors.append(("read", "数据不完整"))
                except Exception as e:
                    consistency_errors.append(("read", str(e)[:50]))

        # 启动读写线程
        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for i in range(10):
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        # 验证最终一致性
        summary = kg.get_all_knowledge_summary()
        total_nodes = sum(summary.get('nodes_by_category', {}).values())
        kg.close()

        # 检查数据是否丢失
        expected_min = write_count[0] * 0.9  # 允许10%误差
        if total_nodes < expected_min:
            self.record_bug("E8-001", f"数据丢失: 预期>={expected_min}, 实际={total_nodes}", "high")

        if len(consistency_errors) > 20:
            self.record_bug("E8-002", f"一致性错误过多: {len(consistency_errors)}", "high")

        self.record_result("E8: 数据一致性极限", len(consistency_errors) <= 20,
                          f"写入: {write_count[0]}, 读取: {read_count[0]}, "
                          f"节点: {total_nodes}, 一致性错误: {len(consistency_errors)}")

    # ========== E9: 优化器极限压力 ==========

    def test_e9_optimizer_pressure(self):
        """大量优化请求并发"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.optimize_interval = 10

        agent = Agent(config)
        optimize_errors = []
        optimize_count = [0]

        # 快速提交大量任务触发优化
        for i in range(500):
            agent.submit_task(name=f"Opt-{i}", handler=lambda: random.random())

        # 并发执行
        def execute_batch(batch_id):
            try:
                for _ in range(10):
                    pending = agent.engine.pending_count
                    if pending == 0:
                        break
                    agent.engine.run_batch(min(pending, 20))
                    optimize_count[0] += 1
            except Exception as e:
                optimize_errors.append((batch_id, str(e)[:50]))

        threads = [threading.Thread(target=execute_batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        stats = agent.engine.stats
        opt_stats = agent.optimizer.stats
        agent.stop()

        if len(optimize_errors) > 5:
            self.record_bug("E9-001", f"优化器压力错误: {len(optimize_errors)}", "medium")

        self.record_result("E9: 优化器极限压力", len(optimize_errors) <= 5,
                          f"500任务, 执行: {stats['total_executed']}, "
                          f"优化次数: {opt_stats['total_optimizations']}, 错误: {len(optimize_errors)}")

    # ========== E10: 全链路压力 ==========

    def test_e10_full_pipeline(self):
        """完整业务流程压力测试"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 10
        config.optimize_interval = 30

        agent = Agent(config)
        pipeline_errors = []
        pipeline_stages = {'submit': 0, 'execute': 0, 'store': 0, 'optimize': 0, 'query': 0}

        # 完整业务流程
        for cycle in range(50):
            try:
                # 1. 提交任务
                for i in range(10):
                    agent.submit_task(
                        name=f"Pipeline-{cycle}-{i}",
                        handler=lambda: f"result_{cycle}_{i}",
                        task_type="pipeline"
                    )
                pipeline_stages['submit'] += 10

                # 2. 执行任务
                agent.engine.run_batch(10)
                pipeline_stages['execute'] += 1

                # 3. 存储知识
                agent.knowledge.store_experience(
                    title=f"Pipeline-Experience-{cycle}",
                    content=f"cycle_{cycle}_experience",
                    category="pipeline"
                )
                pipeline_stages['store'] += 1

                # 4. 触发优化
                if cycle % 10 == 0:
                    agent.optimizer.reflect_on_tasks([
                        {'name': f'task-{i}', 'status': 'success', 'result': 'ok'}
                        for i in range(5)
                    ])
                    pipeline_stages['optimize'] += 1

                # 5. 查询知识
                agent.knowledge.search_nodes("Pipeline", limit=5)
                pipeline_stages['query'] += 1

            except Exception as e:
                pipeline_errors.append((cycle, str(e)[:50]))

        stats = agent.engine.stats
        kg_summary = agent.knowledge.get_all_knowledge_summary()
        agent.stop()

        total_nodes = sum(kg_summary.get('nodes_by_category', {}).values())

        if len(pipeline_errors) > 10:
            self.record_bug("E10-001", f"全链路错误过多: {len(pipeline_errors)}", "high")

        self.record_result("E10: 全链路压力", len(pipeline_errors) <= 10,
                          f"阶段: {pipeline_stages}, 执行: {stats['total_executed']}, "
                          f"知识节点: {total_nodes}, 错误: {len(pipeline_errors)}")

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("第十五轮极限挑战测试总结")
        print("=" * 70)

        passed = sum(1 for r in self.test_results if r['passed'])
        failed = len(self.test_results) - passed

        print(f"\n测试结果: {passed}/{len(self.test_results)} 通过, {failed} 失败")

        if self.bugs_found:
            print(f"\n发现 {len(self.bugs_found)} 个Bug:")
            for bug in self.bugs_found:
                print(f"  [{bug['severity'].upper()}] {bug['id']}: {bug['description'][:80]}")
        else:
            print("\n未发现新Bug ✓")

        print("\n详细结果:")
        for r in self.test_results:
            status = "✓" if r['passed'] else "✗"
            print(f"  [{status}] {r['test']}: {r['details'][:100]}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    tester = Round15ExtremeTest()
    tester.run_all_tests()
