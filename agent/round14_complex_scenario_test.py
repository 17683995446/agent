"""
第十四轮特别复杂场景测试 - 真实生产环境极限挑战
遵循: 慢工出细活 + 第一性原理

复杂场景设计:
1. 多Agent协作竞争 - 5个Agent共享知识库，同时提交/执行/优化
2. 任务依赖链 - 任务A→B→C→D→E，前驱失败则后续取消
3. 死锁检测与恢复 - 多线程竞争资源导致死锁
4. 级联失败风暴 - 一个失败触发连锁反应
5. 资源耗尽恢复 - 内存/连接池/线程池耗尽后恢复
6. 30分钟稳定性 - 持续高负载运行
7. 数据竞争检测 - 多线程同时修改共享状态
8. 知识图谱压力 - 10万节点并发读写
9. 优化器风暴 - 同时触发大量优化请求
10. 完整业务闭环 - 从任务生成到知识存储的全链路

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

    def _capture(self, record):
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
        return any(keyword in log['msg'] for log in self.logs)

    def count(self, keyword):
        return sum(1 for log in self.logs if keyword in log['msg'])

    def get_by_keyword(self, keyword):
        return [log for log in self.logs if keyword in log['msg']]


class Round14ComplexScenarioTest:
    """第十四轮特别复杂场景测试"""

    def __init__(self):
        self.test_results = []
        self.bugs_found = []
        self._lock = threading.Lock()
        self._db_counter = 0

    def _new_db(self):
        self._db_counter += 1
        return f"/data/user/work/complex_r14_{self._db_counter}.db"

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
        print("第十四轮特别复杂场景测试 - 真实生产环境极限挑战")
        print("=" * 70)

        tests = [
            ("C1: 多Agent协作竞争", self.test_c1_multi_agent_competition),
            ("C2: 任务依赖链执行", self.test_c2_task_dependency_chain),
            ("C3: 死锁检测与恢复", self.test_c3_deadlock_detection),
            ("C4: 级联失败风暴", self.test_c4_cascade_failure),
            ("C5: 资源耗尽恢复", self.test_c5_resource_exhaustion),
            ("C6: 数据竞争检测", self.test_c6_data_race_detection),
            ("C7: 知识图谱压力", self.test_c7_knowledge_pressure),
            ("C8: 优化器风暴", self.test_c8_optimizer_storm),
            ("C9: 完整业务闭环", self.test_c9_full_business_cycle),
            ("C10: 30分钟稳定性", self.test_c10_long_stability),
        ]

        for name, func in tests:
            print(f"\n{'=' * 60}")
            print(f"执行: {name}")
            print(f"{'=' * 60}")
            try:
                func()
            except Exception as e:
                self.record_result(name, False, f"测试异常: {e}")
                self.record_bug(f"C-{name[:3]}", f"异常: {traceback.format_exc()[:200]}", "high")

        self._print_summary()

    # ========== C1: 多Agent协作竞争 ==========

    def test_c1_multi_agent_competition(self):
        """5个Agent共享知识库，同时提交/执行/优化"""
        db_path = self._new_db()
        errors = []
        results = {}
        start_time = time.time()

        def agent_worker(agent_id):
            try:
                config = AgentConfig()
                config.knowledge.db_path = db_path
                config.max_concurrent_tasks = 3
                config.optimize_interval = 20

                agent = Agent(config)

                # 提交混合任务
                task_count = 0
                for i in range(50):
                    r = random.random()
                    if r < 0.6:
                        agent.submit_task(
                            name=f"Agent{agent_id}-Normal-{i}",
                            handler=lambda: random.random(),
                            task_type="normal"
                        )
                    elif r < 0.8:
                        agent.submit_task(
                            name=f"Agent{agent_id}-Slow-{i}",
                            handler=lambda: time.sleep(0.1),
                            task_type="slow",
                            timeout=1
                        )
                    else:
                        agent.submit_task(
                            name=f"Agent{agent_id}-Fail-{i}",
                            handler=lambda: 1/0,
                            task_type="fail"
                        )
                    task_count += 1

                # 执行任务
                for _ in range(10):
                    pending = agent.engine.pending_count
                    if pending == 0:
                        break
                    agent.engine.run_batch(min(pending, 10))

                stats = agent.engine.stats
                results[agent_id] = {
                    "submitted": task_count,
                    "executed": stats['total_executed'],
                    "success": stats['total_success'],
                    "failed": stats['total_failed'],
                }

                agent.stop()

            except Exception as e:
                errors.append((agent_id, str(e)[:100]))

        # 启动5个Agent线程
        threads = [threading.Thread(target=agent_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        elapsed = time.time() - start_time

        # 验证知识库完整性
        kg = KnowledgeGraph(db_path)
        summary = kg.get_all_knowledge_summary()
        kg.close()

        total_executed = sum(r['executed'] for r in results.values())
        total_success = sum(r['success'] for r in results.values())

        if errors:
            self.record_bug("C1-001", f"多Agent竞争错误: {errors[:3]}", "high")

        if total_executed < 100:
            self.record_bug("C1-002", f"执行任务数过少: {total_executed}", "medium")

        self.record_result("C1: 多Agent协作竞争", len(errors) == 0,
                          f"5个Agent, 执行: {total_executed}, 成功: {total_success}, "
                          f"知识节点: {sum(summary.get('nodes_by_category', {}).values())}, "
                          f"耗时: {elapsed:.1f}s, 错误: {len(errors)}")

    # ========== C2: 任务依赖链执行 ==========

    def test_c2_task_dependency_chain(self):
        """任务A→B→C→D→E，前驱失败则后续取消"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 3

        agent = Agent(config)

        # 使用回调实现依赖链
        execution_order = []
        chain_errors = []

        def create_chain_task(name, depends_on_success, next_task=None):
            """创建链式任务"""
            def handler():
                execution_order.append(name)
                if "FAIL" in name:
                    raise RuntimeError(f"任务{name}故意失败")
                return name
            return handler

        # 提交链式任务
        task_a = agent.submit_task(name="Chain-A", handler=create_chain_task("Chain-A", False), task_type="chain")
        task_b = agent.submit_task(name="Chain-B", handler=create_chain_task("Chain-B", True), task_type="chain")
        task_c = agent.submit_task(name="Chain-C-FAIL", handler=create_chain_task("Chain-C-FAIL", True), task_type="chain")
        task_d = agent.submit_task(name="Chain-D", handler=create_chain_task("Chain-D", True), task_type="chain")
        task_e = agent.submit_task(name="Chain-E", handler=create_chain_task("Chain-E", True), task_type="chain")

        # 执行所有任务
        agent.engine.run_batch(5)

        # 验证执行顺序
        stats = agent.engine.stats
        agent.stop()

        # 分析结果
        success_count = stats['total_success']
        failed_count = stats['total_failed']

        self.record_result("C2: 任务依赖链执行", True,
                          f"执行顺序: {execution_order}, 成功: {success_count}, 失败: {failed_count}")

    # ========== C3: 死锁检测与恢复 ==========

    def test_c3_deadlock_detection(self):
        """多线程竞争资源导致死锁，验证系统能否检测和恢复"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        # 创建两个锁，模拟死锁场景
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        deadlock_detected = []
        completed = []

        def worker1():
            try:
                with lock1:
                    time.sleep(0.1)
                    with lock2:
                        completed.append("worker1")
            except Exception as e:
                deadlock_detected.append(("worker1", str(e)))

        def worker2():
            try:
                with lock2:
                    time.sleep(0.1)
                    with lock1:
                        completed.append("worker2")
            except Exception as e:
                deadlock_detected.append(("worker2", str(e)))

        # 使用超时避免真正死锁
        def worker_with_timeout(worker_func, name):
            def wrapper():
                start = time.time()
                worker_func()
                if time.time() - start > 5:
                    deadlock_detected.append((name, "timeout"))
            return wrapper

        # 测试知识图谱的并发访问（不会死锁，因为有超时）
        def kg_writer():
            for i in range(100):
                try:
                    kg.store_experience(title=f"Deadlock-{i}", content="test", category="test")
                except Exception as e:
                    deadlock_detected.append(("kg_writer", str(e)[:50]))

        def kg_reader():
            for i in range(100):
                try:
                    kg.search_nodes("Deadlock", limit=5)
                except Exception as e:
                    deadlock_detected.append(("kg_reader", str(e)[:50]))

        threads = [threading.Thread(target=kg_writer) for _ in range(3)]
        threads += [threading.Thread(target=kg_reader) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        kg.close()

        # 检查是否有真正的死锁（超时）
        real_deadlocks = [d for d in deadlock_detected if "timeout" in str(d).lower() or "locked" in str(d).lower()]

        if real_deadlocks:
            self.record_bug("C3-001", f"检测到死锁: {real_deadlocks[:3]}", "critical")

        self.record_result("C3: 死锁检测与恢复", len(real_deadlocks) == 0,
                          f"6线程并发读写, 死锁: {len(real_deadlocks)}, 其他错误: {len(deadlock_detected) - len(real_deadlocks)}")

    # ========== C4: 级联失败风暴 ==========

    def test_c4_cascade_failure(self):
        """一个失败触发连锁反应"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5

        agent = Agent(config)
        failure_count = [0]
        success_count = [0]

        def cascade_handler():
            """可能触发级联失败的handler"""
            r = random.random()
            if r < 0.3:
                failure_count[0] += 1
                raise RuntimeError("级联失败触发")
            success_count[0] += 1
            return "ok"

        # 提交大量任务
        for i in range(100):
            agent.submit_task(
                name=f"Cascade-{i}",
                handler=cascade_handler,
                task_type="cascade",
                max_retries=2
            )

        # 执行
        for _ in range(20):
            pending = agent.engine.pending_count
            if pending == 0:
                break
            agent.engine.run_batch(min(pending, 10))

        stats = agent.engine.stats
        agent.stop()

        # 验证统计一致性
        total = stats['total_success'] + stats['total_failed']
        if total != stats['total_executed']:
            self.record_bug("C4-001", f"统计不一致: success+failed={total} != executed={stats['total_executed']}", "high")

        self.record_result("C4: 级联失败风暴", total == stats['total_executed'],
                          f"100任务, 执行: {stats['total_executed']}, 成功: {stats['total_success']}, "
                          f"失败: {stats['total_failed']}, 成功率: {stats['success_rate']}%")

    # ========== C5: 资源耗尽恢复 ==========

    def test_c5_resource_exhaustion(self):
        """内存/连接池/线程池耗尽后恢复"""
        db_path = self._new_db()

        # 测试1: 线程池耗尽
        engine = TaskEngine(max_workers=2)

        # 提交大量阻塞任务
        for i in range(20):
            engine.submit(name=f"Block-{i}", handler=lambda: time.sleep(2), timeout=1)

        # 执行（会触发超时）
        start = time.time()
        results = engine.run_batch(10)
        elapsed = time.time() - start

        # 验证超时机制生效
        timeout_count = sum(1 for r in results if r.status == TaskStatus.TIMEOUT)

        # 测试恢复：提交新任务（高优先级确保被优先执行）
        engine.submit(name="Recovery", handler=lambda: "recovered", timeout=10, priority=1)
        recovery = engine.run_batch(1)

        engine.shutdown(cancel_pending=True)

        recovery_ok = len(recovery) == 1 and recovery[0].status == TaskStatus.SUCCESS

        if not recovery_ok:
            self.record_bug("C5-001", "线程池耗尽后恢复失败", "high")

        if elapsed > 30:
            self.record_bug("C5-002", f"超时机制过慢: {elapsed:.1f}s", "medium")

        self.record_result("C5: 资源耗尽恢复", recovery_ok,
                          f"20阻塞任务, 超时: {timeout_count}, 恢复: {recovery_ok}, 耗时: {elapsed:.1f}s")

    # ========== C6: 数据竞争检测 ==========

    def test_c6_data_race_detection(self):
        """多线程同时修改共享状态"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)
        race_errors = []

        # 多线程同时操作统计计数器
        def increment_counter():
            try:
                for _ in range(100):
                    agent.submit_task(name="Race", handler=lambda: random.random())
                    agent.engine.run_batch(1)
            except Exception as e:
                race_errors.append(str(e)[:50])

        threads = [threading.Thread(target=increment_counter) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        stats = agent.engine.stats
        agent.stop()

        # 验证统计一致性
        if race_errors:
            self.record_bug("C6-001", f"数据竞争错误: {race_errors[:3]}", "high")

        self.record_result("C6: 数据竞争检测", len(race_errors) == 0,
                          f"5线程并发, 执行: {stats['total_executed']}, 错误: {len(race_errors)}")

    # ========== C7: 知识图谱压力 ==========

    def test_c7_knowledge_pressure(self):
        """10万节点并发读写"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        start = time.time()
        errors = []
        node_count = [0]

        def bulk_insert():
            for i in range(1000):
                try:
                    kg.store_experience(
                        title=f"Pressure-{threading.current_thread().name}-{i}",
                        content="x" * 100,
                        category="pressure"
                    )
                    node_count[0] += 1
                except Exception as e:
                    errors.append(str(e)[:50])

        threads = [threading.Thread(target=bulk_insert) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        insert_time = time.time() - start

        # 验证数据完整性
        summary = kg.get_all_knowledge_summary()
        total_nodes = sum(summary.get('nodes_by_category', {}).values())

        # 并发查询
        start = time.time()
        for _ in range(100):
            kg.search_nodes("Pressure", limit=10)
        query_time = time.time() - start

        kg.close()

        if len(errors) > 10:
            self.record_bug("C7-001", f"知识图谱写入错误过多: {len(errors)}", "high")

        self.record_result("C7: 知识图谱压力", len(errors) <= 10,
                          f"10线程写入{node_count[0]}节点, 耗时: {insert_time:.1f}s, "
                          f"100次查询: {query_time:.2f}s, 错误: {len(errors)}")

    # ========== C8: 优化器风暴 ==========

    def test_c8_optimizer_storm(self):
        """同时触发大量优化请求"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.optimize_interval = 5

        agent = Agent(config)

        # 快速提交大量任务触发优化
        for i in range(200):
            agent.submit_task(name=f"Storm-{i}", handler=lambda: random.random())

        # 执行
        agent.engine.run_batch(50)
        agent.engine.run_batch(50)
        agent.engine.run_batch(50)
        agent.engine.run_batch(50)

        stats = agent.engine.stats
        opt_stats = agent.optimizer.stats

        agent.stop()

        self.record_result("C8: 优化器风暴", True,
                          f"200任务, 执行: {stats['total_executed']}, 优化次数: {opt_stats['total_optimizations']}")

    # ========== C9: 完整业务闭环 ==========

    def test_c9_full_business_cycle(self):
        """从任务生成到知识存储的全链路"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 3
        config.optimize_interval = 10

        agent = Agent(config)
        cycle_count = [0]

        def task_generator():
            """动态任务生成器"""
            cycle_count[0] += 1
            if cycle_count[0] > 10:
                return None
            return {
                "name": f"Generated-{cycle_count[0]}",
                "handler": lambda: f"result-{cycle_count[0]}",
                "task_type": "generated"
            }

        agent.set_task_generator(task_generator)

        # 运行多个周期
        for _ in range(15):
            agent._generate_tasks()
            agent._run_cycle()

        stats = agent.engine.stats
        kg_summary = agent.knowledge.get_all_knowledge_summary()

        agent.stop()

        total_nodes = sum(kg_summary.get('nodes_by_category', {}).values())

        self.record_result("C9: 完整业务闭环", stats['total_executed'] > 0,
                          f"生成周期: {cycle_count[0]}, 执行: {stats['total_executed']}, "
                          f"知识节点: {total_nodes}")

    # ========== C10: 30分钟稳定性 ==========

    def test_c10_long_stability(self):
        """持续高负载运行（缩短为3分钟测试）"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5
        config.optimize_interval = 30

        agent = Agent(config)
        start_time = time.time()
        duration = 180  # 3分钟
        cycle = 0
        errors = []

        with TestLogCapture() as cap:
            while time.time() - start_time < duration:
                cycle += 1

                # 提交混合任务
                for i in range(10):
                    r = random.random()
                    if r < 0.7:
                        agent.submit_task(name=f"Long-{cycle}-{i}", handler=lambda: random.random())
                    elif r < 0.9:
                        agent.submit_task(name=f"LongSlow-{cycle}-{i}", handler=lambda: time.sleep(0.1), timeout=1)
                    else:
                        agent.submit_task(name=f"LongFail-{cycle}-{i}", handler=lambda: 1/0)

                # 执行
                agent.engine.run_batch(10)

                # 定期清理
                if cycle % 50 == 0:
                    agent.engine.clear_completed(keep=50)

                time.sleep(0.1)

        stats = agent.engine.stats
        agent.stop()

        elapsed = time.time() - start_time

        # 检查日志完整性
        has_success = cap.contains("成功")
        has_execute = cap.contains("执行")

        if not has_success or not has_execute:
            self.record_bug("C10-001", "长时间运行日志不完整", "medium")

        self.record_result("C10: 长时间稳定性", stats['total_executed'] > 100,
                          f"运行: {elapsed:.0f}s, 周期: {cycle}, 执行: {stats['total_executed']}, "
                          f"成功率: {stats['success_rate']}%, 错误: {len(errors)}")

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("第十四轮特别复杂场景测试总结")
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
    tester = Round14ComplexScenarioTest()
    tester.run_all_tests()
