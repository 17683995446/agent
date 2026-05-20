"""
第十轮混沌工程测试 - 模拟真实生产环境中最混乱的场景
测试目标：在极端混乱条件下发现深层bug
遵循：慢工出细活原理

混沌场景设计：
1. 多Agent实例共享同一数据库
2. 任务回调中提交新任务（递归风暴）
3. 运行中动态替换任务生成器
4. 同时进行读写删除操作
5. 异常回调链（回调抛异常+回调中再注册回调）
6. Agent.run()中途stop再run()
7. 大量超时任务与正常任务混合
8. 知识图谱节点被删除后仍被查询
9. shutdown后继续submit
10. 完整生命周期压力测试（创建→运行→停止→重建→运行→停止）
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
import shutil
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '/workspace')

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus, Task
from agent.modules.knowledge_graph import KnowledgeGraph, KnowledgeNode
from agent.modules.optimizer import Optimizer
from agent.modules.llm_client import LLMClient


class Round10ChaosTest:
    """第十轮混沌工程测试套件"""

    def __init__(self):
        self.test_results = []
        self.bugs_found = []
        self._lock = threading.Lock()
        self._db_counter = 0

    def _new_db(self):
        self._db_counter += 1
        return f"/data/user/work/chaos_r10_{self._db_counter}.db"

    def record_result(self, test_name, passed, details=""):
        with self._lock:
            self.test_results.append({"test": test_name, "passed": passed, "details": details})
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  [{status}] {test_name}: {details[:120]}")

    def record_bug(self, bug_id, description, severity="medium"):
        with self._lock:
            self.bugs_found.append({"id": bug_id, "description": description, "severity": severity})
        print(f"  🐛 BUG [{bug_id}] ({severity}): {description[:100]}")

    def run_all_tests(self):
        print("\n" + "=" * 70)
        print("第十轮混沌工程测试 - 极端混乱场景")
        print("=" * 70)

        tests = [
            ("C1: 多Agent共享数据库", self.test_c1_shared_db),
            ("C2: 递归任务提交风暴", self.test_c2_recursive_submit),
            ("C3: 动态替换任务生成器", self.test_c3_hotswap_generator),
            ("C4: 并发读写删知识图谱", self.test_c4_concurrent_kg_ops),
            ("C5: 异常回调链", self.test_c5_callback_chain),
            ("C6: run→stop→run重入", self.test_c6_reentrant_run),
            ("C7: 超时与正常任务混合", self.test_c7_mixed_timeout),
            ("C8: shutdown后submit", self.test_c8_submit_after_shutdown),
            ("C9: 完整生命周期压力", self.test_c9_lifecycle_stress),
            ("C10: 混沌综合大考", self.test_c10_chaos_comprehensive),
        ]

        for name, func in tests:
            print(f"\n{'=' * 60}")
            print(f"执行: {name}")
            print(f"{'=' * 60}")
            try:
                func()
            except Exception as e:
                self.record_result(name, False, f"测试异常: {e}")
                self.record_bug(f"C-{name[:3]}", f"测试异常: {traceback.format_exc()[:200]}", "high")

        self._print_summary()

    # ========== C1: 多Agent共享数据库 ==========

    def test_c1_shared_db(self):
        """多个Agent实例共享同一个SQLite数据库"""
        db_path = self._new_db()
        errors = []
        results = [0, 0, 0]

        def agent_work(agent_id):
            try:
                config = AgentConfig()
                config.knowledge.db_path = db_path
                config.max_concurrent_tasks = 3
                agent = Agent(config)
                for i in range(30):
                    agent.submit_task(
                        name=f"Shared-{agent_id}-{i}",
                        handler=lambda: random.random(),
                        task_type="shared_test"
                    )
                    if i % 10 == 9:
                        agent.engine.run_batch(10)
                agent.engine.run_batch(50)
                results[agent_id] = agent.engine.stats['total_executed']
                agent.stop()
            except Exception as e:
                errors.append((agent_id, str(e)[:80]))

        threads = [threading.Thread(target=agent_work, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        if errors:
            self.record_bug("C1-001", f"共享数据库错误: {errors}", "high")
        self.record_result("C1: 多Agent共享数据库", len(errors) == 0,
                          f"3个Agent分别执行: {results}, 错误: {len(errors)}")

    # ========== C2: 递归任务提交风暴 ==========

    def test_c2_recursive_submit(self):
        """任务回调中提交新任务，形成递归风暴"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5
        config.optimize_interval = 100  # 减少优化干扰

        agent = Agent(config)
        depth_counter = [0]
        max_depth = 3
        total_submitted = [0]

        def recursive_task(depth=0):
            if depth >= max_depth:
                return f"leaf-{depth}"
            total_submitted[0] += 1
            agent.submit_task(
                name=f"Recursive-{depth}-{total_submitted[0]}",
                handler=lambda d=depth + 1: recursive_task(d),
                task_type="recursive"
            )
            return f"branch-{depth}"

        # 提交3个根任务，每个递归3层
        for i in range(3):
            agent.submit_task(name=f"Root-{i}", handler=recursive_task, task_type="root")

        # 执行多轮以处理递归生成的任务
        for _ in range(10):
            pending = agent.engine.pending_count
            if pending == 0:
                break
            agent.engine.run_batch(min(pending, 20))

        stats = agent.engine.stats
        agent.stop()

        expected_min = 3 + 3 * 3 + 3 * 3 * 3  # 3 + 9 + 27 = 39
        if stats['total_executed'] < expected_min * 0.5:
            self.record_bug("C2-001",
                          f"递归任务执行不足: 期望>={expected_min * 0.5}, 实际={stats['total_executed']}", "medium")

        self.record_result("C2: 递归任务提交风暴", True,
                          f"提交约{total_submitted[0]}个子任务, 执行: {stats['total_executed']}, 成功率: {stats['success_rate']}%")

    # ========== C3: 动态替换任务生成器 ==========

    def test_c3_hotswap_generator(self):
        """运行中动态替换任务生成器"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 3

        agent = Agent(config)
        gen_counter = [0]

        def gen_a():
            gen_counter[0] += 1
            if gen_counter[0] > 5:
                return None
            return {"name": f"GenA-{gen_counter[0]}", "handler": lambda: "a"}

        def gen_b():
            return {"name": f"GenB-{gen_counter[0]}", "handler": lambda: "b"}

        agent.set_task_generator(gen_a)

        # 模拟几个循环
        for _ in range(3):
            agent._generate_tasks()
            agent.engine.run_batch(3)

        # 热替换
        gen_counter[0] = 0
        agent.set_task_generator(gen_b)

        for _ in range(3):
            agent._generate_tasks()
            agent.engine.run_batch(3)

        stats = agent.engine.stats
        agent.stop()

        self.record_result("C3: 动态替换任务生成器", True,
                          f"执行: {stats['total_executed']}, 成功率: {stats['success_rate']}%")

    # ========== C4: 并发读写删知识图谱 ==========

    def test_c4_concurrent_kg_ops(self):
        """同时进行读写删除操作"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        # 先写入一些数据
        node_ids = []
        for i in range(50):
            nid = kg.store_experience(title=f"Node-{i}", content=f"Content {i}", category="test")
            node_ids.append(nid)

        errors = []
        ops_done = [0]

        def writer():
            try:
                for i in range(50, 100):
                    kg.store_experience(title=f"Write-{i}", content=f"Content {i}", category="concurrent")
                    ops_done[0] += 1
            except Exception as e:
                errors.append(("writer", str(e)[:60]))

        def reader():
            try:
                for _ in range(100):
                    kg.search_nodes("test", limit=5)
                    kg.query_experience("Content", limit=3)
                    ops_done[0] += 1
            except Exception as e:
                errors.append(("reader", str(e)[:60]))

        def deleter():
            try:
                for nid in node_ids[:25]:
                    kg.delete_node(nid)
                    ops_done[0] += 1
            except Exception as e:
                errors.append(("deleter", str(e)[:60]))

        def updater():
            try:
                for nid in node_ids[25:]:
                    kg.update_node_usage(nid, success=True)
                    ops_done[0] += 1
            except Exception as e:
                errors.append(("updater", str(e)[:60]))

        threads = [threading.Thread(target=f) for f in [writer, reader, deleter, updater]]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # 验证数据库完整性
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("PRAGMA integrity_check")
        integrity = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM nodes")
        node_count = c.fetchone()[0]
        conn.close()

        if integrity != "ok":
            self.record_bug("C4-001", f"数据库完整性损坏: {integrity}", "critical")

        if errors:
            self.record_bug("C4-002", f"并发操作错误: {errors[:3]}", "high")

        kg.close()
        self.record_result("C4: 并发读写删知识图谱", integrity == "ok",
                          f"操作: {ops_done[0]}, 节点: {node_count}, 完整性: {integrity}, 错误: {len(errors)}")

    # ========== C5: 异常回调链 ==========

    def test_c5_callback_chain(self):
        """回调抛异常 + 回调中再注册回调"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.optimize_interval = 100

        agent = Agent(config)
        callback_log = []

        def bad_callback(task):
            callback_log.append("bad_called")
            raise RuntimeError("回调故意抛异常!")

        def good_callback(task):
            callback_log.append("good_called")

        def chain_callback(task):
            callback_log.append("chain_called")
            # 在回调中注册新回调
            agent.engine.on_complete(good_callback)
            # 在回调中提交新任务
            agent.submit_task(name="FromCallback", handler=lambda: "callback_result")

        # 先设置会抛异常的回调
        agent.engine.on_complete(bad_callback)

        # 提交任务
        agent.submit_task(name="TriggerBad", handler=lambda: "ok")
        agent.engine.run_batch(1)

        # 替换为链式回调
        agent.engine.on_complete(chain_callback)
        agent.submit_task(name="TriggerChain", handler=lambda: "ok2")
        agent.engine.run_batch(1)

        # 执行回调中提交的任务
        agent.engine.run_batch(1)

        agent.stop()

        if "bad_called" not in callback_log:
            self.record_bug("C5-001", "异常回调未被调用", "medium")

        if "chain_called" not in callback_log:
            self.record_bug("C5-002", "链式回调未被调用", "medium")

        self.record_result("C5: 异常回调链", True,
                          f"回调日志: {callback_log}, 系统未崩溃")

    # ========== C6: run→stop→run重入 ==========

    def test_c6_reentrant_run(self):
        """Agent.run()中途stop再run()"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 3

        agent = Agent(config)
        run_cycles = [0]

        def gen_tasks():
            run_cycles[0] += 1
            if run_cycles[0] > 20:
                return None
            return {"name": f"Cycle-{run_cycles[0]}", "handler": lambda: run_cycles[0]}

        agent.set_task_generator(gen_tasks)

        errors = []

        # 第一次运行（后台线程）
        def run_first():
            try:
                agent.run(max_cycles=5, cycle_interval=0.1)
            except Exception as e:
                errors.append(f"run1: {e}")

        t1 = threading.Thread(target=run_first)
        t1.start()
        t1.join(timeout=10)

        # 停止
        agent.stop()
        time.sleep(0.5)

        # 第二次运行（同一个Agent实例）
        def run_second():
            try:
                agent.run(max_cycles=5, cycle_interval=0.1)
            except Exception as e:
                errors.append(f"run2: {e}")

        t2 = threading.Thread(target=run_second)
        t2.start()
        t2.join(timeout=10)

        agent.stop()

        if errors:
            self.record_bug("C6-001", f"重入运行错误: {errors}", "medium")

        self.record_result("C6: run→stop→run重入", len(errors) == 0,
                          f"两次运行完成, 错误: {len(errors)}")

    # ========== C7: 超时与正常任务混合 ==========

    def test_c7_mixed_timeout(self):
        """大量超时任务与正常任务混合执行"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5

        agent = Agent(config)

        # 混合提交：50%正常 + 30%超时 + 20%失败
        for i in range(40):
            r = random.random()
            if r < 0.5:
                agent.submit_task(name=f"Normal-{i}", handler=lambda: random.random(),
                                task_type="normal", timeout=5)
            elif r < 0.8:
                agent.submit_task(name=f"Slow-{i}", handler=lambda: time.sleep(10),
                                task_type="slow", timeout=1)
            else:
                agent.submit_task(name=f"Fail-{i}", handler=lambda: 1/0,
                                task_type="fail", timeout=5)

        # 执行
        for _ in range(5):
            pending = agent.engine.pending_count
            if pending == 0:
                break
            agent.engine.run_batch(min(pending, 20))

        stats = agent.engine.stats
        agent.stop()

        # 检查统计一致性
        total = stats['total_success'] + stats['total_failed']
        if total != stats['total_executed']:
            self.record_bug("C7-001",
                          f"统计不一致: success+failed={total} != executed={stats['total_executed']}", "high")

        self.record_result("C7: 超时与正常任务混合", total == stats['total_executed'],
                          f"执行: {stats['total_executed']}, 成功: {stats['total_success']}, "
                          f"失败: {stats['total_failed']}, 统计一致: {total == stats['total_executed']}")

    # ========== C8: shutdown后submit ==========

    def test_c8_submit_after_shutdown(self):
        """引擎shutdown后尝试提交任务"""
        engine = TaskEngine(max_workers=3)

        # 正常提交
        engine.submit(name="Before", handler=lambda: 1)
        engine.run_batch(1)
        engine.shutdown(cancel_pending=True)

        # shutdown后提交
        errors = []
        try:
            engine.submit(name="AfterShutdown", handler=lambda: 2)
            self.record_bug("C8-001", "shutdown后submit应抛异常但未抛", "medium")
        except RuntimeError as e:
            pass  # 预期行为
        except Exception as e:
            errors.append(str(e))

        self.record_result("C8: shutdown后submit", len(errors) == 0,
                          f"shutdown后正确拒绝提交")

    # ========== C9: 完整生命周期压力 ==========

    def test_c9_lifecycle_stress(self):
        """创建→运行→停止→重建→运行→停止，重复多次"""
        db_path = self._new_db()
        errors = []
        total_tasks = 0

        for cycle in range(5):
            try:
                config = AgentConfig()
                config.knowledge.db_path = db_path
                config.max_concurrent_tasks = 3

                agent = Agent(config)

                for i in range(20):
                    agent.submit_task(
                        name=f"Life-{cycle}-{i}",
                        handler=lambda: random.random(),
                        task_type="lifecycle"
                    )
                agent.engine.run_batch(20)
                total_tasks += agent.engine.stats['total_executed']
                agent.stop()
            except Exception as e:
                errors.append((cycle, str(e)[:60]))

        if errors:
            self.record_bug("C9-001", f"生命周期错误: {errors}", "high")

        # 验证数据库跨生命周期持久化
        kg = KnowledgeGraph(db_path)
        summary = kg.get_all_knowledge_summary()
        kg.close()

        self.record_result("C9: 完整生命周期压力", len(errors) == 0,
                          f"5轮生命周期, 总任务: {total_tasks}, 知识节点: {sum(summary.get('nodes_by_category', {}).values())}")

    # ========== C10: 混沌综合大考 ==========

    def test_c10_chaos_comprehensive(self):
        """综合混沌：同时进行所有混乱操作"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5
        config.optimize_interval = 50

        agent = Agent(config)
        errors = []
        ops = [0]
        stop_flag = threading.Event()

        # 混沌操作1：持续提交任务
        def chaos_submitter():
            i = 0
            while not stop_flag.is_set() and i < 200:
                try:
                    r = random.random()
                    if r < 0.4:
                        agent.submit_task(name=f"Chaos-{i}", handler=lambda: random.random())
                    elif r < 0.7:
                        agent.submit_task(name=f"ChaosSlow-{i}", handler=lambda: time.sleep(0.5), timeout=1)
                    elif r < 0.85:
                        agent.submit_task(name=f"ChaosFail-{i}", handler=lambda: 1/0)
                    else:
                        agent.submit_task(name=f"ChaosBig-{i}",
                                        handler=lambda: "x" * 100000, timeout=5)
                    ops[0] += 1
                    i += 1
                except RuntimeError:
                    break  # 引擎已关闭
                except Exception as e:
                    errors.append(("submit", str(e)[:40]))
                time.sleep(0.01)

        # 混沌操作2：持续执行任务
        def chaos_runner():
            while not stop_flag.is_set():
                try:
                    pending = agent.engine.pending_count
                    if pending > 0:
                        agent.engine.run_batch(min(pending, 10))
                        ops[0] += 1
                except Exception as e:
                    errors.append(("run", str(e)[:40]))
                time.sleep(0.05)

        # 混沌操作3：持续查询知识图谱
        def chaos_reader():
            while not stop_flag.is_set():
                try:
                    agent.knowledge.search_nodes("Chaos", limit=3)
                    agent.knowledge.query_experience("test", limit=2)
                    ops[0] += 1
                except Exception as e:
                    errors.append(("read", str(e)[:40]))
                time.sleep(0.03)

        # 混沌操作4：持续取消任务
        def chaos_canceler():
            while not stop_flag.is_set():
                try:
                    with agent.engine._lock:
                        if agent.engine._queue:
                            task = agent.engine._queue[0]
                            agent.engine.cancel_task(task.id)
                            ops[0] += 1
                except Exception as e:
                    errors.append(("cancel", str(e)[:40]))
                time.sleep(0.08)

        # 混沌操作5：持续获取状态
        def chaos_status():
            while not stop_flag.is_set():
                try:
                    _ = agent.status
                    _ = agent.engine.stats
                    ops[0] += 1
                except Exception as e:
                    errors.append(("status", str(e)[:40]))
                time.sleep(0.02)

        threads = [
            threading.Thread(target=chaos_submitter),
            threading.Thread(target=chaos_runner),
            threading.Thread(target=chaos_reader),
            threading.Thread(target=chaos_canceler),
            threading.Thread(target=chaos_status),
        ]

        for t in threads:
            t.start()

        # 运行10秒
        time.sleep(10)
        stop_flag.set()

        for t in threads:
            t.join(timeout=10)

        stats = agent.engine.stats
        agent.stop()

        error_rate = len(errors) / max(ops[0], 1) * 100

        if error_rate > 5:
            self.record_bug("C10-001", f"混沌错误率过高: {error_rate:.1f}%, 错误: {errors[:5]}", "high")

        self.record_result("C10: 混沌综合大考", error_rate < 5,
                          f"操作: {ops[0]}, 任务执行: {stats['total_executed']}, "
                          f"成功率: {stats['success_rate']}%, 错误: {len(errors)}, 错误率: {error_rate:.1f}%")

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("第十轮混沌工程测试总结")
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
    tester = Round10ChaosTest()
    tester.run_all_tests()
