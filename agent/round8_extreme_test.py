"""
第八轮极端场景测试 - 真实生产环境模拟
测试目标：发现系统在极端条件下的潜在bug
遵循：慢工出细活原理
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

# 添加项目路径
sys.path.insert(0, '/workspace')

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus
from agent.modules.knowledge_graph import KnowledgeGraph


class Round8ExtremeTest:
    """第八轮极端场景测试套件"""

    def __init__(self):
        self.test_results = []
        self.bugs_found = []
        self._lock = threading.Lock()

    def record_result(self, test_name: str, passed: bool, details: str = ""):
        with self._lock:
            self.test_results.append({
                "test": test_name,
                "passed": passed,
                "details": details,
                "timestamp": time.time()
            })
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  [{status}] {test_name}: {details[:100]}")

    def record_bug(self, bug_id: str, description: str, severity: str = "medium"):
        with self._lock:
            self.bugs_found.append({
                "id": bug_id,
                "description": description,
                "severity": severity,
                "timestamp": time.time()
            })
            print(f"  🐛 BUG发现 [{bug_id}]: {description[:80]}... (严重性: {severity})")

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*70)
        print("第八轮极端场景测试 - 真实生产环境模拟")
        print("="*70)

        tests = [
            ("E1: 数据库连接池耗尽测试", self.test_e1_db_connection_exhaustion),
            ("E2: 内存泄漏检测", self.test_e2_memory_leak),
            ("E3: 任务队列溢出测试", self.test_e3_queue_overflow),
            ("E4: 优化线程竞争测试", self.test_e4_optimize_thread_race),
            ("E5: 数据库损坏恢复测试", self.test_e5_db_corruption_recovery),
            ("E6: 极端并发测试", self.test_e6_extreme_concurrency),
            ("E7: 资源耗尽恢复测试", self.test_e7_resource_exhaustion),
            ("E8: 长时间稳定性测试", self.test_e8_long_running_stability),
        ]

        for name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"执行: {name}")
            print(f"{'='*60}")
            try:
                test_func()
            except Exception as e:
                self.record_result(name, False, f"测试异常: {e}")
                self.record_bug(f"E-{name[:3]}", f"测试执行异常: {traceback.format_exc()}", "high")

        self._print_summary()

    def test_e1_db_connection_exhaustion(self):
        """E1: 数据库连接池耗尽测试
        模拟大量线程同时访问数据库，检测连接管理是否正确
        """
        db_path = "/data/user/work/test_e1.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        kg = KnowledgeGraph(db_path)

        # 创建50个线程同时访问数据库
        errors = []
        success_count = [0]

        def db_operation(thread_id):
            try:
                for i in range(10):
                    # 并发写入
                    kg.store_experience(
                        title=f"Thread-{thread_id}-Exp-{i}",
                        content=f"Content from thread {thread_id} iteration {i}",
                        category="test",
                        importance=0.5
                    )
                    # 并发读取
                    results = kg.search_nodes(f"Thread-{thread_id}")
                    time.sleep(0.01)  # 模拟处理时间
                with self._lock:
                    success_count[0] += 1
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = []
        for i in range(50):
            t = threading.Thread(target=db_operation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        kg.close()

        # 验证数据完整性
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM nodes")
        node_count = c.fetchone()[0]
        c.execute("PRAGMA integrity_check")
        integrity = c.fetchone()[0]
        conn.close()

        if len(errors) > 0:
            self.record_bug("E1-001", f"数据库并发访问错误: {errors[:5]}", "high")

        if integrity != "ok":
            self.record_bug("E1-002", f"数据库完整性检查失败: {integrity}", "critical")

        # 检查连接是否正确关闭
        try:
            os.remove(db_path)
            self.record_result("E1: 数据库连接池耗尽测试", True, f"50线程并发成功{success_count[0]}, 节点数{node_count}")
        except Exception as e:
            self.record_bug("E1-003", f"数据库文件无法删除，连接未正确关闭: {e}", "high")
            self.record_result("E1: 数据库连接池耗尽测试", False, f"连接未正确关闭: {e}")

    def test_e2_memory_leak(self):
        """E2: 内存泄漏检测
        执行大量任务后检查内存是否持续增长
        """
        import tracemalloc
        tracemalloc.start()

        db_path = "/data/user/work/test_e2.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5
        config.optimize_interval = 10

        agent = Agent(config)

        # 记录初始内存
        snapshot1 = tracemalloc.take_snapshot()
        initial_mem = sum(stat.size for stat in snapshot1.statistics('lineno'))

        # 执行500个任务
        task_count = 500
        for i in range(task_count):
            agent.submit_task(
                name=f"MemoryTest-{i}",
                handler=lambda: time.sleep(0.01),
                task_type="memory_test"
            )
            if i % 50 == 0:
                agent.engine.run_batch(10)

        # 先停止agent（这会清理所有资源）
        agent.stop()

        # 强制垃圾回收
        gc.collect()
        time.sleep(1)

        # 记录最终内存
        snapshot2 = tracemalloc.take_snapshot()
        final_mem = sum(stat.size for stat in snapshot2.statistics('lineno'))

        # 检查内存增长
        mem_growth = final_mem - initial_mem
        mem_growth_mb = mem_growth / 1024 / 1024

        # 检查内部状态（stop后应该全部清理）
        handler_count = len(agent.engine._task_handlers) if hasattr(agent.engine, '_task_handlers') else 0
        cancel_event_count = len(agent.engine._cancel_events) if hasattr(agent.engine, '_cancel_events') else 0
        completed_count = len(agent.engine._completed)

        # 内存增长超过50MB视为潜在泄漏
        if mem_growth_mb > 50:
            self.record_bug("E2-001", f"内存增长过大: {mem_growth_mb:.2f}MB", "medium")

        # 检查资源清理（stop后应该全部清理）
        if handler_count > 0:
            self.record_bug("E2-002", f"handler泄漏: {handler_count}个未清理", "high")

        if cancel_event_count > 0:
            self.record_bug("E2-003", f"cancel_event泄漏: {cancel_event_count}个未清理", "high")

        tracemalloc.stop()
        self.record_result("E2: 内存泄漏检测", mem_growth_mb < 100,
                          f"内存增长: {mem_growth_mb:.2f}MB, handlers: {handler_count}, events: {cancel_event_count}")

    def test_e3_queue_overflow(self):
        """E3: 任务队列溢出测试
        提交超过系统处理能力的任务数量
        """
        db_path = "/data/user/work/test_e3.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 3
        config.task_queue_size = 100

        engine = TaskEngine(max_workers=3, timeout=5)

        # 提交500个任务（远超队列大小）
        submitted = 0
        failed_submit = 0
        task_ids = []

        for i in range(500):
            try:
                task = engine.submit(
                    name=f"Overflow-{i}",
                    handler=lambda: time.sleep(0.1),
                    task_type="overflow_test"
                )
                task_ids.append(task.id)
                submitted += 1
            except Exception as e:
                failed_submit += 1

        # 执行一批任务
        results = engine.run_batch(50)

        # 检查队列状态
        pending = engine.pending_count
        stats = engine.stats

        engine.shutdown(cancel_pending=True)

        # 检查是否有任务丢失
        if submitted < 400:  # 至少应该能提交80%
            self.record_bug("E3-001", f"任务提交失败过多: 提交{submitted}/500", "high")

        # 检查队列是否正常处理
        if stats['total_executed'] < 10:
            self.record_bug("E3-002", f"任务执行数过低: {stats['total_executed']}", "medium")

        self.record_result("E3: 任务队列溢出测试", submitted >= 400,
                          f"提交: {submitted}/500, 执行: {stats['total_executed']}, 待处理: {pending}")

    def test_e4_optimize_thread_race(self):
        """E4: 优化线程竞争测试
        测试后台优化线程与主线程的资源竞争
        """
        db_path = "/data/user/work/test_e4.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5
        config.optimize_interval = 3

        agent = Agent(config)

        # 并发执行：主线程提交任务，后台优化线程运行
        errors = []
        task_counter = [0]

        def submit_tasks():
            try:
                for i in range(100):
                    agent.submit_task(
                        name=f"Race-{i}",
                        handler=lambda: random.random(),
                        task_type="race_test"
                    )
                    task_counter[0] += 1
                    time.sleep(0.02)
            except Exception as e:
                errors.append(("submit", str(e)))

        def run_cycles():
            try:
                for i in range(20):
                    agent.engine.run_batch(5)
                    time.sleep(0.1)
            except Exception as e:
                errors.append(("run", str(e)))

        def query_knowledge():
            try:
                for i in range(50):
                    agent.knowledge.query_experience("test", limit=5)
                    time.sleep(0.05)
            except Exception as e:
                errors.append(("query", str(e)))

        threads = [
            threading.Thread(target=submit_tasks),
            threading.Thread(target=run_cycles),
            threading.Thread(target=query_knowledge),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=30)

        agent.stop()

        if len(errors) > 0:
            self.record_bug("E4-001", f"线程竞争错误: {errors}", "high")

        # 检查数据一致性
        stats = agent.engine.stats
        kg_summary = agent.knowledge.get_all_knowledge_summary()

        self.record_result("E4: 优化线程竞争测试", len(errors) == 0,
                          f"任务: {task_counter[0]}, 执行: {stats['total_executed']}, 错误: {len(errors)}")

    def test_e5_db_corruption_recovery(self):
        """E5: 数据库损坏恢复测试
        模拟数据库文件损坏，测试恢复能力
        """
        db_path = "/data/user/work/test_e5.db"

        # 创建正常数据库
        if os.path.exists(db_path):
            os.remove(db_path)

        kg = KnowledgeGraph(db_path)
        for i in range(10):
            kg.store_experience(title=f"Test-{i}", content=f"Content {i}", category="test")
        kg.close()

        # 模拟损坏：写入垃圾数据
        with open(db_path, 'ab') as f:
            f.write(b'\x00' * 1000)  # 写入空字节

        # 尝试重新打开
        try:
            kg2 = KnowledgeGraph(db_path)
            # 尝试查询
            results = kg2.search_nodes("Test")
            kg2.close()

            # 如果能正常工作，说明有恢复机制或损坏不严重
            self.record_result("E5: 数据库损坏恢复测试", True,
                              f"数据库部分恢复，查询结果: {len(results)}条")
        except Exception as e:
            # 预期：应该能优雅处理错误
            self.record_bug("E5-001", f"数据库损坏后未优雅处理: {e}", "medium")
            self.record_result("E5: 数据库损坏恢复测试", False, f"异常: {e}")

        # 清理
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_e6_extreme_concurrency(self):
        """E6: 极端并发测试
        100个线程同时提交任务和查询知识库
        """
        db_path = "/data/user/work/test_e6.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 10

        agent = Agent(config)

        errors = []
        operations = [0]

        def concurrent_operation(thread_id):
            try:
                for i in range(5):
                    # 随机选择操作
                    op = random.choice(['submit', 'query', 'status'])
                    if op == 'submit':
                        agent.submit_task(
                            name=f"Concurrent-{thread_id}-{i}",
                            handler=lambda x=random.random(): x,
                            task_type="concurrent"
                        )
                    elif op == 'query':
                        agent.knowledge.query_experience("test", limit=3)
                    else:
                        _ = agent.status
                    operations[0] += 1
            except Exception as e:
                with self._lock:
                    errors.append((thread_id, str(e)[:100]))

        threads = []
        for i in range(100):
            t = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=60)

        agent.stop()

        error_rate = len(errors) / 500 * 100  # 100线程 * 5操作

        if error_rate > 5:  # 错误率超过5%
            self.record_bug("E6-001", f"高并发错误率: {error_rate:.1f}%, 错误: {errors[:3]}", "high")

        self.record_result("E6: 极端并发测试", error_rate < 5,
                          f"操作: {operations[0]}, 错误: {len(errors)}, 错误率: {error_rate:.1f}%")

    def test_e7_resource_exhaustion(self):
        """E7: 资源耗尽恢复测试
        模拟系统资源耗尽后的恢复能力
        """
        db_path = "/data/user/work/test_e7.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5

        agent = Agent(config)

        # 创建大量任务消耗资源
        for i in range(200):
            agent.submit_task(
                name=f"Resource-{i}",
                handler=lambda: [random.random() for _ in range(1000)],  # 分配内存
                task_type="resource_test"
            )

        # 执行任务
        for _ in range(10):
            agent.engine.run_batch(20)

        # 强制垃圾回收
        gc.collect()

        # 检查系统是否仍正常工作
        test_task = agent.submit_task(
            name="PostExhaustion-Test",
            handler=lambda: "recovery_ok",
            task_type="recovery"
        )
        agent.engine.run_batch(1)

        # 验证恢复
        stats = agent.engine.stats
        recovery_ok = stats['total_executed'] > 0

        agent.stop()

        if not recovery_ok:
            self.record_bug("E7-001", "资源耗尽后系统未恢复", "high")

        self.record_result("E7: 资源耗尽恢复测试", recovery_ok,
                          f"执行任务: {stats['total_executed']}, 成功率: {stats['success_rate']}%")

    def test_e8_long_running_stability(self):
        """E8: 长时间稳定性测试
        1000轮循环，模拟长时间运行
        """
        db_path = "/data/user/work/test_e8.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5
        config.optimize_interval = 20

        agent = Agent(config)

        cycles = 1000
        errors = []
        memory_samples = []

        print(f"  开始{cycles}轮稳定性测试...")

        for i in range(cycles):
            try:
                # 随机任务类型
                task_type = random.choice(['quick', 'medium', 'slow', 'fail'])
                if task_type == 'quick':
                    handler = lambda: random.random()
                elif task_type == 'medium':
                    handler = lambda: sum(range(100))
                elif task_type == 'slow':
                    handler = lambda: time.sleep(0.01)
                else:
                    handler = lambda: 1/0  # 故意失败

                agent.submit_task(
                    name=f"Stability-{i}",
                    handler=handler,
                    task_type=task_type,
                    timeout=5
                )

                if i % 10 == 0:
                    agent.engine.run_batch(5)

                if i % 100 == 0:
                    # 记录内存使用
                    import psutil
                    process = psutil.Process()
                    mem_mb = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(mem_mb)
                    print(f"    轮次 {i}/{cycles}, 内存: {mem_mb:.1f}MB")

            except Exception as e:
                errors.append((i, str(e)[:50]))

        # 最终执行
        agent.engine.run_batch(50)
        agent.stop()

        # 分析结果
        stats = agent.engine.stats
        kg_summary = agent.knowledge.get_all_knowledge_summary()

        # 检查内存增长趋势
        if len(memory_samples) >= 2:
            mem_growth = memory_samples[-1] - memory_samples[0]
            if mem_growth > 100:  # 增长超过100MB
                self.record_bug("E8-001", f"长时间运行内存增长: {mem_growth:.1f}MB", "medium")

        # 检查错误率
        error_rate = len(errors) / cycles * 100
        if error_rate > 1:
            self.record_bug("E8-002", f"长时间运行错误率: {error_rate:.1f}%", "medium")

        # 检查知识库增长
        total_nodes = sum(kg_summary.get('nodes_by_category', {}).values())

        self.record_result("E8: 长时间稳定性测试", error_rate < 2,
                          f"轮次: {cycles}, 执行: {stats['total_executed']}, "
                          f"成功率: {stats['success_rate']}%, 知识节点: {total_nodes}, 错误: {len(errors)}")

    def _print_summary(self):
        """打印测试总结"""
        print("\n" + "="*70)
        print("第八轮极端场景测试总结")
        print("="*70)

        passed = sum(1 for r in self.test_results if r['passed'])
        failed = len(self.test_results) - passed

        print(f"\n测试结果: {passed}/{len(self.test_results)} 通过, {failed} 失败")

        if self.bugs_found:
            print(f"\n发现 {len(self.bugs_found)} 个Bug:")
            for bug in self.bugs_found:
                print(f"  [{bug['severity'].upper()}] {bug['id']}: {bug['description'][:60]}...")
        else:
            print("\n未发现新Bug ✓")

        print("\n详细结果:")
        for r in self.test_results:
            status = "✓" if r['passed'] else "✗"
            print(f"  [{status}] {r['test']}: {r['details'][:80]}")

        print("\n" + "="*70)


if __name__ == "__main__":
    tester = Round8ExtremeTest()
    tester.run_all_tests()
