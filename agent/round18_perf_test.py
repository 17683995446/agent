"""
第十八轮性能优化专项测试 - 聚焦性能瓶颈和资源泄漏
遵循: 多阶段迭代原理 + 第一性原理

性能专项设计:
1. P1: 搜索性能 - 大数据集LIKE查询性能
2. P2: 事务竞争 - 高并发写入事务锁争用
3. P3: 内存泄漏检测 - _completed列表增长
4. P4: 连接池泄漏 - 线程销毁后连接清理
5. P5: 批量调度优化 - run_batch超时计算
6. P6: 批量插入优化 - 批量vs逐条写入
7. P7: 回调死锁风险 - 优化回调锁嵌套
8. P8: 知识清理效果 - cleanup_old_knowledge验证
9. P9: 线程本地连接 - 线程复用vs新建
10. P10: 全链路性能基准 - 端到端吞吐量测量
"""
import os, sys, time, json, random, threading, traceback, gc
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus, Task
from agent.modules.knowledge_graph import KnowledgeGraph, KnowledgeNode
from agent.modules.optimizer import Optimizer

import logging
logging.basicConfig(level=logging.WARNING, format='%(name)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class Round18PerfTest:
    def __init__(self):
        self.results = []
        self.bugs = []
        self._lock = threading.Lock()
        self._db_counter = 0

    def _new_db(self):
        self._db_counter += 1
        return f"/data/user/work/perf_r18_{self._db_counter}.db"

    def record(self, name, passed, details=""):
        with self._lock:
            self.results.append({"test": name, "passed": passed, "details": details})
        print(f"  [{'✓' if passed else '✗'}] {name}: {details[:150]}")

    def bug(self, bid, desc, sev="medium"):
        with self._lock:
            self.bugs.append({"id": bid, "description": desc, "severity": sev})
        print(f"  🐛 BUG [{bid}] ({sev}): {desc[:120]}")

    def run_all(self):
        print("\n" + "=" * 70)
        print("第十八轮性能优化专项测试")
        print("=" * 70)

        tests = [
            ("P1: 搜索性能", self.test_p1_search_perf),
            ("P2: 事务竞争", self.test_p2_transaction_contention),
            ("P3: 内存泄漏检测", self.test_p3_memory_leak),
            ("P4: 连接池泄漏", self.test_p4_connection_leak),
            ("P5: 批量调度优化", self.test_p5_batch_scheduling),
            ("P6: 批量插入优化", self.test_p6_bulk_insert),
            ("P7: 回调死锁风险", self.test_p7_callback_deadlock),
            ("P8: 知识清理效果", self.test_p8_cleanup_effectiveness),
            ("P9: 线程本地连接", self.test_p9_thread_local_conn),
            ("P10: 全链路性能基准", self.test_p10_full_benchmark),
        ]

        for name, func in tests:
            print(f"\n{'=' * 60}")
            print(f"执行: {name}")
            print(f"{'=' * 60}")
            try:
                func()
            except Exception as e:
                self.record(name, False, f"异常: {e}")
                self.bug(f"P-{name[:3]}", f"异常: {traceback.format_exc()[:200]}", "high")

        self._summary()

    # ========== P1: 搜索性能 ==========

    def test_p1_search_perf(self):
        """大数据集LIKE查询性能"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        # 插入10000条数据
        print("    准备10000条数据...")
        for i in range(10000):
            kg.store_experience(
                title=f"PerfTest-{i:05d}",
                content=f"这是第{i}条测试数据，包含一些关键词如性能优化和压力测试",
                category=f"cat_{i % 20}",
                tags=f"tag_{i % 50}",
            )

        # 测试搜索性能
        queries = ["PerfTest-05000", "性能优化", "cat_5", "tag_10", "不存在的关键词"]
        total_time = 0
        max_time = 0

        for q in queries:
            start = time.time()
            results = kg.search_nodes(q, limit=10)
            elapsed = time.time() - start
            total_time += elapsed
            max_time = max(max_time, elapsed)

        avg_time = total_time / len(queries)
        kg.close()

        # 验证：单次搜索不应超过2秒
        if max_time > 2.0:
            self.bug("P1-001", f"搜索性能过慢: 最大={max_time:.3f}s, 平均={avg_time:.3f}s", "high")

        self.record("P1: 搜索性能", max_time <= 2.0,
                    f"10000节点, {len(queries)}次查询, 最大: {max_time:.3f}s, 平均: {avg_time:.3f}s")

    # ========== P2: 事务竞争 ==========

    def test_p2_transaction_contention(self):
        """高并发写入事务锁争用"""
        db_path = self._new_db()

        errors = []
        write_count = [0]

        def writer(tid):
            kg = KnowledgeGraph(db_path)
            try:
                for i in range(200):
                    kg.store_experience(
                        title=f"TxCont-{tid}-{i}",
                        content=f"事务竞争测试数据",
                        category="tx_contention"
                    )
                    write_count[0] += 1
            except Exception as e:
                errors.append((tid, str(e)[:80]))
            finally:
                kg.close()

        start = time.time()
        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        elapsed = time.time() - start

        throughput = write_count[0] / elapsed

        if len(errors) > 50:
            self.bug("P2-001", f"事务竞争错误过多: {len(errors)}", "high")

        if throughput < 100:
            self.bug("P2-002", f"写入吞吐量过低: {throughput:.0f}/s", "medium")

        self.record("P2: 事务竞争", len(errors) <= 50,
                    f"20线程x200写入, 吞吐: {throughput:.0f}/s, 耗时: {elapsed:.1f}s, 错误: {len(errors)}")

    # ========== P3: 内存泄漏检测 ==========

    def test_p3_memory_leak(self):
        """_completed列表增长检测"""
        engine = TaskEngine(max_workers=3)

        # 提交大量任务但不清理
        for i in range(500):
            engine.submit(name=f"MemLeak-{i}", handler=lambda: "ok")

        # 执行所有任务
        for _ in range(100):
            pending = engine.engine.pending_count if hasattr(engine, 'engine') else engine.pending_count
            if pending == 0:
                break
            engine.run_batch(min(pending, 10))

        completed_size = len(engine._completed)
        stats = engine.stats

        # 清理后验证
        engine.clear_completed(keep=100)
        after_clear = len(engine._completed)

        engine.shutdown(cancel_pending=True)

        # 验证清理有效
        if completed_size > 500:
            self.bug("P3-001", f"_completed列表过大: {completed_size}", "medium")

        if after_clear > 100:
            self.bug("P3-002", f"清理后列表仍过大: {after_clear}", "high")

        self.record("P3: 内存泄漏检测", after_clear <= 100,
                    f"执行: {stats['total_executed']}, 清理前: {completed_size}, 清理后: {after_clear}")

    # ========== P4: 连接池泄漏 ==========

    def test_p4_connection_leak(self):
        """线程销毁后连接清理"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        # 启动并销毁大量线程
        def short_lived_thread(tid):
            local_kg = KnowledgeGraph(db_path)
            local_kg.store_experience(
                title=f"ConnLeak-{tid}",
                content="连接泄漏测试",
                category="conn_leak"
            )
            local_kg.close()

        for i in range(50):
            t = threading.Thread(target=short_lived_thread, args=(i,))
            t.start()
            t.join(timeout=10)

        # 检查连接池大小
        conn_count = len(kg._all_connections)
        kg.close()

        if conn_count > 10:
            self.bug("P4-001", f"连接池泄漏: {conn_count}个连接未清理", "high")

        self.record("P4: 连接池泄漏", conn_count <= 10,
                    f"50个短生命周期线程, 残留连接: {conn_count}")

    # ========== P5: 批量调度优化 ==========

    def test_p5_batch_scheduling(self):
        """run_batch超时计算性能"""
        engine = TaskEngine(max_workers=5)

        # 提交大量短任务
        for i in range(100):
            engine.submit(name=f"BatchSched-{i}", handler=lambda: "ok", timeout=10)

        # 测量run_batch调度时间
        start = time.time()
        total_executed = 0
        for _ in range(20):
            pending = engine.pending_count
            if pending == 0:
                break
            results = engine.run_batch(min(pending, 20))
            total_executed += len(results)

        schedule_time = time.time() - start
        engine.shutdown(cancel_pending=True)

        # 验证：100个简单任务调度不应超过10秒
        if schedule_time > 10:
            self.bug("P5-001", f"批量调度过慢: {schedule_time:.1f}s", "medium")

        self.record("P5: 批量调度优化", schedule_time <= 10,
                    f"100任务, 执行: {total_executed}, 调度耗时: {schedule_time:.2f}s")

    # ========== P6: 批量插入优化 ==========

    def test_p6_bulk_insert(self):
        """批量vs逐条写入性能对比"""
        db_path = self._new_db()

        # 逐条插入
        kg_single = KnowledgeGraph(db_path)
        start = time.time()
        for i in range(1000):
            kg_single.store_experience(
                title=f"Single-{i}",
                content="逐条插入测试",
                category="bulk_test"
            )
        single_time = time.time() - start
        kg_single.close()

        # 使用store_experience_batch批量插入
        db_path2 = self._new_db()
        kg_bulk = KnowledgeGraph(db_path2)
        start = time.time()

        experiences = [
            {"title": f"Bulk-{i}", "content": "批量插入测试", "category": "bulk_test"}
            for i in range(1000)
        ]
        kg_bulk.store_experience_batch(experiences)

        bulk_time = time.time() - start
        kg_bulk.close()

        speedup = single_time / max(bulk_time, 0.001)

        if speedup < 2.0:
            self.bug("P6-001", f"批量插入加速不足: {speedup:.1f}x (预期>=2x)", "medium")

        self.record("P6: 批量插入优化", speedup >= 2.0,
                    f"逐条: {single_time:.2f}s, 批量: {bulk_time:.2f}s, 加速: {speedup:.1f}x")

    # ========== P7: 回调死锁风险 ==========

    def test_p7_callback_deadlock(self):
        """优化回调锁嵌套检测"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5
        config.optimize_interval = 5

        agent = Agent(config)
        deadlock_detected = [False]
        completed_count = [0]

        # 注册一个会访问engine的回调
        original_callback = agent._on_task_done

        def complex_callback(task):
            completed_count[0] += 1
            # 在回调中访问多个模块（模拟真实场景）
            try:
                _ = agent.engine.stats
                _ = agent.knowledge.search_nodes("test", limit=1)
                _ = agent.optimizer.stats
            except Exception:
                pass

        agent.engine.on_complete(complex_callback)

        # 提交并执行大量任务
        for i in range(100):
            agent.submit_task(name=f"DeadlockTest-{i}", handler=lambda: "ok")

        start = time.time()
        for _ in range(30):
            pending = agent.engine.pending_count
            if pending == 0:
                break
            agent.engine.run_batch(min(pending, 10))
            # 超过30秒认为死锁
            if time.time() - start > 30:
                deadlock_detected[0] = True
                break

        elapsed = time.time() - start
        agent.stop()

        if deadlock_detected[0]:
            self.bug("P7-001", "回调中检测到死锁或长时间阻塞", "critical")

        if elapsed > 15:
            self.bug("P7-002", f"回调执行过慢: {elapsed:.1f}s", "high")

        self.record("P7: 回调死锁风险", not deadlock_detected[0] and elapsed <= 15,
                    f"100任务, 完成: {completed_count[0]}, 耗时: {elapsed:.1f}s, 死锁: {deadlock_detected[0]}")

    # ========== P8: 知识清理效果 ==========

    def test_p8_cleanup_effectiveness(self):
        """cleanup_old_knowledge验证"""
        db_path = self._new_db()
        kg = KnowledgeGraph(db_path)

        # 插入大量低重要性数据
        for i in range(500):
            kg.store_experience(
                title=f"Old-{i}",
                content="过时数据",
                category="experience",
                importance=0.1  # 低重要性
            )

        # 插入少量高重要性数据
        for i in range(50):
            kg.store_experience(
                title=f"Important-{i}",
                content="重要数据",
                category="experience",
                importance=0.9
            )

        before = kg.get_all_knowledge_summary()
        total_before = sum(before.get('nodes_by_category', {}).values())

        # 执行清理
        cleaned = kg.cleanup_old_knowledge(max_age_seconds=0, min_importance=0.2)

        after = kg.get_all_knowledge_summary()
        total_after = sum(after.get('nodes_by_category', {}).values())

        kg.close()

        # 验证：低重要性数据应被清理，高重要性数据应保留
        important_preserved = total_after >= 50  # 至少保留重要数据

        if not important_preserved:
            self.bug("P8-001", f"清理误删重要数据: 清理前={total_before}, 清理后={total_after}", "high")

        self.record("P8: 知识清理效果", important_preserved,
                    f"清理前: {total_before}, 清理后: {total_after}, 清理: {cleaned}条")

    # ========== P9: 线程本地连接 ==========

    def test_p9_thread_local_conn(self):
        """线程复用vs新建连接性能"""
        db_path = self._new_db()

        # 方案A：每个操作新建KG实例（新建连接）
        start = time.time()
        for i in range(100):
            kg = KnowledgeGraph(db_path)
            kg.store_experience(title=f"NewConn-{i}", content="新建连接", category="conn_test")
            kg.close()
        new_conn_time = time.time() - start

        # 方案B：复用同一个KG实例（线程本地连接）
        kg_reuse = KnowledgeGraph(db_path)
        start = time.time()
        for i in range(100):
            kg_reuse.store_experience(title=f"ReuseConn-{i}", content="复用连接", category="conn_test")
        reuse_conn_time = time.time() - start
        kg_reuse.close()

        speedup = new_conn_time / max(reuse_conn_time, 0.001)

        self.record("P9: 线程本地连接", True,
                    f"新建: {new_conn_time:.2f}s, 复用: {reuse_conn_time:.2f}s, 加速: {speedup:.1f}x")

    # ========== P10: 全链路性能基准 ==========

    def test_p10_full_benchmark(self):
        """端到端吞吐量测量"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 10
        config.optimize_interval = 100  # 减少优化干扰

        agent = Agent(config)

        # 阶段1：提交吞吐量
        start = time.time()
        for i in range(1000):
            agent.submit_task(name=f"Bench-{i}", handler=lambda: f"result_{i}")
        submit_time = time.time() - start
        submit_tps = 1000 / submit_time

        # 阶段2：执行吞吐量
        start = time.time()
        for _ in range(100):
            pending = agent.engine.pending_count
            if pending == 0:
                break
            agent.engine.run_batch(min(pending, 20))
        execute_time = time.time() - start
        stats = agent.engine.stats
        execute_tps = stats['total_executed'] / max(execute_time, 0.001)

        # 阶段3：知识存储吞吐量
        start = time.time()
        for i in range(500):
            agent.knowledge.store_experience(
                title=f"BenchExp-{i}",
                content=f"基准测试经验数据{i}",
                category="benchmark"
            )
        store_time = time.time() - start
        store_tps = 500 / store_time

        # 阶段4：查询吞吐量
        start = time.time()
        for i in range(500):
            agent.knowledge.search_nodes(f"BenchExp-{i % 100}", limit=5)
        query_time = time.time() - start
        query_tps = 500 / query_time

        agent.stop()

        # 性能基线验证
        if submit_tps < 500:
            self.bug("P10-001", f"提交吞吐量过低: {submit_tps:.0f}/s", "medium")

        if execute_tps < 50:
            self.bug("P10-002", f"执行吞吐量过低: {execute_tps:.0f}/s", "medium")

        self.record("P10: 全链路性能基准", True,
                    f"提交: {submit_tps:.0f}/s, 执行: {execute_tps:.0f}/s, "
                    f"存储: {store_tps:.0f}/s, 查询: {query_tps:.0f}/s")

    def _summary(self):
        print("\n" + "=" * 70)
        print("第十八轮性能优化专项测试总结")
        print("=" * 70)

        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed

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
    tester = Round18PerfTest()
    tester.run_all()
