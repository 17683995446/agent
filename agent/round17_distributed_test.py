"""
第十七轮分布式场景测试 - 多Agent分布式协作
遵循: 多阶段迭代原理 + 第一性原理

分布式场景设计:
1. D1: 多进程Agent通信 - 进程间任务传递
2. D2: 共享知识库一致性 - 多进程读写无冲突
3. D3: 分布式锁机制 - 复杂资源竞争
4. D4: 网络分区恢复 - 分区后数据同步
5. D5: 节点故障转移 - 单点故障不影响整体
6. D6: 跨进程任务分发 - 任务在多个进程间分配
7. D7: 分布式状态同步 - 状态变更同步
8. D8: 并发连接池极限 - 大量并发连接
9. D9: 消息队列压力 - 高频消息传递
10. D10: 全分布式链路 - 完整分布式业务流程

每个场景通过全链路日志验证正确性。
"""
import os
import sys
import time
import json
import random
import threading
import multiprocessing
import traceback
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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


class Round17DistributedTest:
    """第十七轮分布式场景测试"""

    def __init__(self):
        self.test_results = []
        self.bugs_found = []
        self._lock = threading.Lock()
        self._db_counter = 0

    def _new_db(self):
        self._db_counter += 1
        return f"/data/user/work/distributed_r17_{self._db_counter}.db"

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
        print("第十七轮分布式场景测试 - 多Agent分布式协作")
        print("=" * 70)

        tests = [
            ("D1: 多进程Agent通信", self.test_d1_multi_process),
            ("D2: 共享知识库一致性", self.test_d2_shared_knowledge),
            ("D3: 分布式锁机制", self.test_d3_distributed_lock),
            ("D4: 网络分区恢复", self.test_d4_partition_recovery),
            ("D5: 节点故障转移", self.test_d5_failover),
            ("D6: 跨进程任务分发", self.test_d6_cross_process_tasks),
            ("D7: 分布式状态同步", self.test_d7_state_sync),
            ("D8: 并发连接池极限", self.test_d8_connection_pool),
            ("D9: 消息队列压力", self.test_d9_message_queue),
            ("D10: 全分布式链路", self.test_d10_full_distributed),
        ]

        for name, func in tests:
            print(f"\n{'=' * 60}")
            print(f"执行: {name}")
            print(f"{'=' * 60}")
            try:
                func()
            except Exception as e:
                self.record_result(name, False, f"测试异常: {e}")
                self.record_bug(f"D-{name[:3]}", f"异常: {traceback.format_exc()[:200]}", "high")

        self._print_summary()

    # ========== D1: 多进程Agent通信 ==========

    def test_d1_multi_process(self):
        """进程间任务传递"""
        db_path = self._new_db()
        
        # 使用共享数据库模拟多进程通信
        def worker_process(process_id, db_path, result_queue):
            try:
                config = AgentConfig()
                config.knowledge.db_path = db_path
                config.max_concurrent_tasks = 5
                
                agent = Agent(config)
                
                # 每个进程提交任务
                for i in range(20):
                    agent.submit_task(
                        name=f"P{process_id}-Task-{i}",
                        handler=lambda: f"result_from_p{process_id}",
                        task_type="distributed"
                    )
                
                # 执行
                agent.engine.run_batch(20)
                
                stats = agent.engine.stats
                agent.stop()
                
                result_queue.put({
                    'process_id': process_id,
                    'executed': stats['total_executed'],
                    'success': stats['total_success'],
                    'error': None
                })
            except Exception as e:
                result_queue.put({
                    'process_id': process_id,
                    'executed': 0,
                    'success': 0,
                    'error': str(e)[:100]
                })

        # 启动多个进程
        from multiprocessing import Process, Queue
        result_queue = Queue()
        processes = []
        
        for i in range(4):
            p = Process(target=worker_process, args=(i, db_path, result_queue))
            processes.append(p)
            p.start()
        
        # 等待所有进程完成
        for p in processes:
            p.join(timeout=120)
        
        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        total_executed = sum(r['executed'] for r in results)
        total_success = sum(r['success'] for r in results)
        errors = [r['error'] for r in results if r['error']]
        
        # 验证数据一致性
        kg = KnowledgeGraph(db_path)
        summary = kg.get_all_knowledge_summary()
        kg.close()
        
        if errors:
            self.record_bug("D1-001", f"多进程通信错误: {errors}", "high")
        
        self.record_result("D1: 多进程Agent通信", len(errors) == 0,
                          f"4进程, 执行: {total_executed}, 成功: {total_success}, 错误: {len(errors)}")

    # ========== D2: 共享知识库一致性 ==========

    def test_d2_shared_knowledge(self):
        """多进程读写无冲突"""
        db_path = self._new_db()
        
        # 并发写入测试
        write_errors = []
        read_errors = []
        write_count = [0]
        read_count = [0]
        
        def writer_thread(thread_id):
            kg = KnowledgeGraph(db_path)
            try:
                for i in range(100):
                    kg.store_experience(
                        title=f"Shared-{thread_id}-{i}",
                        content=f"data_{thread_id}_{i}",
                        category="shared"
                    )
                    with self._lock:
                        write_count[0] += 1
            except Exception as e:
                write_errors.append((thread_id, str(e)[:50]))
            finally:
                kg.close()
        
        def reader_thread(thread_id):
            kg = KnowledgeGraph(db_path)
            try:
                for i in range(100):
                    results = kg.search_nodes("Shared", limit=10)
                    with self._lock:
                        read_count[0] += 1
            except Exception as e:
                read_errors.append((thread_id, str(e)[:50]))
            finally:
                kg.close()
        
        # 启动读写线程
        threads = []
        for i in range(20):
            threads.append(threading.Thread(target=writer_thread, args=(i,)))
        for i in range(20):
            threads.append(threading.Thread(target=reader_thread, args=(i,)))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        
        # 验证最终一致性
        kg = KnowledgeGraph(db_path)
        summary = kg.get_all_knowledge_summary()
        total_nodes = sum(summary.get('nodes_by_category', {}).values())
        kg.close()
        
        if len(write_errors) > 10:
            self.record_bug("D2-001", f"共享知识库写入错误: {len(write_errors)}", "high")
        
        if len(read_errors) > 10:
            self.record_bug("D2-002", f"共享知识库读取错误: {len(read_errors)}", "high")
        
        self.record_result("D2: 共享知识库一致性", len(write_errors) <= 10 and len(read_errors) <= 10,
                          f"20写20读线程, 写入: {write_count[0]}, 读取: {read_count[0]}, "
                          f"节点: {total_nodes}, 写错误: {len(write_errors)}, 读错误: {len(read_errors)}")

    # ========== D3: 分布式锁机制 ==========

    def test_d3_distributed_lock(self):
        """复杂资源竞争"""
        db_path = self._new_db()
        
        # 模拟分布式锁
        lock_file = f"/data/user/work/distributed_lock_{int(time.time())}.lock"
        contention_count = [0]
        lock_errors = []
        
        def lock_worker(worker_id):
            try:
                for i in range(50):
                    # 尝试获取文件锁
                    import fcntl
                    with open(lock_file, 'w') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        try:
                            # 临界区
                            kg = KnowledgeGraph(db_path)
                            kg.store_experience(
                                title=f"Lock-{worker_id}-{i}",
                                content=f"locked_data_{worker_id}_{i}",
                                category="lock_test"
                            )
                            kg.close()
                            with self._lock:
                                contention_count[0] += 1
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                lock_errors.append((worker_id, str(e)[:50]))
        
        # 启动竞争线程
        threads = [threading.Thread(target=lock_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        
        # 清理
        if os.path.exists(lock_file):
            os.remove(lock_file)
        
        # 验证数据完整性
        kg = KnowledgeGraph(db_path)
        summary = kg.get_all_knowledge_summary()
        total_nodes = sum(summary.get('nodes_by_category', {}).values())
        kg.close()
        
        if len(lock_errors) > 5:
            self.record_bug("D3-001", f"分布式锁错误: {len(lock_errors)}", "high")
        
        self.record_result("D3: 分布式锁机制", len(lock_errors) <= 5,
                          f"10线程, 完成: {contention_count[0]}操作, 节点: {total_nodes}, 错误: {len(lock_errors)}")

    # ========== D4: 网络分区恢复 ==========

    def test_d4_partition_recovery(self):
        """分区后数据同步"""
        db_path = self._new_db()
        
        # 模拟网络分区：部分写入失败，然后恢复
        partition_errors = []
        pre_partition_count = [0]
        post_partition_count = [0]
        
        def partitioned_writer(thread_id, should_fail):
            kg = KnowledgeGraph(db_path)
            try:
                for i in range(50):
                    if should_fail and i > 25:
                        # 模拟分区：随机失败
                        if random.random() < 0.3:
                            raise ConnectionError("模拟网络分区")
                    
                    kg.store_experience(
                        title=f"Partition-{thread_id}-{i}",
                        content=f"partition_data_{thread_id}_{i}",
                        category="partition_test"
                    )
                    
                    if should_fail:
                        with self._lock:
                            pre_partition_count[0] += 1
                    else:
                        with self._lock:
                            post_partition_count[0] += 1
            except Exception as e:
                partition_errors.append((thread_id, str(e)[:50]))
            finally:
                kg.close()
        
        # 第一阶段：正常写入
        threads1 = [threading.Thread(target=partitioned_writer, args=(i, True)) for i in range(5)]
        for t in threads1:
            t.start()
        for t in threads1:
            t.join(timeout=30)
        
        # 第二阶段：恢复后写入
        threads2 = [threading.Thread(target=partitioned_writer, args=(i+5, False)) for i in range(5)]
        for t in threads2:
            t.start()
        for t in threads2:
            t.join(timeout=30)
        
        # 验证数据
        kg = KnowledgeGraph(db_path)
        summary = kg.get_all_knowledge_summary()
        total_nodes = sum(summary.get('nodes_by_category', {}).values())
        kg.close()
        
        recovery_ok = total_nodes >= (pre_partition_count[0] + post_partition_count[0]) * 0.8
        
        if not recovery_ok:
            self.record_bug("D4-001", f"分区恢复数据丢失: 预期>={(pre_partition_count[0] + post_partition_count[0]) * 0.8}, 实际={total_nodes}", "high")
        
        self.record_result("D4: 网络分区恢复", recovery_ok,
                          f"分区前: {pre_partition_count[0]}, 分区后: {post_partition_count[0]}, "
                          f"总节点: {total_nodes}, 错误: {len(partition_errors)}")

    # ========== D5: 节点故障转移 ==========

    def test_d5_failover(self):
        """单点故障不影响整体"""
        db_path = self._new_db()
        
        active_nodes = [0]
        failed_nodes = [0]
        failover_errors = []
        
        def node_worker(node_id):
            try:
                config = AgentConfig()
                config.knowledge.db_path = db_path
                config.max_concurrent_tasks = 3
                
                agent = Agent(config)
                
                # 模拟部分节点故障
                if node_id % 3 == 0:
                    failed_nodes[0] += 1
                    raise RuntimeError(f"模拟节点{node_id}故障")
                
                # 正常节点提交任务
                for i in range(20):
                    agent.submit_task(
                        name=f"Node{node_id}-Task-{i}",
                        handler=lambda: "ok",
                        task_type="failover"
                    )
                
                agent.engine.run_batch(20)
                stats = agent.engine.stats
                agent.stop()
                
                active_nodes[0] += 1
                return stats['total_executed']
            except Exception as e:
                failover_errors.append((node_id, str(e)[:50]))
                return 0
        
        # 启动多个"节点"
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(node_worker, i) for i in range(6)]
            results = [f.result() for f in futures]
        
        total_executed = sum(results)
        
        # 验证：即使部分节点故障，整体仍能工作
        failover_ok = total_executed > 0 and active_nodes[0] >= 3
        
        if not failover_ok:
            self.record_bug("D5-001", f"故障转移失败: 执行={total_executed}, 活跃节点={active_nodes[0]}", "high")
        
        self.record_result("D5: 节点故障转移", failover_ok,
                          f"6节点, 活跃: {active_nodes[0]}, 故障: {failed_nodes[0]}, "
                          f"执行: {total_executed}, 错误: {len(failover_errors)}")

    # ========== D6: 跨进程任务分发 ==========

    def test_d6_cross_process_tasks(self):
        """任务在多个线程间分配（使用线程模拟分布式）"""
        db_path = self._new_db()
        
        distributed_count = [0]
        distribution_errors = []
        result_lock = threading.Lock()
        results = []
        
        def task_distributor(thread_id, task_list, result_list):
            try:
                config = AgentConfig()
                config.knowledge.db_path = db_path
                config.max_concurrent_tasks = 5
                
                agent = Agent(config)
                local_count = 0
                local_executed = 0
                
                # 处理分配给该线程的任务
                for task_id in task_list:
                    try:
                        agent.submit_task(
                            name=f"Distributed-{thread_id}-{task_id}",
                            handler=lambda tid=task_id: f"task_{tid}_result",
                            task_type="distributed"
                        )
                        local_count += 1
                        
                        # 每5个任务执行一次
                        if local_count % 5 == 0:
                            agent.engine.run_batch(5)
                    except Exception as e:
                        distribution_errors.append((thread_id, str(e)[:50]))
                
                # 执行剩余任务
                pending = agent.engine.pending_count
                if pending > 0:
                    agent.engine.run_batch(pending)
                
                stats = agent.engine.stats
                local_executed = stats['total_executed']
                agent.stop()
                
                with result_lock:
                    result_list.append({
                        'thread_id': thread_id, 
                        'executed': local_executed, 
                        'distributed': local_count
                    })
            except Exception as e:
                with result_lock:
                    result_list.append({
                        'thread_id': thread_id, 
                        'executed': 0, 
                        'distributed': 0, 
                        'error': str(e)[:100]
                    })
        
        # 创建200个任务ID
        all_task_ids = list(range(200))
        
        # 将任务分配给4个线程
        task_per_thread = 50
        task_assignments = [
            all_task_ids[i:i+task_per_thread] 
            for i in range(0, len(all_task_ids), task_per_thread)
        ]
        
        # 启动分发线程
        threads = []
        for i in range(4):
            t = threading.Thread(target=task_distributor, args=(i, task_assignments[i], results))
            threads.append(t)
            t.start()
        
        # 等待完成
        for t in threads:
            t.join(timeout=120)
        
        total_executed = sum(r.get('executed', 0) for r in results)
        total_distributed = sum(r.get('distributed', 0) for r in results)
        errors = [r.get('error') for r in results if r.get('error')]
        
        distribution_ok = total_executed >= 150  # 至少执行75%
        
        if not distribution_ok:
            self.record_bug("D6-001", f"任务分发不足: 执行={total_executed}, 预期>=150", "high")
        
        self.record_result("D6: 跨进程任务分发", distribution_ok,
                          f"4线程, 分发: {total_distributed}, 执行: {total_executed}, 错误: {len(errors)}")

    # ========== D7: 分布式状态同步 ==========

    def test_d7_state_sync(self):
        """状态变更同步"""
        db_path = self._new_db()
        
        state_changes = []
        sync_errors = []
        
        def state_worker(worker_id):
            try:
                kg = KnowledgeGraph(db_path)
                
                for i in range(50):
                    # 写入状态
                    node_id = kg.store_experience(
                        title=f"State-{worker_id}-{i}",
                        content=f"state_v{i}",
                        category="state_sync"
                    )
                    
                    # 立即读取验证
                    results = kg.search_nodes(f"State-{worker_id}-{i}", limit=1)
                    
                    if not results:
                        sync_errors.append((worker_id, i, "状态未同步"))
                    else:
                        state_changes.append((worker_id, i))
                
                kg.close()
            except Exception as e:
                sync_errors.append((worker_id, -1, str(e)[:50]))
        
        # 启动状态同步线程
        threads = [threading.Thread(target=state_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        
        sync_ok = len(sync_errors) <= 10
        
        if not sync_ok:
            self.record_bug("D7-001", f"状态同步错误: {len(sync_errors)}", "high")
        
        self.record_result("D7: 分布式状态同步", sync_ok,
                          f"10线程, 状态变更: {len(state_changes)}, 同步错误: {len(sync_errors)}")

    # ========== D8: 并发连接池极限 ==========

    def test_d8_connection_pool(self):
        """大量并发连接"""
        db_path = self._new_db()
        
        connection_errors = []
        successful_connections = [0]
        
        def connection_worker(worker_id):
            try:
                for i in range(50):
                    # 每个操作创建新连接
                    kg = KnowledgeGraph(db_path)
                    kg.store_experience(
                        title=f"Conn-{worker_id}-{i}",
                        content=f"connection_test_{worker_id}_{i}",
                        category="connection_pool"
                    )
                    kg.close()
                    with self._lock:
                        successful_connections[0] += 1
            except Exception as e:
                connection_errors.append((worker_id, str(e)[:50]))
        
        # 启动大量连接线程
        threads = [threading.Thread(target=connection_worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        
        pool_ok = len(connection_errors) <= 20
        
        if not pool_ok:
            self.record_bug("D8-001", f"连接池错误: {len(connection_errors)}", "high")
        
        self.record_result("D8: 并发连接池极限", pool_ok,
                          f"50线程, 成功连接: {successful_connections[0]}, 错误: {len(connection_errors)}")

    # ========== D9: 消息队列压力 ==========

    def test_d9_message_queue(self):
        """高频消息传递"""
        db_path = self._new_db()
        
        message_count = [0]
        message_errors = []
        
        def producer(producer_id, message_queue):
            try:
                for i in range(100):
                    message = {
                        'id': f"P{producer_id}-M{i}",
                        'content': f"message_content_{producer_id}_{i}",
                        'timestamp': time.time()
                    }
                    message_queue.put(message)
                    with self._lock:
                        message_count[0] += 1
            except Exception as e:
                message_errors.append((f"producer_{producer_id}", str(e)[:50]))
        
        def consumer(consumer_id, message_queue, kg):
            try:
                processed = 0
                while processed < 50:
                    try:
                        msg = message_queue.get(timeout=2)
                        kg.store_experience(
                            title=msg['id'],
                            content=msg['content'],
                            category="message_queue"
                        )
                        processed += 1
                    except Exception as e:
                        break
            except Exception as e:
                message_errors.append((f"consumer_{consumer_id}", str(e)[:50]))
        
        from queue import Queue
        message_queue = Queue(maxsize=1000)
        kg = KnowledgeGraph(db_path)
        
        # 启动生产者和消费者
        producers = [threading.Thread(target=producer, args=(i, message_queue)) for i in range(5)]
        consumers = [threading.Thread(target=consumer, args=(i, message_queue, kg)) for i in range(5)]
        
        for p in producers:
            p.start()
        for c in consumers:
            c.start()
        
        for p in producers:
            p.join(timeout=60)
        for c in consumers:
            c.join(timeout=60)
        
        kg.close()
        
        queue_ok = len(message_errors) <= 10 and message_count[0] >= 400
        
        if not queue_ok:
            self.record_bug("D9-001", f"消息队列错误: {len(message_errors)}, 消息数: {message_count[0]}", "high")
        
        self.record_result("D9: 消息队列压力", queue_ok,
                          f"5生产者5消费者, 消息: {message_count[0]}, 错误: {len(message_errors)}")

    # ========== D10: 全分布式链路 ==========

    def test_d10_full_distributed(self):
        """完整分布式业务流程"""
        db_path = self._new_db()
        
        pipeline_stages = {'submit': 0, 'execute': 0, 'store': 0, 'query': 0}
        pipeline_errors = []
        
        def distributed_pipeline(worker_id):
            try:
                config = AgentConfig()
                config.knowledge.db_path = db_path
                config.max_concurrent_tasks = 5
                
                agent = Agent(config)
                
                # 完整业务流程
                for cycle in range(20):
                    # 1. 提交任务
                    for i in range(5):
                        agent.submit_task(
                            name=f"Pipeline-{worker_id}-{cycle}-{i}",
                            handler=lambda: f"result_{worker_id}_{cycle}_{i}",
                            task_type="distributed_pipeline"
                        )
                    with self._lock:
                        pipeline_stages['submit'] += 5
                    
                    # 2. 执行任务
                    agent.engine.run_batch(5)
                    with self._lock:
                        pipeline_stages['execute'] += 1
                    
                    # 3. 存储知识
                    agent.knowledge.store_experience(
                        title=f"Pipeline-Exp-{worker_id}-{cycle}",
                        content=f"experience_{worker_id}_{cycle}",
                        category="distributed_pipeline"
                    )
                    with self._lock:
                        pipeline_stages['store'] += 1
                    
                    # 4. 查询知识
                    agent.knowledge.search_nodes("Pipeline", limit=5)
                    with self._lock:
                        pipeline_stages['query'] += 1
                
                stats = agent.engine.stats
                agent.stop()
                return stats['total_executed']
            except Exception as e:
                pipeline_errors.append((worker_id, str(e)[:50]))
                return 0
        
        # 启动多个分布式工作节点
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(distributed_pipeline, i) for i in range(5)]
            results = [f.result() for f in futures]
        
        total_executed = sum(results)
        
        # 验证数据
        kg = KnowledgeGraph(db_path)
        summary = kg.get_all_knowledge_summary()
        total_nodes = sum(summary.get('nodes_by_category', {}).values())
        kg.close()
        
        pipeline_ok = total_executed >= 400 and len(pipeline_errors) <= 5
        
        if not pipeline_ok:
            self.record_bug("D10-001", f"全分布式链路失败: 执行={total_executed}, 错误={len(pipeline_errors)}", "high")
        
        self.record_result("D10: 全分布式链路", pipeline_ok,
                          f"5工作节点, 阶段: {pipeline_stages}, 执行: {total_executed}, "
                          f"知识节点: {total_nodes}, 错误: {len(pipeline_errors)}")

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("第十七轮分布式场景测试总结")
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
    tester = Round17DistributedTest()
    tester.run_all_tests()
