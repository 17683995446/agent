"""
第二十三轮长时间运行稳定性测试 - 模拟72小时持续运行
测试场景：内存泄漏、线程泄漏、连接池耗尽、句柄耗尽、日志膨胀、知识图谱膨胀
"""
import os
import sys
import time
import json
import random
import threading
import tempfile
import shutil
import gc
import tracemalloc
import sqlite3
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus
from agent.modules.knowledge_graph import KnowledgeGraph

import logging
# 降低日志级别避免干扰
logging.basicConfig(level=logging.WARNING, format='%(name)s [%(levelname)s] %(message)s')


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.snapshots = []
        self._lock = threading.Lock()
    
    def snapshot(self, label: str = "") -> Dict[str, Any]:
        """获取当前资源快照"""
        import sys
        
        # 内存使用
        mem_info = {}
        try:
            import resource
            mem_info['rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except:
            mem_info['rss_mb'] = 0
        
        # 对象计数
        obj_count = len(gc.get_objects())
        
        # 线程数
        thread_count = threading.active_count()
        
        # SQLite连接数（通过检查打开的文件）
        sqlite_connections = 0
        try:
            import psutil
            proc = psutil.Process()
            sqlite_connections = len([f for f in proc.open_files() if f.path.endswith('.db')])
        except:
            pass
        
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'memory_mb': mem_info['rss_mb'],
            'objects': obj_count,
            'threads': thread_count,
            'sqlite_conns': sqlite_connections,
        }
        
        with self._lock:
            self.snapshots.append(snapshot)
        
        return snapshot
    
    def get_trend(self) -> Dict[str, Any]:
        """获取资源趋势"""
        if len(self.snapshots) < 2:
            return {}
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        return {
            'memory_growth_mb': last['memory_mb'] - first['memory_mb'],
            'object_growth': last['objects'] - first['objects'],
            'thread_growth': last['threads'] - first['threads'],
            'duration_sec': last['timestamp'] - first['timestamp'],
            'snapshot_count': len(self.snapshots),
        }


class Round23StabilityTest:
    """长时间运行稳定性测试套件"""

    def __init__(self):
        self.results = []
        self.bugs = []
        self._db_counter = 0
        self._test_dir = tempfile.mkdtemp(prefix="stability_r23_")
        self.monitor = ResourceMonitor()
        print(f"[初始化] 稳定性测试目录: {self._test_dir}")

    def _new_db(self):
        self._db_counter += 1
        return os.path.join(self._test_dir, f"stability_r23_{self._db_counter}.db")

    def record(self, name, passed, details=""):
        self.results.append({"test": name, "passed": passed, "details": details})
        status = "✓" if passed else "✗"
        print(f"  [{status}] {name}: {details[:150]}")

    def bug(self, bid, desc, sev="medium"):
        self.bugs.append({"id": bid, "description": desc, "severity": sev})
        print(f"  🐛 BUG [{bid}] ({sev}): {desc[:120]}")

    def cleanup(self):
        """清理测试目录"""
        try:
            shutil.rmtree(self._test_dir, ignore_errors=True)
            print(f"[清理] 测试目录已删除: {self._test_dir}")
        except Exception as e:
            print(f"[警告] 清理失败: {e}")

    def run_all(self):
        print("\n" + "=" * 70)
        print("第二十三轮长时间运行稳定性测试")
        print("=" * 70)
        print(f"测试时间: {datetime.now().isoformat()}")
        print(f"测试目录: {self._test_dir}")
        print("=" * 70)

        # 启动内存追踪
        tracemalloc.start()
        self.monitor.snapshot("start")

        tests = [
            ("S1: 内存泄漏-持续任务执行", self.test_s1_memory_leak),
            ("S2: 线程泄漏-反复启停", self.test_s2_thread_leak),
            ("S3: 连接池耗尽-大量数据库操作", self.test_s3_connection_exhaustion),
            ("S4: 句柄泄漏-文件操作", self.test_s4_handle_leak),
            ("S5: 知识图谱膨胀-大量数据存储", self.test_s5_knowledge_bloat),
            ("S6: 任务队列积压-持续提交", self.test_s6_queue_backlog),
            ("S7: 优化器稳定性-长时间运行", self.test_s7_optimizer_stability),
            ("S8: 配置热更新循环-反复更新", self.test_s8_config_reload),
            ("S9: 综合压力测试-模拟生产负载", self.test_s9_production_load),
            ("S10: 资源清理验证-关闭后检查", self.test_s10_resource_cleanup),
        ]

        for name, func in tests:
            print(f"\n{'=' * 60}")
            print(f"执行: {name}")
            print(f"{'=' * 60}")
            try:
                func()
                self.monitor.snapshot(name)
            except Exception as e:
                import traceback
                self.record(name, False, f"异常: {str(e)[:100]}")
                self.bug(f"S-{name[:3]}", f"异常: {traceback.format_exc()[:200]}", "high")

        # 停止内存追踪
        tracemalloc.stop()
        self.monitor.snapshot("end")
        
        self._summary()
        self.cleanup()

    # ========== S1: 内存泄漏-持续任务执行 ==========

    def test_s1_memory_leak(self):
        """持续执行大量任务，检测内存是否持续增长"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)
        errors = []
        
        # 记录初始内存
        gc.collect()
        initial_mem = len(gc.get_objects())
        print(f"    [S1] 初始对象数: {initial_mem}")

        # 执行多轮任务
        rounds = 20
        tasks_per_round = 50
        
        for r in range(rounds):
            # 提交并执行任务
            for i in range(tasks_per_round):
                agent.submit_task(
                    name=f"mem-r{r}-t{i}",
                    handler=lambda x=random.random(): f"result_{x}"
                )
            agent.engine.run_batch(tasks_per_round)
            
            # 每轮后强制GC并检查
            if r % 5 == 0:
                gc.collect()
                current_obj = len(gc.get_objects())
                growth = current_obj - initial_mem
                print(f"    [S1] Round {r}: 对象数={current_obj}, 增长={growth}")

        # 最终检查
        gc.collect()
        final_obj = len(gc.get_objects())
        obj_growth = final_obj - initial_mem
        
        print(f"    [S1] 最终对象数: {final_obj}, 总增长: {obj_growth}")
        
        agent.stop()

        # 判断：对象增长不应超过初始的50%（允许一定增长）
        if obj_growth > initial_mem * 0.5:
            errors.append(f"对象增长过多: {obj_growth} (初始: {initial_mem})")
            self.bug("S1-001", f"内存泄漏嫌疑: 对象从{initial_mem}增长到{final_obj}", "high")

        self.record("S1: 内存泄漏-持续任务执行", not errors,
                    f"对象增长: {obj_growth}, 问题: {errors if errors else '无'}")

    # ========== S2: 线程泄漏-反复启停 ==========

    def test_s2_thread_leak(self):
        """反复启动和停止Agent，检测线程是否泄漏"""
        errors = []
        
        initial_threads = threading.active_count()
        print(f"    [S2] 初始线程数: {initial_threads}")

        # 反复启停
        iterations = 10
        for i in range(iterations):
            db_path = self._new_db()
            config = AgentConfig()
            config.knowledge.db_path = db_path
            
            agent = Agent(config)
            
            # 提交一些任务
            for j in range(10):
                agent.submit_task(name=f"thread-{i}-{j}", handler=lambda: "ok")
            agent.engine.run_batch(10)
            
            # 停止
            agent.stop()
            
            # 检查线程数
            current_threads = threading.active_count()
            if i % 3 == 0:
                print(f"    [S2] 迭代 {i}: 线程数={current_threads}")

        # 等待所有线程清理
        time.sleep(2)
        final_threads = threading.active_count()
        thread_growth = final_threads - initial_threads
        
        print(f"    [S2] 最终线程数: {final_threads}, 增长: {thread_growth}")

        # 判断：线程增长不应超过2
        if thread_growth > 2:
            errors.append(f"线程泄漏: 增长{thread_growth}个线程")
            self.bug("S2-001", f"线程泄漏: 从{initial_threads}增长到{final_threads}", "high")

        self.record("S2: 线程泄漏-反复启停", not errors,
                    f"线程增长: {thread_growth}, 问题: {errors if errors else '无'}")

    # ========== S3: 连接池耗尽-大量数据库操作 ==========

    def test_s3_connection_exhaustion(self):
        """大量并发数据库操作，检测连接是否耗尽"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)
        errors = []
        
        # 并发写入知识图谱
        def write_knowledge(thread_id):
            for i in range(20):
                try:
                    agent.knowledge.store_experience(
                        category="test",
                        title=f"thread-{thread_id}-exp-{i}",
                        content=f"content_{'x'*100}",
                        importance=random.uniform(0.1, 0.9)
                    )
                except Exception as e:
                    return str(e)
            return None

        threads = 20
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(write_knowledge, i) for i in range(threads)]
            results = [f.result() for f in as_completed(futures)]
        
        errors_in_threads = [r for r in results if r]
        if errors_in_threads:
            errors.extend(errors_in_threads[:3])  # 只记录前3个错误

        # 检查数据库状态
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM nodes")
        node_count = cursor.fetchone()[0]
        conn.close()

        agent.stop()

        if errors_in_threads:
            self.bug("S3-001", f"并发数据库操作失败: {errors_in_threads[0][:100]}", "high")

        self.record("S3: 连接池耗尽-大量数据库操作", not errors_in_threads,
                    f"写入节点数: {node_count}, 错误: {len(errors_in_threads)}")

    # ========== S4: 句柄泄漏-文件操作 ==========

    def test_s4_handle_leak(self):
        """反复创建和关闭数据库，检测文件句柄是否泄漏"""
        errors = []
        
        # 获取初始打开文件数
        initial_files = 0
        try:
            import psutil
            proc = psutil.Process()
            initial_files = len(proc.open_files())
        except:
            pass

        print(f"    [S4] 初始打开文件数: {initial_files}")

        # 反复创建和关闭KnowledgeGraph
        iterations = 50
        for i in range(iterations):
            db_path = self._new_db()
            kg = KnowledgeGraph(db_path)
            
            # 执行一些操作
            kg.store_experience("test", f"handle-{i}", "content", 0.5)
            kg.get_all_knowledge_summary()
            
            kg.close()
            
            if i % 10 == 0:
                gc.collect()

        # 强制GC
        gc.collect()
        time.sleep(1)

        # 检查最终打开文件数
        final_files = initial_files
        try:
            import psutil
            proc = psutil.Process()
            final_files = len(proc.open_files())
        except:
            pass

        file_growth = final_files - initial_files
        print(f"    [S4] 最终打开文件数: {final_files}, 增长: {file_growth}")

        # 判断：文件句柄增长不应超过5
        if file_growth > 5:
            errors.append(f"文件句柄泄漏: 增长{file_growth}个")
            self.bug("S4-001", f"句柄泄漏: 打开文件从{initial_files}增长到{final_files}", "high")

        self.record("S4: 句柄泄漏-文件操作", not errors,
                    f"文件句柄增长: {file_growth}, 问题: {errors if errors else '无'}")

    # ========== S5: 知识图谱膨胀-大量数据存储 ==========

    def test_s5_knowledge_bloat(self):
        """存储大量知识，检测数据库是否正常处理"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)
        errors = []

        # 存储大量知识
        batch_size = 100
        batches = 20
        
        for b in range(batches):
            experiences = []
            for i in range(batch_size):
                experiences.append({
                    'category': random.choice(['experience', 'strategy', 'insight']),
                    'title': f"batch-{b}-exp-{i}",
                    'content': f"content_{'x'*random.randint(50, 200)}",
                    'importance': random.uniform(0.1, 0.9),
                })
            
            try:
                # 使用批量插入
                agent.knowledge.store_experience_batch(experiences)
            except Exception as e:
                errors.append(f"批量存储失败: {str(e)[:50]}")

            if b % 5 == 0:
                print(f"    [S5] 批次 {b}: 已存储 {b * batch_size} 条")

        # 检查数据库大小
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        
        # 检查查询性能
        start_time = time.time()
        summary = agent.knowledge.get_all_knowledge_summary()
        query_time = time.time() - start_time

        agent.stop()

        print(f"    [S5] 数据库大小: {db_size_mb:.2f}MB, 查询耗时: {query_time:.3f}s")

        # 判断：查询时间不应超过5秒
        if query_time > 5:
            errors.append(f"查询性能下降: {query_time:.3f}s")
            self.bug("S5-001", f"知识图谱膨胀导致查询缓慢: {query_time:.3f}s", "medium")

        self.record("S5: 知识图谱膨胀-大量数据存储", not errors,
                    f"存储: {batches * batch_size}条, 大小: {db_size_mb:.2f}MB, 查询: {query_time:.3f}s")

    # ========== S6: 任务队列积压-持续提交 ==========

    def test_s6_queue_backlog(self):
        """持续提交任务，检测队列积压处理"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.task_queue_size = 100

        agent = Agent(config)
        errors = []
        
        submitted = 0
        rejected = 0
        
        # 持续提交任务
        for i in range(500):
            try:
                agent.submit_task(
                    name=f"backlog-{i}",
                    handler=lambda: time.sleep(0.01)
                )
                submitted += 1
            except RuntimeError as e:
                if "队列已满" in str(e):
                    rejected += 1

        # 执行部分任务
        pending = agent.engine.pending_count
        print(f"    [S6] 提交: {submitted}, 拒绝: {rejected}, 待处理: {pending}")

        # 执行所有待处理任务
        while pending > 0:
            batch = min(50, pending)
            agent.engine.run_batch(batch)
            pending = agent.engine.pending_count

        stats = agent.engine.stats
        agent.stop()

        # 判断：应该有任务被拒绝，且执行的任务数合理
        if rejected == 0 and submitted == 500:
            errors.append("队列无限制，可能存在OOM风险")
            self.bug("S6-001", "任务队列无限制", "high")

        self.record("S6: 任务队列积压-持续提交", not errors,
                    f"提交: {submitted}, 拒绝: {rejected}, 执行: {stats['total_executed']}")

    # ========== S7: 优化器稳定性-长时间运行 ==========

    def test_s7_optimizer_stability(self):
        """长时间运行优化器，检测稳定性"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.optimize_interval = 1  # 1秒优化一次

        agent = Agent(config)
        errors = []

        # 提交大量任务触发优化
        for i in range(100):
            agent.submit_task(
                name=f"opt-{i}",
                handler=lambda: f"result_{random.random()}",
                task_type=random.choice(['analysis', 'generation', 'review'])
            )

        # 执行任务
        agent.engine.run_batch(100)
        
        # 等待优化器处理
        time.sleep(3)

        # 检查优化器状态
        try:
            # 触发手动优化
            agent.optimizer.optimize_knowledge()
        except Exception as e:
            errors.append(f"优化失败: {str(e)[:50]}")

        stats = agent.engine.stats
        agent.stop()

        self.record("S7: 优化器稳定性-长时间运行", not errors,
                    f"执行: {stats['total_executed']}, 问题: {errors if errors else '无'}")

    # ========== S8: 配置热更新循环-反复更新 ==========

    def test_s8_config_reload(self):
        """反复更新配置，检测稳定性"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)
        errors = []

        # 反复更新配置
        for i in range(20):
            try:
                # 更新配置
                new_value = 2 + (i % 5)
                result = agent.update_config('agent.max_concurrent_tasks', new_value)
                
                if not result and agent.config_manager is not None:
                    errors.append(f"配置更新失败: round {i}")
            except Exception as e:
                errors.append(f"配置更新异常: {str(e)[:50]}")

        agent.stop()

        if errors:
            self.bug("S8-001", f"配置热更新不稳定: {errors[0]}", "medium")

        self.record("S8: 配置热更新循环-反复更新", not errors,
                    f"问题: {errors if errors else '无'}")

    # ========== S9: 综合压力测试-模拟生产负载 ==========

    def test_s9_production_load(self):
        """模拟真实生产负载"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 5
        config.task_queue_size = 200

        agent = Agent(config)
        errors = []
        
        # 模拟生产负载：多线程并发提交任务
        def producer_thread(thread_id, task_count):
            for i in range(task_count):
                try:
                    agent.submit_task(
                        name=f"prod-t{thread_id}-{i}",
                        handler=lambda: time.sleep(random.uniform(0.01, 0.05)),
                        task_type=random.choice(['analysis', 'generation', 'review', 'query'])
                    )
                except RuntimeError:
                    pass  # 队列满
                time.sleep(random.uniform(0.001, 0.01))

        def consumer_thread():
            while True:
                pending = agent.engine.pending_count
                if pending > 0:
                    agent.engine.run_batch(min(10, pending))
                time.sleep(0.1)

        # 启动生产者线程
        producers = [threading.Thread(target=producer_thread, args=(i, 50)) for i in range(5)]
        
        # 启动消费者线程
        consumer = threading.Thread(target=consumer_thread, daemon=True)
        consumer.start()

        for p in producers:
            p.start()

        for p in producers:
            p.join(timeout=30)

        # 等待任务完成
        time.sleep(2)
        
        # 最终执行
        pending = agent.engine.pending_count
        if pending > 0:
            agent.engine.run_batch(pending)

        stats = agent.engine.stats
        agent.stop()

        print(f"    [S9] 执行: {stats['total_executed']}, 成功: {stats['total_success']}, 失败: {stats['total_failed']}")

        # 判断：应该有大量任务成功执行
        if stats['total_executed'] < 50:
            errors.append(f"任务执行不足: {stats['total_executed']}")
            self.bug("S9-001", f"生产负载下任务执行不足: {stats['total_executed']}", "high")

        self.record("S9: 综合压力测试-模拟生产负载", not errors,
                    f"执行: {stats['total_executed']}, 成功: {stats['total_success']}")

    # ========== S10: 资源清理验证-关闭后检查 ==========

    def test_s10_resource_cleanup(self):
        """验证关闭后资源正确清理"""
        db_path = self._new_db()
        config = AgentConfig()
        config.knowledge.db_path = db_path

        # 记录初始状态
        initial_threads = threading.active_count()
        
        agent = Agent(config)
        
        # 提交任务
        for i in range(20):
            agent.submit_task(name=f"cleanup-{i}", handler=lambda: "ok")
        agent.engine.run_batch(20)
        
        # 记录运行时状态
        running_threads = threading.active_count()
        
        # 停止
        agent.stop()
        
        # 等待清理
        time.sleep(2)
        gc.collect()
        
        # 检查最终状态
        final_threads = threading.active_count()
        
        print(f"    [S10] 初始线程: {initial_threads}, 运行时: {running_threads}, 关闭后: {final_threads}")

        # 验证KG已关闭
        kg_closed = agent.knowledge._closed
        
        # 验证引擎已关闭
        engine_shutdown = agent.engine._shutdown

        errors = []
        if not kg_closed:
            errors.append("KnowledgeGraph未正确关闭")
        if not engine_shutdown:
            errors.append("TaskEngine未正确关闭")
        if final_threads > initial_threads + 1:  # 允许1个差异
            errors.append(f"线程未清理: 剩余{final_threads - initial_threads}个")

        if errors:
            self.bug("S10-001", f"资源清理不完整: {errors}", "high")

        self.record("S10: 资源清理验证-关闭后检查", not errors,
                    f"KG关闭: {kg_closed}, 引擎关闭: {engine_shutdown}, 线程: {final_threads}")

    def _summary(self):
        print("\n" + "=" * 70)
        print("第二十三轮长时间运行稳定性测试总结")
        print("=" * 70)

        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed if self.results else 0

        print(f"\n测试结果: {passed}/{len(self.results)} 通过, {failed} 失败")

        # 资源趋势
        trend = self.monitor.get_trend()
        if trend:
            print(f"\n资源趋势:")
            print(f"  内存增长: {trend.get('memory_growth_mb', 0):.2f}MB")
            print(f"  对象增长: {trend.get('object_growth', 0)}")
            print(f"  线程增长: {trend.get('thread_growth', 0)}")
            print(f"  测试时长: {trend.get('duration_sec', 0):.1f}秒")

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
    tester = Round23StabilityTest()
    tester.run_all()
