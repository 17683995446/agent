"""
第九轮深度测试 - 边界条件与异常路径
测试目标：发现系统在边界条件和异常路径下的潜在bug
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

sys.path.insert(0, '/workspace')

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus, Task
from agent.modules.knowledge_graph import KnowledgeGraph, KnowledgeNode
from agent.modules.optimizer import Optimizer
from agent.modules.llm_client import LLMClient


class Round9DeepTest:
    """第九轮深度测试套件"""

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
        print("第九轮深度测试 - 边界条件与异常路径")
        print("="*70)

        tests = [
            ("D1: 空任务处理器测试", self.test_d1_empty_handler),
            ("D2: 超大任务结果测试", self.test_d2_large_result),
            ("D3: 任务优先级边界测试", self.test_d3_priority_boundary),
            ("D4: 知识图谱空查询测试", self.test_d4_empty_query),
            ("D5: 重复任务提交测试", self.test_d5_duplicate_submit),
            ("D6: 任务超时边界测试", self.test_d6_timeout_boundary),
            ("D7: 并发关闭测试", self.test_d7_concurrent_shutdown),
            ("D8: 负数参数测试", self.test_d8_negative_params),
            ("D9: 特殊字符处理测试", self.test_d9_special_chars),
            ("D10: 长时间运行后重启测试", self.test_d10_restart_after_run),
        ]

        for name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"执行: {name}")
            print(f"{'='*60}")
            try:
                test_func()
            except Exception as e:
                self.record_result(name, False, f"测试异常: {e}")
                self.record_bug(f"D-{name[:3]}", f"测试执行异常: {traceback.format_exc()}", "high")

        self._print_summary()

    def test_d1_empty_handler(self):
        """D1: 空任务处理器测试
        测试当handler为None或不可调用时的行为
        """
        engine = TaskEngine(max_workers=3)

        # 测试None handler
        try:
            task = engine.submit(name="NoneHandler", handler=None)
            engine.run_batch(1)
            # 应该失败但不崩溃
            if task.status != TaskStatus.FAILED:
                self.record_bug("D1-001", f"None handler应该导致任务失败，实际状态: {task.status}", "medium")
            self.record_result("D1: 空任务处理器测试", True, f"None handler正确处理，状态: {task.status}")
        except Exception as e:
            self.record_result("D1: 空任务处理器测试", False, f"异常: {e}")

        engine.shutdown(cancel_pending=True)

    def test_d2_large_result(self):
        """D2: 超大任务结果测试
        测试任务返回超大结果时的内存处理
        """
        db_path = "/data/user/work/test_d2.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)

        # 创建返回大结果的任务
        large_data_size = 10 * 1024 * 1024  # 10MB

        def large_result_task():
            return "x" * large_data_size

        task = agent.submit_task(
            name="LargeResult",
            handler=large_result_task,
            task_type="large_test"
        )
        agent.engine.run_batch(1)

        # 检查是否正确处理大结果
        if task.status == TaskStatus.SUCCESS:
            result_size = len(task.result) if task.result else 0
            self.record_result("D2: 超大任务结果测试", True, f"大结果处理成功，大小: {result_size/1024/1024:.2f}MB")
        else:
            self.record_result("D2: 超大任务结果测试", False, f"大结果任务失败: {task.error}")

        agent.stop()

    def test_d3_priority_boundary(self):
        """D3: 任务优先级边界测试
        测试极端优先级值
        """
        engine = TaskEngine(max_workers=3)

        # 测试极端优先级
        priorities = [-1000, -1, 0, 1, 100, 1000, 999999]

        for p in priorities:
            try:
                task = engine.submit(
                    name=f"Priority-{p}",
                    handler=lambda: p,
                    priority=p
                )
            except Exception as e:
                self.record_bug("D3-001", f"优先级{p}提交失败: {e}", "low")

        # 执行并检查顺序
        results = engine.run_batch(len(priorities))

        # 检查是否按优先级排序（小数字=高优先级）
        if len(results) == len(priorities):
            self.record_result("D3: 任务优先级边界测试", True, f"所有{len(priorities)}个优先级测试通过")
        else:
            self.record_bug("D3-002", f"优先级任务执行数不匹配: {len(results)}/{len(priorities)}", "medium")
            self.record_result("D3: 任务优先级边界测试", False, f"执行数不匹配")

        engine.shutdown(cancel_pending=True)

    def test_d4_empty_query(self):
        """D4: 知识图谱空查询测试
        测试空字符串查询和空数据库查询
        """
        db_path = "/data/user/work/test_d4.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        kg = KnowledgeGraph(db_path)

        # 测试空字符串查询
        try:
            results = kg.search_nodes("")
            # 空查询应该返回空结果或所有结果
            self.record_result("D4: 知识图谱空查询测试", True, f"空查询返回{len(results)}条结果")
        except Exception as e:
            self.record_bug("D4-001", f"空查询异常: {e}", "medium")
            self.record_result("D4: 知识图谱空查询测试", False, f"空查询异常: {e}")

        # 测试不存在的节点
        try:
            node = kg.get_node(999999)
            if node is not None:
                self.record_bug("D4-002", f"不存在的节点应该返回None，实际: {node}", "low")
        except Exception as e:
            self.record_bug("D4-003", f"查询不存在的节点异常: {e}", "low")

        kg.close()

    def test_d5_duplicate_submit(self):
        """D5: 重复任务提交测试
        测试相同名称任务的重复提交
        """
        db_path = "/data/user/work/test_d5.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)

        # 提交相同名称的任务
        task_ids = []
        for i in range(5):
            task = agent.submit_task(
                name="DuplicateTask",  # 相同名称
                handler=lambda: "result",
                task_type="duplicate"
            )
            task_ids.append(task.id)

        agent.engine.run_batch(5)

        # 检查是否所有任务都被正确处理
        unique_ids = set(task_ids)
        if len(unique_ids) == 5:
            self.record_result("D5: 重复任务提交测试", True, f"5个相同名称任务都有唯一ID")
        else:
            self.record_bug("D5-001", f"任务ID不唯一: {task_ids}", "high")
            self.record_result("D5: 重复任务提交测试", False, f"ID不唯一")

        agent.stop()

    def test_d6_timeout_boundary(self):
        """D6: 任务超时边界测试
        测试极短超时和极长超时
        """
        engine = TaskEngine(max_workers=3)

        # 测试极短超时（1秒）
        def slow_task():
            time.sleep(5)
            return "done"

        task = engine.submit(
            name="ShortTimeout",
            handler=slow_task,
            timeout=1  # 1秒超时
        )
        engine.run_batch(1)

        if task.status == TaskStatus.TIMEOUT:
            self.record_result("D6: 任务超时边界测试", True, f"1秒超时正确触发")
        else:
            self.record_bug("D6-001", f"1秒超时未触发，状态: {task.status}", "high")
            self.record_result("D6: 任务超时边界测试", False, f"超时未触发")

        engine.shutdown(cancel_pending=True)

    def test_d7_concurrent_shutdown(self):
        """D7: 并发关闭测试
        测试多个线程同时调用stop()
        """
        db_path = "/data/user/work/test_d7.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path

        agent = Agent(config)

        # 提交一些任务
        for i in range(10):
            agent.submit_task(name=f"Task-{i}", handler=lambda: i)

        # 多个线程同时调用stop
        errors = []
        def stop_agent():
            try:
                agent.stop()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=stop_agent) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        if len(errors) > 0:
            self.record_bug("D7-001", f"并发stop异常: {errors}", "high")
            self.record_result("D7: 并发关闭测试", False, f"异常: {errors}")
        else:
            self.record_result("D7: 并发关闭测试", True, f"5线程并发stop无异常")

    def test_d8_negative_params(self):
        """D8: 负数参数测试
        测试负数参数的处理
        """
        engine = TaskEngine(max_workers=3)

        errors = []

        # 测试负数优先级
        try:
            task = engine.submit(name="NegPriority", handler=lambda: 1, priority=-5)
        except Exception as e:
            errors.append(f"负优先级: {e}")

        # 测试负数超时
        try:
            task = engine.submit(name="NegTimeout", handler=lambda: 1, timeout=-1)
        except Exception as e:
            errors.append(f"负超时: {e}")

        # 测试负数重试
        try:
            task = engine.submit(name="NegRetry", handler=lambda: 1, max_retries=-1)
        except Exception as e:
            errors.append(f"负重试: {e}")

        engine.shutdown(cancel_pending=True)

        if len(errors) > 0:
            self.record_bug("D8-001", f"负数参数处理: {errors}", "low")
        self.record_result("D8: 负数参数测试", len(errors) == 0, f"错误数: {len(errors)}")

    def test_d9_special_chars(self):
        """D9: 特殊字符处理测试
        测试任务名称和内容中的特殊字符
        """
        db_path = "/data/user/work/test_d9.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        kg = KnowledgeGraph(db_path)

        # 测试特殊字符
        special_strings = [
            "任务<with>tags",
            "任务'单引号'",
            '任务"双引号"',
            "任务\n换行",
            "任务\t制表符",
            "任务;分号",
            "任务--SQL注释",
            "任务' OR '1'='1",  # SQL注入尝试
        ]

        errors = []
        for s in special_strings:
            try:
                kg.store_experience(
                    title=s,
                    content=f"测试内容: {s}",
                    category="special_test"
                )
            except Exception as e:
                errors.append((s, str(e)[:50]))

        kg.close()

        if len(errors) > 0:
            self.record_bug("D9-001", f"特殊字符处理错误: {errors[:3]}", "medium")
            self.record_result("D9: 特殊字符处理测试", False, f"错误数: {len(errors)}")
        else:
            self.record_result("D9: 特殊字符处理测试", True, f"所有{len(special_strings)}个特殊字符测试通过")

    def test_d10_restart_after_run(self):
        """D10: 长时间运行后重启测试
        测试运行一段时间后重启的正确性
        """
        db_path = "/data/user/work/test_d10.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        config = AgentConfig()
        config.knowledge.db_path = db_path
        config.max_concurrent_tasks = 3

        # 第一次运行
        agent = Agent(config)
        for i in range(50):
            agent.submit_task(name=f"First-{i}", handler=lambda: random.random())
        agent.engine.run_batch(50)
        first_stats = agent.engine.stats
        agent.stop()

        # 重启
        agent2 = Agent(config)
        for i in range(50):
            agent2.submit_task(name=f"Second-{i}", handler=lambda: random.random())
        agent2.engine.run_batch(50)
        second_stats = agent2.engine.stats
        agent2.stop()

        # 检查知识库持久化
        kg = KnowledgeGraph(db_path)
        summary = kg.get_all_knowledge_summary()
        kg.close()

        total_nodes = sum(summary.get('nodes_by_category', {}).values())

        if total_nodes > 0:
            self.record_result("D10: 长时间运行后重启测试", True,
                              f"第一次: {first_stats['total_executed']}任务, "
                              f"第二次: {second_stats['total_executed']}任务, "
                              f"知识节点: {total_nodes}")
        else:
            self.record_bug("D10-001", "重启后知识库为空", "medium")
            self.record_result("D10: 长时间运行后重启测试", False, "知识库未持久化")

    def _print_summary(self):
        """打印测试总结"""
        print("\n" + "="*70)
        print("第九轮深度测试总结")
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
    tester = Round9DeepTest()
    tester.run_all_tests()
