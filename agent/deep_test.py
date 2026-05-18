#!/usr/bin/env python3
"""
第二轮深度压力测试 - 聚焦模块交互、边界条件、数据完整性

测试目标：
B1: 回调重复触发检测
B2: 统计计数准确性
B3: 完成列表重复检测
B4: LLM故障注入与恢复
B5: 非预期JSON结构处理
B6: 各种类型结果序列化
B7: 看门狗重启资源泄漏
B8: 知识图谱膨胀与清理
"""
import sys
import os
import time
import random
import json
import threading
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, Task, TaskStatus
from agent.modules.knowledge_graph import KnowledgeGraph
from agent.modules.llm_client import LLMClient
from agent.modules.optimizer import Optimizer


class DeepTestMetrics:
    def __init__(self):
        self.issues = []
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.lock = threading.Lock()

    def record_issue(self, test_name, severity, description):
        with self.lock:
            self.issues.append({
                "test": test_name,
                "severity": severity,  # critical / major / minor
                "description": description,
            })

    def pass_test(self, name):
        self.tests_run += 1
        self.tests_passed += 1
        print(f"  ✓ {name}")

    def fail_test(self, name, reason):
        self.tests_run += 1
        self.tests_failed += 1
        self.record_issue(name, "major", reason)
        print(f"  ✗ {name}: {reason}")

    def report(self):
        print("\n" + "=" * 70)
        print("深度测试报告")
        print("=" * 70)
        print(f"测试总数: {self.tests_run}")
        print(f"通过: {self.tests_passed}")
        print(f"失败: {self.tests_failed}")
        if self.issues:
            print(f"\n发现的问题 ({len(self.issues)}个):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. [{issue['severity'].upper()}] {issue['test']}: {issue['description']}")
        print("=" * 70)


M = DeepTestMetrics()


def clean_db():
    for f in ["data/knowledge.db", "data/stress_test.db", "data/test_deep.db"]:
        if os.path.exists(f):
            os.remove(f)


# ==================== B1+B2+B3: 回调重复/统计准确性/列表重复 ====================

def test_callback_and_stats():
    """测试B1:回调重复触发 B2:统计计数 B3:完成列表重复"""
    print("\n--- 测试B1+B2+B3: 回调重复触发与统计准确性 ---")

    callback_count = [0]
    callback_task_ids = []

    def counting_callback(task):
        callback_count[0] += 1
        callback_task_ids.append(task.id)

    engine = TaskEngine(max_workers=3, retry_limit=2, timeout=10)
    engine.on_complete(counting_callback)

    # 提交会失败然后重试的任务
    attempt = [0]
    def flaky_task():
        attempt[0] += 1
        if attempt[0] <= 1:
            raise ValueError("第一次失败")
        return "success"

    engine.submit("会重试的任务", flaky_task, max_retries=3)
    engine.run_batch(1)

    stats = engine.stats
    completed = engine.get_recent_completed(limit=100)

    # 检查B2: 统计准确性
    # 每个任务只应被计数一次（无论重试多少次）
    # _total_executed 应该等于 completed 列表长度
    if stats["total_executed"] == len(completed):
        M.pass_test(f"B2 统计一致性: executed={stats['total_executed']}, completed={len(completed)}")
    else:
        M.fail_test(f"B2 统计一致性",
                    f"executed={stats['total_executed']} != completed列表长度={len(completed)}")

    # 检查B3: 完成列表无重复
    completed_ids = [t.id for t in completed]
    unique_ids = set(completed_ids)
    if len(completed_ids) == len(unique_ids):
        M.pass_test("B3 完成列表无重复")
    else:
        dup_count = len(completed_ids) - len(unique_ids)
        M.fail_test("B3 完成列表无重复", f"发现{dup_count}个重复任务ID")

    # 检查B1: 回调触发次数
    # 每个完成的任务应只触发一次回调
    if callback_count[0] == len(completed):
        M.pass_test(f"B1 回调次数正确: {callback_count[0]}次")
    else:
        M.fail_test("B1 回调次数",
                    f"回调{callback_count[0]}次但只有{len(completed)}个完成任务")

    # 检查回调任务ID无重复
    if len(callback_task_ids) == len(set(callback_task_ids)):
        M.pass_test("B1 回调任务ID无重复")
    else:
        M.fail_test("B1 回调任务ID", f"回调收到重复ID: {callback_task_ids}")

    engine.shutdown()


# ==================== B4: LLM故障注入 ====================

def test_llm_fault_injection():
    """测试B4: LLM客户端故障处理"""
    print("\n--- 测试B4: LLM故障注入与恢复 ---")

    from agent.core.config import LLMConfig

    # 测试1: analyze_json对非JSON回复的处理
    config = LLMConfig(provider="openai", api_key="fake")
    llm = LLMClient(config)

    # 模拟模式返回的JSON可能不包含期望字段
    result = llm.analyze_json("返回: hello world")
    if isinstance(result, dict) and ("parsed" in result or "action" in result):
        M.pass_test("B4 非JSON回复处理: 返回了有效字典结构")
    else:
        M.fail_test("B4 非JSON回复处理", f"未返回有效字典: {result}")

    # 测试2: analyze_json对空回复的处理
    result2 = llm.analyze_json("")
    if result2 is not None:
        M.pass_test("B4 空回复处理: 返回了非None结果")
    else:
        M.fail_test("B4 空回复处理", "返回了None")

    # 测试3: chat对异常的处理
    result3 = llm.chat([{"role": "user", "content": "test"}])
    if result3 and len(result3) > 0:
        M.pass_test("B4 chat异常恢复: 返回了有效响应")
    else:
        M.fail_test("B4 chat异常恢复", f"返回了无效响应: {result3}")


# ==================== B5: 非预期JSON结构 ====================

def test_optimizer_unexpected_json():
    """测试B5: 优化器处理非预期JSON结构"""
    print("\n--- 测试B5: 优化器非预期JSON结构处理 ---")

    clean_db()
    kg = KnowledgeGraph("data/test_deep.db")

    # 创建一个模拟LLM，返回各种非预期结构
    class FaultyLLM:
        call_count = 0
        def analyze_json(self, prompt, system_prompt=""):
            self.call_count += 1
            # 返回缺少关键字段的结构
            return {"action": "skip"}  # 没有 experience_to_store, new_strategies 等

        def chat(self, messages, system_prompt="", **kwargs):
            return '{"action": "skip"}'

    optimizer = Optimizer(kg, FaultyLLM())

    # 测试reflect_on_tasks不崩溃
    try:
        from agent.modules.task_engine import Task
        task = Task(id="test1", name="测试任务", task_type="test", priority=5,
                     status=TaskStatus.SUCCESS, created_at=time.time(), max_retries=0, timeout=5)
        task.status = TaskStatus.SUCCESS
        task.duration = 1.0
        task.result = "ok"

        result = optimizer.reflect_on_tasks([task])
        M.pass_test("B5 reflect_on_tasks容错: 不崩溃")
    except Exception as e:
        M.fail_test("B5 reflect_on_tasks容错", str(e))

    # 测试analyze_failure不崩溃
    try:
        task2 = Task(id="test2", name="失败任务", task_type="test", priority=5,
                      status=TaskStatus.FAILED, created_at=time.time(), max_retries=1, timeout=5)
        task2.status = TaskStatus.FAILED
        task2.error = "测试错误"
        task2.retries = 1

        result2 = optimizer.analyze_failure(task2)
        M.pass_test("B5 analyze_failure容错: 不崩溃")
    except Exception as e:
        M.fail_test("B5 analyze_failure容错", str(e))

    kg.close()


# ==================== B6: 各种类型结果序列化 ====================

def test_result_serialization():
    """测试B6: 各种类型任务结果的序列化"""
    print("\n--- 测试B6: 各种类型结果序列化 ---")

    clean_db()
    kg = KnowledgeGraph("data/test_deep.db")

    class MockLLM:
        def analyze_json(self, prompt, system_prompt=""):
            return {"suggestions": ["test"]}
        def chat(self, messages, **kwargs):
            return '{"suggestions": ["test"]}'

    optimizer = Optimizer(kg, MockLLM())

    test_results = [
        ("字符串", "hello world"),
        ("字典", {"key": "value", "nested": {"a": 1}}),
        ("列表", [1, 2, 3] * 1000),  # 大列表
        ("None", None),
        ("数字", 42),
        ("布尔", True),
        ("空字符串", ""),
        ("特殊字符", "中文🚀\n\t\r特殊字符" * 100),
        ("大字典", {f"key_{i}": f"value_{i}" * 100 for i in range(100)}),
    ]

    for idx, (name, result_value) in enumerate(test_results):
        try:
            task = Task(id=f"ser_{idx}", name=f"序列化测试-{name}", task_type="test",
                        priority=5, status=TaskStatus.SUCCESS, created_at=time.time(),
                        max_retries=0, timeout=5)
            task.status = TaskStatus.SUCCESS
            task.duration = 1.0
            task.result = result_value

            optimizer.reflect_on_tasks([task])
            M.pass_test(f"B6 {name}序列化")
        except Exception as e:
            M.fail_test(f"B6 {name}序列化", str(e)[:100])

    kg.close()


# ==================== B7: 看门狗重启资源泄漏 ====================

def test_watchdog_resource_leak():
    """测试B7: 看门狗重启后资源泄漏"""
    print("\n--- 测试B7: 看门狗重启资源泄漏 ---")

    clean_db()
    config = AgentConfig()
    config.knowledge.db_path = "data/test_deep.db"

    # 创建一个会崩溃的Agent
    agent = Agent(config)

    crash_count = [0]
    def crashing_generator():
        crash_count[0] += 1
        if crash_count[0] <= 2:
            raise RuntimeError("模拟崩溃")
        return None  # 第三次正常

    agent.set_task_generator(crashing_generator)

    # 尝试运行（应该触发崩溃）
    try:
        agent.run(max_cycles=3, cycle_interval=0.1)
    except:
        pass

    # 检查数据库连接是否正常关闭
    # 通过尝试重新打开同一数据库来验证
    try:
        kg2 = KnowledgeGraph("data/test_deep.db")
        kg2.store_experience("重启后测试", "连接正常")
        count = len(kg2.search_nodes("重启后测试"))
        kg2.close()
        if count > 0:
            M.pass_test("B7 重启后数据库可正常访问")
        else:
            M.fail_test("B7 重启后数据库", "无法写入新数据")
    except Exception as e:
        M.fail_test("B7 重启后数据库", str(e)[:100])


# ==================== B8: 知识图谱膨胀与清理 ====================

def test_knowledge_bloat():
    """测试B8: 知识图谱无限膨胀"""
    print("\n--- 测试B8: 知识图谱膨胀与清理 ---")

    clean_db()
    kg = KnowledgeGraph("data/test_bloat.db")

    # 插入大量数据
    for i in range(500):
        kg.store_experience(
            title=f"膨胀测试_{i}",
            content="x" * 500,
            tags="bloat,test",
            importance=0.1,
        )

    summary_before = kg.get_all_knowledge_summary()
    total_before = sum(summary_before["nodes_by_category"].values())
    M.pass_test(f"B8 插入500条: 当前{total_before}条")

    # 测试清理低重要性节点
    conn = kg._get_connection()
    with kg._lock:
        c = conn.cursor()
        c.execute("DELETE FROM nodes WHERE importance < 0.2")
        deleted = c.rowcount
        conn.commit()

    summary_after = kg.get_all_knowledge_summary()
    total_after = sum(summary_after["nodes_by_category"].values())

    if total_after < total_before:
        M.pass_test(f"B8 清理低重要性: {total_before} → {total_after} (删除{deleted}条)")
    else:
        M.fail_test("B8 清理低重要性", "未成功清理")

    # 测试VACUUM
    try:
        kg.vacuum()
        M.pass_test("B8 VACUUM成功")
    except Exception as e:
        M.fail_test("B8 VACUUM", str(e)[:100])

    kg.close()


# ==================== 综合集成测试 ====================

def test_full_integration():
    """综合集成测试：长时间运行+模块交互"""
    print("\n--- 综合集成测试 ---")

    clean_db()
    config = AgentConfig()
    config.max_concurrent_tasks = 5
    config.task_retry_limit = 2
    config.task_timeout = 5
    config.optimize_interval = 8

    agent = Agent(config)

    # 混合任务生成器
    cycle = [0]
    def mixed_generator():
        cycle[0] += 1
        tasks = []

        # 正常任务
        tasks.append({
            "name": f"正常任务-{cycle[0]}",
            "handler": lambda: {"status": "ok", "cycle": cycle[0]},
            "task_type": "normal",
            "priority": 5,
        })

        # 偶尔失败任务
        if random.random() < 0.3:
            tasks.append({
                "name": f"可能失败-{cycle[0]}",
                "handler": lambda: (_ for _ in ()).throw(ValueError("随机失败")),
                "task_type": "flaky",
                "priority": 5,
                "max_retries": 2,
            })

        # 各种结果类型
        result_types = [
            lambda: {"dict": True},
            lambda: [1, 2, 3],
            lambda: None,
            lambda: "string result",
            lambda: 42,
        ]
        tasks.append({
            "name": f"结果类型测试-{cycle[0]}",
            "handler": random.choice(result_types),
            "task_type": "result_test",
            "priority": 7,
        })

        return tasks

    agent.set_task_generator(mixed_generator)

    try:
        agent.run(max_cycles=30, cycle_interval=0.1)
    except Exception as e:
        M.fail_test("综合集成", f"Agent崩溃: {e}")
        return

    stats = agent.status["engine"]
    if stats["success_rate"] > 50:
        M.pass_test(f"综合集成: {stats['total_executed']}任务, {stats['success_rate']}%成功")
    else:
        M.fail_test("综合集成", f"成功率过低: {stats['success_rate']}%")

    # 验证知识图谱数据完整性
    try:
        kg = KnowledgeGraph(config.knowledge.db_path)
        summary = kg.get_all_knowledge_summary()
        total_nodes = sum(summary["nodes_by_category"].values())
        if total_nodes > 0:
            M.pass_test(f"综合集成-知识图谱: {total_nodes}条知识")
        else:
            M.fail_test("综合集成-知识图谱", "知识图谱为空")
        kg.close()
    except Exception as e:
        M.fail_test("综合集成-知识图谱", str(e)[:100])


# ==================== 主入口 ====================

def main():
    print("=" * 70)
    print("第二轮深度压力测试")
    print("聚焦: 模块交互、边界条件、数据完整性")
    print("=" * 70)

    test_callback_and_stats()
    test_llm_fault_injection()
    test_optimizer_unexpected_json()
    test_result_serialization()
    test_watchdog_resource_leak()
    test_knowledge_bloat()
    test_full_integration()

    M.report()
    return len(M.issues)


if __name__ == "__main__":
    issues = main()
    sys.exit(1 if issues > 0 else 0)
