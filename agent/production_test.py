#!/usr/bin/env python3
"""
第四轮生产级压力测试 - 聚焦资源耗尽、累积效应、真实故障

E1: run_batch统计计数竞争
E2: RETRY任务状态共享
E3: get_recent_completed与clear_completed竞争
E4: 事务回滚在autocommit下无效
E5: 多线程连接泄漏
E6: 看门狗重启后任务生成器状态
"""
import sys, os, time, random, json, threading, traceback, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, Task, TaskStatus
from agent.modules.knowledge_graph import KnowledgeGraph
from agent.modules.llm_client import LLMClient
from agent.modules.optimizer import Optimizer


class M:
    def __init__(self):
        self.issues = []
        self.passed = 0
        self.failed = 0

    def ok(self, name):
        self.passed += 1
        print(f"  ✓ {name}")

    def fail(self, name, reason):
        self.failed += 1
        self.issues.append({"test": name, "reason": reason})
        print(f"  ✗ {name}: {reason}")

    def report(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"生产级测试: {self.passed}/{total} 通过, {self.failed} 失败")
        if self.issues:
            for i, iss in enumerate(self.issues, 1):
                print(f"  {i}. {iss['test']}: {iss['reason']}")
        print(f"{'='*60}")
        return self.failed

metrics = M()

def clean():
    for f in os.listdir("data"):
        if f.endswith(".db") or f.endswith(".db-wal") or f.endswith(".db-shm"):
            try: os.remove(f"data/{f}")
            except: pass


# ========== E1: run_batch统计计数竞争 ==========

def test_e1_stats_race():
    """E1: 高并发下run_batch的统计计数准确性"""
    print("\n--- E1: 统计计数竞争 ---")
    clean()

    engine = TaskEngine(max_workers=10, retry_limit=0, timeout=1)

    # 提交大量会超时/失败的任务
    for i in range(200):
        engine.submit(f"竞争任务-{i}", lambda: time.sleep(2), timeout=0.5, max_retries=0)

    # 多线程同时执行run_batch
    results = []
    lock = threading.Lock()

    def batch_runner():
        for _ in range(5):
            batch = engine.run_batch(10)
            with lock:
                results.extend(batch)

    threads = [threading.Thread(target=batch_runner) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    stats = engine.stats
    # 验证: total_executed == total_success + total_failed
    expected = stats["total_success"] + stats["total_failed"]

    if stats["total_executed"] == expected and stats["total_executed"] > 0:
        metrics.ok(f"E1 统计一致性: executed={stats['total_executed']}, expected={expected}")
    else:
        metrics.fail("E1 统计一致性",
                    f"executed={stats['total_executed']} != success({stats['total_success']}) + failed({stats['total_failed']}) = {expected}")

    engine.shutdown()


# ========== E2: RETRY任务状态共享 ==========

def test_e2_retry_shared_state():
    """E2: RETRY任务被多个线程共享修改"""
    print("\n--- E2: RETRY任务状态共享 ---")
    clean()

    engine = TaskEngine(max_workers=5, retry_limit=3, timeout=5)

    attempt_count = {}
    lock = threading.Lock()

    def flaky_task(task_id):
        with lock:
            attempt_count[task_id] = attempt_count.get(task_id, 0) + 1
            current = attempt_count[task_id]
        if current < 3:
            raise ValueError(f"第{current}次失败")
        return f"第{current}次成功"

    # 提交会重试的任务
    for i in range(20):
        engine.submit(f"重试任务-{i}", lambda tid=i: flaky_task(tid), max_retries=3)

    results = engine.run_batch(5)

    # 验证每个任务的重试次数正确
    errors = []
    for task_id, count in attempt_count.items():
        if count < 1 or count > 4:  # 至少1次，最多4次（初始+3重试）
            errors.append(f"任务{task_id}执行了{count}次")

    if not errors:
        metrics.ok(f"E2 RETRY状态正确: {len(attempt_count)}个任务, 重试次数正常")
    else:
        metrics.fail("E2 RETRY状态", errors[0])

    engine.shutdown()


# ========== E3: get_recent_completed与clear_completed竞争 ==========

def test_e3_completed_race():
    """E3: 读取completed列表时触发清理"""
    print("\n--- E3: completed列表竞争 ---")
    clean()

    engine = TaskEngine(max_workers=3, retry_limit=0, timeout=2)

    # 快速产生大量完成任务
    for i in range(150):
        engine.submit(f"快速任务-{i}", lambda: "ok", max_retries=0)

    # 同时执行读取和清理
    errors = []
    stop_event = threading.Event()

    def reader():
        while not stop_event.is_set():
            try:
                recent = engine.get_recent_completed(limit=50)
                # 验证返回的列表一致性
                if len(recent) > 50:
                    errors.append(f"返回了{len(recent)}个，超过限制的50")
            except Exception as e:
                errors.append(f"读取异常: {e}")

    def cleaner():
        while not stop_event.is_set():
            try:
                engine.clear_completed(keep=30)
            except Exception as e:
                errors.append(f"清理异常: {e}")

    # 先执行一批任务
    engine.run_batch(10)

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=cleaner)
    t1.start()
    t2.start()

    time.sleep(3)
    stop_event.set()
    t1.join(timeout=5)
    t2.join(timeout=5)

    critical_errors = [e for e in errors if "index" in e.lower() or "range" in e.lower()]
    if not critical_errors:
        metrics.ok(f"E3 completed竞争: {len(errors)}个非严重错误")
    else:
        metrics.fail("E3 completed竞争", critical_errors[0][:100])

    engine.shutdown()


# ========== E4: 事务回滚测试 ==========

def test_e4_transaction_rollback():
    """E4: 验证事务回滚行为"""
    print("\n--- E4: 事务回滚 ---")
    clean()

    kg = KnowledgeGraph("data/e4.db")

    # 记录初始数量
    initial = kg.get_all_knowledge_summary()
    initial_nodes = sum(initial.get("nodes_by_category", {}).values())

    # 触发一个会失败的事务
    try:
        with kg._transaction() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO nodes (category, title, content, importance, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                     ("test", "临时节点", "内容", 0.5, time.time(), time.time()))
            # 故意触发错误
            c.execute("INVALID SQL SYNTAX")
    except Exception:
        pass  # 预期失败

    # 验证没有残留数据
    after = kg.get_all_knowledge_summary()
    after_nodes = sum(after.get("nodes_by_category", {}).values())

    if after_nodes == initial_nodes:
        metrics.ok(f"E4 事务回滚: 失败事务未残留数据")
    else:
        metrics.fail("E4 事务回滚", f"事务失败后数据从{initial_nodes}变为{after_nodes}")

    kg.close()


# ========== E5: 多线程连接泄漏 ==========

def test_e5_connection_leak():
    """E5: 长时间运行后连接泄漏"""
    print("\n--- E5: 连接泄漏 ---")
    clean()

    kg = KnowledgeGraph("data/e5.db")

    # 模拟多线程长时间使用
    def worker(thread_id):
        for i in range(100):
            kg.store_experience(f"线程{thread_id}-{i}", "内容" * 100, importance=0.5)
            if i % 10 == 0:
                kg.get_all_knowledge_summary()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # 关闭后重新打开验证完整性
    kg.close()

    try:
        kg2 = KnowledgeGraph("data/e5.db")
        summary = kg2.get_all_knowledge_summary()
        total = sum(summary.get("nodes_by_category", {}).values())
        kg2.close()

        if total >= 1000:  # 10线程*100任务
            metrics.ok(f"E5 连接管理: {total}条数据完整")
        else:
            metrics.fail("E5 连接管理", f"数据丢失: 期望>=1000, 实际{total}")
    except Exception as e:
        metrics.fail("E5 连接管理", str(e)[:100])


# ========== E6: 看门狗重启后状态一致性 ==========

def test_e6_watchdog_state():
    """E6: 看门狗重启后任务生成器状态"""
    print("\n--- E6: 看门狗重启状态 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/e6.db"
    config.max_concurrent_tasks = 3

    agent = Agent(config)

    # 使用闭包变量
    counter = {"value": 0}

    def generator():
        counter["value"] += 1
        if counter["value"] > 5:
            raise RuntimeError("模拟崩溃")
        return [{"name": f"任务-{counter['value']}", "handler": lambda: "ok", "max_retries": 0}]

    agent.set_task_generator(generator)

    try:
        agent.run(max_cycles=10, cycle_interval=0.1)
    except:
        pass

    # 验证counter被正确捕获
    if counter["value"] >= 5:
        metrics.ok(f"E6 看门狗状态: counter={counter['value']}")
    else:
        metrics.fail("E6 看门狗状态", f"counter={counter['value']}, 期望>=5")


# ========== 综合: 500轮极限压力测试 ==========

def test_extreme_500_cycles():
    """综合: 500轮极限压力测试"""
    print("\n--- 综合: 500轮极限压力 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/extreme.db"
    config.max_concurrent_tasks = 15
    config.task_retry_limit = 2
    config.task_timeout = 2
    config.optimize_interval = 5

    agent = Agent(config)

    cycle = [0]
    failure_injected = [0]

    def extreme_generator():
        cycle[0] += 1
        tasks = []

        # 每轮生成大量任务
        for j in range(10):
            tasks.append({
                "name": f"E-{cycle[0]}-{j}",
                "handler": lambda: {"v": random.random()},
                "task_type": "normal",
                "priority": random.randint(1, 10),
                "timeout": 1,
                "max_retries": 1,
            })

        # 随机注入故障
        if random.random() < 0.1:
            failure_injected[0] += 1
            tasks.append({
                "name": f"FAIL-{cycle[0]}",
                "handler": lambda: (_ for _ in ()).throw(RuntimeError("注入故障")),
                "task_type": "fail",
                "max_retries": 2,
            })

        return tasks

    agent.set_task_generator(extreme_generator)

    start = time.time()
    try:
        agent.run(max_cycles=500, cycle_interval=0.02)
    except Exception as e:
        metrics.fail("综合极限测试", f"崩溃: {e}")
        return

    elapsed = time.time() - start
    stats = agent.status

    sr = stats["engine"]["success_rate"]
    total = stats["engine"]["total_executed"]
    nodes = sum(stats["knowledge"].get("nodes_by_category", {}).values())

    if sr > 85 and total > 4000:
        metrics.ok(f"极限测试: {elapsed:.1f}秒, {total}任务, {sr}%成功, {nodes}条知识")
    else:
        metrics.fail("极限测试", f"成功率{sr}%或任务数{total}不达标")


# ========== 主入口 ==========

def main():
    print("=" * 60)
    print("第四轮生产级压力测试")
    print("=" * 60)

    test_e1_stats_race()
    test_e2_retry_shared_state()
    test_e3_completed_race()
    test_e4_transaction_rollback()
    test_e5_connection_leak()
    test_e6_watchdog_state()
    test_extreme_500_cycles()

    return metrics.report()


if __name__ == "__main__":
    sys.exit(1 if main() > 0 else 0)
