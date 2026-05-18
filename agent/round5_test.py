#!/usr/bin/env python3
"""
第五轮测试 - 重构回归 + 新缺陷验证

F1: _task_handlers/_cancel_events内存泄漏
F2: _generate_tasks访问不存在的Task属性
F3: run_batch统计竞争
F4: _execute_wrapper中started_at为0
"""
import sys, os, time, random, threading, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus
from agent.core.interfaces import Task


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
        print(f"第五轮测试: {self.passed}/{total} 通过, {self.failed} 失败")
        if self.issues:
            for i, iss in enumerate(self.issues, 1):
                print(f"  {i}. {iss['test']}: {iss['reason']}")
        print(f"{'='*60}")
        return self.failed

m = M()

def clean():
    for f in os.listdir("data"):
        if f.endswith(".db") or f.endswith(".db-wal") or f.endswith(".db-shm"):
            try: os.remove(f"data/{f}")
            except: pass


# ========== F1: _task_handlers内存泄漏 ==========

def test_f1_handler_memory_leak():
    """F1: 大量任务后_task_handlers是否增长"""
    print("\n--- F1: _task_handlers内存泄漏 ---")
    clean()

    engine = TaskEngine(max_workers=3, retry_limit=0, timeout=2)

    # 提交并执行1000个任务
    for i in range(1000):
        engine.submit(f"任务-{i}", lambda: "ok", max_retries=0)

    # 分批执行
    for _ in range(200):
        engine.run_batch(5)

    handlers_size = len(engine._task_handlers)
    cancel_size = len(engine._cancel_events)
    completed_size = len(engine._completed)

    # 理想情况：handlers应该被清理，与completed数量一致
    if handlers_size > completed_size + 50:  # 允许一些延迟
        m.fail("F1 handler泄漏",
               f"handlers={handlers_size}, completed={completed_size}, 差={handlers_size - completed_size}")
    else:
        m.ok(f"F1 handler管理: handlers={handlers_size}, completed={completed_size}")

    if cancel_size > completed_size + 50:
        m.fail("F1 cancel_events泄漏",
               f"cancel_events={cancel_size}, completed={completed_size}")
    else:
        m.ok(f"F1 cancel_events管理: cancel_events={cancel_size}")

    engine.shutdown()


# ========== F2: Task对象属性访问 ==========

def test_f2_task_attribute_access():
    """F2: _generate_tasks中Task对象属性访问"""
    print("\n--- F2: Task属性访问 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/f2.db"
    agent = Agent(config)

    # 测试1: 生成器返回Task对象（新接口）
    def task_gen_returns_task():
        t = Task(
            id="test1",
            name="测试Task对象",
            task_type="test",
            priority=5,
            status=TaskStatus.PENDING,
            created_at=time.time(),
            max_retries=0,
            timeout=5,
        )
        # Task对象不再有handler属性
        return [t]

    agent.set_task_generator(task_gen_returns_task)

    try:
        agent.run(max_cycles=1, cycle_interval=0.1)
        m.ok("F2 Task对象生成器不崩溃")
    except AttributeError as e:
        m.fail("F2 Task对象属性", f"AttributeError: {e}")
    except Exception as e:
        # 可能是其他错误，但不应该是AttributeError
        m.ok(f"F2 Task对象生成器: 其他错误(可接受): {type(e).__name__}")

    # 测试2: 生成器返回dict（正常路径）
    def task_gen_returns_dict():
        return [{"name": "dict任务", "handler": lambda: "ok", "max_retries": 0}]

    agent.set_task_generator(task_gen_returns_dict)
    try:
        agent.run(max_cycles=1, cycle_interval=0.1)
        m.ok("F2 dict生成器正常工作")
    except Exception as e:
        m.fail("F2 dict生成器", str(e)[:100])


# ========== F3: run_batch统计竞争 ==========

def test_f3_stats_race():
    """F3: 高并发超时下的统计准确性"""
    print("\n--- F3: 统计竞争 ---")
    clean()

    engine = TaskEngine(max_workers=20, retry_limit=0, timeout=1)

    # 提交大量会超时的任务
    for i in range(200):
        engine.submit(f"超时-{i}", lambda: time.sleep(5), timeout=0.2, max_retries=0)

    # 多线程同时执行
    threads = []
    for _ in range(5):
        t = threading.Thread(target=lambda: engine.run_batch(10))
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=30)

    stats = engine.stats
    expected = stats["total_success"] + stats["total_failed"]

    if stats["total_executed"] == expected:
        m.ok(f"F3 统计一致: executed={stats['total_executed']}, expected={expected}")
    else:
        m.fail("F3 统计竞争",
               f"executed={stats['total_executed']} != success+failed={expected}")

    engine.shutdown()


# ========== F4: _execute_wrapper started_at ==========

def test_f4_wrapper_started_at():
    """F4: wrapper中started_at为0时的duration"""
    print("\n--- F4: wrapper started_at ---")
    clean()

    engine = TaskEngine(max_workers=2, retry_limit=0, timeout=2)

    # 正常任务
    engine.submit("正常", lambda: "ok", max_retries=0)
    results = engine.run_batch(1)

    task = results[0]
    if task.duration >= 0 and task.duration < 10:
        m.ok(f"F4 duration正常: {task.duration:.3f}s")
    else:
        m.fail("F4 duration", f"异常值: {task.duration}")

    if task.started_at > 0:
        m.ok(f"F4 started_at正常: {task.started_at}")
    else:
        m.fail("F4 started_at", f"值为0")

    engine.shutdown()


# ========== 综合: 300轮高负载测试 ==========

def test_comprehensive_300():
    """综合: 300轮高负载"""
    print("\n--- 综合: 300轮高负载 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/f_comp.db"
    config.max_concurrent_tasks = 10
    config.task_retry_limit = 1
    config.task_timeout = 2
    config.optimize_interval = 10

    agent = Agent(config)

    cycle = [0]
    def gen():
        cycle[0] += 1
        tasks = []
        for j in range(8):
            tasks.append({
                "name": f"C-{cycle[0]}-{j}",
                "handler": lambda: {"v": random.random()},
                "task_type": "fast",
                "priority": random.randint(1, 10),
                "timeout": 1,
                "max_retries": 1,
            })
        if random.random() < 0.12:
            tasks.append({
                "name": f"C-fail-{cycle[0]}",
                "handler": lambda: (_ for _ in ()).throw(RuntimeError("故障")),
                "task_type": "fail",
                "max_retries": 1,
            })
        return tasks

    agent.set_task_generator(gen)

    start = time.time()
    try:
        agent.run(max_cycles=300, cycle_interval=0.02)
    except Exception as e:
        m.fail("综合300轮", f"崩溃: {e}")
        return

    elapsed = time.time() - start
    stats = agent.status

    sr = stats["engine"]["success_rate"]
    total = stats["engine"]["total_executed"]
    nodes = sum(stats["knowledge"].get("nodes_by_category", {}).values())

    if sr > 85 and total > 2000:
        m.ok(f"综合300轮: {elapsed:.1f}秒, {total}任务, {sr}%成功, {nodes}条知识")
    else:
        m.fail("综合300轮", f"成功率{sr}%或任务数{total}不达标")


def main():
    print("=" * 60)
    print("第五轮测试 - 重构回归 + 新缺陷验证")
    print("=" * 60)

    test_f1_handler_memory_leak()
    test_f2_task_attribute_access()
    test_f3_stats_race()
    test_f4_wrapper_started_at()
    test_comprehensive_300()

    return m.report()


if __name__ == "__main__":
    sys.exit(1 if main() > 0 else 0)
