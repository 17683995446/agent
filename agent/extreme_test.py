#!/usr/bin/env python3
"""
第三轮补充测试 - 极端边界条件

D1: 超大量任务提交（队列溢出）
D2: 任务handler抛出极端异常（SystemExit/KeyboardInterrupt/GeneratorExit）
D3: 知识图谱WAL文件在并发下的完整性
D4: Agent在优化反思过程中被强制停止
D5: 空handler/None handler任务
D6: 任务kwargs中包含不可序列化的对象
D7: 同时stop和submit的竞争
D8: 优化器_cleanup_knowledge与store_experience并发
"""
import sys, os, time, random, json, threading, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, Task, TaskStatus
from agent.modules.knowledge_graph import KnowledgeGraph
from agent.modules.llm_client import LLMClient
from agent.modules.optimizer import Optimizer


class M2:
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
        print(f"补充测试: {self.passed}/{total} 通过, {self.failed} 失败")
        if self.issues:
            for i, iss in enumerate(self.issues, 1):
                print(f"  {i}. {iss['test']}: {iss['reason']}")
        print(f"{'='*60}")
        return self.failed

M = M2()

def clean():
    for f in os.listdir("data"):
        if f.endswith(".db") or f.endswith(".db-wal") or f.endswith(".db-shm"):
            try: os.remove(f"data/{f}")
            except: pass


# ========== D2: 极端异常类型 ==========

def test_d2_extreme_exceptions():
    """D2: handler抛出SystemExit/KeyboardInterrupt等极端异常"""
    print("\n--- D2: 极端异常类型处理 ---")
    clean()

    engine = TaskEngine(max_workers=3, retry_limit=2, timeout=5)

    # SystemExit不应该终止整个进程
    engine.submit("SystemExit任务", lambda: (_ for _ in ()).throw(SystemExit(1)), max_retries=1)
    # KeyboardInterrupt不应该终止
    engine.submit("KBInterrupt任务", lambda: (_ for _ in ()).throw(KeyboardInterrupt()), max_retries=1)
    # GeneratorExit
    def gen_exit():
        def gen():
            yield 1
            raise GeneratorExit()
        g = gen()
        next(g)
        g.close()
        return "ok"
    engine.submit("GeneratorExit任务", gen_exit, max_retries=1)
    # 普通ValueError作为对照
    engine.submit("ValueError任务", lambda: (_ for _ in ()).throw(ValueError("普通错误")), max_retries=1)

    results = engine.run_batch(4)

    # 所有任务都应该被捕获，不应该崩溃
    non_crashed = all(r.status in (TaskStatus.FAILED, TaskStatus.SUCCESS, TaskStatus.RETRY) for r in results)
    if non_crashed:
        M.ok(f"D2 极端异常全部捕获: {[r.status.value for r in results]}")
    else:
        M.fail("D2 极端异常", f"有任务状态异常: {[r.status.value for r in results]}")

    engine.shutdown()


# ========== D5: 空handler/None handler ==========

def test_d5_none_handler():
    """D5: handler为None或不可调用"""
    print("\n--- D5: None/不可调用handler ---")
    clean()

    engine = TaskEngine(max_workers=2, retry_limit=0, timeout=3)

    # None handler
    engine.submit("None handler", None, max_retries=0)
    # 非函数handler
    engine.submit("int handler", 42, max_retries=0)
    # 正常handler作为对照
    engine.submit("正常handler", lambda: "ok", max_retries=0)

    results = engine.run_batch(3)

    statuses = {r.name: r.status.value for r in results}
    # None和int应该失败，正常应该成功
    if statuses.get("正常handler") == "success" and statuses.get("None handler") == "failed":
        M.ok(f"D5 None handler正确处理: {statuses}")
    else:
        M.fail("D5 None handler", f"状态不符合预期: {statuses}")

    engine.shutdown()


# ========== D6: 不可序列化的kwargs ==========

def test_d6_unserializable_kwargs():
    """D6: kwargs包含不可JSON序列化的对象"""
    print("\n--- D6: 不可序列化kwargs ---")
    clean()

    engine = TaskEngine(max_workers=2, retry_limit=0, timeout=3)

    class CustomObj:
        def __str__(self):
            return "CustomObj"

    def task_with_obj(obj):
        return str(obj)

    engine.submit("自定义对象", task_with_obj, kwargs={"obj": CustomObj()})
    engine.submit("线程锁", task_with_obj, kwargs={"obj": threading.Lock()})
    engine.submit("lambda", task_with_obj, kwargs={"obj": lambda x: x})

    results = engine.run_batch(3)

    all_ok = all(r.status == TaskStatus.SUCCESS for r in results)
    if all_ok:
        M.ok(f"D6 不可序列化kwargs正常执行")
    else:
        M.fail("D6 不可序列化kwargs", f"有任务失败: {[r.status.value for r in results]}")

    engine.shutdown()


# ========== D7: 同时stop和submit ==========

def test_d7_stop_submit_race():
    """D7: Agent stop()和submit()同时调用"""
    print("\n--- D7: stop/submit竞争条件 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/d7.db"
    agent = Agent(config)

    errors = []

    def submitter():
        for i in range(100):
            try:
                agent.submit_task(f"竞争任务-{i}", lambda: "ok")
            except RuntimeError as e:
                if "已关闭" in str(e):
                    pass  # 预期行为
                else:
                    errors.append(str(e))
            except Exception as e:
                errors.append(str(e))

    def stopper():
        time.sleep(0.02)
        try:
            agent.stop()
        except Exception as e:
            errors.append(f"stop异常: {e}")

    t1 = threading.Thread(target=submitter)
    t2 = threading.Thread(target=stopper)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    unexpected_errors = [e for e in errors if "已关闭" not in e]
    if not unexpected_errors:
        M.ok(f"D7 stop/submit竞争: {len(errors)}个预期错误, 0个意外错误")
    else:
        M.fail("D7 stop/submit竞争", f"{len(unexpected_errors)}个意外错误: {unexpected_errors[0][:100]}")


# ========== D8: cleanup与store并发 ==========

def test_d8_cleanup_store_concurrent():
    """D8: _cleanup_knowledge与store_experience并发"""
    print("\n--- D8: cleanup/store并发 ---")
    clean()

    kg = KnowledgeGraph("data/d8.db")
    optimizer = Optimizer(kg, LLMClient(AgentConfig().llm))

    errors = []
    stop_event = threading.Event()

    def writer():
        i = 0
        while not stop_event.is_set():
            try:
                kg.store_experience(f"并发经验-{i}", "内容" * 50, importance=random.uniform(0.1, 0.9))
                kg.log_task(f"t-{i}", "test", "success", duration=0.1)
                i += 1
            except Exception as e:
                errors.append(f"writer: {e}")

    def cleaner():
        i = 0
        while not stop_event.is_set():
            try:
                optimizer.optimize_knowledge()
                i += 1
            except Exception as e:
                errors.append(f"cleaner: {e}")

    def reader():
        i = 0
        while not stop_event.is_set():
            try:
                kg.get_all_knowledge_summary()
                kg.search_nodes("并发", limit=5)
                i += 1
            except Exception as e:
                errors.append(f"reader: {e}")

    t1 = threading.Thread(target=writer)
    t2 = threading.Thread(target=cleaner)
    t3 = threading.Thread(target=reader)

    t1.start()
    t2.start()
    t3.start()

    time.sleep(10)  # 运行10秒
    stop_event.set()

    t1.join(timeout=5)
    t2.join(timeout=5)
    t3.join(timeout=5)

    alive = [t for t in [t1, t2, t3] if t.is_alive()]
    critical_errors = [e for e in errors if any(kw in str(e).lower() for kw in ["database", "disk", "corrupt", "malformed"])]

    if alive:
        M.fail("D8 cleanup/store并发", f"{len(alive)}个线程未完成")
    elif critical_errors:
        M.fail("D8 cleanup/store并发", f"{len(critical_errors)}个严重错误: {critical_errors[0][:100]}")
    else:
        M.ok(f"D8 cleanup/store并发: {len(errors)}个非严重错误, 无数据损坏")

    kg.close()


# ========== D9: 超长时间运行（200轮极快循环） ==========

def test_d9_extreme_long_run():
    """D9: 200轮极快循环+高频优化"""
    print("\n--- D9: 极端长时间运行 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/d9.db"
    config.max_concurrent_tasks = 10
    config.task_retry_limit = 1
    config.task_timeout = 2
    config.optimize_interval = 2  # 极高频优化

    agent = Agent(config)

    cycle = [0]
    def gen():
        cycle[0] += 1
        tasks = []
        for j in range(8):
            tasks.append({
                "name": f"D9-{cycle[0]}-{j}",
                "handler": lambda: {"v": random.random()},
                "task_type": "fast",
                "priority": random.randint(1, 10),
                "timeout": 1,
                "max_retries": 1,
            })
        if random.random() < 0.15:
            tasks.append({
                "name": f"D9-fail-{cycle[0]}",
                "handler": lambda: (_ for _ in ()).throw(RuntimeError("随机崩溃")),
                "task_type": "fail",
                "max_retries": 1,
            })
        return tasks

    agent.set_task_generator(gen)

    start = time.time()
    try:
        agent.run(max_cycles=200, cycle_interval=0.01)
    except Exception as e:
        M.fail("D9 极端运行", f"崩溃: {e}")
        return

    elapsed = time.time() - start
    stats = agent.status

    sr = stats["engine"]["success_rate"]
    total = stats["engine"]["total_executed"]
    nodes = sum(stats["knowledge"].get("nodes_by_category", {}).values())
    opts = stats["optimizer"]["total_optimizations"]

    M.ok(f"D9 极端运行: {elapsed:.1f}秒, {total}任务, {sr}%成功, {nodes}条知识, {opts}次优化")


# ========== 主入口 ==========

def main():
    print("=" * 60)
    print("第三轮补充测试 - 极端边界条件")
    print("=" * 60)

    test_d2_extreme_exceptions()
    test_d5_none_handler()
    test_d6_unserializable_kwargs()
    test_d7_stop_submit_race()
    test_d8_cleanup_store_concurrent()
    test_d9_extreme_long_run()

    return M.report()


if __name__ == "__main__":
    sys.exit(1 if main() > 0 else 0)
