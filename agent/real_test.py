#!/usr/bin/env python3
"""
第三轮真实场景深度测试

聚焦：长时间运行的真实bug
C1: 回调线程与主线程并发写知识图谱（死锁风险）
C2: _cleanup_knowledge绕过事务管理器
C3: run_batch的except块无锁修改统计
C4: 任务链状态共享问题
C5: stop()后访问已关闭引擎
C6: 混合事务模式
C7: 多线程连接泄漏
"""
import sys, os, time, random, json, threading, traceback, gc, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, Task, TaskStatus
from agent.modules.knowledge_graph import KnowledgeGraph
from agent.modules.llm_client import LLMClient
from agent.modules.optimizer import Optimizer


class Metrics:
    def __init__(self):
        self.issues = []
        self.passed = 0
        self.failed = 0
        self.lock = threading.Lock()

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
        print(f"第三轮测试: {self.passed}/{total} 通过, {self.failed} 失败")
        if self.issues:
            for i, iss in enumerate(self.issues, 1):
                print(f"  {i}. {iss['test']}: {iss['reason']}")
        print(f"{'='*60}")
        return self.failed

M = Metrics()

def clean():
    for f in os.listdir("data"):
        if f.endswith(".db") or f.endswith(".db-wal") or f.endswith(".db-shm"):
            try: os.remove(f"data/{f}")
            except: pass


# ========== C1: 回调线程与主线程并发写知识图谱 ==========

def test_c1_concurrent_kg_write():
    """C1: 回调线程和主线程同时写知识图谱是否死锁"""
    print("\n--- C1: 并发知识图谱写入（死锁检测）---")
    clean()
    kg = KnowledgeGraph("data/c1.db")

    deadlock_detected = [False]
    write_count = [0]

    def writer_thread(thread_id):
        """模拟回调线程持续写入"""
        for i in range(200):
            try:
                kg.store_experience(
                    title=f"线程{thread_id}-经验{i}",
                    content=f"内容{i}",
                    tags=f"thread{thread_id}",
                    importance=0.5,
                )
                kg.log_task(f"t{thread_id}-{i}", "test", "success")
                write_count[0] += 1
            except Exception as e:
                if "lock" in str(e).lower() or "deadlock" in str(e).lower():
                    deadlock_detected[0] = True
                raise

    def reader_thread():
        """模拟主线程持续读取"""
        for i in range(200):
            try:
                kg.get_all_knowledge_summary()
                kg.search_nodes(f"线程", limit=5)
                kg.get_active_strategies()
            except Exception as e:
                if "lock" in str(e).lower() or "deadlock" in str(e).lower():
                    deadlock_detected[0] = True
                raise

    def cleanup_thread():
        """模拟优化器清理"""
        for i in range(50):
            try:
                kg.get_all_knowledge_summary()
                # 模拟_cleanup_knowledge
                conn = kg._get_connection()
                with kg._lock:
                    conn.execute("DELETE FROM nodes WHERE importance < 0.1 AND usage_count = 0")
                    conn.commit()
            except Exception as e:
                if "lock" in str(e).lower() or "deadlock" in str(e).lower():
                    deadlock_detected[0] = True
                raise

    threads = [
        threading.Thread(target=writer_thread, args=(1,)),
        threading.Thread(target=writer_thread, args=(2,)),
        threading.Thread(target=writer_thread, args=(3,)),
        threading.Thread(target=reader_thread),
        threading.Thread(target=cleanup_thread),
    ]

    for t in threads:
        t.start()

    deadline = time.time() + 30  # 30秒超时=死锁
    for t in threads:
        t.join(timeout=max(0, deadline - time.time()))

    alive = [t for t in threads if t.is_alive()]
    if alive:
        M.fail("C1 并发写入死锁检测", f"{len(alive)}个线程未完成(可能死锁)")
    else:
        M.ok(f"C1 并发写入无死锁 ({write_count[0]}次写入)")

    kg.close()


# ========== C3: run_batch的except块无锁修改统计 ==========

def test_c3_stats_race():
    """C3: 并发修改统计计数器"""
    print("\n--- C3: 统计计数器竞争条件 ---")
    clean()

    engine = TaskEngine(max_workers=5, retry_limit=1, timeout=2)

    # 提交大量会失败的任务
    for i in range(50):
        engine.submit(f"失败任务-{i}", lambda: (_ for _ in ()).throw(ValueError("test")),
                      max_retries=1, timeout=1)

    results = engine.run_batch(5)

    stats = engine.stats
    # 验证: total_executed == total_success + total_failed
    expected = stats["total_success"] + stats["total_failed"]
    if stats["total_executed"] == expected:
        M.ok(f"C3 统计一致性: executed={stats['total_executed']} = success({stats['total_success']}) + failed({stats['total_failed']})")
    else:
        M.fail("C3 统计一致性",
               f"executed={stats['total_executed']} != success({stats['total_success']}) + failed({stats['total_failed']}) = {expected}")

    engine.shutdown()


# ========== C4: 任务链状态共享 ==========

def test_c4_task_chain_state():
    """C4: 任务链中前后任务共享状态"""
    print("\n--- C4: 任务链状态共享 ---")
    clean()

    shared_counter = {"value": 0}
    lock = threading.Lock()

    def increment_task(amount):
        with lock:
            shared_counter["value"] += amount
            return shared_counter["value"]

    engine = TaskEngine(max_workers=5, retry_limit=0, timeout=5)

    # 提交100个递增任务
    for i in range(100):
        engine.submit(f"递增-{i}", increment_task, args=(1,), timeout=2)

    results = engine.run_batch(10)

    # 验证最终值
    success_count = sum(1 for r in results if r.status == TaskStatus.SUCCESS)
    if shared_counter["value"] == success_count:
        M.ok(f"C4 任务链状态一致: 100个递增, {success_count}成功, 最终值={shared_counter['value']}")
    else:
        M.fail("C4 任务链状态一致",
               f"成功{success_count}个但最终值={shared_counter['value']}")

    engine.shutdown()


# ========== C5: stop()后访问已关闭引擎 ==========

def test_c5_stop_then_access():
    """C5: Agent stop()后访问引擎"""
    print("\n--- C5: stop()后访问引擎 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/c5.db"
    agent = Agent(config)

    agent.submit_task("测试", lambda: "ok")
    agent.run(max_cycles=1, cycle_interval=0.1)

    # stop()已在run()的finally中调用
    # 尝试再次stop()——应该幂等
    try:
        agent.stop()
        M.ok("C5 重复stop()幂等")
    except Exception as e:
        M.fail("C5 重复stop()", str(e)[:100])

    # 尝试访问status
    try:
        status = agent.status
        M.ok(f"C5 stop后访问status: running={status['running']}")
    except Exception as e:
        M.fail("C5 stop后访问status", str(e)[:100])


# ========== C6: 混合事务模式 ==========

def test_c6_mixed_transaction():
    """C6: _transaction()与直接commit()混合使用"""
    print("\n--- C6: 混合事务模式 ---")
    clean()

    kg = KnowledgeGraph("data/c6.db")

    errors = []

    def mixed_operations(thread_id):
        for i in range(100):
            try:
                # 交替使用事务和直接操作
                if i % 3 == 0:
                    kg.store_experience(f"事务写入-{thread_id}-{i}", "内容", importance=0.5)
                elif i % 3 == 1:
                    # 模拟_cleanup_knowledge的直接操作
                    conn = kg._get_connection()
                    with kg._lock:
                        conn.execute("INSERT INTO nodes (category, title, content, importance, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                                    ("test", f"直接写入-{thread_id}-{i}", "", 0.5, time.time(), time.time()))
                        conn.commit()
                else:
                    # 读取
                    kg.search_nodes(f"{thread_id}", limit=3)
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=mixed_operations, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()

    deadline = time.time() + 20
    for t in threads:
        t.join(timeout=max(0, deadline - time.time()))

    alive = [t for t in threads if t.is_alive()]
    transaction_errors = [e for e in errors if "transaction" in e.lower() or "lock" in e.lower()]

    if alive:
        M.fail("C6 混合事务", f"{len(alive)}个线程未完成(可能死锁)")
    elif transaction_errors:
        M.fail("C6 混合事务", f"{len(transaction_errors)}个事务错误: {transaction_errors[0][:100]}")
    else:
        M.ok(f"C6 混合事务无错误 ({len(errors)}个其他错误可接受)")

    kg.close()


# ========== C7: 多线程连接泄漏 ==========

def test_c7_connection_leak():
    """C7: 多线程环境下的连接管理"""
    print("\n--- C7: 多线程连接泄漏 ---")
    clean()

    kg = KnowledgeGraph("data/c7.db")

    # 在多个线程中使用知识图谱
    def kg_user(thread_id):
        for i in range(50):
            kg.store_experience(f"线程{thread_id}-{i}", "内容", importance=0.3)
            kg.get_all_knowledge_summary()

    threads = [threading.Thread(target=kg_user, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    # close()后检查是否还能正常操作
    kg.close()

    try:
        # 重新打开验证数据库完整性
        kg2 = KnowledgeGraph("data/c7.db")
        summary = kg2.get_all_knowledge_summary()
        total = sum(summary.get("nodes_by_category", {}).values())
        kg2.close()

        if total > 0:
            M.ok(f"C7 连接管理: {total}条数据完整")
        else:
            M.fail("C7 连接管理", "数据丢失")
    except Exception as e:
        M.fail("C7 连接管理", str(e)[:100])


# ========== 综合: 长时间运行稳定性 ==========

def test_long_running_stability():
    """综合: 模拟长时间运行（快速循环+高并发+频繁优化）"""
    print("\n--- 综合: 长时间运行稳定性 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/c7.db"
    config.max_concurrent_tasks = 8
    config.task_retry_limit = 1
    config.task_timeout = 3
    config.optimize_interval = 3  # 每3个任务就优化一次（高频）

    agent = Agent(config)

    cycle = [0]
    def aggressive_generator():
        cycle[0] += 1
        tasks = []
        # 每轮生成多个不同类型的任务
        for j in range(5):
            tasks.append({
                "name": f"快速任务-{cycle[0]}-{j}",
                "handler": lambda n=cycle[0]: {"cycle": n, "rand": random.random()},
                "task_type": "fast",
                "priority": random.randint(1, 10),
                "timeout": 2,
                "max_retries": 1,
            })
        # 偶尔插入会失败的任务
        if random.random() < 0.2:
            tasks.append({
                "name": f"失败任务-{cycle[0]}",
                "handler": lambda: (_ for _ in ()).throw(ConnectionError("模拟连接失败")),
                "task_type": "flaky",
                "priority": 3,
                "max_retries": 2,
            })
        return tasks

    agent.set_task_generator(aggressive_generator)

    try:
        agent.run(max_cycles=50, cycle_interval=0.05)
    except Exception as e:
        M.fail("综合长时间运行", f"Agent崩溃: {e}")
        traceback.print_exc()
        return

    stats = agent.status
    engine = stats["engine"]
    kg = stats["knowledge"]

    checks = []

    # 检查1: 成功率 > 80%
    sr = engine["success_rate"]
    checks.append(("成功率", sr > 80, f"{sr}%"))

    # 检查2: 有知识积累
    total_nodes = sum(kg.get("nodes_by_category", {}).values())
    checks.append(("知识积累", total_nodes > 0, f"{total_nodes}条"))

    # 检查3: 有优化记录
    opt_count = stats["optimizer"].get("total_optimizations", 0)
    checks.append(("优化执行", opt_count > 0, f"{opt_count}次"))

    for name, passed, detail in checks:
        if passed:
            M.ok(f"综合-{name}: {detail}")
        else:
            M.fail(f"综合-{name}", detail)


# ========== 主入口 ==========

def main():
    print("=" * 60)
    print("第三轮真实场景深度测试")
    print("=" * 60)

    test_c1_concurrent_kg_write()
    test_c3_stats_race()
    test_c4_task_chain_state()
    test_c5_stop_then_access()
    test_c6_mixed_transaction()
    test_c7_connection_leak()
    test_long_running_stability()

    return M.report()


if __name__ == "__main__":
    sys.exit(1 if main() > 0 else 0)
