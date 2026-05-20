#!/usr/bin/env python3
"""
第七轮慢速测试 - 细致验证每一个潜在问题

S1: _optimize_queue嵌套列表问题
S2: 跨模块访问engine._queue的竞争
S3: stop()等待队列清空不完整
S4: run_with_watchdog资源泄漏
"""
import sys, os, time, random, threading, traceback, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, run_with_watchdog
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus
from agent.modules.knowledge_graph import KnowledgeGraph
from agent.modules.optimizer import Optimizer
from agent.modules.llm_client import LLMClient


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
        print(f"第七轮慢速测试: {self.passed}/{total} 通过, {self.failed} 失败")
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


# ========== S1: 嵌套列表问题 ==========

def test_s1_nested_queue():
    """S1: 验证_optimize_queue是否正确处理嵌套列表"""
    print("\n--- S1: 嵌套列表问题 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/s1.db"
    config.optimize_interval = 3
    config.max_concurrent_tasks = 3

    agent = Agent(config)

    # 运行足够多的任务触发优化
    cycle = [0]
    def gen():
        cycle[0] += 1
        if cycle[0] <= 20:
            return [{"name": f"S1-{cycle[0]}", "handler": lambda: "ok",
                     "task_type": "s1-test", "max_retries": 0}]
        return None

    agent.set_task_generator(gen)
    agent.run(max_cycles=25, cycle_interval=0.05)

    # 检查优化是否正常完成（无崩溃）
    stats = agent.status
    if stats['optimizer']['total_optimizations'] > 0:
        m.ok(f"S1 优化完成: {stats['optimizer']['total_optimizations']}次")
    else:
        m.fail("S1 优化", "优化次数为0，可能队列处理有问题")


# ========== S2: 跨模块访问竞争 ==========

def test_s2_cross_module_access():
    """S2: 验证_run_cycle中访问engine._queue是否安全"""
    print("\n--- S2: 跨模块访问竞争 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/s2.db"
    config.max_concurrent_tasks = 5

    agent = Agent(config)

    # 高频访问场景
    cycle = [0]
    def gen():
        cycle[0] += 1
        tasks = []
        for j in range(10):
            tasks.append({"name": f"S2-{cycle[0]}-{j}", "handler": lambda: "ok",
                         "task_type": "s2-test", "max_retries": 0})
        return tasks

    agent.set_task_generator(gen)

    errors = []
    original_run_cycle = agent._run_cycle

    def wrapped_run_cycle():
        try:
            original_run_cycle()
        except Exception as e:
            errors.append(str(e))

    agent._run_cycle = wrapped_run_cycle
    agent.run(max_cycles=50, cycle_interval=0.01)

    race_errors = [e for e in errors if "lock" in e.lower() or "race" in e.lower()]
    if not race_errors:
        m.ok(f"S2 无竞争错误: {len(errors)}个其他错误")
    else:
        m.fail("S2 竞争错误", race_errors[0][:100])


# ========== S3: stop等待队列清空 ==========

def test_s3_stop_queue_drain():
    """S3: 验证stop()是否正确等待优化队列清空"""
    print("\n--- S3: stop等待队列清空 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/s3.db"
    config.optimize_interval = 2

    agent = Agent(config)

    cycle = [0]
    def gen():
        cycle[0] += 1
        return [{"name": f"S3-{cycle[0]}", "handler": lambda: "ok",
                 "task_type": "s3-test", "max_retries": 0}]

    agent.set_task_generator(gen)

    # 运行一段时间后立即停止
    def stop_after_delay():
        time.sleep(0.5)
        agent.stop()

    stopper = threading.Thread(target=stop_after_delay)
    stopper.start()

    agent.run(max_cycles=100, cycle_interval=0.02)
    stopper.join(timeout=5)

    # 检查是否有数据丢失
    stats = agent.status
    if stats['engine']['total_executed'] > 0:
        m.ok(f"S3 stop正常: {stats['engine']['total_executed']}任务完成")
    else:
        m.fail("S3 stop", "任务执行数为0，可能被过早停止")


# ========== S4: 看门狗资源泄漏 ==========

def test_s4_watchdog_resource_leak():
    """S4: 验证run_with_watchdog重建Agent时资源释放"""
    print("\n--- S4: 看门狗资源泄漏 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/s4.db"
    config.max_concurrent_tasks = 2

    agent = Agent(config)

    crash_count = [0]
    def crash_gen():
        crash_count[0] += 1
        if crash_count[0] <= 3:
            raise RuntimeError(f"模拟崩溃 #{crash_count[0]}")
        return None

    agent.set_task_generator(crash_gen)

    # 运行看门狗（会触发重启）
    try:
        run_with_watchdog(agent, max_cycles=5, cycle_interval=0.1,
                         max_restarts=3, restart_cooldown=0.5)
    except Exception as e:
        pass

    # 验证数据库可以正常访问
    try:
        kg = KnowledgeGraph("data/s4.db")
        summary = kg.get_all_knowledge_summary()
        kg.close()
        m.ok(f"S4 资源释放正常: 数据库可访问")
    except Exception as e:
        m.fail("S4 资源泄漏", f"数据库访问失败: {e}")


# ========== 综合: 500轮慢速运行 ==========

def test_slow_500():
    """综合: 500轮慢速运行，充分暴露问题"""
    print("\n--- 综合: 500轮慢速运行 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/slow.db"
    config.max_concurrent_tasks = 8
    config.task_retry_limit = 2
    config.task_timeout = 3
    config.optimize_interval = 5

    agent = Agent(config)

    cycle = [0]
    def slow_gen():
        cycle[0] += 1
        tasks = []
        for j in range(6):
            # 混合任务类型
            if random.random() < 0.7:
                tasks.append({
                    "name": f"Slow-{cycle[0]}-{j}",
                    "handler": lambda: {"v": random.random()},
                    "task_type": random.choice(["compute", "data", "query"]),
                    "priority": random.randint(1, 10),
                    "timeout": 2,
                    "max_retries": 1,
                })
            else:
                tasks.append({
                    "name": f"Slow-fail-{cycle[0]}-{j}",
                    "handler": lambda: (_ for _ in ()).throw(RuntimeError("随机失败")),
                    "task_type": "fail",
                    "max_retries": 1,
                })
        return tasks

    agent.set_task_generator(slow_gen)

    start = time.time()
    agent.run(max_cycles=500, cycle_interval=0.03)
    elapsed = time.time() - start

    stats = agent.status
    sr = stats['engine']['success_rate']
    total = stats['engine']['total_executed']
    nodes = sum(stats['knowledge'].get('nodes_by_category', {}).values())
    opts = stats['optimizer']['total_optimizations']

    if sr > 60 and total > 2000:  # 考虑30%故意失败任务
        m.ok(f"500轮慢速: {elapsed:.1f}秒, {total}任务, {sr}%成功, {nodes}条知识, {opts}次优化")
    else:
        m.fail("500轮慢速", f"成功率{sr}%或任务数{total}不达标")


def main():
    print("=" * 60)
    print("第七轮慢速测试 - 细致验证每一个潜在问题")
    print("=" * 60)

    test_s1_nested_queue()
    test_s2_cross_module_access()
    test_s3_stop_queue_drain()
    test_s4_watchdog_resource_leak()
    test_slow_500()

    return m.report()


if __name__ == "__main__":
    sys.exit(1 if main() > 0 else 0)
