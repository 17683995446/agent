#!/usr/bin/env python3
"""
压力测试与缺陷发现模块

测试场景：
1. 高并发（10+并发任务）
2. 长时间运行（100+轮循环）
3. 复杂依赖任务链
4. 高失败率场景（模拟API不稳定）
5. 内存压力测试（大量数据积累）
6. 异常注入（随机崩溃、超时、死锁）
"""
import sys
import os
import time
import random
import threading
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus

# 测试统计
class TestMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.total_cycles = 0
        self.total_tasks = 0
        self.success_tasks = 0
        self.failed_tasks = 0
        self.retried_tasks = 0
        self.timeouts = 0
        self.errors = []
        self.memory_samples = []
        self.latency_samples = []
        self.lock = threading.Lock()

    def record_task(self, status, duration, retried=False):
        with self.lock:
            self.total_tasks += 1
            self.latency_samples.append(duration)
            if status == TaskStatus.SUCCESS:
                self.success_tasks += 1
            elif status == TaskStatus.FAILED:
                self.failed_tasks += 1
            elif status == TaskStatus.TIMEOUT:
                self.timeouts += 1
            if retried:
                self.retried_tasks += 1

    def record_error(self, error_msg):
        with self.lock:
            self.errors.append({
                "time": datetime.now().isoformat(),
                "error": error_msg
            })

    def sample_memory(self):
        import psutil
        process = psutil.Process()
        with self.lock:
            self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB

    def report(self):
        elapsed = time.time() - self.start_time
        print("\n" + "="*70)
        print("压力测试报告")
        print("="*70)
        print(f"运行时间: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        print(f"总循环数: {self.total_cycles}")
        print(f"总任务数: {self.total_tasks}")
        print(f"成功: {self.success_tasks} ({self.success_tasks/max(self.total_tasks,1)*100:.1f}%)")
        print(f"失败: {self.failed_tasks}")
        print(f"超时: {self.timeouts}")
        print(f"重试: {self.retried_tasks}")
        if self.latency_samples:
            print(f"平均延迟: {sum(self.latency_samples)/len(self.latency_samples):.3f}s")
            print(f"最大延迟: {max(self.latency_samples):.3f}s")
        if self.memory_samples:
            print(f"内存使用: {self.memory_samples[0]:.1f}MB → {self.memory_samples[-1]:.1f}MB")
        print(f"错误记录: {len(self.errors)}条")
        for e in self.errors[-5:]:
            print(f"  - {e['time']}: {e['error'][:100]}")
        print("="*70)

metrics = TestMetrics()


# ========== 缺陷注入任务 ==========

def task_with_random_failure():
    """随机失败任务 - 测试重试机制"""
    if random.random() < 0.3:  # 30%失败率
        raise ConnectionError("模拟网络连接失败")
    time.sleep(random.uniform(0.01, 0.1))
    return {"status": "success", "data": random.randint(1, 1000)}


def task_with_timeout_risk():
    """可能超时的任务 - 测试超时控制"""
    sleep_time = random.uniform(0.05, 0.5)
    if sleep_time > 0.3:  # 可能超时
        time.sleep(sleep_time)
    else:
        time.sleep(sleep_time)
    return {"sleep": sleep_time}


def task_memory_intensive():
    """内存密集型任务 - 测试内存管理"""
    data = []
    for _ in range(random.randint(100, 1000)):
        data.append("x" * random.randint(100, 1000))
    result = len(data)
    del data  # 尝试释放
    return {"items": result}


def task_with_exception():
    """随机异常任务 - 测试异常处理"""
    exception_type = random.choice([
        ValueError("无效值"),
        KeyError("missing_key"),
        IndexError("索引越界"),
        ZeroDivisionError("除以零"),
        RuntimeError("运行时错误"),
    ])
    if random.random() < 0.25:
        raise exception_type
    return {"safe": True}


def task_deadlock_risk():
    """死锁风险任务 - 测试并发安全"""
    lock_a = threading.Lock()
    lock_b = threading.Lock()

    def acquire_both():
        with lock_a:
            time.sleep(0.001)
            with lock_b:
                return "acquired"

    # 简化的死锁模拟
    try:
        result = acquire_both()
        return {"result": result}
    except Exception as e:
        raise RuntimeError(f"死锁风险: {e}")


def task_database_stress():
    """数据库压力任务 - 测试知识图谱并发"""
    from agent.modules.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph("data/stress_test.db")
    try:
        node_id = kg.store_experience(
            title=f"压力测试_{random.randint(1, 10000)}",
            content="x" * random.randint(100, 5000),
            tags="stress,test",
            importance=random.random()
        )
        return {"node_id": node_id}
    finally:
        kg.close()


def task_recursive_depth(depth=0):
    """递归深度任务 - 测试栈深度"""
    if depth > 100:
        return {"depth": depth}
    return task_recursive_depth(depth + 1)


def task_large_result():
    """大结果返回 - 测试结果存储"""
    return {"data": [random.random() for _ in range(random.randint(1000, 10000))]}


def task_nested_callback():
    """嵌套回调任务 - 测试回调机制"""
    results = []
    for i in range(random.randint(3, 10)):
        results.append({"step": i, "value": random.random()})
    return {"steps": len(results), "results": results}


def task_external_resource():
    """外部资源任务 - 测试资源泄漏"""
    resources = []
    try:
        for _ in range(random.randint(5, 20)):
            resources.append(open("/dev/null" if os.name != "nt" else "nul"))
        return {"resources": len(resources)}
    finally:
        for r in resources:
            try:
                r.close()
            except:
                pass


# ========== 复杂任务链 ==========

def create_task_chain():
    """创建依赖任务链"""
    chain_results = []

    def step1():
        result = {"step": 1, "data": random.randint(1, 100)}
        chain_results.append(result)
        return result

    def step2():
        if not chain_results:
            raise ValueError("step1未完成")
        result = {"step": 2, "input": chain_results[-1]["data"], "output": random.randint(1, 100)}
        chain_results.append(result)
        return result

    def step3():
        if len(chain_results) < 2:
            raise ValueError("前置步骤未完成")
        result = {"step": 3, "sum": sum(r.get("data", r.get("output", 0)) for r in chain_results)}
        return result

    return [step1, step2, step3]


# ========== 压力测试生成器 ==========

def create_stress_task_generator(metrics_ref):
    """创建压力测试任务生成器"""
    task_types = [
        ("随机失败", task_with_random_failure, 5),
        ("超时风险", task_with_timeout_risk, 5),
        ("内存密集", task_memory_intensive, 6),
        ("随机异常", task_with_exception, 5),
        ("死锁风险", task_deadlock_risk, 7),
        ("数据库压力", task_database_stress, 4),
        ("递归深度", task_recursive_depth, 8),
        ("大结果", task_large_result, 6),
        ("嵌套回调", task_nested_callback, 5),
        ("外部资源", task_external_resource, 5),
    ]

    call_count = [0]

    def generator():
        call_count[0] += 1
        metrics_ref.total_cycles = call_count[0]

        # 每10轮采样内存
        if call_count[0] % 10 == 0:
            try:
                metrics_ref.sample_memory()
            except:
                pass

        # 生成任务批次（随循环增加任务量）
        batch_size = min(3 + call_count[0] // 10, 15)  # 逐渐增加到15个任务

        tasks = []
        for i in range(batch_size):
            name, handler, priority = random.choice(task_types)
            task_name = f"{name} #{call_count[0]}-{i+1}"

            # 动态调整超时
            timeout = random.choice([1, 2, 5, 10])

            tasks.append({
                "name": task_name,
                "handler": handler,
                "task_type": "stress",
                "priority": priority,
                "timeout": timeout,
                "max_retries": random.randint(1, 3),
            })

        # 偶尔添加任务链
        if random.random() < 0.2:
            chain = create_task_chain()
            for j, step in enumerate(chain):
                tasks.append({
                    "name": f"任务链 #{call_count[0]}-链{j+1}",
                    "handler": step,
                    "task_type": "chain",
                    "priority": 3,
                    "timeout": 5,
                })

        return tasks

    return generator


# ========== 主测试 ==========

def run_stress_test(cycles=100, interval=0.5):
    """运行压力测试"""
    print("="*70)
    print("开始压力测试")
    print(f"目标: {cycles}轮循环, 间隔{interval}秒")
    print("="*70)

    # 清理旧数据
    for f in ["data/knowledge.db", "data/stress_test.db"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"清理: {f}")

    config = AgentConfig()
    config.max_concurrent_tasks = 10  # 提高并发
    config.task_retry_limit = 3
    config.task_timeout = 10
    config.optimize_interval = 10  # 每10个任务优化一次

    agent = Agent(config)

    # 设置压力测试任务生成器
    global metrics
    agent.set_task_generator(create_stress_task_generator(metrics))

    # 包装任务完成回调以收集指标
    original_callback = agent.engine._on_task_complete

    def metrics_callback(task):
        metrics.record_task(task.status, task.duration, task.retries > 0)
        if task.error:
            metrics.record_error(task.error)
        if original_callback:
            original_callback(task)

    agent.engine._on_task_complete = metrics_callback

    try:
        agent.run(max_cycles=cycles, cycle_interval=interval)
    except Exception as e:
        metrics.record_error(f"Agent崩溃: {str(e)}")
        raise
    finally:
        metrics.report()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    run_stress_test(cycles=args.cycles, interval=args.interval)
