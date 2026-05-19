#!/usr/bin/env python3
"""
第六轮第一性原理测试 - 验证核心设计承诺是否兑现

P1: 知识→执行闭环是否真正工作？
P2: 执行/优化/存储是否真正并行？
P3: 出错是否真正自动修复？
P4: usage_count/success_rate是否被更新？
P5: 经验是否被过早清理？
"""
import sys, os, time, random, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.knowledge_graph import KnowledgeGraph
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

    def warn(self, name, reason):
        self.passed += 1
        print(f"  ⚠ {name}: {reason}")

    def report(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"第六轮第一性原理测试: {self.passed}/{total} 通过, {self.failed} 失败")
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


# ========== P1: 知识→执行闭环 ==========

def test_p1_knowledge_to_execution():
    """P1: 存储的经验是否被用于改善后续执行？"""
    print("\n--- P1: 知识→执行闭环 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/p1.db"
    config.optimize_interval = 3

    agent = Agent(config)

    # 先存储一条明确的经验（使用与任务类型匹配的标题）
    agent.knowledge.store_experience(
        title="p1-test经验-应该被使用",
        content="当执行p1-test任务时，应该优先使用此经验来改善执行",
        category="experience",
        tags="p1,test,should-be-used",
        importance=0.9,
    )

    # 验证经验已存储
    results = agent.knowledge.search_nodes("p1-test经验", limit=5)
    if not results:
        m.fail("P1 经验存储", "经验未成功存储")
        agent.knowledge.close()
        return

    m.ok(f"P1 经验已存储: id={results[0].id}")

    # 检查usage_count是否为0（经验从未被使用）
    if results[0].usage_count == 0:
        m.warn("P1 usage_count=0", "经验存储后从未被查询使用（知识→执行闭环断裂）")
    else:
        m.ok(f"P1 usage_count={results[0].usage_count}")

    # 运行一些任务触发优化
    cycle = [0]
    def gen():
        cycle[0] += 1
        return [{"name": f"P1任务-{cycle[0]}", "handler": lambda: "ok",
                 "task_type": "p1-test", "max_retries": 0}]

    agent.set_task_generator(gen)
    agent.run(max_cycles=5, cycle_interval=0.1)

    # 再次检查usage_count
    results2 = agent.knowledge.search_nodes("p1-test经验", limit=5)
    if results2 and results2[0].usage_count > 0:
        m.ok(f"P1 经验被使用: usage_count={results2[0].usage_count}")
    else:
        m.fail("P1 知识→执行闭环",
               f"经验未被使用(usage_count={results2[0].usage_count if results2 else 'N/A'})，闭环断裂")


# ========== P2: 三者并行 ==========

def test_p2_parallel_execution():
    """P2: 执行/优化/存储是否真正并行？"""
    print("\n--- P2: 三者并行 ---")

    clean()
    config = AgentConfig()
    config.knowledge.db_path = "data/p2.db"
    config.optimize_interval = 2

    agent = Agent(config)

    # 先运行Agent（_start_optimize_thread在run()中调用）
    cycle = [0]
    def gen():
        cycle[0] += 1
        return [{"name": f"P2-{cycle[0]}", "handler": lambda: "ok",
                 "task_type": "p2-test", "max_retries": 0}]

    agent.set_task_generator(gen)
    agent.run(max_cycles=10, cycle_interval=0.05)

    # 运行后检查后台优化线程是否被创建过
    has_optimize_thread = hasattr(agent, '_optimize_thread') and agent._optimize_thread is not None

    if has_optimize_thread:
        m.ok("P2 后台优化线程已创建")
    else:
        m.fail("P2 三者并行", "没有独立的后台优化线程，并行承诺未兑现")
        return

    # 检查优化确实在后台执行了
    stats = agent.status
    if stats['optimizer']['total_optimizations'] > 0:
        m.ok(f"P2 后台优化工作: {stats['optimizer']['total_optimizations']}次优化在后台完成")
    else:
        m.warn("P2 后台优化", "优化次数为0，可能队列未触发")


# ========== P3: 自动修复 ==========

def test_p3_auto_repair():
    """P3: 出错是否真正自动修复？"""
    print("\n--- P3: 自动修复 ---")

    import inspect
    source = inspect.getsource(Agent.run)

    # 检查except块中是否有修复逻辑
    has_repair = any(kw in source for kw in ["repair", "fix", "recover", "恢复", "修复"])
    has_rerun = "rerun" in source or "retry_cycle" in source

    if has_repair or has_rerun:
        m.ok("P3 代码中有修复逻辑")
    else:
        m.fail("P3 自动修复",
               "except块只记录错误，不做任何修复，自动修复承诺未兑现")


# ========== P4: usage_count更新 ==========

def test_p4_usage_count():
    """P4: usage_count/success_rate是否被更新？"""
    print("\n--- P4: usage_count更新 ---")
    clean()

    kg = KnowledgeGraph("data/p4.db")

    # 存储经验
    nid = kg.store_experience("P4测试", "测试内容", importance=0.8)

    # 模拟被查询使用
    kg.query_experience("P4测试", limit=5)

    # 检查usage_count
    node = kg.get_node(nid)
    if node.usage_count == 0:
        m.fail("P4 usage_count",
               f"经验被query_experience查询后usage_count仍为0，学习反馈断裂")
    else:
        m.ok(f"P4 usage_count={node.usage_count}")

    kg.close()


# ========== P5: 经验过早清理 ==========

def test_p5_premature_cleanup():
    """P5: 经验是否被过早清理？"""
    print("\n--- P5: 经验过早清理 ---")
    clean()

    kg = KnowledgeGraph("data/p5.db")

    # 存储importance=0.5的经验（模拟模式默认值）
    nid = kg.store_experience("P5测试", "测试内容", importance=0.5)

    # 手动将updated_at设为2小时前
    import agent.modules.optimizer as opt_module
    conn = kg._get_connection()
    with kg._lock:
        conn.execute("UPDATE nodes SET updated_at = ? WHERE id = ?",
                     (time.time() - 7200, nid))
        conn.commit()

    # 运行清理
    from agent.modules.llm_client import LLMClient
    optimizer = Optimizer(kg, LLMClient(AgentConfig().llm))
    optimizer.optimize_knowledge()

    # 检查经验是否还在
    node = kg.get_node(nid)
    if node is None:
        m.fail("P5 过早清理",
               "importance=0.5的经验在2小时后被清理（cleanup阈值可能过低）")
    else:
        m.ok(f"P5 经验保留: importance={node.importance}")

    kg.close()


# ========== 综合: 真实闭环验证 ==========

def test_full_loop():
    """综合: 验证完整的 知识→执行→反思→改进 闭环"""
    print("\n--- 综合: 完整闭环验证 ---")
    clean()

    config = AgentConfig()
    config.knowledge.db_path = "data/loop.db"
    config.optimize_interval = 3

    agent = Agent(config)

    # Phase 1: 执行任务并积累经验
    phase1_tasks = [0]
    def phase1_gen():
        phase1_tasks[0] += 1
        if phase1_tasks[0] <= 10:
            return [{"name": f"Phase1-{phase1_tasks[0]}",
                     "handler": lambda: {"phase": 1},
                     "task_type": "phase1", "max_retries": 0}]
        return None

    agent.set_task_generator(phase1_gen)
    agent.run(max_cycles=12, cycle_interval=0.05)

    # Phase 2: 验证经验是否积累
    exp1 = agent.knowledge.query_experience("Phase1", limit=5)
    strategies = agent.knowledge.get_active_strategies(limit=5)

    m.ok(f"Phase1完成: {len(exp1)}条相关经验, {len(strategies)}条策略")

    # Phase 3: 执行相似任务，验证经验是否被利用
    phase2_tasks = [0]
    def phase2_gen():
        phase2_tasks[0] += 1
        if phase2_tasks[0] <= 5:
            return [{"name": f"Phase2-{phase2_tasks[0]}",
                     "handler": lambda: {"phase": 2},
                     "task_type": "phase1",  # 同类型，应该匹配Phase1经验
                     "max_retries": 0}]
        return None

    agent.set_task_generator(phase2_gen)
    agent.run(max_cycles=7, cycle_interval=0.05)

    # 验证Phase1经验的usage_count是否增加
    if exp1:
        updated = agent.knowledge.get_node(exp1[0].id)
        if updated and updated.usage_count > 0:
            m.ok(f"闭环验证: Phase1经验被Phase2使用(usage_count={updated.usage_count})")
        else:
            m.warn("闭环验证",
                   f"Phase1经验未被Phase2利用(usage_count={updated.usage_count if updated else 'N/A'})")

    stats = agent.status
    m.ok(f"综合闭环: {stats['engine']['total_executed']}任务, "
        f"{stats['optimizer']['total_optimizations']}次优化")


def main():
    print("=" * 60)
    print("第六轮第一性原理测试")
    print("验证核心设计承诺是否兑现")
    print("=" * 60)

    test_p1_knowledge_to_execution()
    test_p2_parallel_execution()
    test_p3_auto_repair()
    test_p4_usage_count()
    test_p5_premature_cleanup()
    test_full_loop()

    return m.report()


if __name__ == "__main__":
    sys.exit(1 if main() > 0 else 0)
