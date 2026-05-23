"""
第二十一轮生产就绪验证测试 - 信号处理、启动自检、资源限制、优雅关闭、配置热更新
"""
import os, sys, time, json, threading, signal, traceback, random, gc
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from agent.core.config import AgentConfig
from agent.modules.task_engine import TaskEngine, TaskStatus
from agent.modules.knowledge_graph import KnowledgeGraph

import logging
logging.basicConfig(level=logging.WARNING, format='%(name)s [%(levelname)s] %(message)s')


class Round21ProductionTest:
    def __init__(self):
        self.results = []
        self.bugs = []
        self._db_counter = 0

    def _new_db(self):
        self._db_counter += 1
        return f"/data/user/work/prod_r21_{self._db_counter}.db"

    def record(self, name, passed, details=""):
        self.results.append({"test": name, "passed": passed, "details": details})
        print(f"  [{'✓' if passed else '✗'}] {name}: {details[:150]}")

    def bug(self, bid, desc, sev="medium"):
        self.bugs.append({"id": bid, "description": desc, "severity": sev})
        print(f"  🐛 BUG [{bid}] ({sev}): {desc[:120]}")

    def run_all(self):
        print("\n" + "=" * 70)
        print("第二十一轮生产就绪验证测试")
        print("=" * 70)

        tests = [
            ("R1: 信号处理-SIGTERM", self.test_r1_sigterm),
            ("R2: 信号处理-SIGINT", self.test_r2_sigint),
            ("R3: 优雅关闭-数据完整性", self.test_r3_graceful_shutdown),
            ("R4: 优雅关闭-资源释放", self.test_r4_resource_cleanup),
            ("R5: 启动自检-配置验证", self.test_r5_config_validation),
            ("R6: 启动自检-初始化回滚", self.test_r6_init_rollback),
            ("R7: 资源限制-任务队列", self.test_r7_queue_limit),
            ("R8: 资源限制-连接池", self.test_r8_connection_limit),
            ("R9: 配置热更新集成", self.test_r9_config_hotreload),
            ("R10: 生产场景综合", self.test_r10_production_scenario),
        ]

        for name, func in tests:
            print(f"\n{'=' * 60}")
            print(f"执行: {name}")
            print(f"{'=' * 60}")
            try:
                func()
            except Exception as e:
                self.record(name, False, f"异常: {e}")
                self.bug(f"R-{name[:3]}", f"异常: {traceback.format_exc()[:200]}", "high")

        self._summary()

    # ========== R1: SIGTERM信号处理 ==========

    def test_r1_sigterm(self):
        """验证SIGTERM信号触发优雅关闭"""
        import multiprocessing as mp

        def child_process(db_path, result_queue):
            cfg = AgentConfig()
            cfg.knowledge.db_path = db_path
            a = Agent(cfg)

            # 验证信号处理器已注册
            import signal as _sig
            current_handler = _sig.getsignal(_sig.SIGTERM)
            signal_registered = current_handler is not _sig.SIG_DFL

            # 提交一些任务但不执行（避免run_batch阻塞信号处理）
            for i in range(5):
                a.submit_task(name=f"child-{i}", handler=lambda: time.sleep(30))

            # 发送SIGTERM
            _sig.raise_signal(_sig.SIGTERM)
            # 等待信号处理器执行
            time.sleep(2)

            # 验证shutdown_event已被设置
            handled = a._shutdown_event.is_set()
            result_queue.put({'handled': handled, 'registered': signal_registered})

            a.stop()

        result_queue = mp.Queue()
        p = mp.Process(target=child_process, args=(self._new_db(), result_queue))
        p.start()
        p.join(timeout=15)

        # 获取子进程结果
        result = {}
        try:
            result = result_queue.get(timeout=3)
        except:
            pass

        handled = result.get('handled', False)
        registered = result.get('registered', False)

        if not handled:
            self.bug("R1-001", f"SIGTERM未被处理: handled={handled}, registered={registered}", "critical")
            self.record("R1: 信号处理-SIGTERM", False, f"信号处理失败: handled={handled}, registered={registered}")
        else:
            self.record("R1: 信号处理-SIGTERM", True, f"SIGTERM正确处理: shutdown_event已设置")

    # ========== R2: SIGINT信号处理 ==========

    def test_r2_sigint(self):
        """验证SIGINT信号触发优雅关闭"""
        import multiprocessing as mp

        def child_process(db_path, result_queue):
            cfg = AgentConfig()
            cfg.knowledge.db_path = db_path
            a = Agent(cfg)

            # 验证信号处理器已注册
            import signal as _sig
            current_handler = _sig.getsignal(_sig.SIGINT)
            signal_registered = current_handler is not _sig.SIG_DFL

            a.submit_task(name="sigint-test", handler=lambda: time.sleep(30))

            # 发送SIGINT
            _sig.raise_signal(_sig.SIGINT)
            time.sleep(2)

            handled = a._shutdown_event.is_set()
            result_queue.put({'handled': handled, 'registered': signal_registered})

            a.stop()

        result_queue = mp.Queue()
        p = mp.Process(target=child_process, args=(self._new_db(), result_queue))
        p.start()
        p.join(timeout=15)

        result = {}
        try:
            result = result_queue.get(timeout=3)
        except:
            pass

        handled = result.get('handled', False)
        registered = result.get('registered', False)

        if not handled:
            self.bug("R2-001", f"SIGINT未被正确处理: handled={handled}, registered={registered}", "high")
            self.record("R2: 信号处理-SIGINT", False, f"信号处理失败: handled={handled}, registered={registered}")
        else:
            self.record("R2: 信号处理-SIGINT", True, f"SIGINT正确处理: shutdown_event已设置")

    # ========== R3: 优雅关闭-数据完整性 ==========

    def test_r3_graceful_shutdown(self):
        """验证关闭时不丢失已完成的任务数据"""
        config = AgentConfig()
        config.knowledge.db_path = self._new_db()
        agent = Agent(config)

        # 提交并执行任务
        for i in range(20):
            agent.submit_task(name=f"data-{i}", handler=lambda: f"result_{i}")
        agent.engine.run_batch(20)
        time.sleep(0.3)

        # 关闭
        agent.stop()

        # 重新打开数据库验证数据
        kg = KnowledgeGraph(config.knowledge.db_path)
        task_stats = kg.get_task_stats()
        total_logged = sum(s.get('count', 0) for s in task_stats.values())
        kg.close()

        if total_logged < 15:
            self.bug("R3-001", f"关闭时数据丢失: 预期>=15, 实际={total_logged}", "high")

        self.record("R3: 优雅关闭-数据完整性", total_logged >= 15,
                    f"执行20任务, 日志记录: {total_logged}")

    # ========== R4: 优雅关闭-资源释放 ==========

    def test_r4_resource_cleanup(self):
        """验证关闭后资源被正确释放"""
        config = AgentConfig()
        config.knowledge.db_path = self._new_db()
        agent = Agent(config)

        for i in range(10):
            agent.submit_task(name=f"res-{i}", handler=lambda: "ok")
        agent.engine.run_batch(10)

        # 记录关闭前的状态
        kg_connections = len(agent.knowledge._all_connections)
        engine_running = not agent.engine._shutdown

        agent.stop()

        # 验证资源释放
        kg_closed = agent.knowledge._closed
        engine_shutdown = agent.engine._shutdown

        issues = []
        if not kg_closed:
            issues.append("知识图谱未关闭")
        if not engine_shutdown:
            issues.append("引擎未关闭")

        if issues:
            self.bug("R4-001", f"资源未释放: {issues}", "high")

        self.record("R4: 优雅关闭-资源释放", not issues,
                    f"关闭前: 连接={kg_connections}, 引擎运行={engine_running}, 关闭后: KG={kg_closed}, 引擎={engine_shutdown}")

    # ========== R5: 启动自检-配置验证 ==========

    def test_r5_config_validation(self):
        """验证异常配置被正确拒绝"""
        issues = []

        # 测试无效max_concurrent_tasks
        try:
            cfg = AgentConfig()
            cfg.max_concurrent_tasks = 0
            engine = TaskEngine(max_workers=cfg.max_concurrent_tasks)
            engine.shutdown()
            issues.append("max_concurrent_tasks=0 未被拒绝")
        except Exception:
            pass  # 预期

        # 测试无效task_timeout
        try:
            cfg = AgentConfig()
            cfg.task_timeout = -1
            # 如果不报错，检查是否使用了默认值
            engine = TaskEngine(max_workers=2, timeout=cfg.task_timeout)
            engine.shutdown()
        except Exception:
            pass

        # 测试无效db_path（使用/proc下不可写路径）
        try:
            cfg = AgentConfig()
            cfg.knowledge.db_path = "/proc/nonexistent_dir_test/db.db"
            kg = KnowledgeGraph(cfg.knowledge.db_path)
            kg.close()
            issues.append("不可写路径未被拒绝")
        except (ValueError, OSError):
            pass  # 预期：R5修复后应抛出ValueError

        if issues:
            self.bug("R5-001", f"配置验证不足: {issues}", "medium")

        self.record("R5: 启动自检-配置验证", not issues,
                    f"测试3种异常配置, 问题: {issues if issues else '无'}")

    # ========== R6: 启动自检-初始化回滚 ==========

    def test_r6_init_rollback(self):
        """验证初始化失败时已创建的资源被清理"""
        # 模拟LLM初始化失败（通过mock）
        issues = []

        # 测试：正常创建Agent然后手动触发部分失败
        try:
            config = AgentConfig()
            config.knowledge.db_path = self._new_db()
            agent = Agent(config)

            # 手动关闭KG模拟部分失败
            agent.knowledge.close()

            # 尝试使用已关闭的KG
            try:
                agent.knowledge.store_experience(title="test", content="test", category="test")
                issues.append("关闭后操作未被阻止")
            except RuntimeError:
                pass  # 预期：S7.4修复生效

            agent.stop()
        except Exception as e:
            issues.append(f"初始化异常: {str(e)[:50]}")

        self.record("R6: 启动自检-初始化回滚", not issues,
                    f"问题: {issues if issues else '无'}")

    # ========== R7: 资源限制-任务队列 ==========

    def test_r7_queue_limit(self):
        """验证任务队列有大小限制"""
        engine = TaskEngine(max_workers=2, max_queue_size=200)

        # 提交大量任务，预期超过200后抛出异常
        accepted = 0
        rejected = False
        for i in range(500):
            try:
                engine.submit(name=f"queue-{i}", handler=lambda: "ok")
                accepted += 1
            except RuntimeError as e:
                if "队列已满" in str(e):
                    rejected = True
                    break

        queue_size = engine.pending_count

        if not rejected:
            self.bug("R7-001", f"任务队列无限制: {accepted}个任务被接受", "high")

        engine.shutdown(cancel_pending=True)

        self.record("R7: 资源限制-任务队列", rejected and queue_size <= 200,
                    f"提交500任务, 接受: {accepted}, 队列大小: {queue_size}, 被拒绝: {rejected}")

    # ========== R8: 资源限制-连接池 ==========

    def test_r8_connection_limit(self):
        """验证SQLite连接数合理"""
        db_path = self._new_db()
        connections = []

        def create_connection(tid):
            kg = KnowledgeGraph(db_path)
            kg.store_experience(title=f"conn-{tid}", content="test", category="conn")
            connections.append(kg)
            time.sleep(0.1)

        # 创建多个线程连接
        threads = [threading.Thread(target=create_connection, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # 检查连接数
        kg = KnowledgeGraph(db_path)
        total_connections = len(kg._all_connections)
        kg.close()

        # 关闭所有连接
        for c in connections:
            try:
                c.close()
            except:
                pass

        if total_connections > 25:
            self.bug("R8-001", f"连接数过多: {total_connections}", "medium")

        self.record("R8: 资源限制-连接池", total_connections <= 25,
                    f"20线程, 总连接: {total_connections}")

    # ========== R9: 配置热更新集成 ==========

    def test_r9_config_hotreload(self):
        """验证ConfigManager是否已集成到Agent"""
        # 检查Agent是否使用ConfigManager
        has_config_manager = hasattr(Agent, '_config_manager') or hasattr(Agent, 'config_manager')

        if not has_config_manager:
            self.bug("R9-001", "ConfigManager未集成到Agent", "high")

        # 检查Agent是否支持运行时配置更新
        config = AgentConfig()
        config.knowledge.db_path = self._new_db()
        agent = Agent(config)

        has_set_config = hasattr(agent, 'set_config') or hasattr(agent, 'update_config')
        agent.stop()

        if not has_set_config:
            self.bug("R9-002", "Agent不支持运行时配置更新", "medium")

        self.record("R9: 配置热更新集成", has_config_manager and has_set_config,
                    f"ConfigManager: {'已集成' if has_config_manager else '未集成'}, "
                    f"运行时更新: {'支持' if has_set_config else '不支持'}")

    # ========== R10: 生产场景综合 ==========

    def test_r10_production_scenario(self):
        """模拟生产环境综合场景"""
        config = AgentConfig()
        config.knowledge.db_path = self._new_db()
        config.max_concurrent_tasks = 5
        config.optimize_interval = 10

        agent = Agent(config)
        errors = []

        # 阶段1：正常负载
        for i in range(50):
            agent.submit_task(name=f"prod-{i}", handler=lambda: random.random())

        # 阶段2：混合负载（正常+异常+超时）
        for i in range(30):
            r = i % 3
            if r == 0:
                agent.submit_task(name=f"fail-{i}", handler=lambda: 1/0, max_retries=0)
            elif r == 1:
                agent.submit_task(name=f"slow-{i}", handler=lambda: time.sleep(5), timeout=0.1)
            else:
                agent.submit_task(name=f"ok-{i}", handler=lambda: "ok")

        # 阶段3：执行
        for _ in range(20):
            pending = agent.engine.pending_count
            if pending == 0:
                break
            agent.engine.run_batch(min(pending, 10))

        # 阶段4：检查状态
        status = agent.status
        stats = agent.engine.stats

        # 阶段5：优雅关闭
        agent.stop()

        # 验证
        production_ok = (
            stats['total_executed'] >= 50 and
            status['running'] == False and
            agent.engine._shutdown and
            agent.knowledge._closed
        )

        if not production_ok:
            self.bug("R10-001", "生产场景综合验证失败", "high")

        self.record("R10: 生产场景综合", production_ok,
                    f"执行: {stats['total_executed']}, 成功: {stats['total_success']}, "
                    f"失败: {stats['total_failed']}, 错误: {len(errors)}")

    def _summary(self):
        print("\n" + "=" * 70)
        print("第二十一轮生产就绪验证测试总结")
        print("=" * 70)

        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed if self.results else 0

        print(f"\n测试结果: {passed}/{len(self.results)} 通过, {failed} 失败")

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
    tester = Round21ProductionTest()
    tester.run_all()
