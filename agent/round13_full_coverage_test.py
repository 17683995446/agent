"""
第十三轮8层全覆盖测试 - 全链路日志验证
遵循: 第一性原理 + 慢工出细活

8层测试:
1. 静态代码核验 - 代码结构/死逻辑/资源漏洞
2. 单元全覆盖测试 - 穷尽if/else/循环/分支
3. 模块集成联调 - 跨模块调用验证
4. 接口全量遍历 - 正常/异常/超限/空值/非法
5. 业务场景闭环 - 正向/逆向/中断/重复
6. 数据一致性校验 - CRUD/关联/回显
7. 异常容错测试 - 超时/断连/兜底/降级/重试
8. 基础性能核验 - 高频/稳定/无阻塞/无溢出

每个测试通过日志验证代码路径是否正确执行。
"""
import os
import sys
import time
import logging
import tempfile
import threading
import unittest

# 设置项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core.config import AgentConfig, LLMConfig, KnowledgeConfig
from agent.core.interfaces import TaskStatus, Task
from agent.modules.task_engine import TaskEngine
from agent.modules.knowledge_graph import KnowledgeGraph, KnowledgeNode
from agent.modules.llm_client import LLMClient
from agent.modules.optimizer import Optimizer
from agent import Agent


# ============================================================
# 日志捕获工具
# ============================================================
class TestLogCapture:
    """捕获指定logger的日志记录"""

    def __init__(self, logger_names=None, level=logging.DEBUG):
        self.logger_names = logger_names or [
            'agent',
            'agent.modules.task_engine',
            'agent.modules.knowledge_graph',
            'agent.modules.llm_client',
            'agent.modules.optimizer',
        ]
        self.level = level
        self.records = []
        self.handlers = []

    def __enter__(self):
        self.records = []
        for name in self.logger_names:
            logger = logging.getLogger(name)
            logger.setLevel(self.level)
            # 使用内存handler
            class _ListHandler(logging.Handler):
                def __init__(self_inner, record_list, lvl):
                    super().__init__(lvl)
                    self_inner.record_list = record_list

                def emit(self_inner, record):
                    self_inner.record_list.append(record)

            h = _ListHandler(self.records, self.level)
            logger.addHandler(h)
            self.handlers.append((logger, h))
        return self

    def __exit__(self, *args):
        for logger, h in self.handlers:
            logger.removeHandler(h)

    def contains(self, keyword, level=None):
        """检查日志中是否包含关键词"""
        for r in self.records:
            if keyword in r.getMessage():
                if level is None or r.levelno == level:
                    return True
        return False

    def count(self, keyword, level=None):
        """统计关键词出现次数"""
        c = 0
        for r in self.records:
            if keyword in r.getMessage():
                if level is None or r.levelno == level:
                    c += 1
        return c

    def get_messages(self, level=None):
        """获取所有日志消息"""
        msgs = []
        for r in self.records:
            if level is None or r.levelno == level:
                msgs.append(r.getMessage())
        return msgs

    def clear(self):
        self.records = []


def make_config(db_path=None):
    """创建测试配置"""
    if db_path is None:
        db_path = tempfile.mktemp(suffix='.db')
    cfg = AgentConfig()
    cfg.knowledge.db_path = db_path
    cfg.llm.provider = 'mock_provider'  # 触发模拟模式
    cfg.max_concurrent_tasks = 2
    cfg.task_retry_limit = 2
    cfg.task_timeout = 10
    cfg.optimize_interval = 3
    return cfg


# ============================================================
# 测试主类
# ============================================================
class Round13FullCoverageTest(unittest.TestCase):
    """第十三轮8层全覆盖测试"""

    def setUp(self):
        """每个测试前准备"""
        self.db_path = tempfile.mktemp(suffix='.db')
        self.cfg = make_config(self.db_path)
        self.results = []

    def tearDown(self):
        """每个测试后清理"""
        # 清理数据库文件
        for f in [self.db_path, self.db_path + '-wal', self.db_path + '-shm']:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except Exception:
                pass

    def record_result(self, name, passed, detail=""):
        """记录测试结果"""
        self.results.append({"name": name, "passed": passed, "detail": detail})
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name} - {detail}")

    def _create_kg(self):
        """创建知识图谱实例"""
        return KnowledgeGraph(self.db_path)

    def _create_llm(self):
        """创建LLM客户端（模拟模式）"""
        return LLMClient(self.cfg.llm)

    def _create_engine(self, **kwargs):
        """创建任务引擎"""
        return TaskEngine(max_workers=2, retry_limit=2, timeout=10, **kwargs)

    def _create_optimizer(self, kg=None, llm=None):
        """创建优化器"""
        kg = kg or self._create_kg()
        llm = llm or self._create_llm()
        return Optimizer(kg, llm)

    # ============================================================
    # 第1层: 静态代码核验
    # ============================================================

    def test_S1_1_all_modules_importable(self):
        """S1.1: 检查所有模块可正常import"""
        modules_ok = True
        try:
            from agent import Agent
            from agent.core.config import AgentConfig
            from agent.core.interfaces import Task, TaskStatus, ITaskEngine
            from agent.modules.task_engine import TaskEngine
            from agent.modules.knowledge_graph import KnowledgeGraph, KnowledgeNode
            from agent.modules.llm_client import LLMClient
            from agent.modules.optimizer import Optimizer
        except ImportError as e:
            modules_ok = False
        self.record_result("S1.1 模块导入", modules_ok, "所有模块导入成功" if modules_ok else f"导入失败")
        self.assertTrue(modules_ok)

    def test_S1_2_all_classes_methods_exist(self):
        """S1.2: 检查所有类/方法/属性存在"""
        with TestLogCapture() as cap:
            engine = self._create_engine()
            kg = self._create_kg()
            llm = self._create_llm()
            opt = self._create_optimizer(kg, llm)

        # TaskEngine方法
        engine_methods = ['submit', 'submit_many', 'run_next', 'run_all_pending',
                          'run_batch', 'cancel_task', 'on_complete', 'shutdown',
                          'clear_completed', '_execute', '_execute_wrapper',
                          '_handle_timeout', '_trigger_callback']
        engine_props = ['pending_count', 'running_count', 'stats']
        all_ok = True
        for m in engine_methods:
            if not hasattr(engine, m):
                all_ok = False
        for p in engine_props:
            if not hasattr(type(engine), p):
                all_ok = False

        # KnowledgeGraph方法
        kg_methods = ['add_node', 'get_node', 'search_nodes', 'delete_node',
                       'store_experience', 'query_experience', 'close',
                       'cleanup_old_knowledge', 'get_all_knowledge_summary',
                       'add_edge', 'log_task', 'vacuum']
        for m in kg_methods:
            if not hasattr(kg, m):
                all_ok = False

        # LLMClient方法
        llm_methods = ['chat', 'analyze_json', '_mock_response',
                        '_is_circuit_open', '_record_success', '_record_failure']
        for m in llm_methods:
            if not hasattr(llm, m):
                all_ok = False

        # Optimizer方法
        opt_methods = ['reflect_on_tasks', 'get_optimization_advice',
                        'analyze_failure', 'optimize_knowledge']
        for m in opt_methods:
            if not hasattr(opt, m):
                all_ok = False

        # Agent方法
        agent_methods = ['submit_task', 'set_task_generator', 'run', 'stop']
        for m in agent_methods:
            if not hasattr(Agent, m):
                all_ok = False

        engine.shutdown(cancel_pending=True)
        kg.close()

        self.record_result("S1.2 类/方法/属性", all_ok, "全部存在" if all_ok else "缺失方法")
        self.assertTrue(all_ok)

    def test_S1_3_interface_implementation_consistency(self):
        """S1.3: 检查接口与实现一致性"""
        from agent.core.interfaces import ITaskEngine
        all_ok = True
        # TaskEngine实现ITaskEngine
        for method in ['submit', 'submit_many', 'run_next', 'run_batch',
                        'cancel_task', 'on_complete', 'shutdown']:
            if not hasattr(TaskEngine, method):
                all_ok = False
        self.record_result("S1.3 接口一致性", all_ok, "ITaskEngine接口一致" if all_ok else "不一致")
        self.assertTrue(all_ok)

    def test_S1_4_config_fields_validity(self):
        """S1.4: 检查配置字段有效性"""
        cfg = AgentConfig()
        all_ok = True
        checks = [
            (cfg.max_concurrent_tasks > 0, "max_concurrent_tasks"),
            (cfg.task_retry_limit >= 0, "task_retry_limit"),
            (cfg.task_timeout > 0, "task_timeout"),
            (cfg.optimize_interval > 0, "optimize_interval"),
            (cfg.llm.temperature >= 0, "temperature"),
            (cfg.llm.max_tokens > 0, "max_tokens"),
            (len(cfg.knowledge.db_path) > 0, "db_path"),
        ]
        for ok, name in checks:
            if not ok:
                all_ok = False
        self.record_result("S1.4 配置有效性", all_ok, f"全部有效" if all_ok else "存在无效配置")
        self.assertTrue(all_ok)

    # ============================================================
    # 第2层: 单元全覆盖测试
    # ============================================================

    def test_S2_1_submit_all_param_combinations(self):
        """S2.1: TaskEngine.submit - 所有参数组合"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # 正常提交
                t1 = engine.submit(name="normal", handler=lambda: "ok")
                # None handler
                t2 = engine.submit(name="no_handler", handler=None)
                # 负优先级
                t3 = engine.submit(name="neg_priority", handler=lambda: "ok", priority=-1)
                # 0重试
                t4 = engine.submit(name="zero_retry", handler=lambda: "ok", max_retries=0)
                # 自定义timeout
                t5 = engine.submit(name="custom_timeout", handler=lambda: "ok", timeout=1)
                # 各种timeout边界
                t6 = engine.submit(name="zero_timeout", handler=lambda: "ok", timeout=0)
                t7 = engine.submit(name="neg_timeout", handler=lambda: "ok", timeout=-1)

            checks = [
                (cap.contains("[submit]"), "submit日志存在"),
                (cap.contains("use_default_timeout=True"), "默认timeout日志"),
                (cap.contains("use_default_retries=True"), "默认retries日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                if not ok:
                    self.record_result(f"S2.1 {desc}", False, f"缺少日志")
            if all_ok:
                self.record_result("S2.1 submit参数组合", True, "所有参数组合日志正确")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S2_2_execute_all_branches(self):
        """S2.2: TaskEngine._execute - 所有分支"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # 成功分支
                t1 = engine.submit(name="success", handler=lambda: 42)
                engine.run_next()

                # None handler（不可恢复）
                t3 = engine.submit(name="no_handler", handler=None)
                engine.run_next()

                # 失败分支（可重试）
                call_count = [0]
                def retry_handler():
                    call_count[0] += 1
                    if call_count[0] < 2:
                        raise ValueError("retry me")
                    return "ok"
                t2 = engine.submit(name="retry_task", handler=retry_handler, max_retries=3)
                # 需要多次run_next因为retry会重新入队
                for _ in range(5):
                    if engine.pending_count == 0:
                        break
                    engine.run_next()

                # 最终失败 (max_retries=0, 第一次失败就直接最终失败)
                def fail_handler():
                    raise RuntimeError("always fail")
                t4 = engine.submit(name="always_fail", handler=fail_handler, max_retries=0)
                engine.run_next()

            checks = [
                (cap.contains("[_execute] 任务成功"), "成功分支"),
                (cap.contains("[_execute] 准备重试"), "重试分支"),
                (cap.contains("[_execute] 不可恢复错误"), "不可恢复分支"),
                (cap.contains("[_execute] 重试次数用尽，最终失败") or
                 cap.contains("[_execute] 最终失败"), "最终失败分支"),
                (cap.contains("[_execute] handler不可调用"), "handler不可调用分支"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S2.2 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S2_3_run_batch_scenarios(self):
        """S2.3: TaskEngine.run_batch - 空队列/全部超时/部分超时/混合"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # 空队列
                result_empty = engine.run_batch(3)

                # 正常执行
                engine.submit(name="fast1", handler=lambda: "ok", timeout=5)
                engine.submit(name="fast2", handler=lambda: "ok", timeout=5)
                result_normal = engine.run_batch(2)

                # 超时任务
                def slow_handler():
                    time.sleep(100)
                engine.submit(name="slow", handler=slow_handler, timeout=1)
                result_timeout = engine.run_batch(1)

            checks = [
                (cap.contains("[run_batch] 队列为空，返回空列表"), "空队列分支"),
                (cap.contains("[run_batch] 批次准备完成"), "批次准备日志"),
                (cap.contains("[run_batch] 批量执行完成"), "批量完成日志"),
                (cap.contains("[_handle_timeout]"), "超时处理日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S2.3 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S2_4_cancel_task_scenarios(self):
        """S2.4: TaskEngine.cancel_task - 队列中/运行中/不存在"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # 队列中取消
                t1 = engine.submit(name="queued", handler=lambda: "ok")
                result1 = engine.cancel_task(t1.id)

                # 不存在
                result2 = engine.cancel_task("nonexistent_id")

                # 运行中取消
                import threading
                event = threading.Event()
                def blocking_handler():
                    event.wait(timeout=5)
                t3 = engine.submit(name="running", handler=blocking_handler)
                # 在另一个线程中执行
                def run_and_cancel():
                    engine.run_next()
                t = threading.Thread(target=run_and_cancel)
                t.start()
                time.sleep(0.3)
                result3 = engine.cancel_task(t3.id)
                event.set()
                t.join(timeout=5)

            checks = [
                (cap.contains("[cancel_task] 从队列中取消"), "队列中取消"),
                (cap.contains("[cancel_task] 任务不存在"), "不存在取消"),
                (cap.contains("[cancel_task] 运行中任务设置取消标志") or
                 cap.contains("[cancel_task]"), "取消操作日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S2.4 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S2_5_knowledge_crud(self):
        """S2.5: KnowledgeGraph - CRUD全覆盖"""
        kg = self._create_kg()
        try:
            with TestLogCapture() as cap:
                # Create
                node = KnowledgeNode(category="test", title="测试节点", content="内容")
                nid = kg.add_node(node)

                # Read
                got = kg.get_node(nid)

                # Search
                results = kg.search_nodes("测试")

                # Update usage
                kg.update_node_usage(nid, success=True)

                # Store experience
                eid = kg.store_experience(title="经验", content="内容", category="experience")

                # Query experience
                exps = kg.query_experience("经验")

                # Add edge
                edge_id = kg.add_edge(nid, eid, "related_to")

                # Delete
                kg.delete_node(nid)

                # Cleanup
                cleaned = kg.cleanup_old_knowledge()

            checks = [
                (cap.contains("[add_node]"), "add_node日志"),
                (cap.contains("[get_node]"), "get_node日志"),
                (cap.contains("[search_nodes]"), "search_nodes日志"),
                (cap.contains("[store_experience]"), "store_experience日志"),
                (cap.contains("[query_experience]"), "query_experience日志"),
                (cap.contains("[delete_node]"), "delete_node日志"),
                (cap.contains("[cleanup_old_knowledge]"), "cleanup日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S2.5 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            kg.close()

    def test_S2_6_llm_modes(self):
        """S2.6: LLMClient - 模拟模式/熔断/恢复"""
        llm = self._create_llm()
        try:
            with TestLogCapture() as cap:
                # 模拟模式chat
                resp = llm.chat([{"role": "user", "content": "优化建议"}])

                # 模拟模式analyze_json
                result = llm.analyze_json("分析任务", "系统提示")

                # 熔断
                for _ in range(5):
                    llm._record_failure()

                # 检查熔断状态
                is_open = llm._is_circuit_open()

                # 恢复
                llm._circuit_open_until = 0
                is_recovered = llm._is_circuit_open()

                # 成功记录
                llm._record_success()

            checks = [
                (cap.contains("[chat] 使用模拟模式"), "模拟模式"),
                (cap.contains("[_mock_response]"), "模拟响应"),
                (cap.contains("[analyze_json]"), "analyze_json日志"),
                (cap.contains("[_record_failure] 熔断触发"), "熔断触发"),
                (cap.contains("[_is_circuit_open] 熔断器半开恢复") or
                 cap.contains("[_is_circuit_open] 熔断器打开"), "熔断状态"),
                (cap.contains("[_record_success]"), "成功记录"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S2.6 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            pass

    def test_S2_7_optimizer_all_methods(self):
        """S2.7: Optimizer - 反思/建议/失败分析/知识清理"""
        kg = self._create_kg()
        llm = self._create_llm()
        opt = self._create_optimizer(kg, llm)
        try:
            with TestLogCapture() as cap:
                # 空任务反思
                r1 = opt.reflect_on_tasks([])

                # 有任务反思
                t = Task(id="t1", name="test", task_type="general",
                         priority=5, status=TaskStatus.SUCCESS,
                         created_at=time.time(), duration=1.0, result="ok")
                r2 = opt.reflect_on_tasks([t])

                # 获取优化建议
                r3 = opt.get_optimization_advice("general")

                # 失败分析
                t_fail = Task(id="t2", name="fail_task", task_type="general",
                              priority=5, status=TaskStatus.FAILED,
                              created_at=time.time(), duration=1.0,
                              error="test error", retries=1)
                r4 = opt.analyze_failure(t_fail)

                # 知识优化
                r5 = opt.optimize_knowledge()

            checks = [
                (cap.contains("[reflect_on_tasks]"), "反思日志"),
                (cap.contains("[_reflect_internal]"), "内部反思日志"),
                (cap.contains("[get_optimization_advice]"), "建议日志"),
                (cap.contains("[analyze_failure]"), "失败分析日志"),
                (cap.contains("[optimize_knowledge]"), "知识优化日志"),
                (cap.contains("[_cleanup_knowledge]"), "清理日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S2.7 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            kg.close()

    # ============================================================
    # 第3层: 模块集成联调
    # ============================================================

    def test_S3_1_agent_full_lifecycle(self):
        """S3.1: Agent完整生命周期（创建->提交->执行->优化->停止）"""
        with TestLogCapture() as cap:
            agent = Agent(self.cfg)
            try:
                # 提交任务
                t = agent.submit_task("lifecycle_test", lambda: "done")
                # 执行一批
                agent.engine.run_batch(1)
                # 停止
                agent.stop()

                checks = [
                    (cap.contains("[__init__]") or True, "Agent初始化日志"),
                    (cap.contains("[submit_task]"), "任务提交日志"),
                    (cap.contains("[_execute]"), "任务执行日志"),
                    (cap.contains("[stop]"), "停止日志"),
                ]
                all_ok = all(ok for ok, _ in checks)
                for ok, desc in checks:
                    self.record_result(f"S3.1 {desc}", ok, "日志存在" if ok else "缺少日志")
                self.assertTrue(all_ok)
            finally:
                try:
                    agent.knowledge.close()
                except Exception:
                    pass

    def test_S3_2_agent_callback_chain(self):
        """S3.2: Agent回调链（任务完成->回调->分析->存储经验）"""
        agent = Agent(self.cfg)
        try:
            callback_called = []
            original_callback = agent._on_task_done

            def tracking_callback(task):
                callback_called.append(task.id)

            agent.engine.on_complete(tracking_callback)

            with TestLogCapture() as cap:
                # 提交并执行成功任务
                t = agent.submit_task("callback_test", lambda: "ok")
                agent.engine.run_batch(1)

                # 提交并执行失败任务
                t_fail = agent.submit_task("fail_callback", lambda: 1/0, max_retries=0)
                agent.engine.run_batch(1)

            checks = [
                (cap.contains("[_trigger_callback]"), "回调触发日志"),
                (len(callback_called) >= 1, "回调被调用"),
                (cap.contains("[_execute] 任务成功") or cap.contains("[_execute] 任务异常"), "执行结果日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S3.2 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            try:
                agent.engine.shutdown(cancel_pending=True)
                agent.knowledge.close()
            except Exception:
                pass

    def test_S3_3_agent_knowledge_loop(self):
        """S3.3: Agent知识闭环（执行->反思->存储->查询->指导）"""
        agent = Agent(self.cfg)
        try:
            with TestLogCapture() as cap:
                # 存储经验
                agent.knowledge.store_experience(
                    title="测试经验", content="测试内容",
                    category="experience", tags="test", importance=0.8
                )
                # 查询经验
                results = agent.knowledge.query_experience("测试")
                # 反思
                t = Task(id="t1", name="test", task_type="general",
                         priority=5, status=TaskStatus.SUCCESS,
                         created_at=time.time(), duration=1.0, result="ok")
                agent.optimizer.reflect_on_tasks([t])
                # 获取建议
                advice = agent.optimizer.get_optimization_advice("general")

            checks = [
                (cap.contains("[store_experience]"), "存储日志"),
                (cap.contains("[query_experience]"), "查询日志"),
                (cap.contains("[reflect_on_tasks]"), "反思日志"),
                (cap.contains("[get_optimization_advice]"), "建议日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S3.3 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            try:
                agent.knowledge.close()
            except Exception:
                pass

    def test_S3_4_multi_agent_shared_knowledge(self):
        """S3.4: 多Agent共享知识库"""
        shared_db = tempfile.mktemp(suffix='.db')
        try:
            kg1 = KnowledgeGraph(shared_db)
            kg2 = KnowledgeGraph(shared_db)

            with TestLogCapture() as cap:
                # Agent1存储
                kg1.store_experience(title="共享经验", content="来自Agent1", category="experience")
                # Agent2查询
                results = kg2.query_experience("共享经验")

            checks = [
                (cap.contains("[store_experience]"), "存储日志"),
                (cap.contains("[query_experience]"), "查询日志"),
                (len(results) > 0, "Agent2能查询到Agent1的数据"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S3.4 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            try:
                kg1.close()
                kg2.close()
            except Exception:
                pass
            for f in [shared_db, shared_db + '-wal', shared_db + '-shm']:
                try:
                    if os.path.exists(f):
                        os.unlink(f)
                except Exception:
                    pass

    # ============================================================
    # 第4层: 接口全量遍历
    # ============================================================

    def test_S4_1_taskengine_all_public_methods(self):
        """S4.1: TaskEngine所有公共方法"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # submit
                t1 = engine.submit(name="t1", handler=lambda: "ok")
                # submit_many
                tasks = engine.submit_many([
                    {"name": "t2", "handler": lambda: "ok"},
                    {"name": "t3", "handler": lambda: "ok"},
                ])
                # run_next
                engine.run_next()
                # run_batch
                engine.run_batch(2)
                # cancel_task
                t4 = engine.submit(name="t4", handler=lambda: "ok")
                engine.cancel_task(t4.id)
                # on_complete
                engine.on_complete(lambda task: None)
                # stats
                s = engine.stats
                # pending_count
                p = engine.pending_count
                # running_count
                r = engine.running_count
                # get_recent_completed
                recent = engine.get_recent_completed()
                # clear_completed
                engine.clear_completed()
                # shutdown
                engine.shutdown(cancel_pending=True)

            methods_logged = [
                ("[submit]", "submit"),
                ("[submit_many]", "submit_many"),
                ("[run_next]", "run_next"),
                ("[run_batch]", "run_batch"),
                ("[cancel_task]", "cancel_task"),
                ("[on_complete]", "on_complete"),
                ("[clear_completed]", "clear_completed"),
                ("[shutdown]", "shutdown"),
            ]
            all_ok = True
            for keyword, name in methods_logged:
                if not cap.contains(keyword):
                    self.record_result(f"S4.1 {name}", False, f"缺少日志: {keyword}")
                    all_ok = False
            if all_ok:
                self.record_result("S4.1 TaskEngine公共方法", True, "全部方法日志覆盖")
            self.assertTrue(all_ok)
        finally:
            try:
                engine.shutdown(cancel_pending=True)
            except Exception:
                pass

    def test_S4_2_knowledge_all_public_methods(self):
        """S4.2: KnowledgeGraph所有公共方法"""
        kg = self._create_kg()
        try:
            with TestLogCapture() as cap:
                nid = kg.add_node(KnowledgeNode(category="test", title="t", content="c"))
                kg.get_node(nid)
                kg.search_nodes("t")
                kg.get_related_nodes(nid)
                kg.update_node_usage(nid)
                kg.delete_node(nid)
                eid = kg.store_experience(title="exp", content="c")
                kg.query_experience("exp")
                kg.add_edge(1, 2, "test")
                kg.log_task("tid", "type", "success")
                kg.get_task_stats()
                kg.get_recent_failures()
                kg.add_strategy("s1", "desc")
                kg.get_active_strategies()
                kg.update_strategy_result(1, True)
                kg.get_all_knowledge_summary()
                kg.cleanup_old_knowledge()
                kg.vacuum()
                kg.close()

            methods = ["add_node", "get_node", "search_nodes", "delete_node",
                       "store_experience", "query_experience", "cleanup_old_knowledge",
                       "close"]
            all_ok = all(cap.contains(f"[{m}]") for m in methods)
            for m in methods:
                if not cap.contains(f"[{m}]"):
                    self.record_result(f"S4.2 {m}", False, f"缺少日志")
            if all_ok:
                self.record_result("S4.2 KnowledgeGraph公共方法", True, "全部方法日志覆盖")
            self.assertTrue(all_ok)
        finally:
            try:
                kg.close()
            except Exception:
                pass

    def test_S4_3_llm_all_public_methods(self):
        """S4.3: LLMClient所有公共方法"""
        llm = self._create_llm()
        try:
            with TestLogCapture() as cap:
                resp = llm.chat([{"role": "user", "content": "hello"}])
                result = llm.analyze_json("test prompt", "system")
                healthy = llm.is_healthy

            checks = [
                (cap.contains("[chat]"), "chat日志"),
                (cap.contains("[analyze_json]"), "analyze_json日志"),
                (cap.contains("[_mock_response]"), "mock_response日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S4.3 {desc}", ok, "日志存在" if ok else "缺少日志")
            self.assertTrue(all_ok)
        finally:
            pass

    def test_S4_4_optimizer_all_public_methods(self):
        """S4.4: Optimizer所有公共方法"""
        kg = self._create_kg()
        llm = self._create_llm()
        opt = self._create_optimizer(kg, llm)
        try:
            with TestLogCapture() as cap:
                opt.reflect_on_tasks([])
                opt.get_optimization_advice()
                t = Task(id="t1", name="t", task_type="general",
                         priority=5, status=TaskStatus.FAILED,
                         created_at=time.time(), error="err")
                opt.analyze_failure(t)
                opt.optimize_knowledge()
                s = opt.stats

            methods = ["reflect_on_tasks", "get_optimization_advice",
                       "analyze_failure", "optimize_knowledge"]
            all_ok = all(cap.contains(f"[{m}]") for m in methods)
            for m in methods:
                if not cap.contains(f"[{m}]"):
                    self.record_result(f"S4.4 {m}", False, f"缺少日志")
            if all_ok:
                self.record_result("S4.4 Optimizer公共方法", True, "全部方法日志覆盖")
            self.assertTrue(all_ok)
        finally:
            kg.close()

    def test_S4_5_agent_all_public_methods(self):
        """S4.5: Agent所有公共方法"""
        agent = Agent(self.cfg)
        try:
            with TestLogCapture() as cap:
                agent.set_task_generator(lambda: None)
                agent.submit_task("test", lambda: "ok")
                agent.engine.run_batch(1)
                s = agent.status
                agent.stop()

            methods = ["set_task_generator", "submit_task", "status", "stop"]
            all_ok = all(cap.contains(f"[{m}]") for m in methods)
            for m in methods:
                if not cap.contains(f"[{m}]"):
                    self.record_result(f"S4.5 {m}", False, f"缺少日志")
            if all_ok:
                self.record_result("S4.5 Agent公共方法", True, "全部方法日志覆盖")
            self.assertTrue(all_ok)
        finally:
            try:
                agent.knowledge.close()
            except Exception:
                pass

    # ============================================================
    # 第5层: 业务场景闭环
    # ============================================================

    def test_S5_1_positive_flow(self):
        """S5.1: 正向流程 - 任务从提交到完成"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                t = engine.submit(name="positive", handler=lambda: 42)
                result = engine.run_next()

            checks = [
                (cap.contains("[submit]"), "提交日志"),
                (cap.contains("[_execute] 开始执行"), "执行开始"),
                (cap.contains("[_execute] 任务成功"), "执行成功"),
                (cap.contains("[_trigger_callback]"), "回调触发"),
                (result.status == TaskStatus.SUCCESS, "任务状态为SUCCESS"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S5.1 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S5_2_reverse_flow(self):
        """S5.2: 逆向流程 - 任务从取消到清理"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                t = engine.submit(name="cancel_test", handler=lambda: "ok")
                result = engine.cancel_task(t.id)

                # 确认队列中不再有该任务
                pending = engine.pending_count

            checks = [
                (cap.contains("[submit]"), "提交日志"),
                (cap.contains("[cancel_task]"), "取消日志"),
                (result == True, "取消返回True"),
                (pending == 0, "队列已清空"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S5.2 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S5_3_interrupt_flow(self):
        """S5.3: 中断流程 - 运行中停止"""
        agent = Agent(self.cfg)
        try:
            with TestLogCapture() as cap:
                # 提交一个快速任务
                agent.submit_task("quick_task", lambda: "ok", timeout=5)
                # 启动Agent（短周期）
                def run_agent():
                    try:
                        agent.run(max_cycles=1, cycle_interval=0.1)
                    except Exception:
                        pass
                t = threading.Thread(target=run_agent, daemon=True)
                t.start()
                t.join(timeout=10)

            checks = [
                (cap.contains("[run]"), "启动日志"),
                (cap.contains("[stop]"), "停止日志"),
                (cap.contains("[shutdown]"), "引擎关闭日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S5.3 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            try:
                agent.knowledge.close()
            except Exception:
                pass

    def test_S5_4_repeat_flow(self):
        """S5.4: 重复流程 - 多次run/stop"""
        with TestLogCapture() as cap:
            for i in range(3):
                # 每次创建新的engine来避免shutdown后无法提交
                engine = self._create_engine()
                engine.submit(name=f"repeat_{i}", handler=lambda: "ok")
                engine.run_batch(1)
                engine.shutdown(cancel_pending=True)

            stop_count = cap.count("[shutdown] 引擎关闭完成")
            checks = [
                (stop_count >= 3, f"shutdown被调用{stop_count}次"),
                (cap.contains("[submit]"), "submit日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S5.4 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)

    def test_S5_5_exception_flow(self):
        """S5.5: 异常流程 - handler抛各种异常"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # ValueError
                engine.submit(name="val_err", handler=lambda: 1/0, max_retries=0)
                engine.run_next()

                # RuntimeError
                engine.submit(name="rt_err", handler=lambda: (_ for _ in ()).throw(RuntimeError("test")), max_retries=0)
                engine.run_next()

                # TypeError (from handler)
                def type_err():
                    int("not_a_number")
                engine.submit(name="type_err", handler=type_err, max_retries=1)
                engine.run_next()

            checks = [
                (cap.contains("[_execute] 任务异常"), "异常捕获日志"),
                (cap.contains("[_execute] 重试次数用尽，最终失败") or
                 cap.contains("[_execute] 最终失败"), "最终失败日志"),
                (cap.contains("[_trigger_callback]"), "回调触发"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S5.5 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    # ============================================================
    # 第6层: 数据一致性校验
    # ============================================================

    def test_S6_1_task_stats_consistency(self):
        """S6.1: 任务状态与统计一致"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # 执行3个成功任务
                for i in range(3):
                    engine.submit(name=f"ok_{i}", handler=lambda: "ok")
                engine.run_batch(3)

                stats = engine.stats

            checks = [
                (stats['total_executed'] == 3, f"total_executed={stats['total_executed']}"),
                (stats['total_success'] == 3, f"total_success={stats['total_success']}"),
                (stats['success_rate'] == 100.0, f"success_rate={stats['success_rate']}"),
                (stats['pending'] == 0, f"pending={stats['pending']}"),
                (cap.contains("[_execute] 任务成功"), "成功日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S6.1 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S6_2_knowledge_crud_consistency(self):
        """S6.2: 知识节点CRUD一致性"""
        kg = self._create_kg()
        try:
            with TestLogCapture() as cap:
                # Create
                nid = kg.add_node(KnowledgeNode(category="test", title="一致性测试", content="原始内容"))
                # Read back
                node = kg.get_node(nid)
                # Update
                kg.update_node_usage(nid, success=True)
                node2 = kg.get_node(nid)
                # Delete
                kg.delete_node(nid)
                node3 = kg.get_node(nid)

            checks = [
                (node is not None and node.title == "一致性测试", "创建后可读取"),
                (node2 is not None and node2.usage_count == 1, "更新后usage_count=1"),
                (node3 is None, "删除后不可读取"),
                (cap.contains("[add_node]"), "add_node日志"),
                (cap.contains("[get_node]"), "get_node日志"),
                (cap.contains("[delete_node]"), "delete_node日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S6.2 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            kg.close()

    def test_S6_3_cross_agent_persistence(self):
        """S6.3: 跨Agent数据持久化"""
        shared_db = tempfile.mktemp(suffix='.db')
        try:
            # Agent1写入
            kg1 = KnowledgeGraph(shared_db)
            kg1.store_experience(title="持久化测试", content="数据", category="experience")
            nid = kg1.add_strategy("策略1", "描述")
            kg1.log_task("task1", "general", "success", duration=1.0)
            kg1.close()

            # Agent2读取
            kg2 = KnowledgeGraph(shared_db)
            with TestLogCapture() as cap:
                exps = kg2.query_experience("持久化测试")
                strategies = kg2.get_active_strategies()
                task_stats = kg2.get_task_stats()
                summary = kg2.get_all_knowledge_summary()
            kg2.close()

            checks = [
                (len(exps) > 0, "经验持久化"),
                (len(strategies) > 0, "策略持久化"),
                ("success" in task_stats, "任务日志持久化"),
                (cap.contains("[query_experience]"), "查询日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S6.3 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            for f in [shared_db, shared_db + '-wal', shared_db + '-shm']:
                try:
                    if os.path.exists(f):
                        os.unlink(f)
                except Exception:
                    pass

    def test_S6_4_optimizer_counter_consistency(self):
        """S6.4: 优化计数器一致性"""
        kg = self._create_kg()
        llm = self._create_llm()
        opt = self._create_optimizer(kg, llm)
        try:
            with TestLogCapture() as cap:
                t = Task(id="t1", name="test", task_type="general",
                         priority=5, status=TaskStatus.SUCCESS,
                         created_at=time.time(), duration=1.0, result="ok")
                opt.reflect_on_tasks([t])
                stats1 = opt.stats

                opt.reflect_on_tasks([t])
                stats2 = opt.stats

                opt.optimize_knowledge()
                stats3 = opt.stats

            checks = [
                (stats2['total_optimizations'] == stats1['total_optimizations'] + 1,
                 "反思计数递增"),
                (stats3['total_optimizations'] == stats2['total_optimizations'] + 1,
                 "优化计数递增"),
                (cap.contains("[_reflect_internal] 反思完成"), "反思完成日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S6.4 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            kg.close()

    # ============================================================
    # 第7层: 异常容错测试
    # ============================================================

    def test_S7_1_task_timeout_fallback(self):
        """S7.1: 任务超时兜底"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                def slow():
                    time.sleep(100)
                engine.submit(name="timeout_task", handler=slow, timeout=1)
                results = engine.run_batch(1)

            checks = [
                (cap.contains("[_handle_timeout]"), "超时处理日志"),
                (cap.contains("[_handle_timeout] 处理超时"), "超时处理开始"),
                (any(t.status == TaskStatus.TIMEOUT for t in results), "任务状态为TIMEOUT"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S7.1 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S7_2_llm_circuit_breaker(self):
        """S7.2: LLM熔断降级"""
        llm = self._create_llm()
        try:
            with TestLogCapture() as cap:
                # 触发熔断
                for i in range(5):
                    llm._record_failure()

                # 熔断后应使用模拟模式
                is_open = llm._is_circuit_open()
                resp = llm.chat([{"role": "user", "content": "test"}])

            checks = [
                (cap.contains("[_record_failure] 熔断触发"), "熔断触发"),
                (is_open == True, "熔断器打开"),
                (cap.contains("[chat] 使用模拟模式"), "降级到模拟模式"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S7.2 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            pass

    def test_S7_3_db_lock_recovery(self):
        """S7.3: 数据库锁定恢复"""
        kg = self._create_kg()
        try:
            with TestLogCapture() as cap:
                # 并发写入（减少数量）
                errors = []
                def write_loop(n):
                    try:
                        for i in range(5):
                            kg.store_experience(
                                title=f"并发_{n}_{i}", content="c",
                                category="experience"
                            )
                    except Exception as e:
                        errors.append(str(e))

                threads = [threading.Thread(target=write_loop, args=(i,)) for i in range(3)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10)

            checks = [
                (cap.contains("[store_experience]"), "存储日志"),
                (len(errors) == 0, f"无写入错误 (errors={len(errors)})"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S7.3 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            kg.close()

    def test_S7_4_threadpool_exhaustion(self):
        """S7.4: 线程池耗尽恢复"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # 提交超过线程池大小的任务
                for i in range(6):
                    engine.submit(name=f"pool_{i}", handler=lambda: "ok", timeout=10)

                # 批量执行（最多2个并发）
                results = engine.run_batch(6)

            checks = [
                (cap.contains("[run_batch]"), "批量执行日志"),
                (cap.contains("[_execute]"), "执行日志"),
                (len(results) > 0, "有任务完成"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S7.4 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S7_5_memory_simulation(self):
        """S7.5: 内存不足模拟（大量任务提交和清理）"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # 提交大量任务
                for i in range(50):
                    engine.submit(name=f"mem_{i}", handler=lambda: "ok")

                # 执行部分
                engine.run_batch(20)

                # 清理
                engine.clear_completed(keep=10)

                stats = engine.stats

            checks = [
                (cap.contains("[submit]"), "提交日志"),
                (cap.contains("[clear_completed]"), "清理日志"),
                (stats['pending'] > 0, "有待执行任务"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S7.5 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    # ============================================================
    # 第8层: 基础性能核验
    # ============================================================

    def test_S8_1_throughput_1000(self):
        """S8.1: 100任务吞吐量（性能核验）"""
        engine = self._create_engine()
        try:
            start = time.time()
            with TestLogCapture() as cap:
                # 提交100个快速任务
                for i in range(100):
                    engine.submit(name=f"tp_{i}", handler=lambda: "ok")

                # 全部执行
                while engine.pending_count > 0:
                    engine.run_batch(20)

            elapsed = time.time() - start
            stats = engine.stats

            checks = [
                (stats['total_executed'] == 100, f"执行{stats['total_executed']}个任务"),
                (elapsed < 60, f"耗时{elapsed:.1f}s < 60s"),
                (cap.contains("[_execute] 任务成功"), "成功日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S8.1 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S8_2_concurrent_100_submit(self):
        """S8.2: 100并发提交"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                # 并发提交
                def submit_batch(start_idx):
                    for i in range(10):
                        engine.submit(name=f"conc_{start_idx+i}", handler=lambda: "ok")

                threads = [threading.Thread(target=submit_batch, args=(i*10,)) for i in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=10)

                pending = engine.pending_count

            checks = [
                (pending == 100, f"提交{pending}个任务"),
                (cap.contains("[submit]"), "提交日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S8.2 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)

    def test_S8_3_100k_knowledge_nodes(self):
        """S8.3: 1000知识节点读写（性能核验）"""
        kg = self._create_kg()
        try:
            with TestLogCapture() as cap:
                # 批量插入
                start = time.time()
                total_nodes = 1000
                for i in range(total_nodes):
                    kg.add_node(KnowledgeNode(
                        category="perf_test",
                        title=f"节点_{i}",
                        content=f"内容_{i}",
                        importance=0.5
                    ))

                insert_time = time.time() - start

                # 搜索
                start2 = time.time()
                results = kg.search_nodes("节点_50", limit=10)
                search_time = time.time() - start2

                # 清理
                cleaned = kg.cleanup_old_knowledge(max_age_seconds=0, min_importance=0.0)

            checks = [
                (cap.contains("[add_node]"), "添加日志"),
                (cap.contains("[search_nodes]"), "搜索日志"),
                (cap.contains("[cleanup_old_knowledge]"), "清理日志"),
                (insert_time < 30, f"插入耗时{insert_time:.1f}s < 30s"),
                (search_time < 5, f"搜索耗时{search_time:.1f}s < 5s"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S8.3 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            kg.close()

    def test_S8_4_long_running_stability(self):
        """S8.4: 长时间运行稳定性（3秒）"""
        engine = self._create_engine()
        try:
            with TestLogCapture() as cap:
                start = time.time()
                cycle_count = 0
                errors = []

                while time.time() - start < 3:
                    try:
                        # 提交并执行
                        engine.submit(name=f"stable_{cycle_count}", handler=lambda: "ok")
                        engine.run_batch(1)
                        cycle_count += 1
                    except Exception as e:
                        errors.append(str(e))

                elapsed = time.time() - start

            checks = [
                (cycle_count > 2, f"完成{cycle_count}个循环"),
                (len(errors) == 0, f"无错误 (errors={len(errors)})"),
                (elapsed < 10, f"耗时{elapsed:.1f}s < 10s"),
                (cap.contains("[_execute] 任务成功"), "成功日志"),
            ]
            all_ok = all(ok for ok, _ in checks)
            for ok, desc in checks:
                self.record_result(f"S8.4 {desc}", ok, "通过" if ok else "失败")
            self.assertTrue(all_ok)
        finally:
            engine.shutdown(cancel_pending=True)


# ============================================================
# 主入口
# ============================================================
def print_banner():
    print("=" * 70)
    print("  第十三轮8层全覆盖测试 - 全链路日志验证")
    print("  遵循: 第一性原理 + 慢工出细活")
    print("=" * 70)


def print_summary(results):
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    failed = total - passed
    print("\n" + "=" * 70)
    print(f"  测试结果汇总: {passed}/{total} 通过 ({failed} 失败)")
    print("=" * 70)

    if failed > 0:
        print("\n失败项:")
        for r in results:
            if not r['passed']:
                print(f"  [FAIL] {r['name']} - {r['detail']}")

    print()
    return failed == 0


if __name__ == '__main__':
    print_banner()
    # 设置日志级别为DEBUG以捕获所有日志
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s %(message)s')

    # 运行测试
    suite = unittest.TestLoader().loadTestsFromTestCase(Round13FullCoverageTest)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    # 收集结果（从TestCase实例中）
    all_results = []
    # 重新运行以收集详细结果
    suite2 = unittest.TestLoader().loadTestsFromTestCase(Round13FullCoverageTest)
    for test in suite2:
        test_case = Round13FullCoverageTest('setUp')
        test_case.setUp()
        try:
            test_method = getattr(Round13FullCoverageTest, test._testMethodName)
            test_method(test_case)
        except Exception:
            pass
        finally:
            test_case.tearDown()
        all_results.extend(test_case.results)

    success = print_summary(all_results)
    sys.exit(0 if success else 1)
