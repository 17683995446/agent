"""
持续运行Agent - 核心主类（增强版：线程安全、资源管理、内存优化）
整合：执行引擎 + 优化模块 + 知识图谱 + 保活机制
"""
import time
import logging
import signal
import sys
import traceback
import threading
from typing import Optional, Callable, List, Dict, Any

from .core.config import AgentConfig
from .modules.knowledge_graph import KnowledgeGraph
from .modules.llm_client import LLMClient
from .modules.task_engine import TaskEngine, Task, TaskStatus
from .modules.optimizer import Optimizer

logger = logging.getLogger(__name__)


class Agent:
    """
    极简全能持续运行Agent（增强版）

    三大同步工作：
    1. 持续做事：一直完成既定任务，不停推进
    2. 自动优化：边做边调整做法，越做越好
    3. 存入知识库：把做法、经验、项目资料全部放进知识图谱
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig.from_env()
        self._running = False
        self._cycle_count = 0
        self._start_time = 0
        self._tasks_since_last_optimize = 0
        self._lock = threading.RLock()  # 状态锁
        self._shutdown_event = threading.Event()

        # 初始化三大模块
        self.knowledge = KnowledgeGraph(self.config.knowledge.db_path)
        self.llm = LLMClient(self.config.llm)
        self.engine = TaskEngine(
            max_workers=self.config.max_concurrent_tasks,
            retry_limit=self.config.task_retry_limit,
            timeout=self.config.task_timeout,
        )
        self.optimizer = Optimizer(self.knowledge, self.llm)

        # 注册任务完成回调
        self.engine.on_complete(self._on_task_done)

        # 自定义任务生成器
        self._task_generator: Optional[Callable] = None

        # P2修复：优化专用后台线程
        self._optimize_thread: Optional[threading.Thread] = None
        self._optimize_queue: List = []
        self._optimize_queue_lock = threading.Lock()
        self._optimize_stop_event = threading.Event()

        # 初始经验
        self._seed_initial_knowledge()

        logger.info("Agent初始化完成 ✓")

    def _start_optimize_thread(self):
        """P2修复：启动后台优化线程"""
        self._optimize_stop_event.clear()

        def _optimize_worker():
            while not self._optimize_stop_event.is_set():
                tasks_to_optimize = []
                with self._optimize_queue_lock:
                    if self._optimize_queue:
                        tasks_to_optimize = self._optimize_queue[:]
                        self._optimize_queue.clear()

                if tasks_to_optimize:
                    try:
                        self.optimizer.reflect_on_tasks(tasks_to_optimize)
                    except Exception as e:
                        logger.error(f"后台优化异常: {e}")

                # 短暂休眠避免空转
                self._optimize_stop_event.wait(timeout=0.5)

        self._optimize_thread = threading.Thread(
            target=_optimize_worker, name="OptimizeWorker", daemon=True
        )
        self._optimize_thread.start()
        logger.info("后台优化线程已启动")

    def _seed_initial_knowledge(self):
        """注入初始知识"""
        try:
            existing = self.knowledge.search_nodes("初始经验", limit=1)
            if not existing:
                self.knowledge.store_experience(
                    title="Agent启动经验",
                    content="Agent已初始化，知识图谱、执行引擎、优化模块就绪。持续运行中会自动积累经验。",
                    category="experience",
                    tags="init,agent,startup",
                    importance=0.9,
                )
                self.knowledge.store_experience(
                    title="错误处理原则",
                    content="遇到错误时：1.记录完整错误信息 2.分析根本原因 3.尝试自动修复 4.更新知识图谱 5.调整策略避免重复",
                    category="method",
                    tags="error-handling,best-practice",
                    importance=0.9,
                )
                self.knowledge.store_experience(
                    title="优化循环方法",
                    content="每完成一批任务后进行反思：分析成功模式、识别失败原因、提取可复用策略、更新知识库",
                    category="method",
                    tags="optimization,reflection,loop",
                    importance=0.8,
                )
                logger.info("初始知识已注入知识图谱")
        except Exception as e:
            logger.error(f"初始知识注入失败: {e}")

    def set_task_generator(self, generator: Callable):
        """
        设置任务生成器函数

        Args:
            generator: 无参函数，返回Task或List[Task]或None
                      每个循环会被调用，用于动态生成新任务
        """
        with self._lock:
            self._task_generator = generator
        logger.info("任务生成器已设置")

    def submit_task(self, name: str, handler: Callable, **kwargs) -> Task:
        """快捷提交任务"""
        return self.engine.submit(name=name, handler=handler, **kwargs)

    def _on_task_done(self, task: Task):
        """任务完成回调 - 触发优化和知识存储（P2修复：异步优化）"""
        with self._lock:
            self._tasks_since_last_optimize += 1
            should_optimize = self._tasks_since_last_optimize >= self.config.optimize_interval

        # 失败任务立即分析（轻量操作，保持同步）
        if task.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT):
            try:
                self.optimizer.analyze_failure(task)
            except Exception as e:
                logger.error(f"失败分析异常: {e}")

        # 达到优化间隔，将任务放入后台优化队列（P2修复：不再阻塞执行线程）
        if should_optimize:
            with self._lock:
                self._tasks_since_last_optimize = 0
            try:
                recent = self.engine.get_recent_completed(limit=self.config.optimize_interval)
                if recent:
                    with self._optimize_queue_lock:
                        self._optimize_queue.append(recent)
            except Exception as e:
                logger.error(f"优化入队异常: {e}")

        # 定期清理完成任务记录（防止内存泄漏）
        if self.engine.stats['total_executed'] % 50 == 0:
            self.engine.clear_completed(keep=100)

    def _generate_tasks(self):
        """调用任务生成器获取新任务"""
        with self._lock:
            generator = self._task_generator

        if generator is None:
            return

        try:
            result = generator()

            if result is None:
                return

            if isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        self.engine.submit(**item)
                    elif isinstance(item, Task):
                        # F2修复：新Task是纯数据对象，无handler属性
                        # 需要通过metadata或外部注册handler，否则跳过并警告
                        logger.warning(f"任务生成器返回了Task数据对象但无handler，已跳过: {getattr(item, 'name', 'unknown')}")
                    else:
                        logger.warning(f"任务生成器返回了不支持的类型: {type(item)}")
            elif isinstance(result, Task):
                logger.warning(f"任务生成器返回了Task数据对象但无handler，已跳过: {getattr(result, 'name', 'unknown')}")
            elif isinstance(result, dict):
                self.engine.submit(**result)

        except Exception as e:
            logger.error(f"任务生成异常: {e}")
            logger.debug(traceback.format_exc())

    def _run_cycle(self):
        """执行一个工作循环（线程安全）"""
        with self._lock:
            self._cycle_count += 1
            current_cycle = self._cycle_count

        # 1. 从知识图谱调取经验（P1修复：主动查询并利用经验）
        if current_cycle == 1 or current_cycle % 10 == 0:
            try:
                advice = self.optimizer.get_optimization_advice()
                if advice.get("has_advice"):
                    logger.info(f"优化建议: {advice.get('advice', 'N/A')}")
            except Exception as e:
                logger.error(f"获取优化建议异常: {e}")

        # 2. 生成新任务
        self._generate_tasks()

        # 3. 执行任务
        pending = self.engine.pending_count
        if pending > 0:
            # P1修复：执行前查询相关经验，形成知识→执行闭环
            try:
                pending_types = set()
                with self._lock:
                    for task in self.engine._queue[:5]:
                        pending_types.add(getattr(task, 'task_type', 'general'))
                for tt in list(pending_types)[:3]:
                    self.knowledge.query_experience(tt, limit=2)
            except Exception:
                pass

            batch_size = min(self.config.max_concurrent_tasks, pending)
            self.engine.run_batch(batch_size)
        else:
            # 没有任务时，进行知识优化
            if current_cycle % 5 == 0:
                try:
                    self.optimizer.optimize_knowledge()
                except Exception as e:
                    logger.error(f"知识优化异常: {e}")

        # 4. 打印状态
        if current_cycle % 3 == 0:
            self._print_status()

    def _print_status(self):
        """打印运行状态"""
        try:
            engine_stats = self.engine.stats
            kg_summary = self.knowledge.get_all_knowledge_summary()
            opt_stats = self.optimizer.stats

            with self._lock:
                uptime = time.time() - self._start_time if self._start_time else 0
                current_cycle = self._cycle_count

            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)

            logger.info(
                f"[循环 #{current_cycle}] "
                f"运行时间: {hours}h{minutes}m | "
                f"任务: {engine_stats['total_executed']}完成 "
                f"({engine_stats['success_rate']}%成功) | "
                f"待执行: {engine_stats['pending']} | "
                f"知识: {sum(kg_summary.get('nodes_by_category', {}).values())}条 | "
                f"策略: {kg_summary.get('active_strategies', 0)}条 | "
                f"优化: {opt_stats.get('total_optimizations', 0)}次"
            )
        except Exception as e:
            logger.error(f"状态打印异常: {e}")

    def run(self, max_cycles: int = 0, cycle_interval: float = 2.0):
        """
        启动Agent持续运行

        Args:
            max_cycles: 最大循环次数，0=无限运行
            cycle_interval: 循环间隔(秒)
        """
        with self._lock:
            if self._running:
                logger.warning("Agent已在运行中")
                return
            self._running = True
            self._start_time = time.time()
            self._shutdown_event.clear()

        logger.info("=" * 60)
        logger.info("🚀 极简全能持续运行Agent 已启动")
        logger.info(f"   最大并发: {self.config.max_concurrent_tasks}")
        logger.info(f"   优化间隔: 每{self.config.optimize_interval}个任务")
        logger.info(f"   重试限制: {self.config.task_retry_limit}次")
        logger.info(f"   知识库: {self.config.knowledge.db_path}")
        logger.info(f"   LLM: {self.config.llm.provider}/{self.config.llm.model}")
        logger.info(f"   模式: {'无限运行' if max_cycles == 0 else f'最多{max_cycles}轮'}")
        logger.info("=" * 60)

        # P2修复：启动后台优化线程
        self._start_optimize_thread()

        try:
            while True:
                with self._lock:
                    if not self._running:
                        break
                    current_cycle = self._cycle_count

                if max_cycles > 0 and current_cycle >= max_cycles:
                    logger.info(f"已达最大循环次数 {max_cycles}，Agent停止运行")
                    break

                try:
                    self._run_cycle()
                except KeyboardInterrupt:
                    logger.info("收到中断信号，正在优雅关闭...")
                    break
                except Exception as e:
                    logger.error(f"循环异常: {e}")
                    logger.debug(traceback.format_exc())
                    # 自动修复：记录错误，继续运行
                    try:
                        self.knowledge.store_experience(
                            title=f"运行时错误 #{current_cycle}",
                            content=f"错误: {str(e)}\n堆栈: {traceback.format_exc()[:500]}",
                            category="error",
                            tags="runtime-error,auto-recovery",
                            importance=0.7,
                        )
                    except Exception as ke:
                        logger.error(f"错误记录失败: {ke}")

                # 使用事件等待以支持快速退出
                if self._shutdown_event.wait(timeout=cycle_interval):
                    break

        finally:
            self.stop()

    def stop(self):
        """停止Agent（线程安全）"""
        # E2修复：无论_running状态如何，都要清理资源
        was_running = False
        with self._lock:
            if self._running:
                was_running = True
                self._running = False
                self._shutdown_event.set()

        if not was_running:
            # Agent未运行，但仍需清理资源
            logger.info("Agent未运行，执行资源清理...")
        else:
            logger.info("正在停止Agent...")

        # P2修复：停止后台优化线程
        self._optimize_stop_event.set()
        if self._optimize_thread and self._optimize_thread.is_alive():
            self._optimize_thread.join(timeout=5)

        # 等待优化队列清空
        for _ in range(10):
            with self._optimize_queue_lock:
                if not self._optimize_queue:
                    break
            time.sleep(0.2)

        # 关闭执行引擎（总是执行，清理所有资源）
        try:
            self.engine.shutdown(cancel_pending=True)
        except Exception as e:
            logger.error(f"引擎关闭异常: {e}")

        # 最终优化
        with self._lock:
            pending_optimize = self._tasks_since_last_optimize

        if pending_optimize > 0:
            try:
                recent = self.engine.get_recent_completed(limit=pending_optimize)
                if recent:
                    self.optimizer.reflect_on_tasks(recent)
            except Exception as e:
                logger.error(f"最终优化异常: {e}")

        # 保存运行总结
        try:
            stats = self.engine.stats
            opt_stats = self.optimizer.stats
            self.knowledge.store_experience(
                title=f"运行总结 #{self._cycle_count}轮",
                content=f"总任务: {stats.get('total_executed', 0)}, "
                        f"成功率: {stats.get('success_rate', 0)}%, "
                        f"优化次数: {opt_stats.get('total_optimizations', 0)}",
                category="experience",
                tags="run-summary,agent",
                importance=0.6,
            )
        except Exception as e:
            logger.error(f"运行总结保存失败: {e}")

        # 数据库优化
        try:
            self.knowledge.vacuum()
        except Exception as e:
            logger.error(f"数据库优化失败: {e}")

        # 关闭知识图谱
        try:
            self.knowledge.close()
        except Exception as e:
            logger.error(f"知识图谱关闭失败: {e}")

        logger.info("Agent已优雅关闭 ✓")

    @property
    def status(self) -> Dict[str, Any]:
        """获取Agent当前状态（线程安全）"""
        with self._lock:
            running = self._running
            cycle_count = self._cycle_count
            uptime = time.time() - self._start_time if self._start_time else 0

        kg_summary = {}
        try:
            kg_summary = self.knowledge.get_all_knowledge_summary()
        except Exception:
            pass

        return {
            "running": running,
            "cycle_count": cycle_count,
            "uptime": uptime,
            "engine": self.engine.stats,
            "knowledge": kg_summary,
            "optimizer": self.optimizer.stats,
        }


def run_with_watchdog(agent: Agent, max_cycles: int = 0, cycle_interval: float = 2.0,
                      max_restarts: int = 10, restart_cooldown: int = 5):
    """
    带看门狗的Agent运行（保活模块）

    崩了自动重启，全程不停止。

    Args:
        agent: Agent实例
        max_cycles: 每次启动的最大循环数
        cycle_interval: 循环间隔
        max_restarts: 最大重启次数
        restart_cooldown: 重启冷却时间
    """
    restart_count = 0
    task_generator = None

    # 保存原始任务生成器
    with agent._lock:
        task_generator = agent._task_generator

    while restart_count <= max_restarts:
        try:
            agent.run(max_cycles=max_cycles, cycle_interval=cycle_interval)
            break  # 正常退出

        except Exception as e:
            restart_count += 1
            logger.error(f"Agent崩溃! (重启 {restart_count}/{max_restarts}): {e}")
            logger.debug(traceback.format_exc())

            if restart_count > max_restarts:
                logger.critical(f"超过最大重启次数 {max_restarts}，彻底停止")
                break

            logger.info(f"等待 {restart_cooldown} 秒后重启...")
            time.sleep(restart_cooldown)

            # 重建Agent（先释放旧资源）
            try:
                old_agent = agent
                old_config = old_agent.config
                try:
                    old_agent.knowledge.close()
                except Exception:
                    pass
                agent = Agent(old_config)
                if task_generator:
                    agent.set_task_generator(task_generator)
                logger.info(f"Agent重启成功 (第{restart_count}次)")
            except Exception as re:
                logger.critical(f"重启失败: {re}")
                break
