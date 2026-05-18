"""
执行引擎模块 - 任务执行核心（增强版：真正超时控制、资源管理）
负责：任务调度、执行、结果收集、错误处理、超时控制
"""
import time
import uuid
import logging
import traceback
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)


from ..core.interfaces import ITaskEngine, Task, TaskStatus


class TaskEngine(ITaskEngine):
    """任务执行引擎（增强版）"""

    def __init__(self, max_workers: int = 3, retry_limit: int = 3, timeout: int = 300):
        self.max_workers = max_workers
        self.retry_limit = retry_limit
        self.default_timeout = timeout
        self._queue: List[Task] = []
        self._running: Dict[str, Task] = {}
        self._completed: List[Task] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TaskWorker")
        self._total_executed = 0
        self._total_success = 0
        self._total_failed = 0
        self._on_task_complete: Optional[Callable] = None
        self._lock = threading.RLock()
        self._shutdown = False
        self._cancel_events: Dict[str, threading.Event] = {}

    def submit(self, name: str, handler: Callable, args: tuple = (), kwargs: Dict = None,
               task_type: str = "general", priority: int = 5, timeout: int = 0,
               max_retries: int = 0, metadata: Dict = None) -> Task:
        """提交新任务"""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("执行引擎已关闭，无法提交新任务")

            task = Task(
                id=str(uuid.uuid4())[:8],
                name=name,
                task_type=task_type,
                priority=priority,
                status=TaskStatus.PENDING,
                created_at=time.time(),
                max_retries=max_retries or self.retry_limit,
                timeout=timeout or self.default_timeout,
                metadata=metadata or {},
            )
            # 存储handler和args/kwargs在内部字典中
            if not hasattr(self, '_task_handlers'):
                self._task_handlers: Dict[str, tuple] = {}
            self._task_handlers[task.id] = (handler, args, kwargs or {})
            self._cancel_events[task.id] = threading.Event()
            self._queue.append(task)
            logger.info(f"任务已提交: [{task.task_type}] {task.name} (id={task.id}, 优先级={task.priority}, 超时={task.timeout}s)")
            return task

    def submit_many(self, tasks: List[Dict]) -> List[Task]:
        """批量提交任务"""
        return [self.submit(**t) for t in tasks]

    def run_next(self) -> Optional[Task]:
        """执行下一个任务（优先级排序）"""
        with self._lock:
            if not self._queue:
                return None
            self._queue.sort(key=lambda t: t.priority)
            task = self._queue.pop(0)
        return self._execute(task)

    def run_all_pending(self) -> List[Task]:
        """执行所有待处理任务"""
        results = []
        while True:
            with self._lock:
                if not self._queue:
                    break
            task = self.run_next()
            if task:
                results.append(task)
        return results

    def run_batch(self, count: int = 3) -> List[Task]:
        """并行执行一批任务（真正超时控制）"""
        with self._lock:
            if not self._queue:
                return []
            self._queue.sort(key=lambda t: t.priority)
            batch_size = min(count, len(self._queue))
            batch = [self._queue.pop(0) for _ in range(batch_size)]

        futures = {}
        for task in batch:
            future = self._executor.submit(self._execute_wrapper, task)
            futures[future] = task

        results = []
        for future in as_completed(futures):
            task = futures[future]
            try:
                # 等待任务完成，带真正超时
                result = future.result(timeout=task.timeout)
                results.append(result)
            except FutureTimeoutError:
                # 任务执行超时
                task.status = TaskStatus.TIMEOUT
                task.error = f"任务执行超时 ({task.timeout}s)"
                task.finished_at = time.time()
                task.duration = task.finished_at - task.started_at
                self._total_failed += 1
                self._total_executed += 1
                with self._lock:
                    if task.id in self._running:
                        del self._running[task.id]
                    self._completed.append(task)
                self._trigger_callback(task)
                results.append(task)
                logger.warning(f"任务超时: {task.name}")
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.finished_at = time.time()
                task.duration = task.finished_at - task.started_at
                self._total_failed += 1
                self._total_executed += 1
                with self._lock:
                    if task.id in self._running:
                        del self._running[task.id]
                    self._completed.append(task)
                self._trigger_callback(task)
                results.append(task)

        return results

    def _execute_wrapper(self, task: Task) -> Task:
        """执行包装器（用于线程池）- 捕获所有异常包括BaseException"""
        try:
            return self._execute(task)
        except BaseException as e:
            # 捕获KeyboardInterrupt/SystemExit等，防止worker线程崩溃传播到主线程
            task.status = TaskStatus.FAILED
            task.error = f"[{type(e).__name__}] {str(e)}"
            task.finished_at = time.time()
            task.duration = task.finished_at - task.started_at
            with self._lock:
                self._total_failed += 1
                self._total_executed += 1
                if task.id in self._running:
                    del self._running[task.id]
                self._completed.append(task)
            self._trigger_callback(task)
            logger.error(f"任务捕获极端异常: {task.name} - {type(e).__name__}: {e}")
            return task

    def _execute(self, task: Task) -> Task:
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        with self._lock:
            self._running[task.id] = task
            # 获取handler信息
            handler_info = self._task_handlers.get(task.id, (None, (), {}))
            handler, args, kwargs = handler_info

        logger.info(f"开始执行任务: [{task.task_type}] {task.name} (id={task.id}, 超时={task.timeout}s)")

        should_count = True  # 是否应该计入统计和触发回调

        try:
            # 检查是否已取消
            if self._cancel_events.get(task.id, threading.Event()).is_set():
                raise RuntimeError("任务已被取消")

            # 防御：handler为None或不可调用
            if handler is None or not callable(handler):
                raise TypeError(f"任务handler不可调用: {type(handler)}")

            # 执行任务
            result = handler(*args, **kwargs)

            # 再次检查取消状态
            if self._cancel_events.get(task.id, threading.Event()).is_set():
                raise RuntimeError("任务执行期间被取消")

            task.result = result
            task.status = TaskStatus.SUCCESS
            task.error = ""
            self._total_success += 1
            logger.info(f"任务成功: {task.name} (耗时={task.duration:.2f}s)")

        except Exception as e:
            task.error = str(e)
            task.retries += 1

            if task.retries < task.max_retries:
                task.status = TaskStatus.RETRY
                should_count = False  # RETRY不计数，等最终结果再计数
                logger.warning(f"任务失败，准备重试 ({task.retries}/{task.max_retries}): {task.name} - {e}")
                with self._lock:
                    self._queue.append(task)  # 重新入队
            else:
                task.status = TaskStatus.FAILED
                self._total_failed += 1
                logger.error(f"任务最终失败: {task.name} - {e}")

        finally:
            task.finished_at = time.time()
            task.duration = task.finished_at - task.started_at
            with self._lock:
                if task.id in self._running:
                    del self._running[task.id]

            # 只有非RETRY状态才计入统计和触发回调
            if should_count:
                self._total_executed += 1
                with self._lock:
                    self._completed.append(task)
                self._trigger_callback(task)

        return task

    def _trigger_callback(self, task: Task):
        """触发任务完成回调"""
        if self._on_task_complete:
            try:
                self._on_task_complete(task)
            except Exception as e:
                logger.error(f"任务完成回调异常: {e}")

    def cancel_task(self, task_id: str) -> bool:
        """取消指定任务"""
        with self._lock:
            # 检查是否在队列中
            for task in self._queue:
                if task.id == task_id:
                    if task_id in self._cancel_events:
                        self._cancel_events[task_id].set()
                    task.status = TaskStatus.CANCELLED
                    self._queue.remove(task)
                    logger.info(f"任务已取消: {task.name}")
                    return True

            # 检查是否正在运行
            if task_id in self._running:
                if task_id in self._cancel_events:
                    self._cancel_events[task_id].set()
                return True

        return False

    def on_complete(self, callback: Callable):
        """注册任务完成回调"""
        self._on_task_complete = callback

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def running_count(self) -> int:
        with self._lock:
            return len(self._running)

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_executed": self._total_executed,
                "total_success": self._total_success,
                "total_failed": self._total_failed,
                "pending": len(self._queue),
                "running": len(self._running),
                "success_rate": round(self._total_success / max(self._total_executed, 1) * 100, 1),
            }

    def get_recent_completed(self, limit: int = 10) -> List[Task]:
        """获取最近完成的任务"""
        with self._lock:
            return self._completed[-limit:]

    def clear_completed(self, keep: int = 100):
        """清理已完成任务记录（防止内存无限增长）"""
        with self._lock:
            if len(self._completed) > keep:
                self._completed = self._completed[-keep:]
                logger.debug(f"已清理完成任务记录，保留最近{keep}条")

    def shutdown(self, wait: bool = True, cancel_pending: bool = False):
        """关闭执行引擎"""
        with self._lock:
            self._shutdown = True

            if cancel_pending:
                for task in self._queue:
                    if task.id in self._cancel_events:
                        self._cancel_events[task.id].set()
                    task.status = TaskStatus.CANCELLED
                logger.info(f"已取消 {len(self._queue)} 个待处理任务")
                self._queue.clear()

        self._executor.shutdown(wait=wait)
        logger.info("执行引擎已关闭")
