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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError, wait, FIRST_COMPLETED, ALL_COMPLETED

logger = logging.getLogger(__name__)


from ..core.interfaces import ITaskEngine, Task, TaskStatus


class TaskEngine(ITaskEngine):
    """任务执行引擎（增强版）"""

    def __init__(self, max_workers: int = 3, retry_limit: int = 3, timeout: int = 300, max_queue_size: int = 1000):
        logger.debug(f"[__init__] 初始化TaskEngine | max_workers={max_workers}, retry_limit={retry_limit}, timeout={timeout}, max_queue_size={max_queue_size}")
        self.max_workers = max_workers
        self.retry_limit = retry_limit
        self.default_timeout = timeout
        # R7修复：任务队列最大容量限制
        self.max_queue_size = max_queue_size if max_queue_size > 0 else 1000
        self._queue: List[Task] = []
        self._running: Dict[str, Task] = {}
        self._completed: List[Task] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="TaskWorker")
        # C5修复：添加紧急执行器，用于主线程池被阻塞时的恢复
        self._emergency_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="EmergencyWorker")
        self._total_executed = 0
        self._total_success = 0
        self._total_failed = 0
        self._on_task_complete: Optional[Callable] = None
        self._task_handlers: Dict[str, tuple] = {}
        self._lock = threading.RLock()
        self._shutdown = False
        self._cancel_events: Dict[str, threading.Event] = {}
        # C5修复：跟踪主线程池中提交的任务数（即使超时也计数，直到线程真正完成）
        self._main_pool_submitted = 0  # 提交到主线程池的任务总数
        self._main_pool_completed = 0  # 主线程池实际完成的任务数
        logger.debug(f"[__init__] TaskEngine初始化完成")

    def submit(self, name: str, handler: Callable, args: tuple = (), kwargs: Dict = None,
               task_type: str = "general", priority: int = 5, timeout: int = None,
               max_retries: int = None, metadata: Dict = None) -> Task:
        """提交新任务"""
        logger.debug(f"[submit] 提交任务 | name={name}, task_type={task_type}, priority={priority}, timeout={timeout}, max_retries={max_retries}")
        
        # S2.1修复：输入验证
        if not isinstance(name, str) or not name.strip():
            name = f"unnamed_{int(time.time())}"
        if len(name) > 200:
            name = name[:200]
        if not callable(handler):
            raise TypeError("handler 必须是可调用对象")
        
        use_default_timeout = timeout is None or timeout <= 0
        use_default_retries = max_retries is None
        with self._lock:
            if self._shutdown:
                logger.debug(f"[submit] 引擎已关闭，拒绝提交")
                raise RuntimeError("执行引擎已关闭，无法提交新任务")

            # R7修复：队列大小限制检查
            if len(self._queue) >= self.max_queue_size:
                logger.warning(f"任务队列已满 ({len(self._queue)}/{self.max_queue_size})，拒绝提交: {name}")
                raise RuntimeError(f"任务队列已满 ({self.max_queue_size})，无法提交新任务")

            task = Task(
                id=str(uuid.uuid4().hex[:12]),
                name=name,
                task_type=task_type,
                priority=priority,
                status=TaskStatus.PENDING,
                created_at=time.time(),
                max_retries=max_retries if max_retries is not None else self.retry_limit,
                timeout=timeout if timeout is not None and timeout > 0 else self.default_timeout,
                metadata=metadata or {},
            )
            # 存储handler和args/kwargs在内部字典中
            self._task_handlers[task.id] = (handler, args, kwargs or {})
            self._cancel_events[task.id] = threading.Event()
            self._queue.append(task)
            logger.info(f"任务已提交: [{task.task_type}] {task.name} (id={task.id}, 优先级={task.priority}, 超时={task.timeout}s)")
            logger.debug(f"[submit] 任务入队完成 | id={task.id}, queue_size={len(self._queue)}, use_default_timeout={use_default_timeout}, use_default_retries={use_default_retries}")
            return task

    def submit_many(self, tasks: List[Dict]) -> List[Task]:
        """批量提交任务"""
        logger.debug(f"[submit_many] 批量提交 | count={len(tasks)}")
        # S4.2修复：限制批量大小
        if len(tasks) > 1000:
            raise ValueError(f"批量提交上限为1000个任务，当前{len(tasks)}个")
        result = [self.submit(**t) for t in tasks]
        logger.debug(f"[submit_many] 批量提交完成 | submitted={len(result)}")
        return result

    def run_next(self) -> Optional[Task]:
        """执行下一个任务（优先级排序）"""
        with self._lock:
            if not self._queue:
                logger.debug(f"[run_next] 队列为空，无任务可执行")
                return None
            self._queue.sort(key=lambda t: t.priority)
            task = self._queue.pop(0)
        logger.debug(f"[run_next] 取出任务 | id={task.id}, name={task.name}, queue_remaining={len(self._queue)}")
        return self._execute(task)

    def run_all_pending(self) -> List[Task]:
        """执行所有待处理任务"""
        logger.debug(f"[run_all_pending] 开始执行所有待处理任务")
        results = []
        while True:
            with self._lock:
                if not self._queue:
                    break
            task = self.run_next()
            if task:
                results.append(task)
        logger.debug(f"[run_all_pending] 执行完成 | count={len(results)}")
        return results

    def run_batch(self, count: int = 3) -> List[Task]:
        """并行执行一批任务（真正超时控制 + 线程池阻塞恢复）"""
        logger.debug(f"[run_batch] 开始批量执行 | requested={count}")
        with self._lock:
            if not self._queue:
                logger.debug(f"[run_batch] 队列为空，返回空列表")
                return []
            self._queue.sort(key=lambda t: t.priority)
            batch_size = min(count, len(self._queue))
            batch = [self._queue.pop(0) for _ in range(batch_size)]
        logger.debug(f"[run_batch] 批次准备完成 | batch_size={batch_size}, queue_remaining={len(self._queue)}")

        # C5修复：检测主线程池是否被阻塞
        # 如果提交数 - 完成数 >= max_workers，说明所有worker都被阻塞
        use_emergency = False
        with self._lock:
            blocked_in_pool = self._main_pool_submitted - self._main_pool_completed
            if blocked_in_pool >= self.max_workers:
                use_emergency = True
                logger.debug(f"[run_batch] 检测到主线程池被阻塞 | submitted={self._main_pool_submitted}, completed={self._main_pool_completed}, blocked={blocked_in_pool}, max_workers={self.max_workers}, 使用紧急执行器")

        futures = {}
        executor = self._emergency_executor if use_emergency else self._executor
        for task in batch:
            future = executor.submit(self._execute_wrapper, task)
            futures[future] = task
            # C5修复：跟踪主线程池提交数
            if not use_emergency:
                with self._lock:
                    self._main_pool_submitted += 1
        logger.debug(f"[run_batch] 已提交到线程池 | futures={len(futures)}, executor={'emergency' if use_emergency else 'main'}")

        results = []
        # D6修复：使用wait代替as_completed，实现真正的超时控制
        start_time = time.time()
        pending = set(futures.keys())
        
        while pending:
            # 计算最小超时时间
            min_timeout = float('inf')
            for f in pending:
                task = futures[f]
                elapsed = time.time() - start_time
                remaining = task.timeout - elapsed
                min_timeout = min(min_timeout, remaining)
            
            if min_timeout <= 0:
                # 所有剩余任务都已超时
                logger.debug(f"[run_batch] 所有剩余任务超时 | pending={len(pending)}")
                for f in list(pending):
                    task = futures[f]
                    self._handle_timeout(task)
                    results.append(task)
                break
            
            # 等待任务完成，带超时
            try:
                done, pending = wait(pending, timeout=min_timeout, return_when=FIRST_COMPLETED)
                logger.debug(f"[run_batch] wait返回 | done={len(done)}, pending={len(pending)}")
            except Exception as e:
                logger.error(f"等待任务异常: {e}")
                logger.debug(f"[run_batch] wait异常 | error={e}")
                break
            
            # 处理完成的任务
            for f in done:
                task = futures[f]
                try:
                    result = f.result(timeout=0)
                    results.append(result)
                    logger.debug(f"[run_batch] 任务完成 | id={task.id}, status={task.status}")
                except Exception as e:
                    with self._lock:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.finished_at = time.time()
                        task.duration = max(0, task.finished_at - task.started_at)
                        self._total_failed += 1
                        self._total_executed += 1
                        if task.id in self._running:
                            del self._running[task.id]
                        self._completed.append(task)
                        self._task_handlers.pop(task.id, None)
                        self._cancel_events.pop(task.id, None)
                    self._trigger_callback(task)
                    results.append(task)
                    logger.debug(f"[run_batch] 任务future异常 | id={task.id}, error={e}")
            
            # 检查超时
            current_time = time.time()
            for f in list(pending):
                task = futures[f]
                if current_time - start_time >= task.timeout:
                    self._handle_timeout(task)
                    results.append(task)
                    pending.discard(f)
                    logger.debug(f"[run_batch] 任务超时 | id={task.id}, timeout={task.timeout}s")

        logger.debug(f"[run_batch] 批量执行完成 | results={len(results)}")
        return results

    def _handle_timeout(self, task: Task):
        """处理任务超时"""
        logger.debug(f"[_handle_timeout] 处理超时 | id={task.id}, name={task.name}, current_status={task.status}")
        with self._lock:
            if task.status == TaskStatus.TIMEOUT:
                logger.debug(f"[_handle_timeout] 已处理过，跳过 | id={task.id}")
                return  # 已经处理过
            task.status = TaskStatus.TIMEOUT
            task.error = f"任务执行超时 ({task.timeout}s)"
            task.finished_at = time.time()
            task.duration = max(0, task.finished_at - task.started_at)
            self._total_failed += 1
            self._total_executed += 1
            if task.id in self._running:
                del self._running[task.id]
            self._completed.append(task)
            self._task_handlers.pop(task.id, None)
            self._cancel_events.pop(task.id, None)
        self._trigger_callback(task)
        logger.warning(f"任务超时: {task.name}")
        logger.debug(f"[_handle_timeout] 超时处理完成 | id={task.id}")

    def _execute_wrapper(self, task: Task) -> Task:
        """执行包装器（用于线程池）- 捕获所有异常包括BaseException"""
        logger.debug(f"[_execute_wrapper] 开始执行包装 | id={task.id}, name={task.name}")
        # F4修复：确保started_at已设置
        if not task.started_at:
            task.started_at = time.time()
        try:
            result = self._execute(task)
            logger.debug(f"[_execute_wrapper] 执行完成 | id={task.id}, status={task.status}")
            return result
        except BaseException as e:
            # C7修复：如果已被超时处理，不再重复计数
            with self._lock:
                if task.status == TaskStatus.TIMEOUT:
                    logger.debug(f"[_execute_wrapper] 已超时，跳过异常处理 | id={task.id}")
                    return task
            # 捕获KeyboardInterrupt/SystemExit等，防止worker线程崩溃传播到主线程
            task.status = TaskStatus.FAILED
            task.error = f"[{type(e).__name__}] {str(e)}"
            task.finished_at = time.time()
            task.duration = max(0, task.finished_at - task.started_at)
            with self._lock:
                self._total_failed += 1
                self._total_executed += 1
                if task.id in self._running:
                    del self._running[task.id]
                self._completed.append(task)
                # F1修复：清理
                self._task_handlers.pop(task.id, None)
                self._cancel_events.pop(task.id, None)
            self._trigger_callback(task)
            logger.error(f"任务捕获极端异常: {task.name} - {type(e).__name__}: {e}")
            logger.debug(f"[_execute_wrapper] 极端异常处理完成 | id={task.id}, exception_type={type(e).__name__}")
            return task
        finally:
            # C5修复：无论成功/失败/超时，线程最终都会完成，更新计数
            with self._lock:
                self._main_pool_completed += 1
                logger.debug(f"[_execute_wrapper] 线程完成 | id={task.id}, main_pool_completed={self._main_pool_completed}")

    def _execute(self, task: Task) -> Task:
        """执行单个任务"""
        logger.debug(f"[_execute] 开始执行 | id={task.id}, name={task.name}")
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        with self._lock:
            self._running[task.id] = task
            # 获取handler信息
            handler_info = self._task_handlers.get(task.id, (None, (), {}))
            handler, args, kwargs = handler_info

        logger.info(f"开始执行任务: [{task.task_type}] {task.name} (id={task.id}, 超时={task.timeout}s)")
        logger.debug(f"[_execute] handler信息 | handler_type={type(handler).__name__ if handler else 'None'}, args={args}, kwargs_keys={list(kwargs.keys()) if kwargs else []}")

        should_count = True  # 是否应该计入统计和触发回调

        try:
            # 检查是否已取消
            if self._cancel_events.get(task.id, threading.Event()).is_set():
                logger.debug(f"[_execute] 任务已取消（执行前检查） | id={task.id}")
                raise RuntimeError("任务已被取消")

            # 防御：handler为None或不可调用
            if handler is None or not callable(handler):
                logger.debug(f"[_execute] handler不可调用 | handler_type={type(handler).__name__ if handler else 'None'}")
                raise TypeError(f"任务handler不可调用: {type(handler)}")

            # 执行任务
            result = handler(*args, **kwargs)
            logger.debug(f"[_execute] handler执行完成 | id={task.id}")

            # M4修复：即使已超时，也保存结果（任务实际已完成）
            with self._lock:
                if task.status == TaskStatus.TIMEOUT:
                    logger.warning(f"任务已完成但已被标记超时，保存结果: {task.name}")
                    logger.debug(f"[_execute] 任务已完成但已超时 | id={task.id}")
                    task.result = result  # 保存实际结果
                    return task

            # 再次检查取消状态
            if self._cancel_events.get(task.id, threading.Event()).is_set():
                logger.debug(f"[_execute] 任务执行期间被取消 | id={task.id}")
                raise RuntimeError("任务执行期间被取消")

            # D6修复：使用锁更新状态，防止竞态条件
            with self._lock:
                # 再次检查超时状态
                if task.status == TaskStatus.TIMEOUT:
                    logger.debug(f"[_execute] 锁内再次检查发现已超时 | id={task.id}")
                    return task
                task.result = result
                task.status = TaskStatus.SUCCESS
                task.error = ""
                self._total_success += 1
            # 在锁外计算duration用于日志
            duration = time.time() - task.started_at
            logger.info(f"任务成功: {task.name} (耗时={duration:.2f}s)")
            logger.debug(f"[_execute] 任务成功 | id={task.id}, duration={duration:.2f}s")

        except Exception as e:
            # D6修复：使用锁检查是否已被标记为超时
            with self._lock:
                if task.status == TaskStatus.TIMEOUT:
                    logger.debug(f"[_execute] 异常处理中发现已超时 | id={task.id}")
                    return task

            task.error = str(e)
            task.retries += 1
            logger.debug(f"[_execute] 任务异常 | id={task.id}, error={e}, retries={task.retries}/{task.max_retries}")

            # D1修复：不可恢复的错误直接失败，不重试
            # handler为None或不可调用 → TypeError → 不可恢复
            # handler正常但执行时TypeError（如参数错误）→ 可重试
            is_unrecoverable = isinstance(e, TypeError) and (handler is None or not callable(handler))

            if is_unrecoverable:
                # 不可恢复的错误，直接失败
                logger.debug(f"[_execute] 不可恢复错误 | id={task.id}, error={e}")
                with self._lock:
                    if task.status == TaskStatus.TIMEOUT:
                        return task
                    task.status = TaskStatus.FAILED
                    task.retries = task.max_retries  # 标记为已用完重试
                    self._total_failed += 1
                logger.error(f"任务不可恢复错误，直接失败: {task.name} - {e}")
            elif task.retries < task.max_retries:
                logger.debug(f"[_execute] 准备重试 | id={task.id}, retries={task.retries}/{task.max_retries}")
                task.status = TaskStatus.RETRY
                should_count = False  # RETRY不计数，等最终结果再计数
                logger.warning(f"任务失败，准备重试 ({task.retries}/{task.max_retries}): {task.name} - {e}")
                with self._lock:
                    self._queue.append(task)  # 重新入队
            else:
                logger.debug(f"[_execute] 重试次数用尽，最终失败 | id={task.id}, retries={task.retries}/{task.max_retries}")
                with self._lock:
                    if task.status == TaskStatus.TIMEOUT:
                        return task
                    task.status = TaskStatus.FAILED
                    self._total_failed += 1
                logger.error(f"任务最终失败: {task.name} - {e}")

        finally:
            task.finished_at = time.time()
            task.duration = task.finished_at - task.started_at
            with self._lock:
                if task.id in self._running:
                    del self._running[task.id]

                # C7修复：如果已被超时处理，不再重复计数
                if task.status == TaskStatus.TIMEOUT:
                    should_count = False
                    logger.debug(f"[_execute] finally中发现已超时，不计数 | id={task.id}")

                # 只有非RETRY状态才计入统计和触发回调
                if should_count:
                    self._total_executed += 1
                    self._completed.append(task)
                    # F1修复：清理已完成任务handler和cancel_event
                    self._task_handlers.pop(task.id, None)
                    self._cancel_events.pop(task.id, None)
                    logger.debug(f"[_execute] 计入统计 | id={task.id}, status={task.status}, total_executed={self._total_executed}")
                else:
                    logger.debug(f"[_execute] 不计入统计 | id={task.id}, status={task.status}, should_count={should_count}")

            if should_count:
                self._trigger_callback(task)

        return task

    def _trigger_callback(self, task: Task):
        """触发任务完成回调"""
        logger.debug(f"[_trigger_callback] 触发回调 | id={task.id}, status={task.status}")
        with self._lock:
            callback = self._on_task_complete
        if callback:
            try:
                callback(task)
                logger.debug(f"[_trigger_callback] 回调完成 | id={task.id}")
            except Exception as e:
                logger.error(f"任务完成回调异常: {e}")
                logger.debug(f"[_trigger_callback] 回调异常 | id={task.id}, error={e}")
        else:
            logger.debug(f"[_trigger_callback] 无回调注册 | id={task.id}")

    def cancel_task(self, task_id: str) -> bool:
        """取消指定任务"""
        logger.debug(f"[cancel_task] 取消请求 | task_id={task_id}")
        with self._lock:
            # 检查是否在队列中
            for task in self._queue:
                if task.id == task_id:
                    if task_id in self._cancel_events:
                        self._cancel_events[task_id].set()
                    task.status = TaskStatus.CANCELLED
                    self._queue.remove(task)
                    # E2修复：清理取消任务的资源
                    self._task_handlers.pop(task_id, None)
                    self._cancel_events.pop(task_id, None)
                    logger.info(f"任务已取消: {task.name}")
                    logger.debug(f"[cancel_task] 从队列中取消 | task_id={task_id}, name={task.name}")
                    return True

            # 检查是否正在运行
            if task_id in self._running:
                if task_id in self._cancel_events:
                    self._cancel_events[task_id].set()
                # 注意：运行中的任务不立即清理资源，等执行完成后清理
                logger.debug(f"[cancel_task] 运行中任务设置取消标志 | task_id={task_id}")
                return True

        logger.debug(f"[cancel_task] 任务不存在 | task_id={task_id}")
        return False

    def on_complete(self, callback: Callable):
        """注册任务完成回调"""
        logger.debug(f"[on_complete] 注册回调 | callback={callback}")
        with self._lock:
            self._on_task_complete = callback
        logger.debug(f"[on_complete] 回调注册完成")

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
                logger.debug(f"[clear_completed] 清理 | current={len(self._completed)}, keep={keep}")
                self._completed = self._completed[-keep:]
                logger.debug(f"已清理完成任务记录，保留最近{keep}条")
            else:
                logger.debug(f"[clear_completed] 无需清理 | current={len(self._completed)}, keep={keep}")

    def shutdown(self, wait: bool = True, cancel_pending: bool = False):
        """关闭执行引擎"""
        logger.debug(f"[shutdown] 关闭引擎 | wait={wait}, cancel_pending={cancel_pending}")
        with self._lock:
            self._shutdown = True

            if cancel_pending:
                # E2修复：取消待处理任务时清理所有相关资源
                cancelled_ids = []
                for task in self._queue:
                    if task.id in self._cancel_events:
                        self._cancel_events[task.id].set()
                    task.status = TaskStatus.CANCELLED
                    cancelled_ids.append(task.id)
                logger.info(f"已取消 {len(self._queue)} 个待处理任务")
                logger.debug(f"[shutdown] 取消待处理任务 | count={len(self._queue)}")
                self._queue.clear()

                # E2修复：清理取消任务的handler和cancel_event
                for task_id in cancelled_ids:
                    self._task_handlers.pop(task_id, None)
                    self._cancel_events.pop(task_id, None)
            else:
                logger.debug(f"[shutdown] 保留待处理任务 | pending={len(self._queue)}")

        self._executor.shutdown(wait=wait)
        # C5修复：关闭紧急执行器
        self._emergency_executor.shutdown(wait=wait)
        logger.info("执行引擎已关闭")
        logger.debug(f"[shutdown] 引擎关闭完成")
