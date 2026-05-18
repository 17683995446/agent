"""
示例插件：MetricsReporter - 任务执行指标报告插件

功能：
- 收集任务执行指标
- 定期输出统计报告
- 响应系统事件
"""
import time
import threading
from typing import Dict, Any
from agent.core.plugin_system import IPlugin, PluginInfo, PluginState


class MetricsReporter(IPlugin):
    """任务执行指标报告插件"""

    def __init__(self):
        self._info = PluginInfo(
            name="metrics_reporter",
            version="1.0.0",
            description="任务执行指标收集与报告",
            author="Agent System",
            dependencies=[],
        )
        self._context: Dict[str, Any] = {}
        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_duration": 0.0,
        }
        self._lock = threading.Lock()
        self._running = False
        self._report_thread: threading.Thread = None

    @property
    def info(self) -> PluginInfo:
        return self._info

    def initialize(self, context: Dict[str, Any]) -> bool:
        """初始化插件"""
        self._context = context
        # 注册事件钩子
        agent = context.get("agent")
        if agent:
            agent.engine.on_complete(self._on_task_complete)
        return True

    def start(self) -> bool:
        """启动报告线程"""
        self._running = True
        self._report_thread = threading.Thread(target=self._report_loop, daemon=True)
        self._report_thread.start()
        return True

    def stop(self) -> bool:
        """停止报告线程"""
        self._running = False
        if self._report_thread:
            self._report_thread.join(timeout=5)
        return True

    def destroy(self) -> bool:
        """销毁插件"""
        self._stats.clear()
        self._context.clear()
        return True

    def on_event(self, event_type: str, event_data: Any):
        """处理系统事件"""
        if event_type == "agent_stopped":
            self._print_final_report()

    def _on_task_complete(self, task):
        """任务完成回调"""
        with self._lock:
            self._stats["tasks_completed"] += 1
            if task.error:
                self._stats["tasks_failed"] += 1
            self._stats["total_duration"] += getattr(task, "duration", 0)

    def _report_loop(self):
        """定期报告循环"""
        while self._running:
            time.sleep(30)  # 每30秒报告一次
            self._print_report()

    def _print_report(self):
        """打印当前统计"""
        with self._lock:
            completed = self._stats["tasks_completed"]
            failed = self._stats["tasks_failed"]
            duration = self._stats["total_duration"]
            avg = duration / max(completed, 1)
            success_rate = (completed - failed) / max(completed, 1) * 100

        print(f"\n[MetricsReporter] 任务统计:")
        print(f"  已完成: {completed}, 失败: {failed}, 成功率: {success_rate:.1f}%")
        print(f"  总耗时: {duration:.1f}s, 平均耗时: {avg:.2f}s\n")

    def _print_final_report(self):
        """打印最终报告"""
        print("\n" + "=" * 50)
        print("[MetricsReporter] 最终报告")
        self._print_report()
        print("=" * 50 + "\n")
