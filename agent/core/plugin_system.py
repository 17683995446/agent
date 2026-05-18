"""
插件系统 - 动态加载与生命周期管理

支持：
- 动态加载/卸载插件
- 插件生命周期管理（init/start/stop/destroy）
- 插件间通信
- 热更新支持
"""
import os
import sys
import importlib
import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)


class PluginState:
    """插件状态"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"
    ERROR = "error"


@dataclass
class PluginInfo:
    """插件元信息"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    state: str = PluginState.UNLOADED
    loaded_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    error_message: str = ""


class IPlugin(ABC):
    """插件接口 - 所有插件必须实现"""

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """返回插件信息"""
        pass

    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """
        初始化插件
        Args:
            context: 包含agent、config、logger等共享资源
        Returns:
            初始化是否成功
        """
        pass

    @abstractmethod
    def start(self) -> bool:
        """启动插件"""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """停止插件"""
        pass

    @abstractmethod
    def destroy(self) -> bool:
        """销毁插件，释放资源"""
        pass

    def on_event(self, event_type: str, event_data: Any):
        """
        事件回调 - 插件可重写以响应系统事件
        Events: task_completed, optimization_done, agent_stopped, etc.
        """
        pass


class PluginManager:
    """插件管理器 - 负责插件的加载、生命周期管理和通信"""

    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or ["plugins"]
        self._plugins: Dict[str, IPlugin] = {}
        self._infos: Dict[str, PluginInfo] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._context: Dict[str, Any] = {}
        self._running = False

        # 确保插件目录存在
        for d in self.plugin_dirs:
            os.makedirs(d, exist_ok=True)

    def set_context(self, context: Dict[str, Any]):
        """设置插件共享上下文"""
        self._context = context

    def discover_plugins(self) -> List[str]:
        """发现所有可用的插件"""
        discovered = []
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
            for item in os.listdir(plugin_dir):
                full_path = os.path.join(plugin_dir, item)
                if os.path.isdir(full_path):
                    # 检查是否有__init__.py或plugin.py
                    if os.path.exists(os.path.join(full_path, "__init__.py")):
                        discovered.append(item)
                    elif os.path.exists(os.path.join(full_path, "plugin.py")):
                        discovered.append(item)
                elif item.endswith(".py") and not item.startswith("_"):
                    discovered.append(item[:-3])
        return discovered

    def load_plugin(self, name: str) -> bool:
        """加载指定插件"""
        with self._lock:
            if name in self._plugins:
                logger.warning(f"插件 {name} 已加载")
                return True

            try:
                info = PluginInfo(name=name, version="", description="", author="")
                info.state = PluginState.LOADING
                self._infos[name] = info

                # 查找插件模块
                plugin_module = self._find_plugin_module(name)
                if not plugin_module:
                    raise ImportError(f"找不到插件 {name}")

                # 查找插件类
                plugin_class = self._find_plugin_class(plugin_module)
                if not plugin_class:
                    raise TypeError(f"插件 {name} 中没有找到IPlugin实现类")

                # 实例化
                plugin = plugin_class()
                info = plugin.info
                info.state = PluginState.LOADED
                info.loaded_at = datetime.now()

                self._plugins[name] = plugin
                self._infos[name] = info

                logger.info(f"插件 {name} v{info.version} 加载成功")
                return True

            except Exception as e:
                logger.error(f"加载插件 {name} 失败: {e}")
                if name in self._infos:
                    self._infos[name].state = PluginState.ERROR
                    self._infos[name].error_message = str(e)
                return False

    def _find_plugin_module(self, name: str):
        """查找并导入插件模块"""
        for plugin_dir in self.plugin_dirs:
            # 尝试作为包导入
            init_path = os.path.join(plugin_dir, name, "__init__.py")
            if os.path.exists(init_path):
                spec = importlib.util.spec_from_file_location(
                    f"{plugin_dir}.{name}", init_path
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module

            # 尝试单文件插件
            py_path = os.path.join(plugin_dir, f"{name}.py")
            if os.path.exists(py_path):
                spec = importlib.util.spec_from_file_location(name, py_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                return module

        return None

    def _find_plugin_class(self, module) -> Optional[Type[IPlugin]]:
        """在模块中查找IPlugin实现类"""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, IPlugin) and obj != IPlugin:
                return obj
        return None

    def initialize_plugin(self, name: str) -> bool:
        """初始化指定插件"""
        with self._lock:
            if name not in self._plugins:
                logger.error(f"插件 {name} 未加载")
                return False

            plugin = self._plugins[name]
            info = self._infos[name]

            try:
                info.state = PluginState.INITIALIZING
                success = plugin.initialize(self._context)
                if success:
                    info.state = PluginState.INITIALIZED
                    logger.info(f"插件 {name} 初始化成功")
                    return True
                else:
                    info.state = PluginState.ERROR
                    info.error_message = "初始化返回False"
                    return False
            except Exception as e:
                info.state = PluginState.ERROR
                info.error_message = str(e)
                logger.error(f"初始化插件 {name} 失败: {e}")
                return False

    def start_plugin(self, name: str) -> bool:
        """启动指定插件"""
        with self._lock:
            if name not in self._plugins:
                return False

            plugin = self._plugins[name]
            info = self._infos[name]

            try:
                info.state = PluginState.STARTING
                success = plugin.start()
                if success:
                    info.state = PluginState.RUNNING
                    info.started_at = datetime.now()
                    logger.info(f"插件 {name} 启动成功")
                    return True
                else:
                    info.state = PluginState.ERROR
                    return False
            except Exception as e:
                info.state = PluginState.ERROR
                info.error_message = str(e)
                logger.error(f"启动插件 {name} 失败: {e}")
                return False

    def stop_plugin(self, name: str) -> bool:
        """停止指定插件"""
        with self._lock:
            if name not in self._plugins:
                return False

            plugin = self._plugins[name]
            info = self._infos[name]

            try:
                info.state = PluginState.STOPPING
                success = plugin.stop()
                info.state = PluginState.STOPPED
                logger.info(f"插件 {name} 已停止")
                return success
            except Exception as e:
                info.state = PluginState.ERROR
                info.error_message = str(e)
                logger.error(f"停止插件 {name} 失败: {e}")
                return False

    def unload_plugin(self, name: str) -> bool:
        """卸载指定插件"""
        with self._lock:
            if name not in self._plugins:
                return True

            # 先停止
            self.stop_plugin(name)

            plugin = self._plugins[name]
            info = self._infos[name]

            try:
                info.state = PluginState.DESTROYING
                plugin.destroy()
                info.state = PluginState.DESTROYED

                del self._plugins[name]
                del self._infos[name]

                logger.info(f"插件 {name} 已卸载")
                return True
            except Exception as e:
                info.state = PluginState.ERROR
                info.error_message = str(e)
                logger.error(f"卸载插件 {name} 失败: {e}")
                return False

    def load_and_start_all(self) -> Dict[str, bool]:
        """加载并启动所有发现的插件"""
        results = {}
        discovered = self.discover_plugins()

        for name in discovered:
            # 加载
            if not self.load_plugin(name):
                results[name] = False
                continue

            # 初始化
            if not self.initialize_plugin(name):
                results[name] = False
                continue

            # 启动
            results[name] = self.start_plugin(name)

        return results

    def stop_all(self):
        """停止所有运行的插件"""
        with self._lock:
            for name in list(self._plugins.keys()):
                self.stop_plugin(name)

    def unload_all(self):
        """卸载所有插件"""
        with self._lock:
            for name in list(self._plugins.keys()):
                self.unload_plugin(name)

    def emit_event(self, event_type: str, event_data: Any):
        """向所有插件广播事件"""
        with self._lock:
            for name, plugin in self._plugins.items():
                if self._infos[name].state == PluginState.RUNNING:
                    try:
                        plugin.on_event(event_type, event_data)
                    except Exception as e:
                        logger.error(f"插件 {name} 处理事件 {event_type} 失败: {e}")

        # 触发注册的hooks
        if event_type in self._hooks:
            for hook in self._hooks[event_type]:
                try:
                    hook(event_data)
                except Exception as e:
                    logger.error(f"Hook处理事件 {event_type} 失败: {e}")

    def register_hook(self, event_type: str, callback: Callable):
        """注册事件钩子"""
        with self._lock:
            if event_type not in self._hooks:
                self._hooks[event_type] = []
            self._hooks[event_type].append(callback)

    def unregister_hook(self, event_type: str, callback: Callable):
        """注销事件钩子"""
        with self._lock:
            if event_type in self._hooks:
                if callback in self._hooks[event_type]:
                    self._hooks[event_type].remove(callback)

    def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        return self._infos.get(name)

    def list_plugins(self) -> List[PluginInfo]:
        """列出所有插件信息"""
        return list(self._infos.values())

    def get_running_plugins(self) -> List[str]:
        """获取正在运行的插件名称列表"""
        with self._lock:
            return [
                name for name, info in self._infos.items()
                if info.state == PluginState.RUNNING
            ]
