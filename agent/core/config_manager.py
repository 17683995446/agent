"""
配置管理器 - 支持热更新

功能：
- 配置文件监听（文件变更自动加载）
- 动态配置更新（运行时修改配置）
- 配置变更回调通知
- 配置验证与回滚
"""
import os
import json
import yaml
import time
import logging
import threading
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field, asdict

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

logger = logging.getLogger(__name__)


class ConfigChangeEvent:
    """配置变更事件"""
    def __init__(self, key: str, old_value: Any, new_value: Any, source: str):
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.source = source  # 'file' | 'api' | 'code'
        self.timestamp = time.time()


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_llm_config(config: Dict) -> tuple[bool, str]:
        """验证LLM配置"""
        if 'temperature' in config:
            t = config['temperature']
            if not (0 <= t <= 2):
                return False, f"temperature必须在0-2之间: {t}"
        if 'max_tokens' in config:
            m = config['max_tokens']
            if m < 1 or m > 32000:
                return False, f"max_tokens超出范围: {m}"
        return True, ""
    
    @staticmethod
    def validate_engine_config(config: Dict) -> tuple[bool, str]:
        """验证引擎配置"""
        if 'max_concurrent_tasks' in config:
            m = config['max_concurrent_tasks']
            if m < 1 or m > 100:
                return False, f"max_concurrent_tasks必须在1-100之间: {m}"
        if 'task_timeout' in config:
            t = config['task_timeout']
            if t < 1:
                return False, f"task_timeout必须大于0: {t}"
        return True, ""


class ConfigManager:
    """配置管理器 - 支持热更新"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = self._get_defaults()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._observer: Optional[Observer] = None
        self._watching = False
        self._validators = {
            'llm': ConfigValidator.validate_llm_config,
            'engine': ConfigValidator.validate_engine_config,
        }
        
        # 加载初始配置
        self.load()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'agent': {
                'task_queue_size': 100,
                'max_concurrent_tasks': 3,
                'task_retry_limit': 3,
                'task_timeout': 300,
                'optimize_interval': 5,
                'heartbeat_interval': 10,
                'restart_on_error': True,
            },
            'llm': {
                'provider': 'openai',
                'api_key': '',
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-4o-mini',
                'temperature': 0.7,
                'max_tokens': 2048,
                'timeout': 60,
            },
            'knowledge': {
                'db_path': 'data/knowledge.db',
                'auto_save_interval': 30,
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
            },
        }
    
    def load(self) -> bool:
        """从文件加载配置"""
        try:
            if not os.path.exists(self.config_path):
                logger.info(f"配置文件不存在，使用默认配置: {self.config_path}")
                self._config = self._deep_copy(self._defaults)
                self.save()  # 创建默认配置文件
                return True
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    loaded = yaml.safe_load(f) or {}
                else:
                    loaded = json.load(f)
            
            # 合并配置（加载的覆盖默认的）
            self._config = self._deep_merge(self._defaults, loaded)
            logger.info(f"配置加载成功: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            self._config = self._deep_copy(self._defaults)
            return False
    
    def save(self) -> bool:
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点号路径，如 'llm.temperature'）"""
        with self._lock:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
    
    def set(self, key: str, value: Any, validate: bool = True, 
            notify: bool = True, source: str = 'api') -> bool:
        """设置配置值（支持热更新）"""
        with self._lock:
            keys = key.split('.')
            config = self._config
            
            # 导航到父级
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            final_key = keys[-1]
            old_value = config.get(final_key)
            
            # 验证新值
            if validate:
                section = keys[0] if keys else ''
                if section in self._validators:
                    valid, msg = self._validators[section]({final_key: value})
                    if not valid:
                        logger.error(f"配置验证失败: {msg}")
                        return False
            
            # 设置新值
            config[final_key] = value
            
            # 保存到文件
            self.save()
            
            # 触发回调
            if notify and old_value != value:
                event = ConfigChangeEvent(key, old_value, value, source)
                self._notify_callbacks(key, event)
            
            logger.info(f"配置更新: {key} = {value}")
            return True
    
    def update(self, updates: Dict[str, Any], validate: bool = True) -> Dict[str, bool]:
        """批量更新配置"""
        results = {}
        for key, value in updates.items():
            results[key] = self.set(key, value, validate=validate)
        return results
    
    def register_callback(self, key_pattern: str, callback: Callable[[ConfigChangeEvent], None]):
        """注册配置变更回调（支持通配符，如 'llm.*'）"""
        with self._lock:
            if key_pattern not in self._callbacks:
                self._callbacks[key_pattern] = []
            self._callbacks[key_pattern].append(callback)
    
    def unregister_callback(self, key_pattern: str, callback: Callable):
        """注销配置变更回调"""
        with self._lock:
            if key_pattern in self._callbacks:
                if callback in self._callbacks[key_pattern]:
                    self._callbacks[key_pattern].remove(callback)
    
    def _notify_callbacks(self, changed_key: str, event: ConfigChangeEvent):
        """通知所有匹配的回调"""
        for pattern, callbacks in self._callbacks.items():
            if self._key_matches(changed_key, pattern):
                for callback in callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"配置回调执行失败: {e}")
    
    def _key_matches(self, key: str, pattern: str) -> bool:
        """检查key是否匹配pattern（支持*通配符）"""
        if pattern == '*':
            return True
        if pattern.endswith('.*'):
            prefix = pattern[:-2]
            return key.startswith(prefix + '.')
        return key == pattern
    
    def start_watching(self):
        """开始监听配置文件变化"""
        if self._watching:
            return
        
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog库未安装，配置文件监听功能不可用")
            return
        
        class ConfigFileHandler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager
                self.last_reload = 0
            
            def on_modified(self, event):
                if event.src_path == self.manager.config_path:
                    # 防抖：1秒内不重复加载
                    now = time.time()
                    if now - self.last_reload < 1:
                        return
                    self.last_reload = now
                    
                    logger.info("配置文件变更，重新加载...")
                    self.manager.reload_from_file()
        
        self._observer = Observer()
        handler = ConfigFileHandler(self)
        watch_dir = os.path.dirname(os.path.abspath(self.config_path)) or '.'
        self._observer.schedule(handler, watch_dir, recursive=False)
        self._observer.start()
        self._watching = True
        logger.info(f"开始监听配置文件: {self.config_path}")
    
    def stop_watching(self):
        """停止监听配置文件"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._watching = False
            logger.info("停止监听配置文件")
    
    def reload_from_file(self) -> bool:
        """从文件重新加载配置（热更新）"""
        with self._lock:
            old_config = self._deep_copy(self._config)
            
            if not self.load():
                return False
            
            # 找出变更的配置项
            changes = self._find_changes(old_config, self._config)
            
            # 触发变更通知
            for key, (old_val, new_val) in changes.items():
                event = ConfigChangeEvent(key, old_val, new_val, 'file')
                self._notify_callbacks(key, event)
            
            return True
    
    def _find_changes(self, old: Dict, new: Dict, prefix: str = '') -> Dict[str, tuple]:
        """递归找出配置变更"""
        changes = {}
        all_keys = set(old.keys()) | set(new.keys())
        
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)
            
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                changes.update(self._find_changes(old_val, new_val, full_key))
            elif old_val != new_val:
                changes[full_key] = (old_val, new_val)
        
        return changes
    
    def _deep_copy(self, obj: Any) -> Any:
        """深拷贝"""
        import copy
        return copy.deepcopy(obj)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并两个字典"""
        result = self._deep_copy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        with self._lock:
            return self._deep_copy(self._config)
    
    def reset_to_defaults(self):
        """重置为默认配置"""
        with self._lock:
            self._config = self._deep_copy(self._defaults)
            self.save()
            logger.info("配置已重置为默认值")
