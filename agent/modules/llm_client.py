"""
LLM客户端模块 - 统一大模型接口（增强版：重试/熔断/容错）
支持OpenAI兼容API、Ollama本地模型
"""
import json
import logging
import time
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class LLMClient:
    """统一LLM客户端（增强版）"""

    def __init__(self, config):
        self.config = config
        self._client = None
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5  # 熔断阈值
        self._circuit_open = False
        self._circuit_open_until = 0
        self._init_client()

    def _init_client(self):
        """初始化LLM客户端"""
        provider = self.config.provider

        if provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key or "not-needed",
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                )
                logger.info(f"LLM客户端初始化完成: OpenAI兼容 ({self.config.base_url})")
            except ImportError:
                logger.warning("openai库未安装，将使用模拟模式。安装命令: pip install openai")
                self._client = None
        elif provider == "ollama":
            try:
                import httpx
                self._http_client = httpx.Client(base_url=self.config.base_url, timeout=self.config.timeout)
                self._client = "ollama"
                logger.info(f"LLM客户端初始化完成: Ollama ({self.config.base_url})")
            except ImportError:
                logger.warning("httpx库未安装，将使用模拟模式。安装命令: pip install httpx")
                self._client = None
        else:
            logger.warning(f"未知LLM提供商: {provider}，将使用模拟模式")

    def _is_circuit_open(self) -> bool:
        """检查熔断器是否打开"""
        if not self._circuit_open:
            return False
        if time.time() > self._circuit_open_until:
            # 熔断恢复期，尝试半开
            self._circuit_open = False
            self._consecutive_failures = 0
            logger.info("熔断器半开，尝试恢复LLM调用")
            return False
        return True

    def _record_success(self):
        """记录成功，重置熔断"""
        self._consecutive_failures = 0
        self._circuit_open = False

    def _record_failure(self):
        """记录失败，检查是否需要熔断"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            self._circuit_open = True
            # 熔断30秒
            self._circuit_open_until = time.time() + 30
            logger.warning(
                f"LLM熔断器打开! 连续失败{self._consecutive_failures}次，"
                f"将在30秒后尝试恢复"
            )

    def chat(self, messages: List[Dict[str, str]],
             system_prompt: str = "", temperature: float = None,
             max_tokens: int = None, **kwargs) -> str:
        """
        发送对话请求

        Args:
            messages: 对话消息列表
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            模型回复文本
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        if self._client is None or self._is_circuit_open():
            return self._mock_response(full_messages)

        try:
            if self.config.provider == "openai" and self._client != "ollama":
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=full_messages,
                    temperature=temp,
                    max_tokens=tokens,
                    **kwargs
                )
                self._record_success()
                return response.choices[0].message.content

            elif self.config.provider == "ollama":
                result = self._ollama_chat(full_messages, temp, tokens)
                self._record_success()
                return result

        except Exception as e:
            self._record_failure()
            logger.error(f"LLM请求失败: {e}")
            return self._mock_response(full_messages)

    def _ollama_chat(self, messages, temperature, max_tokens) -> str:
        """Ollama API调用"""
        try:
            resp = self._http_client.post("/api/chat", json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            })
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama请求失败: {e}")
            return self._mock_response(messages)

    def _mock_response(self, messages: List[Dict]) -> str:
        """无LLM时的模拟回复"""
        last_msg = messages[-1]["content"] if messages else ""
        logger.debug(f"[模拟模式] 收到请求: {last_msg[:100]}...")

        # 基于关键词的模拟回复 - 确保返回结构符合各调用方期望
        if "优化" in last_msg or "改进" in last_msg or "分析" in last_msg:
            return json.dumps({
                "overall_assessment": "模拟评估：执行流程稳定",
                "success_patterns": ["快速执行", "自动重试有效"],
                "failure_patterns": ["偶发超时", "部分异常类型未覆盖"],
                "suggestions": ["增加缓存机制", "并行处理独立任务", "添加错误重试逻辑"],
                "new_strategies": [
                    {"name": "模拟策略", "description": "模拟生成的策略", "context": "通用"}
                ],
                "experience_to_store": [
                    {"title": "模拟经验", "content": "模拟生成的经验内容", "tags": "mock,test", "importance": 0.5}
                ],
                "priority_improvements": ["优化执行效率"]
            }, ensure_ascii=False)
        elif "经验" in last_msg or "知识" in last_msg:
            return json.dumps({
                "experiences": [
                    {"title": "任务执行经验", "content": "保持简洁，分步执行", "tags": "execution"},
                    {"title": "错误处理经验", "content": "捕获异常并记录，自动重试", "tags": "error-handling"}
                ],
                "summary": "基于历史执行记录的经验总结"
            }, ensure_ascii=False)
        elif "总结" in last_msg or "反思" in last_msg:
            return json.dumps({
                "overall_assessment": "模拟反思结果",
                "success_patterns": ["执行稳定"],
                "failure_patterns": ["部分任务耗时较长"],
                "suggestions": ["引入优先级队列"],
                "new_strategies": [],
                "experience_to_store": [],
                "priority_improvements": []
            }, ensure_ascii=False)
        elif "失败" in last_msg or "错误" in last_msg:
            return json.dumps({
                "root_cause": "模拟分析的根本原因",
                "fix_suggestion": "模拟的修复建议",
                "preventive_measures": ["预防措施1", "预防措施2"],
                "should_retry": False,
                "retry_strategy": "无需重试"
            }, ensure_ascii=False)
        elif "建议" in last_msg or "策略" in last_msg:
            return json.dumps({
                "advice": "模拟综合建议",
                "recommended_strategies": ["模拟策略1"],
                "warnings": ["模拟风险"],
                "tips": ["模拟技巧"],
                "has_advice": True
            }, ensure_ascii=False)
        else:
            return json.dumps({
                "action": "continue",
                "reason": "任务正常执行中",
                "notes": "模拟模式回复"
            }, ensure_ascii=False)

    def analyze_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """
        发送请求并解析JSON回复（增强容错）

        始终返回Dict，即使解析失败也返回安全的fallback结构
        """
        response = self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt
        )
        try:
            # 尝试提取JSON
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text.strip())

            # 确保返回的是字典
            if isinstance(result, dict):
                return result
            else:
                logger.warning(f"LLM返回了非字典类型: {type(result)}")
                return {"raw_response": response, "parsed": True, "value": result}

        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            logger.warning(f"JSON解析失败: {e}, 原始回复: {response[:200]}")
            return {"raw_response": response, "parsed": False, "error": str(e)}

    @property
    def is_healthy(self) -> bool:
        """检查LLM客户端健康状态"""
        return not self._is_circuit_open() and self._client is not None
