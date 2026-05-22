"""
优化模块 - 自我反思与持续改进（增强版：防御性编程、知识清理）
负责：执行结果分析、策略优化、经验提取、知识图谱维护
"""
import json
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _safe_serialize(value: Any, max_length: int = 500) -> str:
    """安全序列化任意类型的值为字符串"""
    if value is None:
        return ""
    try:
        s = str(value)
        if len(s) > max_length:
            return s[:max_length] + f"...(截断,总长{len(s)})"
        return s
    except Exception:
        return f"<无法序列化: {type(value).__name__}>"


def _safe_list(value: Any) -> list:
    """确保值是列表"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class Optimizer:
    """优化器 - 边做边改，越做越好（增强版）"""

    def __init__(self, knowledge_graph, llm_client):
        self.kg = knowledge_graph
        self.llm = llm_client
        self._optimization_count = 0
        self._last_optimization_time = 0
        self._lock = __import__('threading').Lock()  # 线程安全
        logger.debug(f"[__init__] Optimizer初始化完成")

    def reflect_on_tasks(self, tasks: List) -> Dict[str, Any]:
        """对一批已完成任务进行反思分析"""
        logger.debug(f"[reflect_on_tasks] 开始反思 | tasks_count={len(tasks)}")
        if not tasks:
            logger.debug(f"[reflect_on_tasks] 无任务，跳过反思")
            return {"action": "skip", "reason": "没有任务需要反思"}

        with self._lock:
            return self._reflect_internal(tasks)

    def _reflect_internal(self, tasks: List) -> Dict[str, Any]:
        """反思分析内部实现"""
        logger.debug(f"[_reflect_internal] 开始内部反思 | tasks={len(tasks)}")
        # 准备任务摘要
        task_summaries = []
        for t in tasks:
            # E1修复：兼容Task对象和字典两种输入
            if isinstance(t, dict):
                # 字典输入
                status_val = t.get('status', 'unknown')
                status_str = status_val.value if hasattr(status_val, 'value') else str(status_val)
                task_summaries.append({
                    "name": t.get('name', 'unknown'),
                    "type": t.get('type', t.get('task_type', 'unknown')),
                    "status": status_str,
                    "duration": round(t.get('duration', 0), 2),
                    "error": str(t.get('error', ''))[:200] if t.get('error') else '',
                    "retries": t.get('retries', 0),
                })
            else:
                # Task对象输入
                status_val = getattr(t, 'status', 'unknown')
                status_str = status_val.value if hasattr(status_val, 'value') else str(status_val)
                task_summaries.append({
                    "name": getattr(t, 'name', 'unknown'),
                    "type": getattr(t, 'task_type', 'unknown'),
                    "status": status_str,
                    "duration": round(getattr(t, 'duration', 0), 2),
                    "error": str(getattr(t, 'error', '') or '')[:200],
                    "retries": getattr(t, 'retries', 0),
                })
        logger.debug(f"[_reflect_internal] 任务摘要准备完成 | summaries={len(task_summaries)}")

        # 查询相关经验
        task_types = list(set(ts.get("type", "") for ts in task_summaries))
        related_exp = []
        for tt in task_types:
            if not tt:
                continue
            try:
                nodes = self.kg.query_experience(tt, limit=3)
                for n in nodes:
                    related_exp.append({"title": n.title, "content": n.content[:200]})
            except Exception as e:
                logger.error(f"查询经验失败: {e}")
                logger.debug(f"[_reflect_internal] 查询经验失败 | type={tt}, error={e}")
        logger.debug(f"[_reflect_internal] 相关经验获取完成 | experiences={len(related_exp)}")

        # 调用LLM进行反思
        prompt = f"""请分析以下任务执行结果，给出优化建议。

## 任务执行记录
{json.dumps(task_summaries, ensure_ascii=False, indent=2)}

## 相关历史经验
{json.dumps(related_exp, ensure_ascii=False, indent=2)}

## 请以JSON格式返回：
{{
    "overall_assessment": "整体评估（一句话）",
    "success_patterns": ["成功的模式1", "成功的模式2"],
    "failure_patterns": ["失败的模式1", "失败的模式2"],
    "suggestions": ["具体改进建议1", "具体改进建议2", "具体改进建议3"],
    "new_strategies": [
        {{"name": "策略名称", "description": "策略描述", "context": "适用场景"}}
    ],
    "experience_to_store": [
        {{"title": "经验标题", "content": "经验内容", "tags": "标签1,标签2", "importance": 0.7}}
    ],
    "priority_improvements": ["最优先改进项1", "最优先改进项2"]
}}"""

        system_prompt = "你是一个AI Agent的自我反思优化模块。你的任务是分析任务执行结果，提取经验教训，提出改进策略。始终返回有效的JSON格式。"

        logger.debug(f"[_reflect_internal] 调用LLM分析")
        result = self.llm.analyze_json(prompt, system_prompt)
        logger.debug(f"[_reflect_internal] LLM分析完成 | result_type={type(result).__name__}")

        # 防御性：确保result是字典
        if not isinstance(result, dict):
            logger.debug(f"[_reflect_internal] LLM返回非字典，使用fallback")
            result = {"raw_response": str(result), "parsed": False}

        # 存储新经验到知识图谱（防御性处理）
        experiences = _safe_list(result.get("experience_to_store"))
        logger.debug(f"[_reflect_internal] 待存储经验 | count={len(experiences)}")
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
            try:
                self.kg.store_experience(
                    title=str(exp.get("title", "未命名经验"))[:200],
                    content=_safe_serialize(exp.get("content", ""), 2000),
                    category="experience",
                    tags=str(exp.get("tags", ""))[:200],
                    importance=max(0.0, min(1.0, float(exp.get("importance", 0.5)))),
                )
                logger.info(f"存储新经验: {exp.get('title', 'N/A')}")
            except Exception as e:
                logger.error(f"存储经验失败: {e}")
                logger.debug(f"[_reflect_internal] 存储经验失败 | title={exp.get('title', 'N/A')}, error={e}")

        # 存储新策略（防御性处理）
        strategies = _safe_list(result.get("new_strategies"))
        logger.debug(f"[_reflect_internal] 待存储策略 | count={len(strategies)}")
        for strat in strategies:
            if not isinstance(strat, dict):
                continue
            try:
                self.kg.add_strategy(
                    name=str(strat.get("name", "未命名策略"))[:200],
                    description=str(strat.get("description", ""))[:500],
                    context=str(strat.get("context", ""))[:500],
                )
                logger.info(f"添加新策略: {strat.get('name', 'N/A')}")
            except Exception as e:
                logger.error(f"存储策略失败: {e}")
                logger.debug(f"[_reflect_internal] 存储策略失败 | name={strat.get('name', 'N/A')}, error={e}")

        # 记录任务日志
        suggestions = _safe_list(result.get("suggestions"))
        lessons = json.dumps(suggestions, ensure_ascii=False)[:500]
        for t in tasks:
            try:
                self.kg.log_task(
                    task_id=getattr(t, 'id', 'unknown'),
                    task_type=getattr(t, 'task_type', 'unknown'),
                    status=getattr(t, 'status', 'unknown').value if hasattr(t.status, 'value') else str(getattr(t, 'status', '')),
                    result=_safe_serialize(getattr(t, 'result', None), 500),
                    error=getattr(t, 'error', '') or '',
                    duration=getattr(t, 'duration', 0),
                    lessons=lessons,
                )
            except Exception as e:
                logger.error(f"记录任务日志失败: {e}")
                logger.debug(f"[_reflect_internal] 记录任务日志失败 | task_id={getattr(t, 'id', 'unknown')}, error={e}")

        self._optimization_count += 1
        self._last_optimization_time = time.time()

        logger.info(f"反思完成: {result.get('overall_assessment', 'N/A')}")
        logger.debug(f"[_reflect_internal] 反思完成 | optimization_count={self._optimization_count}")
        return result

    def get_optimization_advice(self, task_type: str = "") -> Dict[str, Any]:
        """获取优化建议（在执行任务前调用）"""
        logger.debug(f"[get_optimization_advice] 获取建议 | task_type={task_type}")
        try:
            strategies = self.kg.get_active_strategies(limit=5)
            experiences = self.kg.query_experience(task_type, limit=5) if task_type else []
            failures = self.kg.get_recent_failures(limit=3)
            logger.debug(f"[get_optimization_advice] 数据获取完成 | strategies={len(strategies)}, experiences={len(experiences)}, failures={len(failures)}")
        except Exception as e:
            logger.error(f"获取优化数据失败: {e}")
            logger.debug(f"[get_optimization_advice] 数据获取异常 | error={e}")
            return {"advice": "数据获取失败，跳过优化", "has_advice": False}

        if not strategies and not experiences and not failures:
            logger.debug(f"[get_optimization_advice] 无优化数据")
            return {"advice": "暂无优化建议，继续执行", "has_advice": False}

        # 用LLM综合分析
        prompt = f"""基于以下信息，为即将执行的任务提供优化建议。

任务类型: {task_type or '通用'}

## 活跃策略
{json.dumps(strategies, ensure_ascii=False, indent=2)}

## 相关经验
{[{'title': e.title, 'content': e.content[:100]} for e in experiences]}

## 最近失败
{json.dumps(failures, ensure_ascii=False, indent=2)}

请以JSON格式返回：
{{
    "advice": "综合建议（一句话）",
    "recommended_strategies": ["推荐使用的策略"],
    "warnings": ["需要注意的风险"],
    "tips": ["实用技巧"],
    "has_advice": true
}}"""

        result = self.llm.analyze_json(prompt, "你是AI Agent优化顾问。基于历史经验和策略，提供执行建议。返回JSON。")

        # 防御性处理
        if not isinstance(result, dict):
            result = {"advice": "解析失败", "has_advice": False}

        return result

    def analyze_failure(self, task) -> Dict[str, Any]:
        """深度分析失败任务"""
        logger.debug(f"[analyze_failure] 分析失败 | task_name={getattr(task, 'name', 'unknown')}, task_status={getattr(task, 'status', 'unknown')}")
        prompt = f"""分析以下任务失败原因，并提供修复方案。

任务名称: {getattr(task, 'name', 'unknown')}
任务类型: {getattr(task, 'task_type', 'unknown')}
错误信息: {getattr(task, 'error', 'unknown')}
重试次数: {getattr(task, 'retries', 0)}

请以JSON格式返回：
{{
    "root_cause": "根本原因分析",
    "fix_suggestion": "修复建议",
    "preventive_measures": ["预防措施1", "预防措施2"],
    "should_retry": false,
    "retry_strategy": "重试策略（如需重试）"
}}"""

        result = self.llm.analyze_json(prompt, "你是错误分析专家。分析失败原因并提供修复方案。返回JSON。")
        logger.debug(f"[analyze_failure] LLM分析完成 | result_type={type(result).__name__}")

        if not isinstance(result, dict):
            logger.debug(f"[analyze_failure] LLM返回非字典，使用fallback")
            result = {"root_cause": "分析失败", "fix_suggestion": "N/A"}

        # 存储错误经验
        try:
            self.kg.store_experience(
                title=f"错误分析: {getattr(task, 'name', 'unknown')}"[:200],
                content=f"错误: {getattr(task, 'error', 'N/A')}\n"
                        f"原因: {result.get('root_cause', 'N/A')}\n"
                        f"修复: {result.get('fix_suggestion', 'N/A')}",
                category="error",
                tags=f"error,{getattr(task, 'task_type', 'unknown')}",
                importance=0.8,
            )
            logger.debug(f"[analyze_failure] 错误经验已存储")
        except Exception as e:
            logger.error(f"存储错误分析失败: {e}")
            logger.debug(f"[analyze_failure] 存储错误分析失败 | error={e}")

        logger.info(f"失败分析完成: {getattr(task, 'name', 'N/A')} - {result.get('root_cause', 'N/A')}")
        return result

    def optimize_knowledge(self) -> Dict[str, Any]:
        """
        知识图谱优化：清理过时知识、合并重复、更新重要性
        """
        logger.debug(f"[optimize_knowledge] 开始知识优化")
        try:
            summary = self.kg.get_all_knowledge_summary()
            logger.debug(f"[optimize_knowledge] 获取摘要完成 | summary={json.dumps(summary, ensure_ascii=False)}")

            # 实际清理操作
            cleaned = self._cleanup_knowledge()
            logger.debug(f"[optimize_knowledge] 清理完成 | cleaned={cleaned}")

            self._optimization_count += 1
            self._last_optimization_time = time.time()

            logger.info(f"知识图谱优化完成: 清理{cleaned}条, 摘要: {json.dumps(summary, ensure_ascii=False)}")
            return {
                "action": "knowledge_optimized",
                "summary": summary,
                "cleaned": cleaned,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error(f"知识图谱优化失败: {e}")
            logger.debug(f"[optimize_knowledge] 优化失败 | error={e}")
            return {"action": "optimization_failed", "error": str(e)}

    def _cleanup_knowledge(self) -> int:
        """执行实际的知识清理"""
        logger.debug(f"[_cleanup_knowledge] 执行知识清理")
        try:
            result = self.kg.cleanup_old_knowledge()
            logger.debug(f"[_cleanup_knowledge] 清理完成 | cleaned={result}")
            return result
        except Exception as e:
            logger.error(f"知识清理执行失败: {e}")
            logger.debug(f"[_cleanup_knowledge] 清理失败 | error={e}")
            return 0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_optimizations": self._optimization_count,
            "last_optimization": self._last_optimization_time,
        }
