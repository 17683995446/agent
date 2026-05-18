"""
示例任务定义 - 演示Agent的通用任务框架
"""
import time
import random
import math


def hello_world(name: str = "World") -> str:
    """简单示例任务"""
    result = f"Hello, {name}! 任务执行成功。"
    print(f"  [任务结果] {result}")
    return result


def calculate_fibonacci(n: int = 30) -> dict:
    """计算斐波那契数列"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    result = {"n": n, "fibonacci": a, "ratio": round(b / a, 6) if a else 0}
    print(f"  [计算结果] 斐波那契({n}) = {a}, 黄金比例近似 = {result['ratio']}")
    return result


def process_data(data_size: int = 1000) -> dict:
    """模拟数据处理任务"""
    data = [random.random() for _ in range(data_size)]
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    result = {
        "size": data_size,
        "mean": round(mean, 4),
        "std": round(math.sqrt(variance), 4),
        "min": round(min(data), 4),
        "max": round(max(data), 4),
    }
    print(f"  [数据处理] 样本数={data_size}, 均值={result['mean']}, 标准差={result['std']}")
    return result


def research_topic(topic: str = "AI Agent架构") -> dict:
    """模拟研究任务"""
    findings = [
        f"关于'{topic}'的关键发现：模块化设计是核心",
        f"关于'{topic}'的最佳实践：持续反馈循环",
        f"关于'{topic}'的趋势：多Agent协作成为主流",
    ]
    result = {
        "topic": topic,
        "findings": findings,
        "confidence": round(random.uniform(0.7, 0.95), 2),
    }
    print(f"  [研究完成] {topic}: {len(findings)}条发现, 置信度={result['confidence']}")
    return result


def generate_report(title: str = "运行报告") -> dict:
    """模拟报告生成任务"""
    result = {
        "title": title,
        "sections": ["概述", "数据分析", "结论"],
        "word_count": random.randint(500, 2000),
        "quality_score": round(random.uniform(0.8, 0.99), 2),
    }
    print(f"  [报告生成] {title}: {result['word_count']}字, 质量={result['quality_score']}")
    return result


def simulate_api_call(endpoint: str = "/api/data") -> dict:
    """模拟API调用任务"""
    # 模拟偶尔失败
    if random.random() < 0.15:  # 15%失败率
        raise ConnectionError(f"API调用失败: {endpoint} - 连接超时")

    result = {
        "endpoint": endpoint,
        "status": 200,
        "response_time_ms": random.randint(50, 500),
        "data": {"items": random.randint(1, 100)},
    }
    print(f"  [API调用] {endpoint}: {result['status']}, 耗时={result['response_time_ms']}ms")
    return result


def monitor_health() -> dict:
    """模拟健康检查任务"""
    metrics = {
        "cpu_usage": round(random.uniform(10, 80), 1),
        "memory_usage": round(random.uniform(30, 70), 1),
        "disk_usage": round(random.uniform(20, 60), 1),
        "active_connections": random.randint(5, 50),
        "response_time_ms": random.randint(10, 200),
    }
    healthy = all(v < 85 for k, v in metrics.items() if "usage" in k)
    print(f"  [健康检查] {'✓ 正常' if healthy else '✗ 异常'} CPU={metrics['cpu_usage']}% MEM={metrics['memory_usage']}%")
    return {"healthy": healthy, "metrics": metrics}


# ========== 任务生成器示例 ==========

def create_demo_task_generator():
    """
    创建示例任务生成器
    每次调用返回一批新任务，模拟持续工作场景
    """
    task_templates = [
        {"name": "数据处理", "handler": process_data, "task_type": "process", "priority": 5},
        {"name": "API调用", "handler": simulate_api_call, "task_type": "api", "priority": 3},
        {"name": "健康检查", "handler": monitor_health, "task_type": "monitor", "priority": 1},
        {"name": "研究报告", "handler": research_topic, "task_type": "research", "priority": 6},
        {"name": "报告生成", "handler": generate_report, "task_type": "report", "priority": 7},
        {"name": "斐波那契计算", "handler": calculate_fibonacci, "task_type": "compute", "priority": 8},
    ]

    call_count = [0]

    def generator():
        call_count[0] += 1

        # 每次生成1-3个随机任务
        count = random.randint(1, 3)
        selected = random.sample(task_templates, min(count, len(task_templates)))

        tasks = []
        for tmpl in selected:
            task_dict = tmpl.copy()
            task_dict["name"] = f"{tmpl['name']} #{call_count[0]}"
            # 随机参数
            if tmpl["handler"] == process_data:
                task_dict["kwargs"] = {"data_size": random.randint(100, 5000)}
            elif tmpl["handler"] == research_topic:
                topics = ["机器学习", "知识图谱", "自动化", "LLM应用", "多Agent系统"]
                task_dict["kwargs"] = {"topic": random.choice(topics)}
            elif tmpl["handler"] == simulate_api_call:
                endpoints = ["/api/users", "/api/data", "/api/health", "/api/tasks"]
                task_dict["kwargs"] = {"endpoint": random.choice(endpoints)}
            tasks.append(task_dict)

        return tasks

    return generator
