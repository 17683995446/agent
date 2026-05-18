#!/usr/bin/env python3
"""
极简全能持续运行Agent - 启动入口

使用方法:
  1. 基础运行（模拟模式，无需LLM API）:
     python run.py

  2. 接入OpenAI API:
     LLM_API_KEY=your_key python run.py

  3. 接入Ollama本地模型:
     LLM_PROVIDER=ollama LLM_BASE_URL=http://localhost:11434 LLM_MODEL=qwen2.5:7b python run.py

  4. 自定义运行:
     python run.py --cycles 20 --interval 1
"""
import os
import sys
import logging
import argparse

# 添加项目根目录到路径（agent/的父目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent, run_with_watchdog
from agent.tasks.demo_tasks import create_demo_task_generator


def setup_logging(level: str = "INFO", log_dir: str = "logs"):
    """配置日志"""
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 控制台
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    # 文件
    file_handler = logging.FileHandler(f"{log_dir}/agent.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 配置根日志
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="极简全能持续运行Agent")
    parser.add_argument("--cycles", type=int, default=0, help="最大循环次数 (0=无限)")
    parser.add_argument("--interval", type=float, default=2.0, help="循环间隔(秒)")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    parser.add_argument("--no-watchdog", action="store_true", help="禁用看门狗(保活)")
    parser.add_argument("--demo", action="store_true", default=True, help="运行示例任务")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("  极简全能持续运行Agent")
    logger.info("  一直干活 → 越干越会干 → 把学会的全部存起来 → 无限循环")
    logger.info("=" * 60)

    # 创建Agent
    from agent.core.config import AgentConfig
    config = AgentConfig.from_env()

    agent = Agent(config)

    # 设置示例任务生成器
    if args.demo:
        agent.set_task_generator(create_demo_task_generator())
        logger.info("已加载示例任务生成器")

    # 也可以手动提交任务
    from agent.tasks.demo_tasks import hello_world
    agent.submit_task("启动问候", hello_world, kwargs={"name": "Agent"}, priority=1)

    # 启动运行
    if args.no_watchdog:
        agent.run(max_cycles=args.cycles, cycle_interval=args.interval)
    else:
        run_with_watchdog(
            agent,
            max_cycles=args.cycles,
            cycle_interval=args.interval,
            max_restarts=config.max_restart_attempts,
            restart_cooldown=config.restart_cooldown,
        )


if __name__ == "__main__":
    main()
