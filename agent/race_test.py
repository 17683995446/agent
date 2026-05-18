#!/usr/bin/env python3
"""
竞态条件专项测试 - 高概率触发E1/E3

使用循环屏障同步，最大化并发冲突概率
"""
import sys, os, time, threading, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.modules.task_engine import TaskEngine, TaskStatus


def test_extreme_race():
    """极端并发竞争测试"""
    print("=" * 60)
    print("竞态条件专项测试")
    print("=" * 60)

    engine = TaskEngine(max_workers=20, retry_limit=0, timeout=0.1)

    # 提交大量短超时任务
    for i in range(500):
        engine.submit(f"任务-{i}", lambda: time.sleep(0.5), timeout=0.1, max_retries=0)

    errors = []
    barrier = threading.Barrier(10)

    def stress_runner():
        try:
            barrier.wait(timeout=5)  # 同步开始
            for _ in range(10):
                # 同时执行run_batch
                batch = engine.run_batch(5)
                # 同时读取统计
                stats = engine.stats
                # 同时读取completed
                completed = engine.get_recent_completed(limit=20)
                # 同时触发清理
                engine.clear_completed(keep=50)

                # 验证一致性
                if stats['total_executed'] != stats['total_success'] + stats['total_failed']:
                    errors.append(f"统计不一致: {stats}")
                if len(completed) > 100:
                    errors.append(f"completed返回过多: {len(completed)}")
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=stress_runner) for _ in range(10)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    elapsed = time.time() - start
    alive = [t for t in threads if t.is_alive()]

    print(f"\n运行时间: {elapsed:.1f}秒")
    print(f"存活线程: {len(alive)}/10")
    print(f"错误数量: {len(errors)}")

    stats = engine.stats
    print(f"\n最终统计:")
    print(f"  executed: {stats['total_executed']}")
    print(f"  success: {stats['total_success']}")
    print(f"  failed: {stats['total_failed']}")
    print(f"  rate: {stats['success_rate']}%")

    # 最终验证
    expected = stats['total_success'] + stats['total_failed']
    if stats['total_executed'] == expected and not errors and not alive:
        print("\n✓ 竞态条件测试通过")
        return 0
    else:
        print(f"\n✗ 竞态条件测试失败")
        for e in errors[:5]:
            print(f"  - {e[:100]}")
        return 1


if __name__ == "__main__":
    sys.exit(test_extreme_race())
