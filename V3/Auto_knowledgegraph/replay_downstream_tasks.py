
"""
阶段 F：下游任务回放验证
证明实体层增强对业务任务有真实收益。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


import argparse
import json
from pathlib import Path
from typing import Any
from datetime import datetime

from V3.Auto_knowledgegraph.config import ensure_runtime_dirs, load_config


def generate_mock_replay_report(report_path: Path, config: Any) -> dict[str, Any]:
    """
    生成下游任务回放报告（当前为框架实现）
    
    后续需要接入真实质控流程：
    - NLP_analyze.py 中的各项检测任务
    - api_server.py 的 API 入口
    """
    # 模拟的评估指标
    mock_metrics = {
        "entity_recall": {
            "baseline": 0.85,
            "with_patches": 0.92,
            "improvement": "+8.2%"
        },
        "normalization_accuracy": {
            "baseline": 0.88,
            "with_patches": 0.94,
            "improvement": "+6.8%"
        },
        "partlist_stability": {
            "baseline": 0.91,
            "with_patches": 0.93,
            "improvement": "+2.2%"
        }
    }
    
    # 模拟的任务级收益
    task_gains = [
        {
            "task": "conclusion_missing",
            "baseline_error_rate": 0.12,
            "patched_error_rate": 0.09,
            "improvement": "-25%",
            "notes": "实体召回提升减少了遗漏"
        },
        {
            "task": "orient_error",
            "baseline_error_rate": 0.08,
            "patched_error_rate": 0.07,
            "improvement": "-12.5%",
            "notes": "部位识别更准确"
        },
        {
            "task": "contradiction",
            "baseline_error_rate": 0.15,
            "patched_error_rate": 0.13,
            "improvement": "-13.3%",
            "notes": "隐式部位表达被正确识别"
        }
    ]
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "status": "framework_ready",
        "evaluated_tasks": [
            "conclusion_missing",
            "orient_error", 
            "apply_orient",
            "contradiction"
        ],
        "metrics": mock_metrics,
        "task_gains": task_gains,
        "recommendations": [
            "当前为框架实现，需要接入真实质控流程",
            "建议收集多院区验证集进行 A/B 测试",
            "建议建立持续监控机制"
        ]
    }
    
    # 生成 Markdown 报告
    report_md = f"""# 下游任务回放报告

生成时间：{report_data["timestamp"]}

## 状态

当前状态：`{report_data["status"]}`

这是一个框架实现，展示了报告结构。后续需要接入真实的质控流程。

## 评估任务

{chr(10).join(f"- {task}" for task in report_data["evaluated_tasks"])}

## 核心指标

| 指标 | 基线 | 补丁后 | 提升 |
|------|------|--------|------|
| 实体召回率 | {mock_metrics["entity_recall"]["baseline"]:.2%} | {mock_metrics["entity_recall"]["with_patches"]:.2%} | {mock_metrics["entity_recall"]["improvement"]} |
| 标准化准确率 | {mock_metrics["normalization_accuracy"]["baseline"]:.2%} | {mock_metrics["normalization_accuracy"]["with_patches"]:.2%} | {mock_metrics["normalization_accuracy"]["improvement"]} |
| Partlist 稳定率 | {mock_metrics["partlist_stability"]["baseline"]:.2%} | {mock_metrics["partlist_stability"]["with_patches"]:.2%} | {mock_metrics["partlist_stability"]["improvement"]} |

## 任务级收益

| 任务 | 基线错误率 | 补丁后错误率 | 改善 |
|------|-----------|-------------|------|
{chr(10).join(f"| {g['task']} | {g['baseline_error_rate']:.1%} | {g['patched_error_rate']:.1%} | {g['improvement']} |" for g in task_gains)}

## 后续建议

{chr(10).join(f"{i+1}. {rec}" for i, rec in enumerate(report_data["recommendations"]))}

## 实现说明

真实回放需要：
1. 加载补丁后的抽取器配置
2. 批量处理验证集报告
3. 对比基线和补丁后的质控结果
4. 统计各项任务的改善情况
"""
    
    report_path.write_text(report_md, encoding="utf-8")
    
    return report_data


def replay(report_path: Path, config: Any) -> dict[str, Any]:
    """主函数：下游任务回放"""
    report_data = generate_mock_replay_report(report_path, config)
    print(f"Replay report generated: {report_path}")
    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto KnowledgeGraph downstream replay")
    parser.add_argument("--report", default="reports/downstream_replay_report.md")
    args = parser.parse_args()

    cfg = load_config()
    ensure_runtime_dirs(cfg)
    
    report_path = cfg.project_root / args.report
    replay(report_path, cfg)


if __name__ == "__main__":
    main()
