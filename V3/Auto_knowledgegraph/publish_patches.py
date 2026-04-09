
"""
阶段 E：补丁发布
将验证通过的候选以补丁层方式发布，而不是直接修改主图谱。
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

import pandas as pd
from datetime import datetime

from V3.Auto_knowledgegraph.config import ensure_runtime_dirs, load_config


def load_validated_candidates(validated_jsonl: Path) -> list[dict[str, Any]]:
    """加载验证后的候选"""
    candidates = []
    if validated_jsonl.exists():
        with validated_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    return candidates


def generate_alias_patch(
    candidates: list[dict[str, Any]],
    min_level: str = "A"
) -> pd.DataFrame:
    """生成别名补丁"""
    level_priority = {"A": 3, "B": 2, "C": 1}
    min_priority = level_priority.get(min_level, 2)
    
    alias_records = []
    for c in candidates:
        if level_priority.get(c.get("validation_level", "C"), 0) < min_priority:
            continue
        
        # 别名类型：A/B级且标记为现有节点别名的
        if c.get("is_alias_of_existing_node"):
            alias_records.append({
                "alias_text": c.get("candidate_text"),
                "target_node_id": c.get("top_candidates", [{}])[0].get("candidate_node_id"),
                "target_path": str(c.get("top_candidates", [{}])[0].get("candidate_node_path", [])),
                "validation_level": c.get("validation_level"),
                "confidence": c.get("llm_confidence"),
                "discovery_method": c.get("discovery_method"),
                "created_at": datetime.now().isoformat()
            })
    
    return pd.DataFrame(alias_records)


def generate_node_candidates_patch(
    candidates: list[dict[str, Any]]
) -> pd.DataFrame:
    """生成新节点候选补丁"""
    node_records = []
    for c in candidates:
        if c.get("is_possible_new_node") and c.get("validation_level") in ["A", "B"]:
            node_records.append({
                "proposed_canonical_name": c.get("candidate_text"),
                "context_sentence": c.get("sentence"),
                "StudyPart": c.get("StudyPart"),
                "modality": c.get("modality"),
                "frequency": c.get("frequency_local"),
                "validation_level": c.get("validation_level"),
                "llm_reasoning": c.get("llm_reasoning"),
                "proposed_parent_path": None,  # 需要人工审核
                "status": "pending_review",
                "created_at": datetime.now().isoformat()
            })
    
    return pd.DataFrame(node_records)


def publish_patches(
    validated_jsonl: Path,
    report_path: Path,
    config: Any
) -> dict[str, Any]:
    """主函数：发布补丁"""
    candidates = load_validated_candidates(validated_jsonl)
    
    if not candidates:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "status": "no_candidates",
            "published_aliases": 0,
            "published_node_candidates": 0
        }
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    
    # 生成补丁
    alias_df = generate_alias_patch(candidates, min_level="B")
    node_df = generate_node_candidates_patch(candidates)
    
    # 保存补丁文件
    patches_dir = config.repo_root / "config" / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    
    alias_patch_path = patches_dir / "knowledgegraph_alias_patch.xlsx"
    node_patch_path = patches_dir / "knowledgegraph_node_candidates.xlsx"
    
    if not alias_df.empty:
        alias_df.to_excel(alias_patch_path, index=False)
    
    if not node_df.empty:
        node_df.to_excel(node_patch_path, index=False)
    
    # 生成报告
    summary = {
        "timestamp": datetime.now().isoformat(),
        "status": "published",
        "total_candidates": len(candidates),
        "published_aliases": len(alias_df),
        "published_node_candidates": len(node_df),
        "alias_patch_path": str(alias_patch_path.relative_to(config.repo_root)),
        "node_patch_path": str(node_patch_path.relative_to(config.repo_root)),
        "level_distribution": {
            "A": sum(1 for c in candidates if c.get("validation_level") == "A"),
            "B": sum(1 for c in candidates if c.get("validation_level") == "B"),
            "C": sum(1 for c in candidates if c.get("validation_level") == "C"),
        }
    }
    
    # 写入 Markdown 报告
    report_md = f"""# 知识图谱补丁发布报告

生成时间：{summary["timestamp"]}

## 统计概览

- 总候选数：{summary["total_candidates"]}
- 发布别名数：{summary["published_aliases"]}
- 新节点候选数：{summary["published_node_candidates"]}

## 验证分级分布

- A级（高可信）：{summary["level_distribution"]["A"]}
- B级（中可信）：{summary["level_distribution"]["B"]}
- C级（低可信）：{summary["level_distribution"]["C"]}

## 补丁文件位置

- 别名补丁：`{summary["alias_patch_path"]}`
- 节点候选补丁：`{summary["node_patch_path"]}`

## 注意事项

1. 别名补丁可直接加载到抽取器中
2. 新节点候选需要人工审核后确认
3. 建议定期（如每月）合并补丁到主图谱
"""
    
    report_path.write_text(report_md, encoding="utf-8")
    
    print(f"Published {summary['published_aliases']} aliases and {summary['published_node_candidates']} node candidates")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto KnowledgeGraph patch publishing")
    parser.add_argument("--input", default="data/entity_candidate_validated.jsonl")
    parser.add_argument("--report", default="reports/patch_release_report.md")
    args = parser.parse_args()

    cfg = load_config()
    ensure_runtime_dirs(cfg)
    
    validated_path = cfg.project_root / args.input
    report_path = cfg.project_root / args.report
    
    publish_patches(validated_path, report_path, cfg)


if __name__ == "__main__":
    main()
