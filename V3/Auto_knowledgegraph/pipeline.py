
"""
Auto KnowledgeGraph 流水线主控

完整的六阶段流水线：
A. 知识底座索引化 (build_graph_index.py)
B. 新表达自动发现 (discover_candidates.py)
C. LLM 候选归一化 (normalize_candidates_with_llm.py)
D. 程序化验证 (validate_candidates.py)
E. 补丁发布 (publish_patches.py)
F. 下游任务回放 (replay_downstream_tasks.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from V3.Auto_knowledgegraph.config import ensure_runtime_dirs, load_config
from V3.Auto_knowledgegraph.build_graph_index import main as build_index
from V3.Auto_knowledgegraph.discover_candidates import discover_candidates
from V3.Auto_knowledgegraph.normalize_candidates_with_llm import normalize_candidates
from V3.Auto_knowledgegraph.validate_candidates import validate_candidates
from V3.Auto_knowledgegraph.publish_patches import publish_patches
from V3.Auto_knowledgegraph.replay_downstream_tasks import replay


def run_pipeline(
    config,
    start_stage: str = "all",
    end_stage: str = "all",
    skip_discovery: bool = False
) -> dict:
    """
    运行完整流水线
    
    Args:
        config: 配置对象
        start_stage: 起始阶段 (A/B/C/D/E/F/all)
        end_stage: 结束阶段 (A/B/C/D/E/F/all)
        skip_discovery: 是否跳过候选发现（使用现有数据）
    """
    stages = ["A", "B", "C", "D", "E", "F"]
    stage_names = {
        "A": "知识底座索引化",
        "B": "新表达自动发现",
        "C": "LLM 候选归一化",
        "D": "程序化验证",
        "E": "补丁发布",
        "F": "下游任务回放"
    }
    
    if start_stage == "all":
        start_idx = 0
    else:
        start_idx = stages.index(start_stage)
    
    if end_stage == "all":
        end_idx = len(stages) - 1
    else:
        end_idx = stages.index(end_stage)
    
    results = {}
    
    for i in range(start_idx, end_idx + 1):
        stage = stages[i]
        print(f"\n{'='*60}")
        print(f"阶段 {stage}: {stage_names[stage]}")
        print(f"{'='*60}")
        
        if stage == "A":
            # 阶段 A: 构建图谱索引
            print("Building graph index...")
            build_index()
            results["stage_A"] = "completed"
            
        elif stage == "B":
            # 阶段 B: 候选发现
            if skip_discovery:
                print("Skipping discovery (using existing candidates)")
                results["stage_B"] = "skipped"
            else:
                print("Discovering candidates...")
                input_path = config.project_root / "data" / "source_reports.jsonl"
                output_path = config.project_root / "data" / "entity_candidate_discovery.jsonl"
                alias_index_path = config.project_root / "data" / "graph_alias_index.json"
                count = discover_candidates(input_path, output_path, alias_index_path)
                results["stage_B"] = {"candidates_discovered": count}
                
        elif stage == "C":
            # 阶段 C: LLM 归一化
            print("Normalizing candidates with LLM...")
            input_path = config.project_root / "data" / "entity_candidate_discovery.jsonl"
            output_path = config.project_root / "data" / "entity_candidate_normalized.jsonl"
            graph_index_path = config.project_root / "data" / "graph_index.json"
            count = normalize_candidates(input_path, output_path, graph_index_path, config)
            results["stage_C"] = {"candidates_normalized": count}
            
        elif stage == "D":
            # 阶段 D: 验证
            print("Validating candidates...")
            input_path = config.project_root / "data" / "entity_candidate_normalized.jsonl"
            output_path = config.project_root / "data" / "entity_candidate_validated.jsonl"
            graph_index_path = config.project_root / "data" / "graph_index.json"
            alias_index_path = config.project_root / "data" / "graph_alias_index.json"
            count = validate_candidates(input_path, output_path, graph_index_path, alias_index_path)
            results["stage_D"] = {"candidates_validated": count}
            
        elif stage == "E":
            # 阶段 E: 补丁发布
            print("Publishing patches...")
            validated_path = config.project_root / "data" / "entity_candidate_validated.jsonl"
            report_path = config.project_root / "reports" / "patch_release_report.md"
            summary = publish_patches(validated_path, report_path, config)
            results["stage_E"] = summary
            
        elif stage == "F":
            # 阶段 F: 下游回放
            print("Running downstream replay...")
            report_path = config.project_root / "reports" / "downstream_replay_report.md"
            report_data = replay(report_path, config)
            results["stage_F"] = report_data
    
    print(f"\n{'='*60}")
    print("流水线执行完成")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Auto KnowledgeGraph Pipeline")
    parser.add_argument("--start", default="all", choices=["A", "B", "C", "D", "E", "F", "all"],
                        help="起始阶段")
    parser.add_argument("--end", default="all", choices=["A", "B", "C", "D", "E", "F", "all"],
                        help="结束阶段")
    parser.add_argument("--skip-discovery", action="store_true",
                        help="跳过候选发现阶段（使用现有数据）")
    parser.add_argument("--stage", choices=["A", "B", "C", "D", "E", "F"],
                        help="仅运行单个阶段")
    args = parser.parse_args()
    
    config = load_config()
    ensure_runtime_dirs(config)
    
    if args.stage:
        run_pipeline(config, start_stage=args.stage, end_stage=args.stage)
    else:
        run_pipeline(config, start_stage=args.start, end_stage=args.end,
                    skip_discovery=args.skip_discovery)


if __name__ == "__main__":
    main()
