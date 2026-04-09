
"""
阶段 D：程序化验证
对 LLM 归一化结果做自动验证，只保留高可信补丁候选。
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from V3.Auto_knowledgegraph.config import ensure_runtime_dirs, load_config


@dataclass
class ValidationResult:
    """验证结果数据结构"""
    candidate_text: str
    validation_level: str  # A, B, C
    validation_reasons: list[str]
    downstream_gain: float | None
    conflict_with_existing_mapping: bool | None


class CandidateValidator:
    """候选验证器"""
    
    def __init__(self, graph_index: dict[str, Any], alias_index: dict[str, list[str]]):
        self.graph_index = graph_index
        self.alias_index = alias_index
        self.node_dict = {n["node_id"]: n for n in graph_index.get("nodes", [])}
    
    def validate_hierarchy_consistency(self, candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        """验证层级一致性"""
        reasons = []
        study_part = candidate.get("StudyPart", "")
        modality = candidate.get("modality", "")
        top_candidates = candidate.get("top_candidates", [])
        
        # 检查候选节点是否与 StudyPart 相容
        for tc in top_candidates:
            node_id = tc.get("candidate_node_id", "")
            node = self.node_dict.get(node_id)
            if node:
                path = node.get("path", [])
                # 简单检查：StudyPart 是否出现在路径中
                if study_part and study_part not in path and not any(study_part in p for p in path):
                    reasons.append(f"StudyPart {study_part} 与节点路径不匹配")
        
        return len(reasons) == 0, reasons
    
    def validate_graph_constraints(self, candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        """验证图谱约束"""
        reasons = []
        is_new_node = candidate.get("is_possible_new_node", False)
        top_candidates = candidate.get("top_candidates", [])
        
        if is_new_node:
            # 新节点需要检查是否有足够支撑
            if not top_candidates:
                reasons.append("疑似新节点但无候选参考")
        else:
            # 现有节点映射需要有效节点ID
            if not top_candidates:
                reasons.append("非新节点但无候选节点")
            else:
                for tc in top_candidates:
                    node_id = tc.get("candidate_node_id")
                    if node_id and node_id not in self.node_dict:
                        reasons.append(f"候选节点 {node_id} 不存在于图谱中")
        
        return len(reasons) == 0, reasons
    
    def validate_confidence(self, candidate: dict[str, Any], threshold: float = 0.7) -> tuple[bool, list[str]]:
        """验证置信度"""
        reasons = []
        confidence = candidate.get("llm_confidence") or 0.0
        
        if confidence < threshold:
            reasons.append(f"置信度 {confidence:.2f} 低于阈值 {threshold}")
        
        return len(reasons) == 0, reasons
    
    def validate(self, candidate: dict[str, Any]) -> ValidationResult:
        """执行完整验证"""
        all_reasons = []
        
        # 层级一致性验证
        hier_ok, hier_reasons = self.validate_hierarchy_consistency(candidate)
        all_reasons.extend(hier_reasons)
        
        # 图谱约束验证
        graph_ok, graph_reasons = self.validate_graph_constraints(candidate)
        all_reasons.extend(graph_reasons)
        
        # 置信度验证
        conf_ok, conf_reasons = self.validate_confidence(candidate)
        all_reasons.extend(conf_reasons)
        
        # 分级判断
        failure_count = sum([not hier_ok, not graph_ok, not conf_ok])
        
        if failure_count == 0:
            level = "A"  # 高可信
        elif failure_count == 1:
            level = "B"  # 中可信
        else:
            level = "C"  # 低可信
        
        return ValidationResult(
            candidate_text=candidate.get("candidate_text", ""),
            validation_level=level,
            validation_reasons=all_reasons or ["Validation passed"],
            downstream_gain=None,
            conflict_with_existing_mapping=False
        )


def validate_candidates(input_jsonl: Path, output_jsonl: Path,
                        graph_index_path: Path, alias_index_path: Path) -> int:
    """主函数：验证候选实体"""
    # 加载索引
    graph_index = {}
    if graph_index_path.exists():
        graph_index = json.loads(graph_index_path.read_text(encoding="utf-8"))
    
    alias_index = {}
    if alias_index_path.exists():
        alias_index = json.loads(alias_index_path.read_text(encoding="utf-8"))
    
    validator = CandidateValidator(graph_index, alias_index)
    
    # 读取归一化后的候选
    candidates = []
    if input_jsonl.exists():
        with input_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    
    if not candidates:
        output_jsonl.write_text("", encoding="utf-8")
        return 0
    
    # 验证
    validated = []
    for candidate in tqdm(candidates, desc="Validating candidates"):
        result = validator.validate(candidate)
        validated.append(asdict(result))
    
    # 写入输出
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in validated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 统计
    level_counts = {"A": 0, "B": 0, "C": 0}
    for v in validated:
        level = v.get("validation_level", "C")
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"Validation results: A={level_counts['A']}, B={level_counts['B']}, C={level_counts['C']}")
    return len(validated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto KnowledgeGraph validation")
    parser.add_argument("--input", default="data/entity_candidate_normalized.jsonl")
    parser.add_argument("--output", default="data/entity_candidate_validated.jsonl")
    parser.add_argument("--graph-index", default="data/graph_index.json")
    parser.add_argument("--alias-index", default="data/graph_alias_index.json")
    args = parser.parse_args()

    cfg = load_config()
    ensure_runtime_dirs(cfg)
    
    input_path = cfg.project_root / args.input
    output_path = cfg.project_root / args.output
    graph_index_path = cfg.project_root / args.graph_index
    alias_index_path = cfg.project_root / args.alias_index
    
    count = validate_candidates(input_path, output_path, graph_index_path, alias_index_path)
    print(f"Validated {count} candidates")


if __name__ == "__main__":
    main()
