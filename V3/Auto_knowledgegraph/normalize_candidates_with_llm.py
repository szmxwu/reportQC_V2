
"""
阶段 C：LLM 候选归一化
使用本地 LLM 对候选表达做标准节点候选排序与归一化。
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

import requests
from tqdm import tqdm

from V3.Auto_knowledgegraph.config import ensure_runtime_dirs, load_config


@dataclass
class NormalizedCandidate:
    """归一化后的候选数据结构"""
    candidate_text: str
    sentence: str
    StudyPart: str
    modality: str
    top_candidates: list[dict[str, Any]]
    is_alias_of_existing_node: bool | None
    is_possible_new_node: bool
    normalization_status: str
    llm_confidence: float | None
    llm_reasoning: str | None


class LLMNormalizer:
    """LLM 归一化器"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: int = 30, max_tokens: int = 2048, temperature: float = 0.1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
    
    def available(self) -> bool:
        """检查 LLM 服务是否可用"""
        try:
            resp = self._session.get(f"{self.base_url}/models", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"LLM service unavailable: {e}")
            return False

    def retrieve_candidate_nodes(self, candidate_text: str, graph_index: dict[str, Any], top_k: int = 5) -> list[dict[str, Any]]:
        """从图谱索引中检索候选节点"""
        nodes = graph_index.get("nodes", [])
        scored_nodes = []
        
        for node in nodes:
            score = 0.0
            canonical = node.get("canonical_name", "")
            
            if canonical == candidate_text:
                score = 1.0
            elif candidate_text in canonical or canonical in candidate_text:
                score = 0.8
            else:
                for synonyms in node.get("synonym_levels", []):
                    for syn in synonyms:
                        if syn == candidate_text:
                            score = 0.95
                            break
                        elif candidate_text in syn or syn in candidate_text:
                            score = max(score, 0.7)
            
            if score > 0:
                scored_nodes.append({
                    "node_id": node.get("node_id"),
                    "path": node.get("path"),
                    "canonical_name": canonical,
                    "synonyms": [s for syns in node.get("synonym_levels", []) for s in syns],
                    "score": score
                })
        
        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        return scored_nodes[:top_k]
    
    def build_prompt(self, candidate: dict[str, Any], candidate_nodes: list[dict[str, Any]]) -> str:
        """构建 LLM 提示词"""
        candidate_text = candidate.get("candidate_text", "")
        sentence = candidate.get("sentence", "")
        study_part = candidate.get("StudyPart", "")
        modality = candidate.get("modality", "")
        
        nodes_text = "\n".join([
            f"{i+1}. {n['canonical_name']}" 
            for i, n in enumerate(candidate_nodes)
        ]) if candidate_nodes else "无匹配节点"
        
        prompt = f"任务：将候选解剖表达归一化到标准知识图谱节点。\n\n候选表达：{candidate_text}\n原始句子：{sentence}\n检查部位：{study_part}\n模态：{modality}\n\n候选标准节点：\n{nodes_text}\n\n请按 JSON 格式输出归一化结果。"
        return prompt

    def call_llm(self, prompt: str) -> dict[str, Any]:
        """调用 LLM API"""
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是医学知识图谱归一化专家。请严格按 JSON 格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        
        try:
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except:
                    pass
            return {"error": "Failed to parse", "raw": content[:200]}
    
    def normalize(self, candidate: dict[str, Any], graph_index: dict[str, Any]) -> NormalizedCandidate:
        """归一化单个候选"""
        candidate_nodes = self.retrieve_candidate_nodes(candidate.get("candidate_text", ""), graph_index)
        prompt = self.build_prompt(candidate, candidate_nodes)
        
        try:
            llm_response = self.call_llm(prompt)
            return NormalizedCandidate(
                candidate_text=candidate.get("candidate_text", ""),
                sentence=candidate.get("sentence", ""),
                StudyPart=candidate.get("StudyPart", ""),
                modality=candidate.get("modality", ""),
                top_candidates=llm_response.get("top_candidates", []),
                is_alias_of_existing_node=llm_response.get("is_alias_of_existing_node"),
                is_possible_new_node=llm_response.get("is_possible_new_node", False),
                normalization_status="success",
                llm_confidence=llm_response.get("confidence"),
                llm_reasoning=llm_response.get("reasoning")
            )
        except Exception as e:
            return NormalizedCandidate(
                candidate_text=candidate.get("candidate_text", ""),
                sentence=candidate.get("sentence", ""),
                StudyPart=candidate.get("StudyPart", ""),
                modality=candidate.get("modality", ""),
                top_candidates=[],
                is_alias_of_existing_node=None,
                is_possible_new_node=False,
                normalization_status="failed",
                llm_confidence=0.0,
                llm_reasoning=str(e)
            )


def normalize_candidates(input_jsonl: Path, output_jsonl: Path, 
                         graph_index_path: Path, config: Any) -> int:
    """主函数：归一化候选实体"""
    # 加载图谱索引
    graph_index = {}
    if graph_index_path.exists():
        graph_index = json.loads(graph_index_path.read_text(encoding="utf-8"))
    
    # 初始化 LLM 归一化器
    normalizer = LLMNormalizer(
        base_url=config.llm_base_url,
        model=config.llm_model,
        api_key=config.llm_api_key,
        timeout=config.llm_timeout,
        max_tokens=config.llm_max_tokens
    )
    
    if not normalizer.available():
        print("Warning: LLM service not available, skipping normalization")
        return 0
    
    # 读取候选数据
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
    
    # 批量归一化
    normalized = []
    for candidate in tqdm(candidates, desc="Normalizing candidates"):
        result = normalizer.normalize(candidate, graph_index)
        normalized.append(asdict(result))
    
    # 写入输出
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in normalized:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return len(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto KnowledgeGraph normalization")
    parser.add_argument("--input", default="data/entity_candidate_discovery.jsonl")
    parser.add_argument("--output", default="data/entity_candidate_normalized.jsonl")
    parser.add_argument("--graph-index", default="data/graph_index.json")
    args = parser.parse_args()

    cfg = load_config()
    ensure_runtime_dirs(cfg)
    
    input_path = cfg.project_root / args.input
    output_path = cfg.project_root / args.output
    graph_index_path = cfg.project_root / args.graph_index
    
    count = normalize_candidates(input_path, output_path, graph_index_path, cfg)
    print(f"Normalized {count} candidates")


if __name__ == "__main__":
    main()
