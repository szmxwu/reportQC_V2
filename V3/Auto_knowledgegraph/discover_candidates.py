"""
阶段 B：新表达自动发现
从多院区真实报告中发现规则未覆盖或覆盖不稳定的实体表达。
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
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from V3.Auto_knowledgegraph.config import ensure_runtime_dirs, load_config


@dataclass
class EntityCandidate:
    """实体候选数据结构"""
    hospital_id: str | None
    report_id: str | None
    sentence: str
    candidate_text: str
    context_before: str | None
    context_after: str | None
    StudyPart: str
    modality: str
    matched_by_rule: bool | None
    current_partlist: list | None
    candidate_type: str | None  # unhit, mis_hit, coarse_grain, alias, implicit
    frequency_local: int | None  # 单院区频次
    frequency_global: int | None  # 跨院区频次
    discovery_method: str | None  # ngram, context_pattern, cooccurrence, downstream_failure


class CandidateDiscoveryEngine:
    """候选实体发现引擎"""
    
    # 解剖相关词性/词缀模式
    ANATOMY_PATTERNS = [
        r'[左右上下前后内外侧中]?(?:叶|段|区|室|腔|窝|间隙|裂|沟|回|角|体|头|颈|脚|角|突|弓|根|干|支|叶|面|缘|壁|膜|带|核|节|丛)',
        r'[左右]?[上下]?[前后]?第?[一二三四五六七八九十\d]+[节段肋椎]',
        r'[左右]?[上下]?[前后]?[IVX\d]+[节段]',
        r'[左右]?[腰胸颈骶][\d-]+(?:椎体|椎间盘|椎间隙|椎管)',
        r'[左右]?[一二三四五六][一二三四五六七八九十]指肠',
        r'[左右]?[上下]肢',
    ]
    
    # 高频停用词（应排除的非实体词汇）
    STOP_WORDS = {
        '患者', '检查', '结果', '所示', '未见', '显示', '考虑', '建议', '请',
        '结合', '临床', '诊断', '扫描', '图像', '层面', '范围', '情况', '表现',
        '所示', '指出', '提示', '可见', '可见于', '明显', '轻度', '重度',
        '部分', '局部', '区域', '周围', '邻近', '前方', '后方', '上方', '下方',
        '右侧', '左侧', '双侧', '两侧', '以上', '以下', '以内', '以外',
        '毫米', '厘米', 'mm', 'cm', 'ml', 'HU', 'ms', '大小约', '直径约',
        '对比剂', '造影剂', '增强扫描', '平扫', '复查', '随访', '进一步',
    }
    
    # 上下文模式（用于发现隐含解剖表达）
    CONTEXT_PATTERNS = [
        (r'位于([\u4e00-\u9fa5]{2,8})(?:内|处|边缘|周围|邻近)', 'location_pattern'),
        (r'(?:压迫|推移|侵犯|累及|包绕|包埋)([\u4e00-\u9fa5]{2,8})', 'action_pattern'),
        (r'(?:与|和|同)([\u4e00-\u9fa5]{2,8})(?:分界|界限|边界|相邻)', 'relation_pattern'),
        (r'(?:自|从|由)([\u4e00-\u9fa5]{2,8})(?:起源|发出|延伸|延续)', 'origin_pattern'),
    ]
    
    def __init__(self, alias_index: dict[str, list[str]] | None = None):
        self.alias_index = alias_index or {}
        self.compiled_anatomy_patterns = [re.compile(p) for p in self.ANATOMY_PATTERNS]
        self.compiled_context_patterns = [
            (re.compile(pattern), name) for pattern, name in self.CONTEXT_PATTERNS
        ]
        
    def load_graph_index(self, graph_index_path: Path) -> dict[str, Any]:
        """加载图谱索引"""
        if not graph_index_path.exists():
            return {}
        return json.loads(graph_index_path.read_text(encoding="utf-8"))
    
    def is_known_entity(self, text: str) -> bool:
        """检查文本是否已在知识图谱中"""
        text = text.strip()
        if text in self.alias_index:
            return True
        # 模糊匹配：检查子串
        for alias in self.alias_index:
            if text in alias or alias in text:
                return True
        return False
    
    def extract_ngrams(self, text: str, min_len: int = 2, max_len: int = 8) -> list[tuple[str, int, int]]:
        """提取 n-gram，返回 (ngram, start, end) 列表"""
        ngrams = []
        for n in range(min_len, min(max_len + 1, len(text) + 1)):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                # 过滤条件
                if self._is_valid_ngram(ngram):
                    ngrams.append((ngram, i, i + n))
        return ngrams
    
    def _is_valid_ngram(self, ngram: str) -> bool:
        """判断 n-gram 是否有效候选"""
        # 过滤纯数字
        if ngram.isdigit():
            return False
        # 过滤纯英文（小于3个字符）
        if ngram.isalpha() and len(ngram) < 3:
            return False
        # 过滤停用词
        if ngram in self.STOP_WORDS:
            return False
        # 过滤包含过多标点的
        if sum(1 for c in ngram if c in '，。；：！？、""（）【】《》') > 0:
            return False
        # 必须包含至少一个中文字符
        if not any('\u4e00' <= c <= '\u9fff' for c in ngram):
            return False
        return True
    
    def matches_anatomy_pattern(self, text: str) -> bool:
        """检查文本是否匹配解剖模式"""
        for pattern in self.compiled_anatomy_patterns:
            if pattern.search(text):
                return True
        return False
    
    def extract_context_candidates(self, sentence: str) -> list[tuple[str, str]]:
        """通过上下文模式提取隐含候选"""
        candidates = []
        for pattern, pattern_name in self.compiled_context_patterns:
            for match in pattern.finditer(sentence):
                candidate = match.group(1)
                if candidate and len(candidate) >= 2:
                    candidates.append((candidate, pattern_name))
        return candidates
    def analyze_sentence(
        self,
        sentence: str,
        report_meta: dict[str, Any],
        extracted_entities: list[dict] | None = None
    ) -> list[EntityCandidate]:
        """分析单个句子，发现候选实体"""
        candidates = []
        extracted_entities = extracted_entities or []
        
        # 1. n-gram 挖掘
        ngrams = self.extract_ngrams(sentence)
        for ngram, start, end in ngrams:
            # 跳过已知的
            if self.is_known_entity(ngram):
                continue
            # 只保留匹配解剖模式的
            if not self.matches_anatomy_pattern(ngram):
                continue
            
            candidate = EntityCandidate(
                hospital_id=report_meta.get('hospital_id'),
                report_id=report_meta.get('report_id'),
                sentence=sentence,
                candidate_text=ngram,
                context_before=None,
                context_after=None,
                StudyPart=report_meta.get('StudyPart', ''),
                modality=report_meta.get('modality', ''),
                matched_by_rule=False,
                current_partlist=None,
                candidate_type='unhit',
                frequency_local=1,
                frequency_global=None,
                discovery_method='ngram'
            )
            candidates.append(candidate)
        
        # 2. 上下文模式挖掘
        context_candidates = self.extract_context_candidates(sentence)
        for candidate_text, method in context_candidates:
            if self.is_known_entity(candidate_text):
                continue
            
            candidate = EntityCandidate(
                hospital_id=report_meta.get('hospital_id'),
                report_id=report_meta.get('report_id'),
                sentence=sentence,
                candidate_text=candidate_text,
                context_before=None,
                context_after=None,
                StudyPart=report_meta.get('StudyPart', ''),
                modality=report_meta.get('modality', ''),
                matched_by_rule=False,
                current_partlist=None,
                candidate_type='implicit',
                frequency_local=1,
                frequency_global=None,
                discovery_method=method
            )
            candidates.append(candidate)
        
        return candidates
    
    def discover_from_reports(
        self,
        reports_data: list[dict[str, Any]],
        min_frequency: int = 3
    ) -> list[EntityCandidate]:
        """
        从报告列表中发现候选实体
        
        Args:
            reports_data: 报告数据列表，每项包含 text, hospital_id, StudyPart, modality等
            min_frequency: 最小频次阈值
        """
        all_candidates: list[EntityCandidate] = []
        
        for report in tqdm(reports_data, desc="Discovering candidates"):
            # 分句
            sentences = re.split(r'[。；!！\n]', report.get('text', ''))
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 5:
                    continue
                
                meta = {
                    'hospital_id': report.get('hospital_id'),
                    'report_id': report.get('report_id'),
                    'StudyPart': report.get('StudyPart', ''),
                    'modality': report.get('modality', ''),
                }
                
                candidates = self.analyze_sentence(sent, meta)
                all_candidates.extend(candidates)
        
        # 频次聚合
        candidate_counts = Counter(c.candidate_text for c in all_candidates)
        
        # 过滤低频
        filtered_candidates = []
        seen = set()
        for c in all_candidates:
            if c.candidate_text in seen:
                continue
            if candidate_counts[c.candidate_text] >= min_frequency:
                c.frequency_local = candidate_counts[c.candidate_text]
                filtered_candidates.append(c)
                seen.add(c.candidate_text)
        
        return filtered_candidates


def discover_candidates(
    input_jsonl: Path,
    output_jsonl: Path,
    alias_index_path: Path | None = None,
    min_frequency: int = 3
) -> int:
    """
    主函数：从输入报告中发现候选实体
    
    Args:
        input_jsonl: 输入报告数据，每行一个JSON对象
        output_jsonl: 输出候选实体文件
        alias_index_path: 图谱别名索引路径
        min_frequency: 最小频次阈值
    """
    # 加载别名索引
    alias_index = {}
    if alias_index_path and alias_index_path.exists():
        alias_index = json.loads(alias_index_path.read_text(encoding="utf-8"))
    
    engine = CandidateDiscoveryEngine(alias_index)
    
    # 读取报告数据
    reports_data = []
    if input_jsonl.exists():
        with input_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    reports_data.append(json.loads(line))
    
    if not reports_data:
        output_jsonl.write_text("", encoding="utf-8")
        return 0
    
    # 发现候选
    candidates = engine.discover_from_reports(reports_data, min_frequency=min_frequency)
    
    # 写入输出
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto KnowledgeGraph candidate discovery")
    parser.add_argument("--input", default="data/source_reports.jsonl", help="输入报告文件")
    parser.add_argument("--output", default="data/entity_candidate_discovery.jsonl", help="输出候选文件")
    parser.add_argument("--alias-index", default="data/graph_alias_index.json", help="图谱别名索引")
    parser.add_argument("--min-freq", type=int, default=3, help="最小频次阈值")
    args = parser.parse_args()

    cfg = load_config()
    ensure_runtime_dirs(cfg)
    
    input_path = cfg.project_root / args.input
    output_path = cfg.project_root / args.output
    alias_index_path = cfg.project_root / args.alias_index
    
    count = discover_candidates(input_path, output_path, alias_index_path, args.min_freq)
    print(f"Discovered {count} entity candidates")


if __name__ == "__main__":
    main()
