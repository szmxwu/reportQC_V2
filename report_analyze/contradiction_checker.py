# -*- coding: utf-8 -*-
"""
报告矛盾检测模块

重构目标：
1. 简化方位处理 - 避免重复遍历
2. 收集所有矛盾 - 不只是第一个
3. 使用Rerank验证 - 替换Word2Vec
4. 清晰命名和数据结构
"""

import re
import os
from typing import List, Dict, NamedTuple, Optional, Tuple
from enum import Enum

# 延迟导入避免循环依赖
def _import_from_nlp_analyze():
    """延迟导入避免循环依赖"""
    import sys
    nlp_module = sys.modules.get('NLP_analyze')
    if nlp_module is None:
        import importlib
        nlp_module = importlib.import_module('NLP_analyze')
    return nlp_module


# Rerank配置
USE_BGE_RERANK = os.getenv('USE_BGE_RERANK', 'true').lower() == 'true'
RERANK_CONTRADICTION_THRESHOLD = float(os.getenv('RERANK_CONTRADICTION_THRESHOLD', '0.35'))


class SentimentType(Enum):
    """实体情感类型"""
    POSITIVE = 1    # 阳性/异常 (positive > 1)
    NEGATIVE = 2    # 阴性/正常 (positive <= 1)
    UNKNOWN = 3     # 无法判断


class EntityGroup(NamedTuple):
    """按部位和方位分组的实体"""
    part: str                           # 部位名称
    orientation: str                    # 方位（左/右/双/空）
    positive_entities: List[Dict]       # 阳性实体
    negative_entities: List[Dict]       # 阴性实体


class ContradictionPair(NamedTuple):
    """矛盾对"""
    negative_sentence: str      # 阴性描述
    positive_sentence: str      # 阳性描述
    confidence: float           # 置信度（Rerank得分）
    part: str                   # 所属部位


def _classify_entity_sentiment(entity: Dict) -> SentimentType:
    """
    判断实体的情感类型（阳性/阴性）
    
    阳性：positive > 1 且 不在排除列表
    阴性：positive <= 1 且 无测量值 且 illness有效
    """
    nlp = _import_from_nlp_analyze()
    
    # 获取配置
    exclud = getattr(nlp, 'exclud', '')
    stopwords = getattr(nlp, 'stopwords', '')
    
    sentence = entity.get('original_short_sentence', '')
    
    # 检查排除列表
    if exclud and re.search(rf"{exclud}", sentence):
        return SentimentType.UNKNOWN
    
    positive = entity.get('positive', 0)
    
    if positive > 1:
        return SentimentType.POSITIVE
    
    # 阴性判断：无测量值且illness有效
    measure = entity.get('measure', 0)
    illness = entity.get('illness', '')
    
    if measure == 0 and illness:
        # 去除stopwords后检查是否还有内容
        clean_illness = re.sub(stopwords, "", illness) if stopwords else illness
        if len(clean_illness) > 1:
            return SentimentType.NEGATIVE
    
    return SentimentType.UNKNOWN


def _group_entities_by_part_orient(
    entities: List[Dict],
    key_parts: List[str]
) -> List[EntityGroup]:
    """
    按部位和方位对实体进行分组
    
    优化：避免原实现中左右双各遍历一次的重复
    """
    nlp = _import_from_nlp_analyze()
    groups = []
    
    for part in key_parts:
        # 获取该部位的所有实体
        part_entities = [
            e for e in entities 
            if nlp.position_in_any_partlist(part, e)
        ]
        
        if not part_entities:
            continue
        
        # 按方位分组
        left = [e for e in part_entities if e.get('orientation') == '左']
        right = [e for e in part_entities if e.get('orientation') == '右']
        both = [e for e in part_entities if e.get('orientation') == '双']
        none = [e for e in part_entities if e.get('orientation') == '' or e.get('orientation') is None]
        
        # 左方位组：左 + 双 + 无方位
        left_pos = []
        left_neg = []
        for e in left + both + none:
            sentiment = _classify_entity_sentiment(e)
            if sentiment == SentimentType.POSITIVE:
                left_pos.append(e)
            elif sentiment == SentimentType.NEGATIVE:
                left_neg.append(e)
        
        if left_pos and left_neg:
            groups.append(EntityGroup(part, '左', left_pos, left_neg))
        
        # 右方位组：右 + 双 + 无方位
        right_pos = []
        right_neg = []
        for e in right + both + none:
            sentiment = _classify_entity_sentiment(e)
            if sentiment == SentimentType.POSITIVE:
                right_pos.append(e)
            elif sentiment == SentimentType.NEGATIVE:
                right_neg.append(e)
        
        if right_pos and right_neg:
            groups.append(EntityGroup(part, '右', right_pos, right_neg))
    
    return groups


def _extract_sub_parts(sentence: str, sub_part_pattern: str) -> List[str]:
    """从句子中提取子部位"""
    if not sub_part_pattern:
        return []
    return re.findall(sub_part_pattern, sentence)


def _has_sub_part_overlap(neg_entity: Dict, pos_entity: Dict, sub_part_pattern: str) -> bool:
    """
    检查两个实体是否有子部位重叠
    
    规则：任一为空 或 存在交集 → True
    """
    neg_sub = _extract_sub_parts(neg_entity.get('original_short_sentence', ''), sub_part_pattern)
    pos_sub = _extract_sub_parts(pos_entity.get('original_short_sentence', ''), sub_part_pattern)
    
    # 任一为空，或存在交集
    return not neg_sub or not pos_sub or bool(set(neg_sub) & set(pos_sub))


def _filter_by_aspect(
    positive_entities: List[Dict],
    negative_entity: Dict,
    aspects: List[str]
) -> List[Dict]:
    """
    根据aspect过滤阳性实体
    
    如果阴性实体包含某个aspect，只保留同样包含该aspect的阳性实体
    """
    if not aspects:
        return positive_entities
    
    neg_sentence = negative_entity.get('original_short_sentence', '')
    
    # 检查阴性实体包含哪些aspect
    matched_aspects = []
    for aspect in aspects:
        if re.search(rf"{aspect}", neg_sentence):
            matched_aspects.append(aspect)
    
    if not matched_aspects:
        # 没有匹配到aspect，返回所有
        return positive_entities
    
    # 只保留包含相同aspect的阳性实体
    filtered = []
    for pos_entity in positive_entities:
        pos_sentence = pos_entity.get('original_short_sentence', '')
        if any(re.search(rf"{aspect}", pos_sentence) for aspect in matched_aspects):
            filtered.append(pos_entity)
    
    return filtered if filtered else positive_entities


def _verify_contradiction_with_rerank(
    neg_sentence: str,
    pos_sentence: str
) -> float:
    """
    使用Rerank验证两句话是否矛盾
    
    Returns:
        矛盾置信度（0-1），低于阈值表示不矛盾
        
    Raises:
        RuntimeError: Rerank服务不可用
    """
    if not USE_BGE_RERANK:
        # 不使用Rerank时，返回中等置信度（由调用方决定是否保留）
        return 0.5
    
    nlp = _import_from_nlp_analyze()
    matcher = nlp.get_semantic_matcher()
    
    if not matcher.available():
        raise RuntimeError("BGE Rerank服务不可用，请检查服务状态")
    
    # Step 1: 语义相似度检查
    emb1 = matcher.get_embedding(neg_sentence)
    emb2 = matcher.get_embedding(pos_sentence)
    
    if emb1 is None or emb2 is None:
        return 0.0
    
    sim = matcher.cosine_sim(emb1, emb2)
    
    # 相似度太低，描述的是不同病变 → 不矛盾
    if sim < 0.3:
        return 0.0
    
    # 高度相似，可能是重复描述 → 不矛盾
    if sim > 0.85:
        return 0.0
    
    # Step 2: Rerank判断语义关系
    query = f"描述1：{neg_sentence}\n描述2：{pos_sentence}"
    passages = [
        "这两句话描述矛盾，一个是正常/阴性，一个是异常/阳性",
        "这两句话描述不矛盾，可能是同一事物的不同角度描述"
    ]
    
    scores = matcher.rerank(query, passages)
    
    if not scores or len(scores) < 2:
        return 0.0
    
    contradiction_score = scores[0]
    non_contradiction_score = scores[1]
    
    # 矛盾得分 > 阈值 且 高于不矛盾得分
    if contradiction_score > RERANK_CONTRADICTION_THRESHOLD and \
       contradiction_score > non_contradiction_score + 0.05:
        return contradiction_score
    
    return 0.0


def _find_contradictions_in_group(
    group: EntityGroup,
    aspects: List[str],
    sub_part_pattern: str
) -> List[ContradictionPair]:
    """
    在实体组内查找所有矛盾对
    
    改进：收集所有矛盾，不只是第一个
    """
    contradictions = []
    
    for neg_entity in group.negative_entities:
        # 1. 按aspect过滤阳性实体
        candidates = _filter_by_aspect(group.positive_entities, neg_entity, aspects)
        
        # 2. 按部位匹配（子部位重叠）
        for pos_entity in candidates:
            # 检查部位是否匹配
            nlp = _import_from_nlp_analyze()
            if not nlp.position_in_any_partlist(neg_entity.get('position', ''), pos_entity):
                continue
            
            # 检查子部位重叠
            if not _has_sub_part_overlap(neg_entity, pos_entity, sub_part_pattern):
                continue
            
            # 3. Rerank验证
            try:
                confidence = _verify_contradiction_with_rerank(
                    neg_entity['original_short_sentence'],
                    pos_entity['original_short_sentence']
                )
            except RuntimeError:
                # Rerank不可用，使用基础判断
                confidence = 0.5
            
            if confidence > 0:
                contradictions.append(ContradictionPair(
                    negative_sentence=neg_entity['original_short_sentence'],
                    positive_sentence=pos_entity['original_short_sentence'],
                    confidence=confidence,
                    part=group.part
                ))
    
    # 按置信度降序排序
    return sorted(contradictions, key=lambda x: x.confidence, reverse=True)


def check_contradiction(
    report_analyze: List[Dict],
    conclusion_analyze: List[Dict],
    modality: str
) -> List[str]:
    """
    检查报告描述与结论中的矛盾语句（新实现）
    
    Args:
        report_analyze: 解析后的描述实体列表
        conclusion_analyze: 解析后的结论实体列表
        modality: 设备类型
    
    Returns:
        List[str]: 矛盾列表，格式与原函数兼容 [stmt1, stmt2, stmt3, stmt4, ...]
        按置信度降序排列，每对矛盾连续出现
    
    Raises:
        RuntimeError: 当Rerank服务不可用时（如启用Rerank）
    """
    nlp = _import_from_nlp_analyze()
    
    # 获取配置
    check_modality = getattr(nlp, 'check_modality', 'CT|MR|DR|MG')
    key_part = getattr(nlp, 'key_part', [])
    aspects = getattr(nlp, 'aspects', [])
    sub_part = getattr(nlp, 'sub_part', '')
    
    # 前置检查
    if re.search(check_modality, modality) is None:
        return []
    
    # 合并并过滤实体
    all_entities = []
    all_entities.extend([x for x in report_analyze if x.get('ignore') == False])
    all_entities.extend([x for x in conclusion_analyze if x.get('ignore') == False])
    
    if not all_entities:
        return []
    
    # 按部位和方位分组
    groups = _group_entities_by_part_orient(all_entities, key_part)
    
    # 收集所有矛盾
    all_contradictions = []
    for group in groups:
        pairs = _find_contradictions_in_group(group, aspects, sub_part)
        all_contradictions.extend(pairs)
    
    # 按置信度全局排序
    all_contradictions.sort(key=lambda x: x.confidence, reverse=True)
    
    # 转换为原格式 [stmt1, stmt2, stmt3, stmt4, ...]
    result = []
    for pair in all_contradictions:
        result.extend([pair.negative_sentence, pair.positive_sentence])
    
    return result
