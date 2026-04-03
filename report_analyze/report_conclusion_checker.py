# -*- coding: utf-8 -*-
"""
报告描述与结论匹配检查模块

重构目标：
1. 简化逻辑 - 只保留Rerank，移除Word2Vec回退
2. 清晰职责 - 数据收集、缺失检测、方位错误检测分离
3. 消除状态变量 - 使用纯函数，减少标志位
"""

import re
import os
from typing import List, Dict, NamedTuple, Optional, Tuple
from dataclasses import dataclass

# 从NLP_analyze导入必要的依赖
# 注意：这些是运行时导入，避免循环依赖

def _import_from_nlp_analyze():
    """延迟导入避免循环依赖"""
    import sys
    nlp_module = sys.modules.get('NLP_analyze')
    if nlp_module is None:
        # 动态导入
        import importlib
        nlp_module = importlib.import_module('NLP_analyze')
    return nlp_module


# Rerank配置
USE_BGE_RERANK = os.getenv('USE_BGE_RERANK', 'true').lower() == 'true'
RERANK_THRESHOLD = float(os.getenv('RERANK_THRESHOLD', '0.7'))


class MatchCandidate(NamedTuple):
    """描述实体与结论实体的匹配候选"""
    report_sentence: str          # 描述原文
    report_orientation: str       # 描述方位
    report_position: str          # 描述部位
    conclusion_sentence: str      # 结论原文
    conclusion_orientation: str   # 结论方位
    conclusion_position: str      # 结论部位
    semantic_score: float         # 语义相似度
    is_orient_match: bool         # 方位是否匹配


class OrientationProbability(NamedTuple):
    """方位错误概率计算结果"""
    report_prob: float            # 相对于描述的概率差
    conclusion_prob: float        # 相对于结论的概率差
    match_prob: float             # 综合匹配概率
    is_error: bool                # 是否判定为错误


def _check_match_with_rerank(report_entity: Dict, conclusion_candidates: List[Dict]) -> Tuple[Optional[Dict], float]:
    """
    使用BGE Rerank模型检查描述和结论是否匹配
    
    Returns:
        (best_match, max_score) - 最佳匹配的结论实体和得分
        如果Rerank API不可用，抛出异常
    """
    if not conclusion_candidates:
        return None, 0.0
    
    nlp = _import_from_nlp_analyze()
    matcher = nlp.get_semantic_matcher()
    
    if not matcher.available():
        raise RuntimeError("BGE Rerank服务不可用，请检查服务状态")
    
    # 准备查询和候选
    query = report_entity['original_short_sentence']
    passages = [c['original_short_sentence'] for c in conclusion_candidates]
    
    # 调用Rerank API
    scores = matcher.rerank(query, passages)
    
    if not scores:
        return None, 0.0
    
    max_score = max(scores)
    best_idx = scores.index(max_score)
    
    if max_score >= RERANK_THRESHOLD:
        return conclusion_candidates[best_idx], max_score
    else:
        return None, max_score


def _is_orient_match(orient1: str, orient2: str) -> bool:
    """检查两个方位是否匹配（不是左右相反）"""
    return not ((orient1 == '左' and orient2 == '右') or 
                (orient1 == '右' and orient2 == '左'))


def _position_matches(report_entity: Dict, conclusion_entity: Dict, upper_position_pattern: str) -> bool:
    """
    检查描述实体和结论实体的部位是否匹配
    
    对于上位部位（如"腹"、"胸"等），使用子集匹配
    对于普通部位，也使用子集匹配（任一partlist子列表满足即可）
    """
    nlp = _import_from_nlp_analyze()
    
    # 统一使用子集匹配（与原实现一致）
    # 检查结论的任一partlist是否是描述的任一partlist的子集，或反之
    return (nlp.any_partlist_is_subset(conclusion_entity, report_entity) or 
            nlp.any_partlist_is_subset(report_entity, conclusion_entity))


def _calc_semantic_score(report_entity: Dict, conclusion_entity: Dict) -> float:
    """计算两个实体的语义相似度"""
    nlp = _import_from_nlp_analyze()
    return nlp.struc_sim(conclusion_entity, report_entity)


def _should_skip_for_orient_error(report_entity: Dict, conclusion_entity: Dict, 
                                   orient_ignore_pattern: str) -> bool:
    """检查是否应该跳过方位错误检测"""
    report_sent = report_entity.get('original_short_sentence', '')
    conclusion_sent = conclusion_entity.get('original_short_sentence', '')
    
    if orient_ignore_pattern and (re.search(orient_ignore_pattern, report_sent) or \
       re.search(orient_ignore_pattern, conclusion_sent)):
        return True
    
    # 检查部位是否匹配（使用子集匹配，与_position_matches一致）
    nlp = _import_from_nlp_analyze()
    if not (nlp.any_partlist_is_subset(conclusion_entity, report_entity) or 
            nlp.any_partlist_is_subset(report_entity, conclusion_entity)):
        return True
    
    return False


def collect_match_candidates(
    report_entities: List[Dict],
    conclusion_entities: List[Dict],
    upper_position_pattern: str,
    orient_ignore_pattern: str
) -> List[MatchCandidate]:
    """
    收集描述与结论之间的所有匹配候选
    
    遍历所有描述实体，找到部位匹配的所有结论实体候选，计算语义相似度
    """
    candidates = []
    
    for report_entity in report_entities:
        if not report_entity.get('illness'):
            continue
            
        # 找到所有部位匹配的结论实体
        for conclusion_entity in conclusion_entities:
            if not _position_matches(report_entity, conclusion_entity, upper_position_pattern):
                continue
            
            # 计算语义相似度
            semantic_score = _calc_semantic_score(report_entity, conclusion_entity)
            
            # 判断方位是否匹配
            is_orient_match = _is_orient_match(
                report_entity.get('orientation', ''),
                conclusion_entity.get('orientation', '')
            )
            
            # 判断是否跳过方位错误检测
            skip_orient_check = _should_skip_for_orient_error(
                report_entity, conclusion_entity, orient_ignore_pattern
            )
            
            # 只有positive>1的实体才加入方位错误检测列表
            should_add_for_orient = (report_entity.get('positive', 0) > 1 and 
                                     not skip_orient_check and 
                                     semantic_score > 0)
            
            if should_add_for_orient:
                candidates.append(MatchCandidate(
                    report_sentence=report_entity['original_short_sentence'].split(",")[0],
                    report_orientation=report_entity.get('orientation', ''),
                    report_position=report_entity.get('position', ''),
                    conclusion_sentence=conclusion_entity['original_short_sentence'].split(",")[0],
                    conclusion_orientation=conclusion_entity.get('orientation', ''),
                    conclusion_position=conclusion_entity.get('position', ''),
                    semantic_score=semantic_score,
                    is_orient_match=is_orient_match
                ))
    
    return candidates


def find_position_matches(
    report_entity: Dict,
    conclusion_entities: List[Dict],
    upper_position_pattern: str
) -> List[Dict]:
    """找到与描述实体部位匹配的所有结论实体"""
    matches = []
    for conclusion_entity in conclusion_entities:
        if _position_matches(report_entity, conclusion_entity, upper_position_pattern):
            matches.append(conclusion_entity)
    return matches


def has_valid_rerank_match(
    report_entity: Dict,
    position_matches: List[Dict]
) -> bool:
    """
    使用Rerank判断描述实体是否有有效匹配
    
    Returns:
        True - 有有效匹配（无论方位是否匹配）
        False - 无有效匹配（可能是真缺失）
    
    Raises:
        RuntimeError - Rerank服务不可用
    """
    if not USE_BGE_RERANK:
        # 不使用Rerank时，检查是否有方位匹配的候选
        return any(
            _is_orient_match(report_entity.get('orientation', ''), 
                           c.get('orientation', ''))
            for c in position_matches
        )
    
    best_match, score = _check_match_with_rerank(report_entity, position_matches)
    
    # Rerank找到匹配 = 有对应内容（无论方位是否匹配）
    # Rerank没找到 = 可能是真缺失
    return best_match is not None


def detect_missing_conclusions(
    report_entities: List[Dict],
    conclusion_entities: List[Dict],
    upper_position_pattern: str,
    miss_ignore_pattern: re.Pattern
) -> List[str]:
    """
    检测结论缺失
    
    对于每个positive>1的描述实体，检查在结论中是否有对应
    """
    missing = []
    
    # 按句子start分组处理
    sentenc_starts = set(d.get('start') for d in report_entities if d.get('start') is not None)
    
    for start in sentenc_starts:
        # 获取该start位置的所有实体（排除空illness的）
        entities = [
            d for d in report_entities 
            if d.get('start') == start and d.get('illness')
        ]
        
        if not entities:
            continue
        
        # 检查该句子中是否有实体缺失结论
        has_missing_in_sentence = False
        
        for entity in entities:
            if entity.get('positive', 0) <= 1:
                continue
            if entity.get('ignore', False):
                continue
            
            # 找到部位匹配的结论实体
            position_matches = find_position_matches(entity, conclusion_entities, upper_position_pattern)
            
            if not position_matches:
                # 没有部位匹配，可能是真缺失
                has_missing_in_sentence = True
                continue
            
            # 使用Rerank判断是否有有效匹配
            if not has_valid_rerank_match(entity, position_matches):
                # Rerank认为没有匹配，可能是真缺失
                has_missing_in_sentence = True
        
        if has_missing_in_sentence:
            # 收集该句子中所有可能缺失的实体
            for entity in entities:
                if entity.get('positive', 0) <= 1:
                    continue
                if entity.get('ignore', False):
                    continue
                
                position_matches = find_position_matches(entity, conclusion_entities, upper_position_pattern)
                
                if not position_matches:
                    missing.append(entity['original_short_sentence'])
                elif not has_valid_rerank_match(entity, position_matches):
                    missing.append(entity['original_short_sentence'])
    
    # 去重并过滤
    missing = list(set(missing))
    if miss_ignore_pattern.pattern:
        missing = [x for x in missing if not miss_ignore_pattern.search(x)]
    
    return missing


def calc_orientation_probability(
    mismatch: MatchCandidate,
    orient_matches: List[MatchCandidate]
) -> OrientationProbability:
    """
    计算方位错误的概率
    
    对比方位不匹配的候选与方位匹配的候选的语义相似度差异
    """
    # 获取该描述在方位匹配中的最高语义相似度
    if orient_matches:
        max_report_sim = max(m.semantic_score for m in orient_matches)
    else:
        max_report_sim = 0
    
    # 计算概率差
    report_prob = mismatch.semantic_score - max_report_sim
    
    # 对于结论侧，简化处理：假设最高相似度为0.5（经验值）
    conclusion_prob = 1.0 if not orient_matches else max(0, mismatch.semantic_score - 0.5)
    
    match_prob = report_prob + conclusion_prob
    
    # 判定规则（与原逻辑保持一致）
    is_error = (
        (conclusion_prob > 0.006 and mismatch.semantic_score >= 0.5) or
        (conclusion_prob == 1.0 and mismatch.semantic_score >= 0.4)
    ) and report_prob > 0
    
    return OrientationProbability(
        report_prob=report_prob,
        conclusion_prob=conclusion_prob,
        match_prob=match_prob,
        is_error=is_error
    )


def detect_orientation_errors(candidates: List[MatchCandidate]) -> List[str]:
    """
    检测方位错误
    
    对方位不匹配的候选，计算其与方位匹配候选的相对概率
    """
    if not candidates:
        return []
    
    # 按描述句子分组
    by_report: Dict[str, List[MatchCandidate]] = {}
    for c in candidates:
        if c.report_sentence not in by_report:
            by_report[c.report_sentence] = []
        by_report[c.report_sentence].append(c)
    
    # 收集方位错误
    errors = []
    
    for report_sent, group in by_report.items():
        # 分离方位匹配和不匹配的
        orient_matches = [c for c in group if c.is_orient_match]
        orient_mismatches = [c for c in group if not c.is_orient_match]
        
        for mismatch in orient_mismatches:
            prob = calc_orientation_probability(mismatch, orient_matches)
            
            if prob.is_error:
                # 检查是否包含左右关键字
                has_orient_keyword = (
                    re.search("左|右", mismatch.report_sentence) or 
                    re.search("左|右", mismatch.conclusion_sentence)
                )
                
                if has_orient_keyword:
                    error_text = f"[描述]{mismatch.report_sentence}；[结论]{mismatch.conclusion_sentence}"
                    errors.append(error_text)
    
    # 去重：对于同一个描述，只保留概率最高的错误
    error_by_report: Dict[str, Tuple[str, float]] = {}
    for i, error_text in enumerate(errors):
        # 找到对应的candidate获取概率
        report_part = error_text.split("；[结论]")[0].replace("[描述]", "")
        
        for c in candidates:
            if c.report_sentence == report_part and not c.is_orient_match:
                prob = calc_orientation_probability(c, 
                    [x for x in candidates if x.report_sentence == report_part and x.is_orient_match])
                if report_part not in error_by_report or prob.match_prob > error_by_report[report_part][1]:
                    error_by_report[report_part] = (error_text, prob.match_prob)
                break
    
    return [v[0] for v in error_by_report.values()]


def check_report_conclusion(
    conclusion_analyze: List[Dict],
    report_analyze: List[Dict],
    modality: str
) -> Tuple[List[str], List[str]]:
    """
    检查报告描述与结论的方位、部位匹配
    
    Args:
        conclusion_analyze: 解析后的结论实体列表
        report_analyze: 解析后的描述实体列表
        modality: 设备类型
    
    Returns:
        (conclusion_missing, orient_error)
        - conclusion_missing: 缺失结论的列表
        - orient_error: 方位错误的列表
    
    Raises:
        RuntimeError: 当Rerank服务不可用时
    """
    nlp = _import_from_nlp_analyze()
    
    # 获取配置
    check_modality = nlp.check_modality if hasattr(nlp, 'check_modality') else 'CT|MR|DR|MG'
    miss_ignore_pattern = nlp.miss_ignore_pattern if hasattr(nlp, 'miss_ignore_pattern') else re.compile('')
    orient_ignore = nlp.orient_ignore if hasattr(nlp, 'orient_ignore') else ''
    upper_position = nlp.upper_position if hasattr(nlp, 'upper_position') else ''
    
    # 前置检查
    if re.search(check_modality, modality) is None:
        return [], []
    
    if not report_analyze:
        return [], []
    
    if not conclusion_analyze:
        # 结论为空，所有positive>1的描述都是缺失
        missing = [
            d['original_short_sentence'] 
            for d in report_analyze 
            if d.get('positive', 0) > 1 and not miss_ignore_pattern.search(d.get('original_short_sentence', ''))
        ]
        return list(set(missing)), []
    
    # 步骤1：收集所有匹配候选（用于方位错误检测）
    candidates = collect_match_candidates(
        report_analyze, 
        conclusion_analyze,
        upper_position,
        orient_ignore
    )
    
    # 步骤2：检测结论缺失
    conclusion_missing = detect_missing_conclusions(
        report_analyze,
        conclusion_analyze,
        upper_position,
        miss_ignore_pattern
    )
    
    # 步骤3：检测方位错误
    orient_error = detect_orientation_errors(candidates)
    
    # 步骤4：如果某描述被判为方位错误，则从缺失列表中移除
    for error_text in orient_error:
        report_part = error_text.split("；[结论]")[0].replace("[描述]", "")
        conclusion_missing = [x for x in conclusion_missing if report_part not in x]
    
    return conclusion_missing, orient_error
