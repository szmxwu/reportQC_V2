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

# 性能计时器（用于debug）
_rerank_call_count = 0
_rerank_total_time = 0.0
_embedding_total_time = 0.0

def get_performance_stats():
    """获取性能统计信息"""
    return {
        'rerank_calls': _rerank_call_count,
        'rerank_time': _rerank_total_time,
        'embedding_time': _embedding_total_time
    }

def reset_performance_stats():
    """重置性能统计信息"""
    global _rerank_call_count, _rerank_total_time, _embedding_total_time
    _rerank_call_count = 0
    _rerank_total_time = 0.0
    _embedding_total_time = 0.0


class MatchCandidate(NamedTuple):
    """描述实体与结论实体的匹配候选"""
    report_sentence: str                    # 描述原文(original_short_sentence，用于输出)
    report_sentence_for_match: str          # 描述用于匹配的文本(short_sentence)
    report_orientation: str                 # 描述方位
    report_position: str                    # 描述部位
    conclusion_sentence: str                # 结论原文(original_short_sentence，用于输出)
    conclusion_sentence_for_match: str      # 结论用于匹配的文本(short_sentence)
    conclusion_orientation: str             # 结论方位
    conclusion_position: str                # 结论部位
    semantic_score: float                   # 语义相似度
    is_orient_match: bool                   # 方位是否匹配
    is_forced_match: bool = False           # 是否强制匹配（部位-数量兜底）


class OrientationProbability(NamedTuple):
    """方位错误概率计算结果"""
    report_prob: float            # 相对于描述的概率差
    conclusion_prob: float        # 相对于结论的概率差
    match_prob: float             # 综合匹配概率
    is_error: bool                # 是否判定为错误


def _check_match_with_rerank(report_entity: Dict, conclusion_candidates: List[Dict]) -> Tuple[Optional[Dict], float]:
    """
    使用BGE Rerank模型检查描述和结论是否匹配
    
    使用short_sentence（预处理后补全的文本）进行匹配，
    输出时仍使用original_short_sentence
    
    Returns:
        (best_match, max_score) - 最佳匹配的结论实体和得分
        如果Rerank API不可用，抛出异常
    """
    import time
    global _rerank_call_count, _rerank_total_time, _embedding_total_time
    
    if not conclusion_candidates:
        return None, 0.0
    
    nlp = _import_from_nlp_analyze()
    matcher = nlp.get_semantic_matcher()
    
    if not matcher.available():
        raise RuntimeError("BGE Rerank服务不可用，请检查服务状态")
    
    # 准备查询和候选 - 使用short_sentence进行匹配
    query = report_entity.get('short_sentence', '') or report_entity.get('original_short_sentence', '')
    passages = [c.get('short_sentence', '') or c.get('original_short_sentence', '') for c in conclusion_candidates]
    
    # 计时Rerank调用
    t0 = time.time()
    scores = matcher.rerank(query, passages)
    elapsed = time.time() - t0
    _rerank_call_count += 1
    _rerank_total_time += elapsed
    
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


def _get_part_identifier(entity: Dict) -> str:
    """
    获取部位标识符
    
    优先使用position字段，如果没有则使用partlist的第一个元素
    """
    position = entity.get('position', '')
    if position:
        return position
    
    partlist = entity.get('partlist', [])
    if partlist and partlist[0]:
        if isinstance(partlist[0], (list, tuple)):
            return partlist[0][-1] if partlist[0] else ''
        else:
            return partlist[0]
    return ''


def _count_entities_by_part(entities: List[Dict], part_identifier: str) -> int:
    """
    统计指定部位的实体数量
    
    使用精确匹配，确保只有完全相同的部位才被计数
    （例如：'胆囊' 不会匹配 '胆囊颈'）
    """
    count = 0
    for e in entities:
        e_part = _get_part_identifier(e)
        # 使用精确匹配
        if e_part and part_identifier and e_part == part_identifier:
            count += 1
    return count


def _is_part_match_or_subset(report_entity: Dict, conclusion_entity: Dict) -> bool:
    """
    检查部位是否匹配或是子集关系
    
    支持以下匹配：
    - 完全匹配：'胆囊' = '胆囊'
    - 子部位匹配：'胆囊颈' ⊂ '胆囊'（胆囊颈是胆囊的一部分）
    
    通过比较partlist来判断子集关系
    """
    # 获取partlist
    report_partlist = report_entity.get('partlist', [])
    conclusion_partlist = conclusion_entity.get('partlist', [])
    
    if not report_partlist or not conclusion_partlist:
        # 如果没有partlist，使用position精确匹配
        report_pos = report_entity.get('position', '')
        conclusion_pos = conclusion_entity.get('position', '')
        return report_pos == conclusion_pos
    
    # 检查是否有子集关系
    # 报告实体的partlist是否是结论实体partlist的子集，或反之
    for rpl in report_partlist:
        rset = set(rpl) if isinstance(rpl, (list, tuple)) else {rpl}
        for cpl in conclusion_partlist:
            cset = set(cpl) if isinstance(cpl, (list, tuple)) else {cpl}
            # 检查子集关系（任一是另一者的子集）
            if rset <= cset or cset <= rset:
                return True
    
    return False


def _is_forced_match(
    report_entity: Dict,
    conclusion_entity: Dict,
    all_report_entities: List[Dict],
    all_conclusion_entities: List[Dict]
) -> bool:
    """
    判断是否强制匹配（部位-数量兜底机制）
    
    当满足以下条件时，即使语义相似度低，也默认匹配：
    1. 部位匹配或是子部位（如胆囊颈是胆囊的一部分）
    2. 描述中该部位只有1个实体（避免多实体歧义）
    3. 描述和结论都是阳性（positive>1）
    
    注意：结论中可以有多个同部位实体（如"胆囊结石，胆囊炎"），
    只要描述中只有1个，就默认匹配，解决医学知识不足的问题。
    
    这用于解决医学知识匹配问题，例如：
    - 描述："胆囊体积增大" 或 "胆囊颈部见高密度影"
    - 结论："胆囊炎"（或"胆囊结石，胆囊炎"）
    - Rerank可能认为不匹配，但医学上它们是对应的
    """
    # 检查部位是否匹配或是子集关系
    if not _is_part_match_or_subset(report_entity, conclusion_entity):
        return False
    
    # 获取报告实体的部位标识（用于计数）
    report_part = _get_part_identifier(report_entity)
    if not report_part:
        return False
    
    # 统计该部位在描述中的实体数量
    report_count = _count_entities_by_part(all_report_entities, report_part)
    
    # 描述中只有1个该部位实体，且是阳性
    # 结论中可以有多个（如结石+炎症），默认不是缺失
    if report_count == 1:
        if report_entity.get('positive', 0) > 1 and conclusion_entity.get('positive', 0) > 1:
            return True
    
    return False


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
    """
    计算两个实体的语义相似度
    
    使用short_sentence（预处理后补全的文本）进行匹配，
    以解决略写问题，如：
    - original_short_sentence: "颈部见高密度影"
    - short_sentence: "胆囊颈见高密度影"
    """
    nlp = _import_from_nlp_analyze()
    
    # 优先使用short_sentence（预处理后补全的文本）
    report_sent = report_entity.get('short_sentence', '')
    conclusion_sent = conclusion_entity.get('short_sentence', '')
    
    # 如果short_sentence为空，回退到original_short_sentence
    if not report_sent:
        report_sent = report_entity.get('original_short_sentence', '')
    if not conclusion_sent:
        conclusion_sent = conclusion_entity.get('original_short_sentence', '')
    
    if not report_sent or not conclusion_sent:
        return 0.0
    
    # 创建临时实体用于struc_sim计算
    report_temp = report_entity.copy()
    report_temp['original_short_sentence'] = report_sent
    
    conclusion_temp = conclusion_entity.copy()
    conclusion_temp['original_short_sentence'] = conclusion_sent
    
    return nlp.struc_sim(conclusion_temp, report_temp)


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
                # 获取用于输出和用于匹配的句子
                report_orig = report_entity.get('original_short_sentence', '').split(",")[0]
                report_match = report_entity.get('short_sentence', '') or report_orig
                
                conclusion_orig = conclusion_entity.get('original_short_sentence', '').split(",")[0]
                conclusion_match = conclusion_entity.get('short_sentence', '') or conclusion_orig
                
                # 检查是否强制匹配
                is_forced = _is_forced_match(
                    report_entity, conclusion_entity,
                    report_entities, conclusion_entities
                )
                
                candidates.append(MatchCandidate(
                    report_sentence=report_orig,
                    report_sentence_for_match=report_match,
                    report_orientation=report_entity.get('orientation', ''),
                    report_position=report_entity.get('position', ''),
                    conclusion_sentence=conclusion_orig,
                    conclusion_sentence_for_match=conclusion_match,
                    conclusion_orientation=conclusion_entity.get('orientation', ''),
                    conclusion_position=conclusion_entity.get('position', ''),
                    semantic_score=semantic_score,
                    is_orient_match=is_orient_match,
                    is_forced_match=is_forced
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
    position_matches: List[Dict],
    all_report_entities: List[Dict] = None,
    all_conclusion_entities: List[Dict] = None
) -> bool:
    """
    使用Rerank判断描述实体是否有有效匹配
    
    当Rerank认为不匹配时，会检查是否满足强制匹配条件（部位-数量兜底机制）：
    - 部位匹配
    - 该部位在描述和结论中都只有1个实体
    - 都是阳性实体
    
    Returns:
        True - 有有效匹配（无论方位是否匹配，包括强制匹配）
        False - 无有效匹配（可能是真缺失）
    
    Raises:
        RuntimeError - Rerank服务不可用
    """
    if not position_matches:
        return False
    
    if not USE_BGE_RERANK:
        # 不使用Rerank时，检查是否有方位匹配的候选
        return any(
            _is_orient_match(report_entity.get('orientation', ''), 
                           c.get('orientation', ''))
            for c in position_matches
        )
    
    best_match, score = _check_match_with_rerank(report_entity, position_matches)
    # Rerank找到匹配 = 有对应内容
    if best_match is not None:
        return True
    
    # Rerank没找到 = 检查是否满足强制匹配条件（部位-数量兜底）
    if all_report_entities is not None and all_conclusion_entities is not None:
        for conc_entity in position_matches:
            if _is_forced_match(report_entity, conc_entity, 
                               all_report_entities, all_conclusion_entities):
                return True
    
    # 没有找到匹配
    return False


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
            
            # 使用Rerank判断是否有有效匹配（传递所有实体用于强制匹配检查）
            if not has_valid_rerank_match(entity, position_matches, report_entities, conclusion_entities):
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
                elif not has_valid_rerank_match(entity, position_matches, report_entities, conclusion_entities):
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
) -> Tuple[List[str], List[str], Dict]:
    """
    检查报告描述与结论的方位、部位匹配
    
    Args:
        conclusion_analyze: 解析后的结论实体列表
        report_analyze: 解析后的描述实体列表
        modality: 设备类型
    
    Returns:
        (conclusion_missing, orient_error, performance_stats)
        - conclusion_missing: 缺失结论的列表
        - orient_error: 方位错误的列表
        - performance_stats: 性能统计信息
    
    Raises:
        RuntimeError: 当Rerank服务不可用时
    """
    import time
    
    # 重置性能统计
    reset_performance_stats()
    
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
    
    # 获取性能统计
    perf_stats = get_performance_stats()
    
    return conclusion_missing, orient_error, perf_stats
