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
import numpy as np
from typing import List, Dict, NamedTuple, Optional, Tuple
from dataclasses import dataclass
from scipy import spatial

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


# 术后相关配置
POSTOPERATIVE_THRESHOLD = float(os.getenv('POSTOPERATIVE_THRESHOLD', '0.5'))
POSTOPERATIVE_PATTERNS = ['术后', '术后改变', '术后复查', '术区', '术后状态', '术后所见']


# Word2Vec模型（延迟加载）
_wv_model = None
_jieba = None

def _get_word2vec_model():
    """延迟加载Word2Vec模型"""
    global _wv_model, _jieba
    if _wv_model is None:
        try:
            from gensim.models import Word2Vec
            import jieba
            _jieba = jieba
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'finetuned_word2vec.m')
            if os.path.exists(model_path):
                _wv_model = Word2Vec.load(model_path)
            else:
                print(f"Word2Vec模型文件不存在: {model_path}")
        except Exception as e:
            print(f"Word2Vec模型加载失败: {e}")
    return _wv_model


def _sentence_semantics_w2v(sentence1: str, sentence2: str) -> float:
    """
    使用Word2Vec计算句子语义相似度
    
    Returns:
        相似度分数 (0-1)，-1表示模型不可用
    """
    model = _get_word2vec_model()
    if model is None:
        return -1
    
    # 获取停用词
    from .config import SystemConfig
    stopwords = SystemConfig.semantics_stopwords() if hasattr(SystemConfig, 'semantics_stopwords') else ''
    
    # 清洗句子
    if stopwords:
        s1 = re.sub(rf"{stopwords}", '', sentence1)
        s2 = re.sub(rf"{stopwords}", '', sentence2)
    else:
        s1, s2 = sentence1, sentence2
    
    if not s1 or not s2:
        return 0.0
    
    # 分词并计算平均向量
    def get_avg_vector(text):
        words = _jieba.lcut(text)
        vectors = []
        for word in words:
            if word in model.wv:
                vectors.append(model.wv[word])
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)
    
    vec1 = get_avg_vector(s1)
    vec2 = get_avg_vector(s2)
    
    if np.sum(vec1) == 0 or np.sum(vec2) == 0:
        return 0.0
    
    # 计算余弦相似度
    sim = 1 - spatial.distance.cosine(vec1, vec2)
    
    # 方位一致性调整
    if sim > 0.5:
        s1_o = re.findall("[左右]", sentence1)
        s2_o = re.findall("[左右]", sentence2)
        if s1_o != s2_o:
            sim -= 0.05
    
    return sim


# 性能计时器（用于debug）
_rerank_call_count = 0
_rerank_total_time = 0.0
_embedding_total_time = 0.0
_semantic_sim_count = 0
_semantic_sim_time = 0.0


def _map_positive_value(positive: int) -> int:
    """
    映射 positive 值以兼容原设计逻辑
    
    新版 Extract_Entities:
      positive=0,1 → 阴性/正常 (映射为 0)
      positive=2,3 → 阳性/异常 (映射为 1)
    """
    if positive is None:
        return 0
    return 1 if positive >= 2 else 0


def get_performance_stats():
    """获取性能统计信息"""
    return {
        'rerank_calls': _rerank_call_count,
        'rerank_time': _rerank_total_time,
        'embedding_time': _embedding_total_time,
        'semantic_sim_calls': _semantic_sim_count,
        'semantic_sim_time': _semantic_sim_time
    }

def reset_performance_stats():
    """重置性能统计信息"""
    global _rerank_call_count, _rerank_total_time, _embedding_total_time
    global _semantic_sim_count, _semantic_sim_time
    _rerank_call_count = 0
    _rerank_total_time = 0.0
    _embedding_total_time = 0.0
    _semantic_sim_count = 0
    _semantic_sim_time = 0.0


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


def _is_orient_match(orient1: str, orient2: str) -> bool:
    """检查两个方位是否匹配（不是左右相反）"""
    return not ((orient1 == '左' and orient2 == '右') or 
                (orient1 == '右' and orient2 == '左'))


def _get_part_identifier(entity: Dict) -> str:
    """
    获取部位标识符
    
    优先使用position字段，如果没有则使用partlist的最后一个元素
    """
    position = entity.get('position', '')
    if position:
        return position
    
    partlist = entity.get('partlist', [])
    if partlist and partlist[0]:
        if isinstance(partlist[0], (list, tuple)):
            return partlist[0][-1] if partlist[0] else ''
        else:
            return partlist[-1]
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
    3. 描述和结论都是阳性（映射后的 positive=1）
    4. **方位不相反**（关键：方位相反时不强制匹配，用于方位错误检测）
    
    注意：结论中可以有多个同部位实体（如"胆囊结石，胆囊炎"），
    只要描述中只有1个，就默认匹配，解决医学知识不足的问题。
    
    这用于解决医学知识匹配问题，例如：
    - 描述："胆囊体积增大" 或 "胆囊颈部见高密度影"
    - 结论："胆囊炎"（或"胆囊结石，胆囊炎"）
    - Rerank可能认为不匹配，但医学上它们是对应的
    """
    # 检查方位是否相反（关键：方位相反时不强制匹配）
    report_orient = report_entity.get('orientation', '')
    conclusion_orient = conclusion_entity.get('orientation', '')
    if ((report_orient == '左' and conclusion_orient == '右') or 
        (report_orient == '右' and conclusion_orient == '左')):
        return False
    
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
        if (_map_positive_value(report_entity.get('positive', 0)) == 1 and 
            _map_positive_value(conclusion_entity.get('positive', 0)) == 1):
            return True
    
    return False


def _position_matches(report_entity: Dict, conclusion_entity: Dict, upper_position_pattern: str) -> bool:
    """
    检查描述实体和结论实体的部位是否匹配
    
    原版逻辑：d['position'] in s['partlist'] or s['position'] in d['partlist']
    即：一个实体的 position 应该在另一个实体的 partlist 中
    """
    report_pos = report_entity.get('position', '')
    conclusion_pos = conclusion_entity.get('position', '')
    
    report_partlists = report_entity.get('partlist', [])
    conclusion_partlists = conclusion_entity.get('partlist', [])
    
    # 提取 partlist 中的所有元素（扁平化）
    report_elements = set()
    for pl in report_partlists:
        if isinstance(pl, (list, tuple)):
            report_elements.update(pl)
        else:
            report_elements.add(pl)
    
    conclusion_elements = set()
    for pl in conclusion_partlists:
        if isinstance(pl, (list, tuple)):
            conclusion_elements.update(pl)
        else:
            conclusion_elements.add(pl)
    
    # 原版逻辑：position in partlist
    # 描述的 position 在结论的 partlist 中，或结论的 position 在描述的 partlist 中
    if report_pos and conclusion_elements and report_pos in conclusion_elements:
        return True
    if conclusion_pos and report_elements and conclusion_pos in report_elements:
        return True
    
    # 回退：直接比较 position
    if report_pos and conclusion_pos and report_pos == conclusion_pos:
        return True
    
    return False


def _calc_semantic_score(report_entity: Dict, conclusion_entity: Dict) -> float:
    """
    计算两个实体的语义相似度（使用Word2Vec加权版本）
    
    逻辑：
    1. partlist相似度：使用Word2Vec计算部位文本相似度
    2. illness相似度：使用Word2Vec计算疾病描述相似度  
    3. 加权：position_sim * 0.4 + illness_sim * 0.6
    4. 多partlist：尝试所有组合，取最高分
    5. 方位不一致：position_sim -= 0.1
    """
    import time
    global _semantic_sim_count, _semantic_sim_time
    
    t0 = time.time()
    
    # 获取partlist列表
    report_partlists = report_entity.get('partlist', [])
    conclusion_partlists = conclusion_entity.get('partlist', [])
    
    if not report_partlists or not conclusion_partlists:
        return 0.0
    
    # 计算partlist相似度（取所有组合中的最高分）
    max_position_sim = 0.0
    
    for r_pl in report_partlists:
        r_text = ' '.join(r_pl) if isinstance(r_pl, (list, tuple)) else str(r_pl)
        for c_pl in conclusion_partlists:
            c_text = ' '.join(c_pl) if isinstance(c_pl, (list, tuple)) else str(c_pl)
            
            # 使用Word2Vec计算部位相似度
            sim = _sentence_semantics_w2v(r_text, c_text)
            if sim < 0:  # Word2Vec不可用
                # 回退到简单子集判断
                r_set = set(r_pl) if isinstance(r_pl, (list, tuple)) else {r_pl}
                c_set = set(c_pl) if isinstance(c_pl, (list, tuple)) else {c_pl}
                if r_set & c_set:  # 有交集
                    sim = 0.7
                elif r_set.issubset(c_set) or c_set.issubset(r_set):
                    sim = 0.5
                else:
                    sim = 0.0
            
            if sim > max_position_sim:
                max_position_sim = sim
    
    # 方位不一致惩罚
    if report_entity.get('orientation') != conclusion_entity.get('orientation'):
        max_position_sim -= 0.1
    
    # 计算illness相似度
    report_illness = report_entity.get('illness', '')
    conclusion_illness = conclusion_entity.get('illness', '')
    
    illness_sim = 0.0
    if report_illness and conclusion_illness:
        # 提取所有疾病描述
        r_ill_list = [x.strip() for x in report_illness.split(",") if x.strip()]
        c_ill_list = [x.strip() for x in conclusion_illness.split(",") if x.strip()]
        
        if r_ill_list and c_ill_list:
            # 尝试所有组合，找最高相似度
            max_illness_sim = 0.0
            for r_ill in r_ill_list:
                for c_ill in c_ill_list:
                    # 优先检查包含关系（如"术后改变"包含在"纵隔旁见高密度影,呈术后改变"中）
                    if r_ill in c_ill or c_ill in r_ill:
                        sim = 0.5  # 包含关系给中等分数
                    else:
                        sim = _sentence_semantics_w2v(r_ill, c_ill)
                        if sim < 0:  # Word2Vec不可用
                            sim = 0.0
                    
                    if sim > max_illness_sim:
                        max_illness_sim = sim
            
            illness_sim = max_illness_sim
    
    # positive状态不一致惩罚（使用映射后的值）
    if (_map_positive_value(report_entity.get('positive')) != 
        _map_positive_value(conclusion_entity.get('positive'))):
        illness_sim -= 0.2
    
    # "信医生但防离谱"：阈值0.1，只要不太离谱就接受
    if illness_sim < 0.1:
        return 0.0
    
    # 确保不低于0
    max_position_sim = max(0.0, max_position_sim)
    illness_sim = max(0.0, illness_sim)
    
    # 加权：部位30% + 疾病70%（增加疾病权重）
    final_sim = max_position_sim * 0.3 + illness_sim * 0.7
    
    elapsed = time.time() - t0
    _semantic_sim_count += 1
    _semantic_sim_time += elapsed
    
    return final_sim


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
            
            # 只有阳性实体（映射后positive=1）才加入方位错误检测列表
            # "信医生但防离谱"：阈值从0.3降到0.15，减少假阳性
            # 方位错误检测本身是高置信度判断，门槛应低于缺失检测
            should_add_for_orient = (_map_positive_value(report_entity.get('positive', 0)) == 1 and 
                                     not skip_orient_check and 
                                     semantic_score >= 0.15)
            
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


def has_valid_match(
    report_entity: Dict,
    position_matches: List[Dict],
    all_report_entities: List[Dict] = None,
    all_conclusion_entities: List[Dict] = None
) -> bool:
    """
    判断描述实体是否有有效匹配
    
    采用旧版严格逻辑：
    1. 结论的 position 必须在描述的 partlist 中（严格部位匹配）
    2. 方位匹配：左右方位不相反
    
    Returns:
        True - 有有效匹配（信任医生）
        False - 无有效匹配（可能是真缺失）
    """
    if not position_matches:
        return False
    
    # 获取描述的 partlist 所有元素
    report_partlists = report_entity.get('partlist', [])
    report_elements = set()
    for pl in report_partlists:
        if isinstance(pl, (list, tuple)):
            report_elements.update(pl)
        else:
            report_elements.add(pl)
    
    # 严格过滤：双向匹配（原版逻辑）
    # 原版: position_in_any_partlist(A.position, B) or position_in_any_partlist(B.position, A)
    # 即：A的position在B的partlist中，或B的position在A的partlist中
    valid_matches = []
    for c in position_matches:
        conc_position = c.get('position', '')
        conc_partlists = c.get('partlist', [])
        conc_elements = set()
        for pl in conc_partlists:
            if isinstance(pl, (list, tuple)):
                conc_elements.update(pl)
            else:
                conc_elements.add(pl)
        
        report_position = report_entity.get('position', '')
        # 双向检查：结论position在描述partlist中，或描述position在结论partlist中
        if conc_position in report_elements or report_position in conc_elements:
            valid_matches.append(c)
    
    if not valid_matches:
        return False
    
    # 检查方位匹配
    orient_matches = [
        c for c in valid_matches
        if _is_orient_match(report_entity.get('orientation', ''), 
                           c.get('orientation', ''))
    ]
    
    if orient_matches:
        # 方位匹配 → 有有效匹配
        return True
    
    # 方位不匹配 → 检查强制兜底（单实体场景）
    if all_report_entities is not None and all_conclusion_entities is not None:
        for c in valid_matches:
            if _is_forced_match(report_entity, c, 
                               all_report_entities, all_conclusion_entities):
                return True
    
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
    
    注意：
    - 'start' 是实体在短句中的开始位置，不是句子标识
    - 使用 'sentence_index' + 'sentence_start' 或 'original_short_sentence' 标识句子
    - 同一句话的多个实体，只要有一个匹配成功，整句话都不算缺失
    """
    # 按句子分组：使用 (sentence_index, sentence_start) 作为唯一标识
    # 如果字段不存在，回退到使用 original_short_sentence
    def get_sentence_key(entity):
        sent_idx = entity.get('sentence_index')
        sent_start = entity.get('sentence_start')
        if sent_idx is not None and sent_start is not None:
            return (sent_idx, sent_start)
        # 回退：使用 original_short_sentence
        return entity.get('original_short_sentence', '')
    
    # 按句子 key 分组
    sentence_groups: Dict[Any, List[Dict]] = {}
    for entity in report_entities:
        if not entity.get('illness'):
            continue
        key = get_sentence_key(entity)
        if key not in sentence_groups:
            sentence_groups[key] = []
        sentence_groups[key].append(entity)
    
    # === 第一遍扫描：收集所有匹配成功的句子 key ===
    matched_sentence_keys = set()
    
    for key, entities in sentence_groups.items():
        for entity in entities:
            if _map_positive_value(entity.get('positive', 0)) == 0:
                continue
            if entity.get('ignore', False):
                continue
            
            position_matches = find_position_matches(entity, conclusion_entities, upper_position_pattern)
            
            if position_matches and has_valid_match(entity, position_matches, report_entities, conclusion_entities):
                # 该句子有匹配成功的实体，整句话都不算缺失
                matched_sentence_keys.add(key)
                break  # 跳出该句子的实体循环
    
    # === 第二遍扫描：收集缺失（排除已匹配的句子）===
    missing = []
    
    for key, entities in sentence_groups.items():
        # 如果该句子已匹配，跳过
        if key in matched_sentence_keys:
            continue
        
        # 否则检查每个实体是否真的缺失
        for entity in entities:
            if _map_positive_value(entity.get('positive', 0)) == 0:
                continue
            if entity.get('ignore', False):
                continue
            
            position_matches = find_position_matches(entity, conclusion_entities, upper_position_pattern)
            
            if not position_matches:
                missing.append(entity['original_short_sentence'])
            elif not has_valid_match(entity, position_matches, report_entities, conclusion_entities):
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


def detect_orientation_errors(
    candidates: List[MatchCandidate],
    conclusion_missing: List[str] = None
) -> List[str]:
    """
    检测方位错误
    
    完全按照旧版 origin/NLP_analyze.py Check_report_conclusion 逻辑：
    1. 分离方位匹配(orientation_match)和方位不匹配(non_orientation_match)
    2. 对于每个方位不匹配的项，计算 conclusion_probability 和 report_probability
    3. 判断条件：
       - conclusion_probability > 0.006 且 semantics >= 0.5
       - 或 conclusion_probability == 1 且 (semantics >= 0.4 或 描述在conclusion_missing中)
       - 同时 report_probability > 0
    4. 按描述和结论去重，保留 match_probability 最高的项
    """
    if not candidates:
        return []
    
    if conclusion_missing is None:
        conclusion_missing = []
    
    # 分离方位匹配和不匹配的（与旧版一致）
    non_orientation_match = [c for c in candidates if not c.is_orient_match]
    orientation_match = [c for c in candidates if c.is_orient_match]
    
    # 收集有效的方位错误项（对应旧版 1152-1164 行）
    valid_errors: List[Tuple[MatchCandidate, OrientationProbability]] = []
    
    for mismatch in non_orientation_match:
        # report_df: 同一个描述，其他方位匹配结论的语义相似度列表
        report_df = [x.semantic_score for x in orientation_match 
                     if x.report_sentence == mismatch.report_sentence]
        
        # conclusion_df: 同一个结论，其他方位匹配描述的语义相似度列表
        conclusion_df = [x.semantic_score for x in orientation_match 
                         if x.conclusion_sentence == mismatch.conclusion_sentence]
        
        # 计算 probability（与旧版一致）
        report_probability = 1.0 if not report_df else mismatch.semantic_score - max(report_df)
        conclusion_probability = 1.0 if not conclusion_df else mismatch.semantic_score - max(conclusion_df)
        
        # 检查描述是否在conclusion_missing中（旧版1158-1159行关键逻辑）
        in_conclusion_missing = any(mismatch.report_sentence in x for x in conclusion_missing)
        
        # 判断条件（与旧版 1157-1160 行一致）
        # cond1: conclusion_probability > 0.006 and semantics >= 0.5
        # cond2: conclusion_probability == 1.0 and (semantics >= 0.4 or in_conclusion_missing)
        cond1 = conclusion_probability > 0.006 and mismatch.semantic_score >= 0.5
        cond2 = conclusion_probability == 1.0 and (mismatch.semantic_score >= 0.4 or in_conclusion_missing)
        
        if (cond1 or cond2) and report_probability > 0:
            match_probability = report_probability + conclusion_probability
            prob_info = OrientationProbability(
                report_prob=report_probability,
                conclusion_prob=conclusion_probability,
                match_prob=match_probability,
                is_error=True
            )
            valid_errors.append((mismatch, prob_info))
    
    # 按描述去重，保留 match_probability 最高的（与旧版 1165-1172 行一致）
    by_report: Dict[str, Tuple[MatchCandidate, OrientationProbability]] = {}
    for item, prob in valid_errors:
        name = item.report_sentence
        if name not in by_report or prob.match_prob > by_report[name][1].match_prob:
            by_report[name] = (item, prob)
    
    # 按结论去重，保留 match_probability 最高的（与旧版 1173-1179 行一致）
    by_conclusion: Dict[str, Tuple[MatchCandidate, OrientationProbability]] = {}
    for item, prob in by_report.values():
        name = item.conclusion_sentence
        if name not in by_conclusion or prob.match_prob > by_conclusion[name][1].match_prob:
            by_conclusion[name] = (item, prob)
    
    # 生成错误列表（与旧版 1180-1181 行一致）
    errors = []
    for item, _ in by_conclusion.values():
        # 检查是否包含左右关键字
        has_orient_keyword = (
            re.search("左|右", item.report_sentence) or 
            re.search("左|右", item.conclusion_sentence)
        )
        if has_orient_keyword:
            error_text = f"[描述]{item.report_sentence}；[结论]{item.conclusion_sentence}"
            errors.append(error_text)
    
    return list(set(errors))


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
    
    # 详细计时器
    detailed_timers = {}
    total_start = time.time()
    
    nlp = _import_from_nlp_analyze()
    
    # 获取配置
    check_modality = nlp.check_modality if hasattr(nlp, 'check_modality') else 'CT|MR|DR|MG'
    miss_ignore_pattern = nlp.miss_ignore_pattern if hasattr(nlp, 'miss_ignore_pattern') else re.compile('')
    orient_ignore = nlp.orient_ignore if hasattr(nlp, 'orient_ignore') else ''
    upper_position = nlp.upper_position if hasattr(nlp, 'upper_position') else ''
    
    # 前置检查
    if re.search(check_modality, modality) is None:
        return [], [], {'detailed': {}}
    
    if not report_analyze:
        return [], [], {'detailed': {}}
    
    if not conclusion_analyze:
        # 结论为空，所有阳性（映射后positive=1）的描述都是缺失
        missing = [
            d['original_short_sentence'] 
            for d in report_analyze 
            if _map_positive_value(d.get('positive', 0)) == 1 
            and not miss_ignore_pattern.search(d.get('original_short_sentence', ''))
        ]
        return list(set(missing)), [], {'detailed': {}}
    
    # 步骤1：收集所有匹配候选（用于方位错误检测）
    t0 = time.time()
    candidates = collect_match_candidates(
        report_analyze, 
        conclusion_analyze,
        upper_position,
        orient_ignore
    )
    detailed_timers['收集匹配候选'] = time.time() - t0
    
    # 步骤2：检测结论缺失
    t0 = time.time()
    conclusion_missing = detect_missing_conclusions(
        report_analyze,
        conclusion_analyze,
        upper_position,
        miss_ignore_pattern
    )
    detailed_timers['检测结论缺失'] = time.time() - t0
    
    # 步骤3：检测方位错误
    t0 = time.time()
    orient_error = detect_orientation_errors(candidates, conclusion_missing)
    detailed_timers['检测方位错误'] = time.time() - t0
    
    # 步骤4：如果某描述被判为方位错误，则从缺失列表中移除
    for error_text in orient_error:
        report_part = error_text.split("；[结论]")[0].replace("[描述]", "")
        conclusion_missing = [x for x in conclusion_missing if report_part not in x]
    
    # 获取性能统计
    perf_stats = get_performance_stats()
    total_elapsed = time.time() - total_start
    
    # 计算其他耗时（不包括上述步骤）
    other_time = total_elapsed - sum(detailed_timers.values())
    if other_time > 0:
        detailed_timers['其他（初始化等）'] = other_time
    
    detailed_timers['总计'] = total_elapsed
    perf_stats['detailed'] = detailed_timers
    
    return conclusion_missing, orient_error, perf_stats
