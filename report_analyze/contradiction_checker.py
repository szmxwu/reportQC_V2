# -*- coding: utf-8 -*-
"""
报告矛盾检测模块 - 完全恢复旧实现逻辑

核心逻辑（与NLP_analyze_copy.py完全一致）：
1. 按部位和方位分组遍历
2. 对每个组，分离positive>1（阳性）和positive<=1（阴性）实体
3. aspect匹配过滤
4. 使用sentence_semantics计算相似度
5. 阈值0.78判断矛盾
"""

import re
import os
import numpy as np
from scipy import spatial
from typing import List, Dict

# 加载 Word2Vec 模型（延迟加载）
_wv_model = None
_jieba = None
_semantics_stopwords = None

def _get_word2vec_model():
    """延迟加载Word2Vec模型"""
    global _wv_model, _jieba, _semantics_stopwords
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
                
            # 加载停用词
            from .config import SystemConfig
            _semantics_stopwords = SystemConfig.semantics_stopwords() if hasattr(SystemConfig, 'semantics_stopwords') else ''
        except Exception as e:
            print(f"Word2Vec模型加载失败: {e}")
    return _wv_model


def _avg_feature_vector(sentence: str, model) -> np.ndarray:
    """文本转向量"""
    if _jieba is None:
        return np.zeros((model.wv.vector_size,), dtype='float32')
    
    words = _jieba.lcut(sentence)
    feature_vec = np.zeros((model.wv.vector_size,), dtype='float32')
    n_words = 0
    for word in words:
        try:
            feature_vec = np.add(feature_vec, model.wv[word])
            n_words += 1
        except:
            pass
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


def _sentence_semantics(s1: str, s2: str) -> float:
    """句子的相似度比较 - 完全恢复旧实现"""
    model = _get_word2vec_model()
    if model is None:
        return 0.0
    
    global _semantics_stopwords
    if _semantics_stopwords is None:
        from .config import SystemConfig
        _semantics_stopwords = SystemConfig.semantics_stopwords() if hasattr(SystemConfig, 'semantics_stopwords') else ''
    
    # 清洗句子
    if _semantics_stopwords:
        s1 = re.sub(rf"{_semantics_stopwords}", '', s1)
        s2 = re.sub(rf"{_semantics_stopwords}", '', s2)
    
    if s1 == "" or s2 == '':
        return 0.0
    
    s1_afv = _avg_feature_vector(s1, model)
    s2_afv = _avg_feature_vector(s2, model)
    
    if np.sum(s1_afv) == 0 or np.sum(s2_afv) == 0:
        sim = 0
    else:
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    
    # 方位一致性调整
    if sim > 0.5:
        s1_o = re.findall("[左右]", s1)
        s2_o = re.findall("[左右]", s2)
        if s1_o != s2_o:
            sim -= 0.05
    
    return sim


# 延迟导入避免循环依赖
def _import_from_nlp_analyze():
    """延迟导入避免循环依赖"""
    import sys
    nlp_module = sys.modules.get('NLP_analyze')
    if nlp_module is None:
        import importlib
        nlp_module = importlib.import_module('NLP_analyze')
    return nlp_module


# 导入配置（避免循环依赖）
def _get_config():
    """获取配置"""
    from .config import SystemConfig
    return SystemConfig


def check_contradiction(
    report_analyze: List[Dict],
    conclusion_analyze: List[Dict],
    modality: str
) -> List[str]:
    """
    检查报告矛盾语句 - 完全恢复旧实现逻辑
    
    Args:
        report_analyze: 解析后的描述实体列表
        conclusion_analyze: 解析后的结论实体列表
        modality: 设备类型
    
    Returns:
        List[str]: 矛盾列表 [stmt1, stmt2, stmt3, stmt4, ...]
    """
    nlp = _import_from_nlp_analyze()
    cfg = _get_config()
    
    # 获取配置（优先从report_analyze.config，如果不存在则从NLP_analyze）
    check_modality = cfg.check_modality() if hasattr(cfg, 'check_modality') else getattr(nlp, 'check_modality', 'CT|MR|DR|MG')
    key_part = cfg.key_part() if hasattr(cfg, 'key_part') else getattr(nlp, 'key_part', [])
    
    # 前置检查
    if re.search(check_modality, modality) is None:
        return []
    
    # 合并实体（旧实现使用 ignore == False）
    all_analyze = []
    all_analyze.extend([x for x in report_analyze if x.get('ignore') == False])
    all_analyze.extend([x for x in conclusion_analyze if x.get('ignore') == False])
    
    if len(all_analyze) == 0:
        return []
    
    contradiction = []
    
    # 按部位和方位分组（与旧实现完全一致）
    for part in key_part:
        # 左 + 双 + 无方位
        key_analyze = [d for d in all_analyze 
                       if (nlp.position_in_any_partlist(part, d)) and 
                       (d.get('orientation') == "左" or d.get('orientation') == "双" or d.get('orientation') == "")]
        contradiction.extend(_get_contradiction(key_analyze))
        
        # 右 + 双 + 无方位
        key_analyze = [d for d in all_analyze 
                       if (nlp.position_in_any_partlist(part, d)) and 
                       (d.get('orientation') == "右" or d.get('orientation') == "双" or d.get('orientation') == "")]
        contradiction.extend(_get_contradiction(key_analyze))
        
        # 双 + 无方位（再次检查确保覆盖）
        key_analyze = [d for d in all_analyze 
                       if (nlp.position_in_any_partlist(part, d)) and 
                       (d.get('orientation') == "双" or d.get('orientation') == "")]
        contradiction.extend(_get_contradiction(key_analyze))
    
    # 去重（按对去重，保持原始顺序）
    # 将扁平列表转为对列表 [(neg1, pos1), (neg2, pos2), ...]
    pairs = []
    for i in range(0, len(contradiction), 2):
        if i + 1 < len(contradiction):
            pairs.append((contradiction[i], contradiction[i + 1]))
    
    # 按对去重，保持顺序
    seen_pairs = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_pairs.append(pair)
    
    # 转回扁平列表
    result = []
    for neg, pos in unique_pairs:
        result.extend([neg, pos])
    
    return result


def _get_contradiction(key_analyze: List[Dict]) -> List[str]:
    """
    在实体组内查找所有矛盾对
    
    返回扁平化列表: [neg1, pos1, neg2, pos2, ...]
    避免重复检测同一对矛盾
    """
    if len(key_analyze) == 0:
        return []
    
    nlp = _import_from_nlp_analyze()
    cfg = _get_config()
    
    # 获取配置（优先从report_analyze.config）
    exclud = cfg.exclud() if hasattr(cfg, 'exclud') else getattr(nlp, 'exclud', '')
    stopwords = cfg.stopwords() if hasattr(cfg, 'stopwords') else getattr(nlp, 'stopwords', '')
    aspects = cfg.aspects() if hasattr(cfg, 'aspects') else getattr(nlp, 'aspects', [])
    sub_part = cfg.sub_part() if hasattr(cfg, 'sub_part') else getattr(nlp, 'sub_part', '')
    
    # 分离阳性（positive > 1）和阴性（positive <= 1）
    positive_key = [d for d in key_analyze 
                    if d.get('positive', 0) > 1 and
                    re.search(rf"{exclud}", d.get('original_short_sentence', '')) == None]
    
    negative_key = [d for d in key_analyze 
                    if d.get('positive', 0) <= 1 and 
                    d.get('measure', 0) == 0 and 
                    len(re.sub(stopwords, "", d.get('illness', ''))) > 1 and 
                    re.search(rf"{exclud}", d.get('original_short_sentence', '')) == None]
    
    if len(negative_key) == 0 or len(positive_key) == 0:
        return []
    
    # 收集所有矛盾对，避免重复
    contradictions = []
    detected_pairs = set()  # 用于去重
    
    for neg_sentence in negative_key:
        neg_text = neg_sentence.get('original_short_sentence', '')
        found = False
        
        # 先尝试aspect匹配
        for aspect in aspects:
            if re.search(rf"{aspect}", neg_text):
                positive_str = _get_positive_key(neg_sentence, aspect, positive_key)
                found = True
                if positive_str != "":
                    pair_key = (neg_text, positive_str)
                    if pair_key not in detected_pairs:
                        detected_pairs.add(pair_key)
                        contradictions.extend([neg_text, positive_str])
                    break  # 找到即跳出aspect循环
        
        # 无aspect匹配时，尝试直接匹配
        if not found:
            positive_str = _get_positive_key(neg_sentence, "", positive_key)
            if positive_str != "":
                # 子部位匹配逻辑
                pos_sub_part = re.findall(sub_part, positive_str)
                neg_sub_part = re.findall(sub_part, neg_text)
                if (pos_sub_part == [] or neg_sub_part == [] or 
                    (set(pos_sub_part) & set(neg_sub_part))):  # 子部位存在交集
                    pair_key = (neg_text, positive_str)
                    if pair_key not in detected_pairs:
                        detected_pairs.add(pair_key)
                        contradictions.extend([neg_text, positive_str])
    
    return contradictions


def _get_positive_key(neg_sentence: Dict, aspect: str, positive_key: List[Dict]) -> str:
    """
    获取与阴性句子最匹配的阳性句子 - 完全恢复旧实现逻辑
    """
    nlp = _import_from_nlp_analyze()
    
    # 按部位过滤
    positive_key = [d for d in positive_key 
                    if nlp.position_in_any_partlist(neg_sentence.get('position', ''), d)]
    
    # 按aspect过滤
    if aspect != "":
        positive_key = [d for d in positive_key 
                        if re.search(rf"{aspect}", d.get('original_short_sentence', ''))]
    
    if positive_key:
        # 使用sentence_semantics计算相似度
        for dic in positive_key:
            sim = _sentence_semantics(
                neg_sentence.get('original_short_sentence', ''),
                dic.get('original_short_sentence', '')
            )
            dic['_semantics'] = sim
        
        semantics_min = min([x['_semantics'] for x in positive_key])
        if semantics_min <= 0.78:
            # 返回相似度最小的（即最不相似的阳性描述）
            for x in positive_key:
                if x['_semantics'] == semantics_min:
                    return x.get('original_short_sentence', '')
    
    return ""
