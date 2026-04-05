# -*- coding: utf-8 -*-
"""
LLM验证模块

集中处理所有LLM验证逻辑，包括：
- 结论缺失验证
- 方位错误验证
- 矛盾检测验证
- 性别错误验证
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from .config import LLMConfig

# 延迟导入避免循环依赖
def _get_llm_validator():
    """获取LLM验证器实例"""
    from llm_service import get_llm_validator
    return get_llm_validator()


def batch_validate_with_llm(
    conclusion_missing_list: List[str],
    orient_error_list: List[str],
    contradiction_list: List[str],
    sex_error_list: List[str],
    patient_sex: str,
    description: str,
    conclusion: str
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    统一批量LLM验证函数 - 并发验证所有类型的错误
    
    将结论缺失、方位错误、矛盾检测、性别错误的结果一起提交给LLM进行批量验证，
    减少多次调用的开销，提高处理效率。
    
    Args:
        conclusion_missing_list: 结论缺失列表
        orient_error_list: 方位错误列表
        contradiction_list: 矛盾列表（成对出现）
        sex_error_list: 性别错误列表
        patient_sex: 患者性别（"男"或"女"）
        description: 报告描述原文
        conclusion: 报告结论原文
        
    Returns:
        tuple: (filtered_conclusion_missing, filtered_orient_error, filtered_contradiction, filtered_sex_error)
    """
    # 检查是否需要验证
    has_conclusion_missing = bool(conclusion_missing_list)
    has_orient_error = bool(orient_error_list)
    has_contradiction = bool(contradiction_list and len(contradiction_list) >= 2)
    has_sex_error = bool(sex_error_list)
    
    if not (has_conclusion_missing or has_orient_error or has_contradiction or has_sex_error):
        return conclusion_missing_list, orient_error_list, contradiction_list, sex_error_list
    
    if not LLMConfig.USE_LLM_VALIDATION():
        return conclusion_missing_list, orient_error_list, contradiction_list, sex_error_list
    
    try:
        validator = _get_llm_validator()
        if not validator.available():
            return conclusion_missing_list, orient_error_list, contradiction_list, sex_error_list
        
        # 构建统一的验证候选列表
        candidates = []
        
        # 1. 结论缺失候选
        for item in conclusion_missing_list:
            candidates.append({
                'type': 'conclusion_missing',
                'description': description,
                'conclusion': conclusion,
                'suspected': item,
                '_original': item
            })
        
        # 2. 方位错误候选
        for item in orient_error_list:
            match = re.search(r'\[描述\](.+?)；\[结论\](.+)', item)
            if match:
                candidates.append({
                    'type': 'orient_error',
                    'description': match.group(1),
                    'conclusion': match.group(2),
                    '_original': item
                })
            else:
                # 格式不匹配，直接保留
                candidates.append({
                    'type': 'orient_error',
                    'description': item,
                    'conclusion': '',
                    '_original': item,
                    '_keep': True
                })
        
        # 3. 矛盾候选（成对出现）
        for i in range(0, len(contradiction_list) - 1, 2):
            stmt1 = contradiction_list[i]
            stmt2 = contradiction_list[i + 1] if i + 1 < len(contradiction_list) else ''
            if isinstance(stmt1, str) and isinstance(stmt2, str):
                candidates.append({
                    'type': 'contradiction',
                    'statement1': stmt1,
                    'statement2': stmt2,
                    '_pair_idx': i // 2
                })
        
        # 4. 性别错误候选
        for item in sex_error_list:
            # 解析性别错误字符串，格式："男性报告中出现：关键词1；关键词2" 或 "女性报告中出现：关键词1；关键词2"
            match = re.search(r'([男女])性报告中出现：(.+)', item)
            if match:
                detected_sex = match.group(1)  # 检测到的性别（报告中出现的性别关键词对应的性别）
                keywords_str = match.group(2)
                keywords = [k.strip() for k in keywords_str.split('；') if k.strip()]
                
                # 构建完整的报告内容用于上下文判断
                report_content = f"描述：{description}\n结论：{conclusion}"
                
                candidates.append({
                    'type': 'sex_error',
                    'patient_sex': patient_sex,
                    'detected_sex': detected_sex,
                    'suspected_keywords': keywords,
                    'report_content': report_content,
                    '_original': item
                })
            else:
                # 格式不匹配，直接保留
                candidates.append({
                    'type': 'sex_error',
                    'patient_sex': patient_sex,
                    'suspected_keywords': [item],
                    'report_content': f"描述：{description}\n结论：{conclusion}",
                    '_original': item,
                    '_keep': True
                })
        
        if not candidates:
            return conclusion_missing_list, orient_error_list, contradiction_list, sex_error_list
        
        # 批量LLM验证（一次调用验证所有）
        validated = validator.batch_validate(candidates)
        
        # 分别收集结果
        confirmed_conclusion_missing = []
        conclusion_needs_review = []
        
        confirmed_orient_error = []
        orient_needs_review = []
        
        confirmed_contradiction = []
        contradiction_needs_review = []
        
        confirmed_sex_error = []
        sex_needs_review = []
        
        for v in validated:
            vtype = v.get('type', '')
            
            if vtype == 'conclusion_missing':
                original = v.get('_original', '')
                if v.get('needs_review'):
                    conclusion_needs_review.append(original)
                elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                    confirmed_conclusion_missing.append(original)
                elif v.get('weak_positive'):
                    confirmed_conclusion_missing.append(f"[弱阳性]{original}")
                # 低置信度丢弃
                
            elif vtype == 'orient_error':
                if v.get('_keep'):
                    # 格式不匹配的直接保留
                    confirmed_orient_error.append(v.get('_original', ''))
                    continue
                    
                original = v.get('_original', '')
                if v.get('needs_review'):
                    orient_needs_review.append(original)
                elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                    confirmed_orient_error.append(original)
                elif v.get('weak_positive'):
                    confirmed_orient_error.append(f"[弱阳性]{original}")
                # 低置信度丢弃
                
            elif vtype == 'contradiction':
                stmt1 = v.get('statement1', '')
                stmt2 = v.get('statement2', '')
                if v.get('needs_review'):
                    contradiction_needs_review.append((stmt1, stmt2))
                elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                    confirmed_contradiction.extend([stmt1, stmt2])
                elif v.get('weak_positive'):
                    confirmed_contradiction.extend([f"[弱阳性]{stmt1}", f"[弱阳性]{stmt2}"])
                # 低置信度丢弃
            
            elif vtype == 'sex_error':
                if v.get('_keep'):
                    # 格式不匹配的直接保留
                    confirmed_sex_error.append(v.get('_original', ''))
                    continue
                
                original = v.get('_original', '')
                if v.get('needs_review'):
                    sex_needs_review.append(original)
                elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                    confirmed_sex_error.append(original)
                elif v.get('weak_positive'):
                    confirmed_sex_error.append(f"[弱阳性]{original}")
                # 低置信度丢弃
        
        # 添加待审核标记
        if conclusion_needs_review:
            confirmed_conclusion_missing.append(f"[待审核]{len(conclusion_needs_review)}项")
        if orient_needs_review:
            confirmed_orient_error.append(f"[待审核]{len(orient_needs_review)}项")
        if contradiction_needs_review:
            confirmed_contradiction.append(f"[待审核]{len(contradiction_needs_review)}项")
        if sex_needs_review:
            confirmed_sex_error.append(f"[待审核]{len(sex_needs_review)}项")
        
        return (
            confirmed_conclusion_missing if confirmed_conclusion_missing else [],
            confirmed_orient_error if confirmed_orient_error else [],
            confirmed_contradiction if confirmed_contradiction else [],
            confirmed_sex_error if confirmed_sex_error else []
        )
        
    except Exception as e:
        print(f"批量LLM验证失败: {e}")
        # 失败时保留原结果
        return conclusion_missing_list, orient_error_list, contradiction_list, sex_error_list


# 向后兼容的独立验证函数（内部使用批量验证）
def validate_conclusion_missing(conclusion_missing_list: List[str], description: str, conclusion: str) -> List[str]:
    """验证结论缺失（向后兼容）"""
    result, _, _, _ = batch_validate_with_llm(
        conclusion_missing_list, [], [], [], "", description, conclusion
    )
    return result


def validate_orient_error(orient_error_list: List[str]) -> List[str]:
    """验证方位错误（向后兼容）"""
    _, result, _, _ = batch_validate_with_llm(
        [], orient_error_list, [], [], "", "", ""
    )
    return result


def validate_contradiction(contradiction_list: List[str]) -> List[str]:
    """验证矛盾（向后兼容）"""
    _, _, result, _ = batch_validate_with_llm(
        [], [], contradiction_list, [], "", "", ""
    )
    return result


def validate_sex_error(sex_error_list: List[str], patient_sex: str, description: str, conclusion: str) -> List[str]:
    """验证性别错误（向后兼容）"""
    _, _, _, result = batch_validate_with_llm(
        [], [], [], sex_error_list, patient_sex, description, conclusion
    )
    return result
