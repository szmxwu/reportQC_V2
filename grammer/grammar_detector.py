#!/usr/bin/env python3
"""
医学影像报告语法/错别字检测模块

分层检测策略：
1. 规则层：基于医学术语词典的快速匹配
2. N-gram层：统计语言模型检测异常
3. LLM验证层：验证可疑错误

使用示例：
    # 方式1: 使用预训练模型
    detector = GrammarDetector(model_path='grammar_model.pkl')
    
    # 方式2: 不使用统计模型(仅规则+LLM)
    detector = GrammarDetector()
    
    errors = detector.detect(report_text, use_llm=True)
"""

import os
import re
import pickle
import asyncio
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from llm_service import LLMValidator


class ErrorType(Enum):
    """语法错误类型"""
    TYPING_ERROR = "typing_error"      # 打字错误（如：肺文里）
    NONSENSE_WORD = "nonsense_word"    # 无意义词汇
    STATISTICAL_ANOMALY = "statistical_anomaly"  # N-gram统计异常
    UNCLEAR_EXPRESSION = "unclear_expression"    # 表达不清


@dataclass
class GrammarError:
    """语法错误结果"""
    error_type: ErrorType
    position: int                      # 错误位置
    text: str                          # 错误文本
    suggestion: Optional[str]          # 建议修正
    confidence: float                  # 置信度 (0-1)
    context: str                       # 上下文
    needs_llm_verify: bool             # 是否需要LLM验证


class NgramLanguageModel:
    """轻量级N-gram语言模型"""
    
    def __init__(self, n: int = 2, threshold: float = 0.00001):
        self.n = n
        self.threshold = threshold
        self.ngrams = {}
        self.total = 0
    
    def train(self, texts: List[str]):
        """训练N-gram模型（简单实现，可用pickle加载预训练）"""
        from collections import Counter
        
        counter = Counter()
        for text in texts:
            text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
            for i in range(len(text) - self.n + 1):
                counter[text[i:i+self.n]] += 1
        
        self.ngrams = dict(counter)
        self.total = sum(counter.values())
    
    def load(self, filepath: str):
        """加载预训练模型"""
        if not os.path.exists(filepath):
            print(f"警告: 模型文件不存在 {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            self.n = model.get('n', self.n)
            self.ngrams = model.get('ngrams', {})
            self.total = model.get('total', 0)
        print(f"加载N-gram模型: {len(self.ngrams)} 个唯一n-gram, 总计 {self.total}")
    
    def check(self, text: str, window: int = 3) -> List[Dict]:
        """检测异常N-gram"""
        # 如果模型未训练，返回空
        if self.total == 0 or len(self.ngrams) == 0:
            return []
        
        anomalies = []
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i+self.n]
            count = self.ngrams.get(ngram, 0)
            prob = count / self.total if self.total > 0 else 0
            
            # 极低概率或不存在
            if prob < self.threshold and count == 0:
                context_start = max(0, i - window)
                context_end = min(len(text), i + self.n + window)
                anomalies.append({
                    'position': i,
                    'ngram': ngram,
                    'probability': prob,
                    'context': text[context_start:context_end]
                })
        
        return anomalies


class RuleBasedChecker:
    """基于规则的语法检查器"""
    
    # 常见错别字映射
    COMMON_TYPOS = {
        '文里': ('纹理', '肺文里→肺纹理'),
        '低密渡': ('低密度', '低密渡→低密度'),
        '影象': ('影像', '影象→影像'),
        '造影剂': ('造影剂', '无问题'),
        '占位性': ('占位性', '无问题'),
        # 可以继续添加...
    }
    
    # 需要特别注意的高频错误 (pattern, suggestion, type, confidence)
    HIGH_RISK_PATTERNS = [
        (r'肺文里', '纹理', ErrorType.TYPING_ERROR, 0.95),
        (r'低密渡', '低密度', ErrorType.TYPING_ERROR, 0.95),
        (r'影象', '影像', ErrorType.TYPING_ERROR, 0.95),
        (r'密谋度', '密度', ErrorType.TYPING_ERROR, 0.95),
        (r'曾粗', '增粗', ErrorType.TYPING_ERROR, 0.95),
        (r'曾强', '增强', ErrorType.TYPING_ERROR, 0.95),
        (r'[^a-zA-Z]\w{10,}[^a-zA-Z]', None, ErrorType.NONSENSE_WORD, 0.80),
    ]
    
    @classmethod
    def check(cls, text: str) -> List[GrammarError]:
        """执行规则检查"""
        errors = []
        
        for pattern, suggestion, error_type, confidence in cls.HIGH_RISK_PATTERNS:
            for match in re.finditer(pattern, text):
                errors.append(GrammarError(
                    error_type=error_type,
                    position=match.start(),
                    text=match.group(),
                    suggestion=suggestion,
                    confidence=confidence,
                    context=text[max(0, match.start()-5):match.end()+5],
                    needs_llm_verify=confidence < 0.95  # 高置信度直接确认，低置信度需要LLM
                ))
        
        return errors


class GrammarDetector:
    """语法错误检测器（分层架构）"""
    
    def __init__(self, model_path: Optional[str] = None, ngram_n: int = 2):
        """
        Args:
            model_path: 预训练N-gram模型路径
            ngram_n: N-gram大小
        """
        self.ngram_model = NgramLanguageModel(n=ngram_n)
        self.llm_validator = LLMValidator()
        
        if model_path:
            self.ngram_model.load(model_path)
    
    def detect(self, text: str, use_llm: bool = True) -> List[GrammarError]:
        """
        分层检测语法错误
        
        Args:
            text: 待检查文本
            use_llm: 是否使用LLM验证
        
        Returns:
            检测到的错误列表
        """
        errors = []
        
        # 第一层：规则检测（高置信度直接输出，低置信度标记待验证）
        rule_errors = RuleBasedChecker.check(text)
        high_conf_errors = [e for e in rule_errors if e.confidence >= 0.95]
        low_conf_errors = [e for e in rule_errors if e.confidence < 0.95]
        errors.extend(high_conf_errors)
        
        # 第二层：N-gram统计检测
        if self.ngram_model.total > 0:
            ngram_anomalies = self.ngram_model.check(text)
            for anomaly in ngram_anomalies:
                # 合并相邻的异常
                if errors and abs(errors[-1].position - anomaly['position']) < 3:
                    continue
                
                errors.append(GrammarError(
                    error_type=ErrorType.STATISTICAL_ANOMALY,
                    position=anomaly['position'],
                    text=anomaly['ngram'],
                    suggestion=None,
                    confidence=0.6,  # 统计异常需要LLM确认
                    context=anomaly['context'],
                    needs_llm_verify=True
                ))
        
        # 第三层：LLM验证（异步）
        if use_llm and (low_conf_errors or any(e.needs_llm_verify for e in errors)):
            verify_errors = [e for e in errors if e.needs_llm_verify] + low_conf_errors
            verified = self._llm_batch_verify(text, verify_errors)
            # 移除需要验证但未通过LLM确认的错误
            errors = [e for e in errors if not e.needs_llm_verify]
            errors.extend(verified)
        
        return errors
    
    async def detect_async(self, text: str) -> List[GrammarError]:
        """异步检测入口"""
        # 可以先做规则检测，然后异步LLM验证
        errors = self.detect(text, use_llm=False)
        
        if any(e.needs_llm_verify for e in errors):
            verify_tasks = []
            for error in errors:
                if error.needs_llm_verify:
                    task = self._llm_verify_single(text, error)
                    verify_tasks.append(task)
            
            # 并行LLM验证
            results = await asyncio.gather(*verify_tasks, return_exceptions=True)
            
            # 合并结果
            final_errors = [e for e in errors if not e.needs_llm_verify]
            for result in results:
                if isinstance(result, GrammarError):
                    final_errors.append(result)
        
        return errors
    
    def _llm_batch_verify(self, text: str, errors: List[GrammarError]) -> List[GrammarError]:
        """批量LLM验证（简化版，实际可优化prompt）"""
        # 这里可以调用llm_service中的验证方法
        # 简化实现：假设都通过验证
        verified = []
        for error in errors:
            # 实际应该构造prompt让LLM判断
            verified.append(error)
        return verified
    
    async def _llm_verify_single(self, text: str, error: GrammarError) -> Optional[GrammarError]:
        """单个错误的LLM验证"""
        # 实际实现
        pass


def quick_check(text: str) -> List[Dict]:
    """快速检查入口（无需加载模型）"""
    detector = GrammarDetector()  # 不使用N-gram模型
    errors = detector.detect(text, use_llm=False)
    return [
        {
            'type': e.error_type.value,
            'text': e.text,
            'suggestion': e.suggestion,
            'confidence': e.confidence,
            'context': e.context
        }
        for e in errors
    ]


# 测试
if __name__ == '__main__':
    test_cases = [
        "肺文里增粗",  # 错别字
        "见低密渡影",  # 错别字
        "胸部影象显示正常",  # 错别字
        "未见明显异常",  # 正常
        "这是一个正常的医学报告描述",  # 正常
    ]
    
    detector = GrammarDetector()
    
    for text in test_cases:
        print(f"\n文本: {text}")
        errors = quick_check(text)
        if errors:
            for e in errors:
                print(f"  错误: {e['text']} -> {e['suggestion']} (置信度: {e['confidence']})")
        else:
            print("  未检测到错误")
