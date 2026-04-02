"""
医学报告语法/错别字检测模块 v3.0 - Trigram增强版

分层架构：
    1. 快速召回层：Trigram+Bigram+单字，高召回率
    2. LLM精校层：多线程批量验证，高精度

核心特性：
    - Trigram检测：对医学固定搭配更可靠
    - 多进程训练：充分利用CPU/GPU
    - 多线程LLM：并发验证提高效率

快速开始：
    >>> from grammer import LayeredGrammarDetector
    >>> detector = LayeredGrammarDetector(model_dir='grammer/models')
    >>> errors = detector.detect("肺文里增粗", use_llm=True)
"""

__version__ = '3.1.0'

# 主要接口
from .layered_grammar_detector import LayeredGrammarDetector, GrammarError
from .fast_recover import FastRecoverDetector, SuspiciousFragment

__all__ = [
    'LayeredGrammarDetector',
    'FastRecoverDetector', 
    'GrammarError',
    'SuspiciousFragment',
]
