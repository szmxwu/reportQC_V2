#!/usr/bin/env python3
"""
医学影像报告语法/错别字检测模块 (v2.0)

基于Confusion Set的分层检测策略：
1. 规则层：基于常见错别字库的快速匹配
2. 统计层：基于白名单的Confusion Set检测
3. LLM验证层：验证可疑错误（可选）

相比纯N-gram的优势：
- 只检测已知混淆字，减少假阳性
- 使用jieba+用户词典理解医学术语
- 支持360万+报告训练的白名单

使用示例：
    # 快速检测（仅规则）
    from grammer.grammar_detector import quick_check
    errors = quick_check("肺文里增粗")
    
    # 完整检测（加载白名单）
    from grammer.grammar_detector import GrammarDetector
    detector = GrammarDetector(whitelist_path='models/confusion_whitelist.pkl')
    errors = detector.detect(report_text)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from grammer.typo_database import detect_typos
from grammer.confusion_checker import ConfusionChecker, GrammarError


class GrammarDetector:
    """
    语法错误检测器（v2.0 - 基于Confusion Set）
    
    这是ConfusionChecker的包装类，保持向后兼容
    """
    
    def __init__(self, whitelist_path: str = None, user_dict_path: str = None):
        """
        Args:
            whitelist_path: 白名单文件路径（可选，来自训练数据）
            user_dict_path: jieba用户词典路径
        """
        if user_dict_path is None:
            user_dict_path = str(REPO_ROOT / 'config/user_dic_expand.txt')
        
        self.checker = ConfusionChecker(
            whitelist_path=whitelist_path,
            user_dict_path=user_dict_path
        )
    
    def detect(self, text: str, use_llm: bool = False) -> List[GrammarError]:
        """
        检测语法错误
        
        Args:
            text: 待检测文本
            use_llm: 是否使用LLM验证（暂不支持）
        
        Returns:
            错误列表
        """
        return self.checker.check(text)
    
    def explain(self, error: GrammarError) -> str:
        """解释错误原因"""
        return self.checker.explain(error)


def quick_check(text: str, whitelist_path: str = None) -> List[Dict]:
    """
    快速检查入口
    
    使用：
        from grammer.grammar_detector import quick_check
        errors = quick_check("肺文里增粗")
    """
    detector = GrammarDetector(whitelist_path=whitelist_path)
    errors = detector.detect(text)
    
    return [
        {
            'position': e.position,
            'text': e.text,
            'suggestion': e.suggestion,
            'type': e.error_type,
            'confidence': e.confidence,
            'context': e.context
        }
        for e in errors
    ]


# 测试
if __name__ == '__main__':
    test_cases = [
        "双肺文里增粗、增多",  # 错别字
        "见低密渡影，边界尚清",  # 错别字
        "肝内末见明显异常",  # 错别字
        "追体边缘骨质增生",  # 错别字
        "胸部CT未见明显异常",  # 正常
    ]
    
    print("=" * 60)
    print("Grammar Detector v2.0 测试")
    print("=" * 60)
    
    detector = GrammarDetector()
    
    for text in test_cases:
        print(f"\n原文: {text}")
        errors = detector.detect(text)
        if errors:
            for e in errors:
                print(f"  ⚠ {e.text} -> {e.suggestion} (conf={e.confidence:.2f})")
        else:
            print("  ✓ 未检测到错误")
