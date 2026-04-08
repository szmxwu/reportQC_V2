#!/usr/bin/env python3
"""
词序错误检测器（保守策略）
只检测高频可靠搭配，忽略低频组合避免假阳性
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class WordOrderDetector:
    """
    基于高频词序模板的错误检测器
    
    检测逻辑：
    1. 正向搭配频次 >> 反向搭配频次
    2. 反向搭配几乎不出现（<5次）
    3. 只检测模板中存在的高频搭配
    """
    
    def __init__(self, templates_path: str = 'models/word_order_templates.json'):
        self.templates_path = templates_path
        self.word_patterns: Dict[str, Dict] = {}  # {错误词: 模式信息}
        self.bigram_templates: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self._load_templates()
    
    def _load_templates(self):
        """加载词序模板"""
        if not Path(self.templates_path).exists():
            print(f"警告: 词序模板文件不存在 {self.templates_path}")
            return
        
        with open(self.templates_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 加载词语级模式（主要检测模式）
        for pattern in data.get('word_patterns', []):
            error_word = pattern['error']
            self.word_patterns[error_word] = pattern
        
        # 加载字符级bigram模板（辅助检测）
        self.bigram_templates = data.get('bigram_templates', {})
        
        print(f"词序检测器加载完成: {len(self.word_patterns)}个词语模式, "
              f"{sum(len(v) for v in self.bigram_templates.values())}个bigram模板")
    
    def detect(self, text: str) -> List[Dict]:
        """
        检测文本中的词序错误
        
        Returns:
            List[{
                'error': str,          # 错误词
                'suggestion': str,     # 建议修正
                'type': 'word_order',  # 错误类型
                'position': (int, int), # 位置
                'confidence': float,    # 置信度（正序/反序比值）
                'forward_freq': int,    # 正序频次
                'backward_freq': int    # 反序频次
            }]
        """
        errors = []
        
        # 1. 词语级模式检测（精确匹配）
        for error_word, pattern in self.word_patterns.items():
            if error_word in text:
                # 确保正序不存在（避免误报）
                correct_word = pattern['correct']
                if correct_word not in text:
                    # 找到所有位置
                    for match in re.finditer(re.escape(error_word), text):
                        errors.append({
                            'error': error_word,
                            'suggestion': correct_word,
                            'type': 'word_order',
                            'position': (match.start(), match.end()),
                            'confidence': pattern['ratio'],
                            'forward_freq': pattern['correct_freq'],
                            'backward_freq': pattern['error_freq']
                        })
        
        # 2. Bigram模板检测（字符级，用于覆盖更多情况）
        # 注：当前简化处理，主要依赖词语级模式
        # 如需更细粒度检测，可在此处实现
        
        return errors
    
    def correct(self, text: str) -> str:
        """
        自动修正文本中的词序错误
        
        Returns:
            修正后的文本
        """
        corrected = text
        errors = self.detect(text)
        
        # 按位置倒序替换，避免位置偏移
        for error in sorted(errors, key=lambda x: -x['position'][0]):
            start, end = error['position']
            corrected = corrected[:start] + error['suggestion'] + corrected[end:]
        
        return corrected


def test_detector():
    """测试词序检测器"""
    detector = WordOrderDetector()
    
    test_cases = [
        # 典型词序错误
        ("双肺纹理增多，异常未见。", "未见异常"),
        ("扫描增强后可见强化。", "增强扫描"),
        ("腰椎增生骨质，建议随访。", "骨质增生"),
        ("心包积液少量，关系正常。", "关系正常/心包积液"),
        ("术后改变同前大致。", "大致同前/术后改变"),
        
        # 正确用法（不应检测）
        ("双肺未见异常。", None),  # 正确使用
        ("骨质增生明显。", None),  # 正确使用
    ]
    
    print("="*70)
    print("词序错误检测测试")
    print("="*70)
    
    for text, expected in test_cases:
        errors = detector.detect(text)
        corrected = detector.correct(text)
        
        print(f"\n原文: {text}")
        if errors:
            for e in errors:
                print(f"  ❌ 词序错误: '{e['error']}' → '{e['suggestion']}' "
                      f"(置信度: {e['confidence']:.0f})")
            print(f"  ✅ 修正: {corrected}")
        else:
            print(f"  ✓ 无词序错误")
        
        if expected and not errors:
            print(f"  ⚠️  漏检: 期望检测到 '{expected}'")
        elif not expected and errors:
            print(f"  ⚠️  误检: 不应检测到错误")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    test_detector()
