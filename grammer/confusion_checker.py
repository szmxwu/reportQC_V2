"""
基于Confusion Set的语法错误检测器

核心思想：
1. 不检测所有罕见n-gram（假阳性高）
2. 只检测已知的混淆字（文/纹、渡/度等）
3. 通过白名单过滤正常用法
4. 使用jieba分词理解词边界

使用：
    from grammer.confusion_checker import ConfusionChecker
    
    checker = ConfusionChecker(
        whitelist_path='models/confusion_whitelist.pkl',
        user_dict_path='config/user_dic_expand.txt'
    )
    
    errors = checker.check("肺文里增粗")  # -> [{'text': '肺文里', 'suggestion': '肺纹理'}]
"""

import os
import sys
import pickle
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

import jieba

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from grammer.typo_database import CONFUSION_PAIRS, detect_typos


@dataclass
class GrammarError:
    """语法错误结果"""
    position: int
    text: str
    suggestion: str
    error_type: str
    confidence: float
    context: str
    reason: str


class ConfusionChecker:
    """
    基于混淆集的语法检测器
    
    相比纯N-gram的优势：
    - 只关注已知错误模式，减少假阳性
    - 使用jieba+用户词典理解医学术语
    - 白名单机制过滤正常用法
    """
    
    # 核心混淆对（错字 -> 正确字）
    CORE_CONFUSIONS = {
        '文': '纹',      # 肺文里 -> 肺纹理
        '渡': '度',      # 低密渡 -> 低密度
        '末': '未',      # 末见 -> 未见
        '曾': '增',      # 曾强 -> 增强
        '象': '像',      # 影象 -> 影像
        '也': '野',      # 肺也 -> 肺野
        '追': '椎',      # 追体 -> 椎体
        '胜': '脏',      # 肾胜 -> 肾脏
        '记': '剂',      # 造影记 -> 造影剂
        '岩': '炎',      # 岩症 -> 炎症
        '买': '脉',      # 动买 -> 动脉
        '伟': '位',      # 占伟 -> 占位
        '苗': '描',      # 扫苗 -> 扫描
        '种': '肿',      # 种块 -> 肿块
        '话': '化',      # 机话 -> 机化
    }
    
    def __init__(self, whitelist_path: str = None, user_dict_path: str = None):
        """
        Args:
            whitelist_path: 白名单文件路径（可选）
            user_dict_path: jieba用户词典路径
        """
        self.whitelist: Dict[str, Set[str]] = {}
        
        # 加载白名单
        if whitelist_path and os.path.exists(whitelist_path):
            with open(whitelist_path, 'rb') as f:
                self.whitelist = pickle.load(f)
            print(f"加载白名单: {len(self.whitelist)} 个混淆字")
        else:
            print("警告: 未加载白名单，将使用基础规则检测")
        
        # 加载jieba用户词典
        if user_dict_path and os.path.exists(user_dict_path):
            print(f"加载jieba词典: {user_dict_path}")
            jieba.load_userdict(user_dict_path)
            jieba.initialize()
        else:
            print(f"警告: 用户词典未找到: {user_dict_path}")
    
    def check(self, text: str, min_confidence: float = 0.8) -> List[GrammarError]:
        """
        检测文本中的语法错误
        
        Args:
            text: 待检测文本
            min_confidence: 最小置信度
        
        Returns:
            错误列表
        """
        errors = []
        
        # 第一层：基于规则库的快速检测
        rule_errors = detect_typos(text, min_confidence=min_confidence)
        for err in rule_errors:
            errors.append(GrammarError(
                position=err['position'],
                text=err['text'],
                suggestion=err['suggestion'],
                error_type=err['type'],
                confidence=err['confidence'],
                context=err['context'],
                reason='rule_based'
            ))
        
        # 第二层：基于混淆集的检测（如果有白名单）
        if self.whitelist:
            confusion_errors = self._check_with_whitelist(text, min_confidence)
            # 合并结果，去重
            errors = self._merge_errors(errors, confusion_errors)
        
        # 第三层：jieba分词验证
        jieba_errors = self._check_with_jieba(text, min_confidence)
        errors = self._merge_errors(errors, jieba_errors)
        
        # 按位置排序
        errors.sort(key=lambda x: x.position)
        
        return errors
    
    def _check_with_whitelist(self, text: str, min_confidence: float) -> List[GrammarError]:
        """使用白名单检测"""
        errors = []
        chinese_text = ''.join(c for c in text if '\u4e00' <= c <= '\u9fff')
        
        for i, char in enumerate(chinese_text):
            if char not in self.CORE_CONFUSIONS:
                continue
            
            # 提取上下文（前后各2字）
            context_start = max(0, i - 2)
            context_end = min(len(chinese_text), i + 3)
            context = chinese_text[context_start:context_end]
            
            # 检查是否在白名单中
            if char in self.whitelist:
                # 如果上下文在白名单中，视为正常
                if context in self.whitelist[char]:
                    continue
                
                # 如果上下文不在白名单，可能是错误
                # 构造建议修正
                correct_char = self.CORE_CONFUSIONS[char]
                suggestion = context.replace(char, correct_char, 1)
                
                # 计算置信度：基于白名单覆盖度
                total_contexts = len(self.whitelist[char])
                if total_contexts > 0:
                    confidence = 0.6 + 0.3 * (1 / (total_contexts + 1))
                else:
                    confidence = 0.7
                
                if confidence >= min_confidence:
                    errors.append(GrammarError(
                        position=i,
                        text=context,
                        suggestion=suggestion,
                        error_type='confusion',
                        confidence=confidence,
                        context=text[max(0, i-5):i+len(context)+5],
                        reason=f'{char}可能是{correct_char}的误写'
                    ))
        
        return errors
    
    def _check_with_jieba(self, text: str, min_confidence: float) -> List[GrammarError]:
        """使用jieba分词检测"""
        errors = []
        
        # 分词
        words = list(jieba.cut(text))
        
        # 检查每个词是否包含可疑的单字
        word_positions = self._get_word_positions(text, words)
        
        for word, (start, end) in zip(words, word_positions):
            # 检查词是否包含混淆字
            for char in word:
                if char in self.CORE_CONFUSIONS:
                    # 如果包含混淆字，但整个词不在词典中，可能是错误
                    # 这里简化处理：假设jieba已经用了用户词典
                    # 如果分词结果奇怪（如单字被单独分出），可能是错误
                    if len(word) == 1 and char in self.CORE_CONFUSIONS:
                        # 单字被单独分出，可能是术语识别失败
                        pass  # 暂不处理，避免假阳性
        
        return errors
    
    def _get_word_positions(self, text: str, words: List[str]) -> List[Tuple[int, int]]:
        """获取每个词在原文中的位置"""
        positions = []
        pos = 0
        for word in words:
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            positions.append((start, end))
            pos = end
        return positions
    
    def _merge_errors(self, errors1: List[GrammarError], 
                     errors2: List[GrammarError]) -> List[GrammarError]:
        """合并错误列表，去重"""
        # 使用位置和文本作为键
        seen = set()
        merged = []
        
        for err in errors1 + errors2:
            key = (err.position, err.text)
            if key not in seen:
                seen.add(key)
                merged.append(err)
        
        return merged
    
    def explain(self, error: GrammarError) -> str:
        """解释错误原因"""
        explanations = {
            'typo': '常见错别字',
            'pinyin': '拼音输入错误',
            'confusion': '形近字混淆',
            'pattern': '短语模式匹配',
        }
        return f"{explanations.get(error.error_type, error.error_type)}: {error.reason}"


def quick_check(text: str, whitelist_path: str = None) -> List[Dict]:
    """快速检查入口"""
    user_dict = str(REPO_ROOT / 'config/user_dic_expand.txt')
    checker = ConfusionChecker(whitelist_path, user_dict)
    errors = checker.check(text)
    
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


if __name__ == '__main__':
    # 测试
    test_cases = [
        "双肺文里增粗、增多",
        "见低密渡影，边界尚清",
        "肝内末见明显异常",
        "胸部影象显示正常",
        "追体边缘骨质增生",
        "未见异常",  # 正常文本
    ]
    
    print("=" * 60)
    print("Confusion Set 语法检测器测试")
    print("=" * 60)
    
    checker = ConfusionChecker(
        user_dict_path=str(REPO_ROOT / 'config/user_dic_expand.txt')
    )
    
    for text in test_cases:
        print(f"\n原文: {text}")
        errors = checker.check(text)
        if errors:
            for err in errors:
                print(f"  ⚠ {err.text} -> {err.suggestion} ({checker.explain(err)}, conf={err.confidence:.2f})")
        else:
            print("  ✓ 未检测到错误")
