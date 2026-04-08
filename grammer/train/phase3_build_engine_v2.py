#!/usr/bin/env python3
"""
阶段三（改进版）：AC自动机引擎构建 + 分词后匹配

解决子串误报问题：
- "见一" 不应该匹配 "下见一" 中的子串
- 解决方案：检测时先分词，匹配分词后的词语
"""

import pickle
import jieba
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import ahocorasick
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class MatchResult:
    error: str
    suggestion: str
    error_type: str
    position: Tuple[int, int]
    score: Optional[float] = None


class FastMedicalCorrectorV2:
    """
    改进版医学纠错引擎
    
    支持分词后匹配，避免子串误报
    """
    
    def __init__(self):
        self.automaton = ahocorasick.Automaton()
        self.confusion_pairs: Dict[str, str] = {}  # 错误词 -> 正确词
        self.high_risk_words: Dict[str, float] = {}  # 高危词 -> 风险分
        self.loaded = False
        self.user_dict_loaded = False
        self.blocked_single_char_combine_chars = {
            '及', '和', '与', '或', '并', '后', '前', '约', '内', '外',
            '左', '右', '双', '上', '下', '中', '未', '无', '见', '起'
        }

    def _load_user_dict(self):
        """加载医学词典，改善放射文本分词边界。"""
        if self.user_dict_loaded:
            return

        user_dict_candidates = [
            Path(__file__).resolve().parents[2] / 'config' / 'user_dic_expand.txt',
            Path(__file__).resolve().parents[1] / 'config' / 'user_dic_expand.txt',
        ]

        for user_dict_path in user_dict_candidates:
            if user_dict_path.exists():
                jieba.load_userdict(str(user_dict_path))
                self.user_dict_loaded = True
                break

        if not self.user_dict_loaded:
            self.user_dict_loaded = True

    def _tokenize_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        """返回带原文位置的分词结果，避免空白字符导致的位置偏移。"""
        self._load_user_dict()

        tokens: List[Tuple[str, int, int]] = []
        search_start = 0

        for word in jieba.cut(text):
            if not word:
                continue

            start = text.find(word, search_start)
            if start == -1:
                start = text.find(word)
                if start == -1:
                    continue

            end = start + len(word)
            tokens.append((word, start, end))
            search_start = end

        return tokens
    
    def add_confusion_pair(self, wrong: str, correct: str):
        """添加混淆对"""
        if wrong not in self.confusion_pairs:
            self.confusion_pairs[wrong] = correct
            self.automaton.add_word(wrong, ('typo', wrong, correct))
    
    def add_high_risk_word(self, word: str, score: float):
        """添加高危词"""
        if word not in self.high_risk_words:
            self.high_risk_words[word] = score
            self.automaton.add_word(word, ('general_high_risk', word, None, score))
    
    def build(self):
        """构建AC自动机"""
        self.automaton.make_automaton()
        self.loaded = True
        print(f"AC自动机构建完成: {len(self.confusion_pairs)}个混淆对, {len(self.high_risk_words)}个高危词")
    
    def scan(self, text: str) -> List[MatchResult]:
        """
        扫描文本（支持分词后匹配）
        
        策略：
        1. 先对文本分词
        2. 在分词后的词语中匹配（避免子串误报如"见一"匹配"下见一"）
        3. 对于连续的单字，检查是否构成混淆对（如"扫"+"瞄"="扫瞄"）
        4. 高危词使用原始子串匹配
        """
        if not self.loaded:
            raise RuntimeError("引擎未构建，请先调用build()")
        
        matches = []
        matched_positions = set()
        
        # 策略1：分词后匹配
        tokens = self._tokenize_with_positions(text)
        
        i = 0
        while i < len(tokens):
            word, start, end = tokens[i]
            word_len = len(word)
            
            # 情况A：整个词在混淆对中
            if word in self.confusion_pairs:
                if start not in matched_positions:
                    matches.append(MatchResult(
                        error=word,
                        suggestion=self.confusion_pairs[word],
                        error_type='typo',
                        position=(start, end)
                    ))
                    matched_positions.add(start)
            
            # 情况B：连续单字组合成混淆对（如"扫"+"瞄"）
            if (
                word_len == 1
                and word not in self.blocked_single_char_combine_chars
                and i + 1 < len(tokens)
            ):
                # 尝试组合后续单字
                combined = word
                combined_positions = [(start, end)]
                j = i + 1
                
                while j < len(tokens) and len(tokens[j][0]) == 1 and len(combined) < 4:
                    next_word, next_start, next_end = tokens[j]
                    if next_word in self.blocked_single_char_combine_chars:
                        break
                    if next_start != combined_positions[-1][1]:
                        break

                    combined += next_word
                    combined_positions.append((next_start, next_end))
                    
                    if combined in self.confusion_pairs:
                        start = combined_positions[0][0]
                        if start not in matched_positions:
                            end = combined_positions[-1][1]
                            matches.append(MatchResult(
                                error=combined,
                                suggestion=self.confusion_pairs[combined],
                                error_type='typo',
                                position=(start, end)
                            ))
                            matched_positions.add(start)
                        break
                    j += 1

            i += 1
        
        # 策略2：高危词使用原始子串匹配
        for end_idx, value in self.automaton.iter(text):
            err_type, err_word, suggestion, *score = value
            if err_type == 'general_high_risk':
                start = end_idx - len(err_word) + 1
                if start not in matched_positions:
                    matches.append(MatchResult(
                        error=err_word,
                        suggestion=None,
                        error_type='general_high_risk',
                        position=(start, end_idx + 1),
                        score=score[0] if score else None
                    ))
                    matched_positions.add(start)
        
        matches.sort(key=lambda x: x.position[0])
        return matches
    
    def correct(self, text: str) -> Dict:
        """纠错：返回修正后的文本"""
        matches = self.scan(text)
        
        # 按位置倒序替换
        result = text
        replacements = []
        
        for m in sorted(matches, key=lambda x: -x.position[0]):
            if m.suggestion:  # 只替换有建议的
                start, end = m.position
                result = result[:start] + m.suggestion + result[end:]
                replacements.append({
                    'error': m.error,
                    'suggestion': m.suggestion,
                    'type': m.error_type,
                    'position': m.position
                })
        
        return {
            'source': text,
            'target': result,
            'errors': replacements
        }
    
    def load_blacklists(self, confusion_path: str, high_risk_path: str):
        """从黑名单文件加载"""
        # 加载混淆对
        with open(confusion_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    wrong, correct = parts[0], parts[1]
                    self.add_confusion_pair(wrong, correct)
        
        # 加载高危词
        with open(high_risk_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    word, pos, huqie_freq, radio_freq, score = parts[:5]
                    self.add_high_risk_word(word, float(score))
        
        self.build()
    
    def save(self, path: str):
        """保存引擎"""
        with open(path, 'wb') as f:
            pickle.dump({
                'confusion_pairs': self.confusion_pairs,
                'high_risk_words': self.high_risk_words
            }, f)
        print(f"引擎已保存: {path}")
    
    def load(self, path: str):
        """加载引擎"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.confusion_pairs = data['confusion_pairs']
            self.high_risk_words = data['high_risk_words']
        
        # 重建自动机
        for wrong, correct in self.confusion_pairs.items():
            self.automaton.add_word(wrong, ('typo', wrong, correct))
        for word, score in self.high_risk_words.items():
            self.automaton.add_word(word, ('general_high_risk', word, None, score))
        
        self.build()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'confusion_pairs': len(self.confusion_pairs),
            'high_risk_words': len(self.high_risk_words),
            'total_patterns': len(self.confusion_pairs) + len(self.high_risk_words),
            'loaded': self.loaded
        }


def build_engine_v2():
    """构建改进版引擎"""
    try:
        from utils.config import MEDICAL_CONFUSION, HIGH_RISK_GENERAL, OUTPUT_DIR
    except ImportError:
        from config import MEDICAL_CONFUSION, HIGH_RISK_GENERAL, OUTPUT_DIR
    
    corrector = FastMedicalCorrectorV2()
    corrector.load_blacklists(
        str(MEDICAL_CONFUSION),
        str(HIGH_RISK_GENERAL)
    )
    corrector.save(str(Path(OUTPUT_DIR) / 'ac_automaton_v2.pkl'))
    return corrector


if __name__ == '__main__':
    # 测试
    corrector = build_engine_v2()
    
    test_texts = [
        "左肺上叶前段胸膜下见一微小结节",  # "见一" 不应该被检测
        "双肺文里增粗，异常未见",  # "文里" 应该被检测
        "建议患者前往阿里巴巴公司",  # "阿里巴巴" 应该被检测
    ]
    
    print("\n测试:")
    for text in test_texts:
        print(f"\n原文: {text}")
        matches = corrector.scan(text)
        if matches:
            for m in matches:
                if m.suggestion:
                    print(f"  - {m.error} -> {m.suggestion} ({m.error_type})")
                else:
                    print(f"  - {m.error} (风险: {m.score:.1f})")
        else:
            print("  无错误")
