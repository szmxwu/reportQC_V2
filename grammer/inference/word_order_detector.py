#!/usr/bin/env python3
"""
词序错误检测器（保守策略）
只检测高频可靠搭配，忽略低频组合避免假阳性
"""

import json
import re
import jieba
from typing import List, Dict, Optional, Tuple
from pathlib import Path


SENTENCE_SPLIT_PATTERN = re.compile(r'[。！？；，、：:;,.!?\n\r]+')


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
        self.word_patterns: Dict[str, List[Dict]] = {}  # {错误词: [模式信息]}
        self.bigram_templates: Dict[str, Dict[str, Tuple[int, int]]] = {}
        self.vocab_freq: Dict[str, int] = {}
        self.user_dict_loaded = False
        self.max_window_tokens = 6
        self.max_pattern_chars = 6
        self.min_chunk_freq = 20
        self._load_templates()

    def _load_user_dict(self):
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

    @staticmethod
    def _split_sentences_with_positions(text: str) -> List[Tuple[str, int, int]]:
        sentences = []
        start = 0
        for match in SENTENCE_SPLIT_PATTERN.finditer(text):
            end = match.start()
            if end > start:
                sentence = text[start:end].strip()
                if sentence:
                    sentences.append((sentence, start, end))
            start = match.end()

        if start < len(text):
            sentence = text[start:].strip()
            if sentence:
                sentences.append((sentence, start, len(text)))

        return sentences

    def _tokenize_with_positions(self, text: str, base_offset: int = 0) -> List[Tuple[str, int, int]]:
        self._load_user_dict()
        tokens = []
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
            tokens.append((word, base_offset + start, base_offset + end))
            search_start = end
        return tokens

    @staticmethod
    def _derive_pattern_parts(pattern: Dict) -> Tuple[str, str]:
        if pattern.get('part1') and pattern.get('part2'):
            return pattern['part1'], pattern['part2']

        correct = pattern['correct']
        pattern_type = pattern.get('type', '')

        if pattern_type == '2+2' and len(correct) == 4:
            return correct[:2], correct[2:]
        if pattern_type == '1+2' and len(correct) == 3:
            return correct[:1], correct[1:]
        if pattern_type == '2+1' and len(correct) == 3:
            return correct[:2], correct[2:]

        mid = len(correct) // 2
        return correct[:mid], correct[mid:]

    def _load_vocab_freq(self):
        vocab_candidates = [
            Path(self.templates_path).resolve().parent / 'radiology_vocab.json',
            Path(__file__).resolve().parents[1] / 'models' / 'radiology_vocab.json',
        ]

        for vocab_path in vocab_candidates:
            if not vocab_path.exists():
                continue
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.vocab_freq = data.get('word_freq', {})
                break
            except Exception:
                continue

    def _chunk_freq(self, chunk: str) -> int:
        if not chunk:
            return 0
        if len(chunk) == 1:
            return 1
        return int(self.vocab_freq.get(chunk, 0))

    def _has_token_split(self, token_window: List[Tuple[str, int, int]], left_chunk: str, right_chunk: str) -> bool:
        if len(token_window) < 2:
            return False

        for split_idx in range(1, len(token_window)):
            left = ''.join(token for token, _, _ in token_window[:split_idx])
            right = ''.join(token for token, _, _ in token_window[split_idx:])
            if left == left_chunk and right == right_chunk:
                return True

        return False

    def _single_token_fallback_ok(self, pattern: Dict) -> bool:
        left_chunk, right_chunk = pattern['error_parts']
        left_freq = self._chunk_freq(left_chunk)
        right_freq = self._chunk_freq(right_chunk)

        # 当分词把整个错误短语吞成一个 token 时，仍要求两个 chunk 至少有一侧是稳定短语。
        if len(left_chunk) >= 2 and left_freq >= self.min_chunk_freq:
            return True
        if len(right_chunk) >= 2 and right_freq >= self.min_chunk_freq:
            return True
        return False

    def _is_valid_window(self, sentence: str, token_window: List[Tuple[str, int, int]], pattern: Dict) -> bool:
        if not token_window:
            return False

        candidate = ''.join(token for token, _, _ in token_window)
        if candidate != pattern['error']:
            return False

        if pattern['correct'] in sentence:
            return False

        error_left, error_right = pattern['error_parts']
        if self._has_token_split(token_window, error_left, error_right):
            return True

        if len(token_window) == 1 and self._single_token_fallback_ok(pattern):
            return True

        return False
    
    def _load_templates(self):
        """加载词序模板"""
        if not Path(self.templates_path).exists():
            print(f"警告: 词序模板文件不存在 {self.templates_path}")
            return
        
        with open(self.templates_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._load_vocab_freq()
        
        # 加载词语级模式（主要检测模式）
        for pattern in data.get('word_patterns', []):
            error_word = pattern['error']
            part1, part2 = self._derive_pattern_parts(pattern)
            normalized = dict(pattern)
            normalized['part1'] = part1
            normalized['part2'] = part2
            normalized['correct_parts'] = [part1, part2]
            normalized['error_parts'] = [part2, part1]
            self.word_patterns.setdefault(error_word, []).append(normalized)
        
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

        for sentence, sent_start, _ in self._split_sentences_with_positions(text):
            tokens = self._tokenize_with_positions(sentence, base_offset=sent_start)
            if not tokens:
                continue

            for i in range(len(tokens)):
                window_tokens: List[Tuple[str, int, int]] = []
                for j in range(i, min(len(tokens), i + self.max_window_tokens)):
                    token, start, end = tokens[j]
                    if window_tokens and start != window_tokens[-1][2]:
                        break

                    window_tokens.append((token, start, end))
                    candidate = ''.join(item[0] for item in window_tokens)
                    if len(candidate) > self.max_pattern_chars:
                        break

                    patterns = self.word_patterns.get(candidate, [])
                    if not patterns:
                        continue

                    for pattern in patterns:
                        if not self._is_valid_window(sentence, window_tokens, pattern):
                            continue

                        start_pos = window_tokens[0][1]
                        end_pos = window_tokens[-1][2]
                        errors.append({
                            'error': pattern['error'],
                            'suggestion': pattern['correct'],
                            'type': 'word_order',
                            'position': (start_pos, end_pos),
                            'confidence': pattern['ratio'],
                            'forward_freq': pattern['correct_freq'],
                            'backward_freq': pattern['error_freq'],
                            'pattern_type': pattern.get('type'),
                            'part1': pattern.get('part1'),
                            'part2': pattern.get('part2'),
                        })

        deduped = []
        seen = set()
        for item in errors:
            key = (item['error'], item['suggestion'], item['position'])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped
    
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
