#!/usr/bin/env python3
"""
高召回率快速检测层 - 多策略融合

设计目标：
1. 高召回率（宁可误报，不可漏报）
2. 不依赖人工维护的词典/规则
3. 利用360万未标注报告学习"正常模式"
4. 轻量级，为LLM筛选可疑片段

检测策略（并行执行）：
├── 策略1: 字符级异常（罕见字、非常用字组合）
├── 策略2: 小型语言模型困惑度（检测语义断裂）
├── 策略3: 上下文一致性（左右熵异常）
├── 策略4: 重复/缺失模式（字符重复、标点异常）
└── 策略5: 长度/结构异常（句子过长/过短）

输出：可疑片段列表，供LLM验证
"""

import os
import sys
import re
import math
import pickle
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class SuspiciousFragment:
    """可疑片段"""
    text: str                      # 可疑文本
    position: Tuple[int, int]      # 在原文中的位置 (start, end)
    strategy: str                  # 检测策略
    score: float                   # 可疑度分数 (0-1)
    context: str                   # 上下文
    reason: str                    # 判定理由


class CharacterAnomalyDetector:
    """
    策略1: 字符级异常检测（支持Trigram）
    
    原理：
    - 从360万报告中统计单字/双字/三字频率
    - Trigram对医学文本更可靠，因为医学术语多为固定3字搭配
    - 例如："肺纹理增粗" - "肺纹理"是正常trigram，"肺文里"是异常
    """
    
    def __init__(self, model_path: str = None, use_trigram: bool = True):
        self.char_freq = Counter()       # 单字频率
        self.bigram_freq = Counter()     # 双字频率
        self.trigram_freq = Counter()    # 三字频率（新增）
        self.total_chars = 0
        self.total_bigrams = 0
        self.total_trigrams = 0
        self.use_trigram = use_trigram   # 是否启用trigram
        
        # 罕见字阈值
        self.rare_threshold = 5
        # Trigram阈值（更严格，因为数据更稀疏）
        self.trigram_threshold = 3
        # 异常分数阈值
        self.score_threshold = 0.7
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, texts: List[str]):
        """从文本列表训练"""
        print(f"训练字符级异常检测器... (trigram={'启用' if self.use_trigram else '禁用'})")
        for text in texts:
            # 只保留中文字符
            chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
            self.char_freq.update(chars)
            self.total_chars += len(chars)
            
            # 双字组合
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i+1]
                self.bigram_freq[bigram] += 1
                self.total_bigrams += 1
            
            # 三字组合（trigram）- 对医学文本更可靠
            if self.use_trigram:
                for i in range(len(chars) - 2):
                    trigram = chars[i] + chars[i+1] + chars[i+2]
                    self.trigram_freq[trigram] += 1
                    self.total_trigrams += 1
    
    def save(self, path: str):
        """保存模型"""
        data = {
            'char_freq': dict(self.char_freq),
            'bigram_freq': dict(self.bigram_freq),
            'total_chars': self.total_chars,
            'total_bigrams': self.total_bigrams,
            'use_trigram': self.use_trigram,
        }
        if self.use_trigram:
            data['trigram_freq'] = dict(self.trigram_freq)
            data['total_trigrams'] = self.total_trigrams
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.char_freq = Counter(data['char_freq'])
            self.bigram_freq = Counter(data['bigram_freq'])
            self.total_chars = data['total_chars']
            self.total_bigrams = data['total_bigrams']
            self.use_trigram = data.get('use_trigram', False)
            
            if self.use_trigram and 'trigram_freq' in data:
                self.trigram_freq = Counter(data['trigram_freq'])
                self.total_trigrams = data.get('total_trigrams', 0)
                print(f"加载Trigram模型: {len(self.trigram_freq)} 个唯一trigram")
    
    def detect(self, text: str) -> List[SuspiciousFragment]:
        """检测字符级异常（支持Trigram）"""
        fragments = []
        chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        
        # 优先检测trigram（最可靠）
        if self.use_trigram and len(chars) >= 3:
            for i in range(len(chars) - 2):
                trigram = chars[i] + chars[i+1] + chars[i+2]
                trigram_count = self.trigram_freq.get(trigram, 0)
                
                if trigram_count < self.trigram_threshold:
                    # Trigram异常分数 - 更严格的阈值
                    score = 0.7 + 0.25 * (1 - trigram_count / self.trigram_threshold)
                    
                    start = max(0, i - 2)
                    end = min(len(chars), i + 5)
                    context = ''.join(chars[start:end])
                    
                    fragments.append(SuspiciousFragment(
                        text=trigram,
                        position=(i, i+3),
                        strategy='trigram_rarity',
                        score=min(score, 0.95),
                        context=context,
                        reason=f"罕见三字组合: '{trigram}'在训练数据中出现{trigram_count}次"
                    ))
        
        # 检测bigram（在没有trigram或trigram不明确时）
        if len(chars) >= 2:
            for i in range(len(chars) - 1):
                # 跳过已被trigram覆盖的区域
                if self.use_trigram and any(
                    f.position[0] <= i < f.position[1] 
                    for f in fragments if f.strategy == 'trigram_rarity'
                ):
                    continue
                
                bigram = chars[i] + chars[i+1]
                bigram_count = self.bigram_freq.get(bigram, 0)
                
                if bigram_count < self.rare_threshold:
                    score = 0.6 + 0.3 * (1 - bigram_count / self.rare_threshold)
                    
                    start = max(0, i - 2)
                    end = min(len(chars), i + 4)
                    context = ''.join(chars[start:end])
                    
                    fragments.append(SuspiciousFragment(
                        text=bigram,
                        position=(i, i+2),
                        strategy='bigram_rarity',
                        score=min(score, 0.9),
                        context=context,
                        reason=f"罕见双字组合: '{bigram}'在训练数据中出现{bigram_count}次"
                    ))
        
        # 单字罕见度（最低优先级）
        for i, char in enumerate(chars):
            # 跳过已被覆盖的区域
            if any(f.position[0] <= i < f.position[1] for f in fragments):
                continue
            
            char_count = self.char_freq.get(char, 0)
            
            if char_count < self.rare_threshold:
                score = 0.5 + 0.2 * (1 - char_count / self.rare_threshold)
                
                start = max(0, i - 3)
                end = min(len(chars), i + 4)
                context = ''.join(chars[start:end])
                
                fragments.append(SuspiciousFragment(
                    text=char,
                    position=(i, i+1),
                    strategy='char_rarity',
                    score=min(score, 0.8),
                    context=context,
                    reason=f"罕见字: '{char}'在训练数据中出现{char_count}次"
                ))
        
        return fragments


class ContextEntropyDetector:
    """
    策略2: 上下文熵异常检测
    
    原理：
    - 计算每个字左右的熵（信息熵）
    - 正常字的左右熵在一定范围内
    - 错别字的上下文通常更"混乱"（熵过高或过低）
    
    例如：
    - "纹"在医学报告中经常跟"理"（熵低，正常）
    - "文"如果跟"里"，这个组合很罕见（熵异常）
    """
    
    def __init__(self, model_path: str = None):
        # 每个字的左右邻居分布
        self.left_context = defaultdict(Counter)   # char -> {left_char: count}
        self.right_context = defaultdict(Counter)  # char -> {right_char: count}
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, texts: List[str]):
        """训练上下文模型"""
        print("训练上下文熵模型...")
        for text in texts:
            chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
            
            for i, char in enumerate(chars):
                # 左上下文
                if i > 0:
                    self.left_context[char][chars[i-1]] += 1
                # 右上下文
                if i < len(chars) - 1:
                    self.right_context[char][chars[i+1]] += 1
    
    def _calculate_entropy(self, counter: Counter) -> float:
        """计算熵"""
        total = sum(counter.values())
        if total == 0:
            return 0
        
        entropy = 0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy
    
    def detect(self, text: str) -> List[SuspiciousFragment]:
        """检测熵异常"""
        fragments = []
        chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        
        for i, char in enumerate(chars):
            left_entropy = self._calculate_entropy(self.left_context[char])
            right_entropy = self._calculate_entropy(self.right_context[char])
            
            # 如果该字在训练数据中很少出现，跳过
            if len(self.left_context[char]) < 3 and len(self.right_context[char]) < 3:
                continue
            
            # 熵异常判断
            # - 熵过高：字的上下文非常多变，可能是异常字
            # - 熵过低：字的上下文过于固定，但当前组合不在常见组合中
            
            avg_entropy = (left_entropy + right_entropy) / 2
            
            # 简单启发式：熵>4或<0.5视为异常
            if avg_entropy > 4 or (avg_entropy < 0.5 and avg_entropy > 0):
                # 检查当前上下文是否见过
                left_char = chars[i-1] if i > 0 else None
                right_char = chars[i+1] if i < len(chars) - 1 else None
                
                context_seen = True
                if left_char and left_char not in self.left_context[char]:
                    context_seen = False
                if right_char and right_char not in self.right_context[char]:
                    context_seen = False
                
                if not context_seen:
                    start = max(0, i - 2)
                    end = min(len(chars), i + 3)
                    context = ''.join(chars[start:end])
                    
                    score = 0.5 + 0.3 * min(avg_entropy / 5, 1)
                    
                    fragments.append(SuspiciousFragment(
                        text=char,
                        position=(i, i+1),
                        strategy='entropy_anomaly',
                        score=min(score, 0.9),
                        context=context,
                        reason=f"上下文熵异常({avg_entropy:.2f})，当前组合罕见"
                    ))
        
        return fragments


class PatternAnomalyDetector:
    """
    策略3: 模式异常检测
    
    检测明显的排版/输入错误：
    - 连续重复字符（如"肺纹理理理"）
    - 异常标点（多个句号、缺少标点）
    - 中英文混用异常（如"肺cT"）
    - 数字格式异常
    """
    
    # 常见错误模式
    PATTERNS = [
        (r'(.)\1{2,}', 'char_repeat', '连续重复字符'),  # 3个以上重复
        (r'[a-zA-Z]{5,}', 'long_english', '异常长英文'),  # 5个以上连续英文
        (r'[\u4e00-\u9fa5][a-zA-Z][\u4e00-\u9fff]', 'cn_en_mix', '中英文异常混合'),
        (r'\d+\.\d+\.\d+\.\d+', 'ip_like', '疑似IP地址'),
        (r'[。，；：]{3,}', 'punctuation_repeat', '标点重复'),
    ]
    
    def detect(self, text: str) -> List[SuspiciousFragment]:
        """检测模式异常"""
        fragments = []
        
        for pattern, pattern_type, desc in self.PATTERNS:
            for match in re.finditer(pattern, text):
                fragments.append(SuspiciousFragment(
                    text=match.group(),
                    position=(match.start(), match.end()),
                    strategy='pattern',
                    score=0.85,
                    context=text[max(0, match.start()-5):match.end()+5],
                    reason=desc
                ))
        
        # 句子长度异常检测
        sentences = re.split(r'[。！？\n]', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 100:  # 超长句子
                start = text.find(sent)
                fragments.append(SuspiciousFragment(
                    text=sent[:30] + "...",
                    position=(start, start + len(sent)),
                    strategy='length',
                    score=0.6,
                    context=sent[:50],
                    reason=f"超长句子({len(sent)}字)，可能缺少标点"
                ))
            elif 0 < len(sent) < 5:  # 超短句子
                start = text.find(sent)
                fragments.append(SuspiciousFragment(
                    text=sent,
                    position=(start, start + len(sent)),
                    strategy='length',
                    score=0.5,
                    context=text[max(0, start-10):start+len(sent)+10],
                    reason=f"超短句子({len(sent)}字)，可能是断句错误"
                ))
        
        return fragments


class FastRecoverDetector:
    """
    快速召回检测器 - 主类（支持Trigram）
    
    整合多种策略，高召回率检测可疑片段
    Trigram检测对医学文本更可靠
    """
    
    def __init__(self, model_dir: str = None, use_trigram: bool = True):
        """
        Args:
            model_dir: 预训练模型目录
            use_trigram: 是否使用trigram检测（推荐启用）
        """
        self.model_dir = model_dir
        self.use_trigram = use_trigram
        
        # 初始化各个检测器
        char_model = f"{model_dir}/char_anomaly.pkl" if model_dir else None
        entropy_model = f"{model_dir}/entropy.pkl" if model_dir else None
        
        self.char_detector = CharacterAnomalyDetector(char_model, use_trigram=use_trigram)
        self.entropy_detector = ContextEntropyDetector(entropy_model)
        self.pattern_detector = PatternAnomalyDetector()
        
        # 合并时的重叠阈值
        self.merge_threshold = 5
    
    def train(self, texts: List[str], output_dir: str):
        """训练所有模型"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 训练字符异常检测器
        self.char_detector.train(texts)
        self.char_detector.save(f"{output_dir}/char_anomaly.pkl")
        
        # 训练熵检测器
        self.entropy_detector.train(texts)
        with open(f"{output_dir}/entropy.pkl", 'wb') as f:
            pickle.dump({
                'left_context': dict(self.entropy_detector.left_context),
                'right_context': dict(self.entropy_detector.right_context)
            }, f)
        
        print(f"模型已保存到: {output_dir}")
    
    def detect(self, text: str, min_score: float = 0.5) -> List[SuspiciousFragment]:
        """
        检测可疑片段
        
        Args:
            text: 待检测文本
            min_score: 最小可疑分数
        
        Returns:
            可疑片段列表（已排序，高分在前）
        """
        if not text or len(text) < 3:
            return []
        
        all_fragments = []
        
        # 并行执行多个检测策略
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.char_detector.detect, text),
                executor.submit(self.entropy_detector.detect, text),
                executor.submit(self.pattern_detector.detect, text),
            ]
            
            for future in futures:
                try:
                    fragments = future.result()
                    all_fragments.extend(fragments)
                except Exception as e:
                    print(f"检测器异常: {e}")
        
        # 过滤低分片段
        all_fragments = [f for f in all_fragments if f.score >= min_score]
        
        # 合并重叠片段
        merged = self._merge_fragments(all_fragments)
        
        # 按分数排序
        merged.sort(key=lambda x: x.score, reverse=True)
        
        return merged
    
    def _merge_fragments(self, fragments: List[SuspiciousFragment]) -> List[SuspiciousFragment]:
        """合并重叠的片段"""
        if not fragments:
            return []
        
        # 按位置排序
        fragments = sorted(fragments, key=lambda x: x.position[0])
        
        merged = []
        current = fragments[0]
        
        for frag in fragments[1:]:
            # 检查是否重叠
            if frag.position[0] <= current.position[1] + self.merge_threshold:
                # 合并：取最大范围，分数加权平均
                new_start = min(current.position[0], frag.position[0])
                new_end = max(current.position[1], frag.position[1])
                new_score = max(current.score, frag.score)
                
                # 选择更长的文本
                new_text = current.text if len(current.text) > len(frag.text) else frag.text
                
                current = SuspiciousFragment(
                    text=new_text,
                    position=(new_start, new_end),
                    strategy=f"{current.strategy}+{frag.strategy}",
                    score=new_score,
                    context=current.context if len(current.context) > len(frag.context) else frag.context,
                    reason=f"{current.reason}; {frag.reason}"
                )
            else:
                merged.append(current)
                current = frag
        
        merged.append(current)
        return merged
    
    def get_top_fragments(self, text: str, top_k: int = 3) -> List[SuspiciousFragment]:
        """获取最可疑的top_k个片段"""
        fragments = self.detect(text)
        return fragments[:top_k]


def quick_detect(text: str, model_dir: str = None) -> List[Dict]:
    """快速检测入口"""
    detector = FastRecoverDetector(model_dir)
    fragments = detector.detect(text)
    
    return [
        {
            'text': f.text,
            'position': f.position,
            'score': f.score,
            'strategy': f.strategy,
            'reason': f.reason,
            'context': f.context
        }
        for f in fragments
    ]


# 测试
if __name__ == '__main__':
    test_cases = [
        "双肺纹理增粗",  # 正常
        "双肺文里增粗",  # 罕见组合
        "肺纹理理理紊乱",  # 重复字符
        "这是一个非常长的句子没有任何标点符号一直在描述肺部的各种情况包括纹理增粗密度增高等等",  # 长句
        "胸部CT正常",  # 正常
    ]
    
    print("=" * 60)
    print("快速召回检测器测试（未训练模式）")
    print("=" * 60)
    
    detector = FastRecoverDetector()
    
    for text in test_cases:
        print(f"\n原文: {text[:50]}{'...' if len(text) > 50 else ''}")
        fragments = detector.detect(text, min_score=0.3)
        if fragments:
            for f in fragments[:3]:  # 只显示top3
                print(f"  ⚠ [{f.strategy}] {f.text} (score={f.score:.2f}): {f.reason}")
        else:
            print("  ✓ 无明显异常")
