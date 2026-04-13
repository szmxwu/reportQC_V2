#!/usr/bin/env python3
"""
提取高频词序错误检测模板（保守策略）
只检测高频可靠搭配，忽略低频组合避免假阳性
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import ahocorasick


# 保守参数配置
MIN_FREQ = 100              # 正确搭配最小频次
MAX_WRONG_FREQ = 5          # 错误搭配最大频次（反序最多出现5次）
MIN_LENGTH = 2              # 最小词语长度
MAX_LENGTH = 6              # 最大词语长度
MAX_WRONG_RAW_FREQ = 50     # 反序短语在真实语料中的最大允许频次
MIN_RAW_RATIO = 5.0         # 正序/反序在真实语料中的最小比值


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """加载词频表"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('word_freq', {})


def count_phrases_in_corpus(corpus_path: str, phrases: Set[str]) -> Dict[str, int]:
    """在真实语料中统计短语出现次数（按行去空格后计数）。"""
    counts = {phrase: 0 for phrase in phrases}

    if not phrases:
        return counts

    automaton = ahocorasick.Automaton()
    for phrase in phrases:
        automaton.add_word(phrase, phrase)
    automaton.make_automaton()

    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            text = line.replace(' ', '').strip()
            if not text:
                continue
            for _, phrase in automaton.iter(text):
                counts[phrase] += 1

    return counts


def is_valid_word(word: str) -> bool:
    """检查是否是有效的医学词汇"""
    if not word or len(word) < MIN_LENGTH or len(word) > MAX_LENGTH:
        return False
    # 过滤纯数字、纯英文、包含特殊字符的
    if re.match(r'^[0-9]+$', word):
        return False
    if re.match(r'^[a-zA-Z]+$', word):
        return False
    # 过滤包含标点符号的
    if re.search(r'[，。、；：""''（）【】《》]', word):
        return False
    return True


def extract_bigram_templates(word_freq: Dict[str, int]) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """
    提取高频bigram词序模板
    
    策略：对于每个高频词，检查其内部所有相邻字符对
    如果 (w1,w2) 频次高且 (w2,w1) 频次极低，则认为是固定搭配
    
    Returns:
        {w1: {w2: (forward_freq, backward_freq)}}
    """
    templates = defaultdict(dict)
    
    # 筛选高频词（频次>MIN_FREQ）
    high_freq_words = {
        word: freq for word, freq in word_freq.items()
        if is_valid_word(word) and freq >= MIN_FREQ
    }
    
    print(f"高频词汇数量: {len(high_freq_words)}")
    
    # 遍历所有高频词，提取内部bigram
    processed_pairs = set()
    
    for word, freq in high_freq_words.items():
        # 拆分词为字符序列
        chars = list(word)
        
        # 提取相邻字符对
        for i in range(len(chars) - 1):
            w1, w2 = chars[i], chars[i+1]
            
            # 跳过单字（通常没有语义）
            if len(w1) == 1 and len(w2) == 1:
                # 创建配对标识避免重复处理
                pair_key = (w1, w2)
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # 正序搭配
                forward = w1 + w2
                forward_freq = word_freq.get(forward, 0)
                
                if forward_freq < MIN_FREQ:
                    continue  # 正序频次不够高，跳过
                
                # 反序搭配
                backward = w2 + w1
                backward_freq = word_freq.get(backward, 0)
                
                if backward_freq > MAX_WRONG_FREQ:
                    continue  # 反序也常见，不是固定搭配
                
                # 满足条件：高频正序 + 低频反序
                templates[w1][w2] = (forward_freq, backward_freq)
    
    return dict(templates)


def extract_word_order_patterns(word_freq: Dict[str, int]) -> List[Dict]:
    """
    提取完整的词序模式（完整词语级别，而非字符级别）
    
    例如：
    - "未见异常" vs "异常未见"
    - "斑片影" vs "影斑片"
    """
    patterns = []
    
    # 筛选中等长度的高频词（2-4字）
    candidate_words = [
        (word, freq) for word, freq in word_freq.items()
        if is_valid_word(word) and 4 <= len(word) <= 6 and freq >= MIN_FREQ * 10
    ]
    
    print(f"候选词数量: {len(candidate_words)}")
    
    for word, freq in sorted(candidate_words, key=lambda x: -x[1]):
        # 尝试各种拆分方式
        found_patterns = []
        
        # 2+2 拆分（四字词）
        if len(word) == 4:
            part1, part2 = word[:2], word[2:]
            reverse = part2 + part1
            reverse_freq = word_freq.get(reverse, 0)
            
            if reverse_freq <= MAX_WRONG_FREQ and freq / (reverse_freq + 1) > 10:
                found_patterns.append({
                    'correct': word,
                    'error': reverse,
                    'type': '2+2',
                    'part1': part1,
                    'part2': part2,
                    'correct_freq': freq,
                    'error_freq': reverse_freq,
                    'ratio': freq / (reverse_freq + 1)
                })
        
        # 2+1 或 1+2 拆分（三字词）
        elif len(word) == 3:
            # 尝试 1+2
            part1, part2 = word[0], word[1:]
            reverse = part2 + part1
            reverse_freq = word_freq.get(reverse, 0)
            
            if reverse_freq <= MAX_WRONG_FREQ and freq / (reverse_freq + 1) > 10:
                found_patterns.append({
                    'correct': word,
                    'error': reverse,
                    'type': '1+2',
                    'part1': part1,
                    'part2': part2,
                    'correct_freq': freq,
                    'error_freq': reverse_freq,
                    'ratio': freq / (reverse_freq + 1)
                })
            
            # 尝试 2+1
            part1, part2 = word[:2], word[2:]
            reverse = part2 + part1
            reverse_freq = word_freq.get(reverse, 0)
            
            if reverse_freq <= MAX_WRONG_FREQ and freq / (reverse_freq + 1) > 10:
                found_patterns.append({
                    'correct': word,
                    'error': reverse,
                    'type': '2+1',
                    'part1': part1,
                    'part2': part2,
                    'correct_freq': freq,
                    'error_freq': reverse_freq,
                    'ratio': freq / (reverse_freq + 1)
                })
        
        # 选择最佳模式（ratio最高的）
        if found_patterns:
            best = max(found_patterns, key=lambda x: x['ratio'])
            if best['ratio'] >= 100:  # 至少100倍差异
                patterns.append(best)
    
    return patterns


def validate_patterns_with_raw_corpus(
    patterns: List[Dict],
    corpus_path: str,
) -> List[Dict]:
    """用真实语料频次二次过滤词序模板，抑制高频反序短语误报。"""
    if not patterns:
        return patterns

    phrases = set()
    for pattern in patterns:
        phrases.add(pattern['correct'])
        phrases.add(pattern['error'])

    raw_counts = count_phrases_in_corpus(corpus_path, phrases)

    validated = []
    filtered = 0
    for pattern in patterns:
        raw_correct = raw_counts.get(pattern['correct'], 0)
        raw_error = raw_counts.get(pattern['error'], 0)
        raw_ratio = raw_correct / (raw_error + 1)

        # 若反序在真实语料中本身高频，或正反序差异不明显，则视为可接受表达，不入错序模板。
        if raw_error > MAX_WRONG_RAW_FREQ or raw_ratio < MIN_RAW_RATIO:
            filtered += 1
            continue

        new_pattern = dict(pattern)
        new_pattern['raw_correct_freq'] = raw_correct
        new_pattern['raw_error_freq'] = raw_error
        new_pattern['raw_ratio'] = raw_ratio
        validated.append(new_pattern)

    print(f"真实语料过滤: 过滤 {filtered} 个，保留 {len(validated)} 个")
    return validated


def save_templates(templates: Dict, patterns: List[Dict], output_path: str):
    """保存模板到JSON文件"""
    output = {
        'parameters': {
            'min_freq': MIN_FREQ,
            'max_wrong_freq': MAX_WRONG_FREQ,
            'max_wrong_raw_freq': MAX_WRONG_RAW_FREQ,
            'min_raw_ratio': MIN_RAW_RATIO,
            'min_length': MIN_LENGTH,
            'max_length': MAX_LENGTH
        },
        'bigram_templates': templates,
        'word_patterns': patterns,
        'stats': {
            'bigram_count': sum(len(v) for v in templates.values()),
            'word_pattern_count': len(patterns)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return output['stats']


def main():
    import sys
    
    vocab_path = 'models/radiology_vocab.json'
    corpus_path = 'models/radiology_corpus.txt'
    output_path = 'models/word_order_templates.json'
    
    print("="*60)
    print("提取词序错误检测模板（保守策略）")
    print("="*60)
    print(f"参数: 最小频次={MIN_FREQ}, 最大错误频次={MAX_WRONG_FREQ}")
    print(f"      真实语料反序上限={MAX_WRONG_RAW_FREQ}, 真实语料最小比值={MIN_RAW_RATIO}")
    print()
    
    # 加载词频表
    print(f"加载词频表: {vocab_path}")
    word_freq = load_vocab(vocab_path)
    print(f"总词汇量: {len(word_freq)}")
    print()
    
    # 提取bigram模板
    print("提取字符级bigram模板...")
    bigram_templates = extract_bigram_templates(word_freq)
    
    # 提取词语级模式
    print("提取词语级模式...")
    word_patterns = extract_word_order_patterns(word_freq)

    # 真实语料二次过滤
    print(f"真实语料频次校验: {corpus_path}")
    word_patterns = validate_patterns_with_raw_corpus(word_patterns, corpus_path)
    
    # 保存结果
    print(f"\n保存模板: {output_path}")
    stats = save_templates(bigram_templates, word_patterns, output_path)
    
    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    print(f"Bigram模板数量: {stats['bigram_count']}")
    print(f"词语级模式数量: {stats['word_pattern_count']}")
    
    # 显示Top20词语级模式
    if word_patterns:
        print("\nTop20 高频词序模式:")
        print("-" * 60)
        for p in sorted(word_patterns, key=lambda x: -x['ratio'])[:20]:
            print(f"  错误: {p['error']:<8} → 正确: {p['correct']:<8} "
                  f"(正序{p['correct_freq']:>8}次, 反序{p['error_freq']:>3}次, "
                  f"差异{p['ratio']:>10.0f}倍)")
    
    print("\n" + "="*60)
    print("完成！")


if __name__ == '__main__':
    main()
