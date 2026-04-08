#!/usr/bin/env python3
"""
构建子串频次表（基于原始未分词文本）

用于解决分词粒度不一致导致的假阳性问题
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import RADIOLOGY_VOCAB


def build_substring_vocab(
    data_path: str = '/home/wmx/work/python/Radiology_Entities/radiology_data/all_data_sample.xlsx',
    text_columns: list = ['描述', '结论'],
    min_freq: int = 1,
    output_path: str = None
) -> dict:
    """
    从原始Excel数据构建子串频次表
    
    Args:
        data_path: Excel数据文件路径
        text_columns: 要处理的文本列
        min_freq: 最小频次阈值
        output_path: 输出JSON路径
    
    Returns:
        {子串: 频次}
    """
    print(f"读取数据: {data_path}")
    df = pd.read_excel(data_path)
    print(f"总记录数: {len(df)}")
    
    # 收集所有候选子串（从现有的词汇表）
    with open(RADIOLOGY_VOCAB, 'r') as f:
        vocab_data = json.load(f)
    
    # 提取所有可能的混淆词（2-4字）
    candidate_substrings = set()
    for word in vocab_data['word_freq'].keys():
        if 2 <= len(word) <= 4:
            candidate_substrings.add(word)
    
    print(f"候选子串数量: {len(candidate_substrings)}")
    
    # 在原始文本中统计子串出现次数
    substring_counts = Counter()
    
    total_texts = 0
    for col in text_columns:
        if col not in df.columns:
            continue
        
        texts = df[col].dropna().astype(str).tolist()
        total_texts += len(texts)
        
        for text in tqdm(texts, desc=f"处理 {col}"):
            for substr in candidate_substrings:
                if substr in text:
                    substring_counts[substr] += text.count(substr)
    
    print(f"处理文本总数: {total_texts}")
    print(f"有频次的子串数: {len(substring_counts)}")
    
    # 过滤低频
    result = {k: v for k, v in substring_counts.items() if v >= min_freq}
    
    # 保存
    if output_path is None:
        output_path = 'models/substring_vocab.json'
    
    output_data = {
        'metadata': {
            'source': data_path,
            'text_columns': text_columns,
            'total_texts': total_texts,
            'min_freq': min_freq
        },
        'substring_freq': result
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"子串频次表已保存: {output_path}")
    print(f"总子串数: {len(result)}")
    
    return result


def load_substring_vocab(path: str = 'models/substring_vocab.json') -> dict:
    """加载子串频次表"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['substring_freq']


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='构建子串频次表')
    parser.add_argument('--data', default='/home/wmx/work/python/Radiology_Entities/radiology_data/all_data_sample.xlsx')
    parser.add_argument('-o', '--output', default='models/substring_vocab.json')
    parser.add_argument('--min-freq', type=int, default=1)
    
    args = parser.parse_args()
    
    build_substring_vocab(args.data, output_path=args.output, min_freq=args.min_freq)
