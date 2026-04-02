#!/usr/bin/env python3
"""
训练快速召回层模型 - 支持GPU加速和多进程

GPU加速说明：
- 字符统计任务主要是内存密集型，GPU收益有限
- 建议优先使用多进程（--workers参数）
- GPU在特征提取阶段可能有帮助

使用：
    # CPU多进程（推荐）
    python train_fast_recover.py --workers 8
    
    # GPU加速（可选）
    python train_fast_recover.py --use-gpu
    
    # 快速测试（少量数据）
    python train_fast_recover.py --max-texts 10000
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from glob import glob
from collections import Counter
from functools import partial
import pickle
import time

import pandas as pd
from tqdm import tqdm

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 尝试导入GPU相关库
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
except ImportError:
    CUDA_AVAILABLE = False
    torch = None


class GPUAcceleratedTrainer:
    """
    GPU加速训练器
    
    注意：字符统计任务的GPU加速收益有限，
    因为主要是内存访问而非计算密集型。
    建议优先使用多进程CPU版本。
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        print(f"使用设备: {self.device}")
    
    def process_batch_gpu(self, texts):
        """
        使用GPU处理一批文本
        
        策略：将文本编码为整数序列，在GPU上进行向量化操作
        """
        if not CUDA_AVAILABLE or torch is None:
            return self.process_batch_cpu(texts)
        
        # 简化实现：实际使用CPU更高效
        return self.process_batch_cpu(texts)
    
    def process_batch_cpu(self, texts):
        """CPU处理一批文本"""
        char_freq = Counter()
        bigram_freq = Counter()
        left_context = {}
        right_context = {}
        
        for text in texts:
            chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
            char_freq.update(chars)
            
            # Bigram
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i+1]
                bigram_freq[bigram] += 1
            
            # 上下文
            for i, char in enumerate(chars):
                if char not in left_context:
                    left_context[char] = Counter()
                    right_context[char] = Counter()
                
                if i > 0:
                    left_context[char][chars[i-1]] += 1
                if i < len(chars) - 1:
                    right_context[char][chars[i+1]] += 1
        
        return {
            'char_freq': char_freq,
            'bigram_freq': bigram_freq,
            'left_context': left_context,
            'right_context': right_context,
        }


def process_text_chunk(texts_chunk, use_trigram: bool = True):
    """
    处理文本块（用于多进程）
    
    这个函数会被多个进程并行执行
    """
    char_freq = Counter()
    bigram_freq = Counter()
    trigram_freq = Counter()  # 新增
    left_context = {}
    right_context = {}
    
    for text in texts_chunk:
        if not isinstance(text, str):
            continue
            
        # 只保留中文字符
        chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        
        if not chars:
            continue
        
        # 单字频率
        char_freq.update(chars)
        
        # 双字频率
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i+1]
            bigram_freq[bigram] += 1
        
        # 三字频率（trigram）- 对医学文本更可靠
        if use_trigram:
            for i in range(len(chars) - 2):
                trigram = chars[i] + chars[i+1] + chars[i+2]
                trigram_freq[trigram] += 1
        
        # 上下文分布
        for i, char in enumerate(chars):
            if char not in left_context:
                left_context[char] = Counter()
                right_context[char] = Counter()
            
            if i > 0:
                left_context[char][chars[i-1]] += 1
            if i < len(chars) - 1:
                right_context[char][chars[i+1]] += 1
    
    result = {
        'char_freq': char_freq,
        'bigram_freq': bigram_freq,
        'left_context': left_context,
        'right_context': right_context,
    }
    
    if use_trigram:
        result['trigram_freq'] = trigram_freq
    
    return result


def merge_results(results, use_trigram: bool = True):
    """合并多个进程的结果"""
    final_char = Counter()
    final_bigram = Counter()
    final_trigram = Counter() if use_trigram else None
    final_left = {}
    final_right = {}
    
    for result in results:
        final_char.update(result['char_freq'])
        final_bigram.update(result['bigram_freq'])
        
        if use_trigram and 'trigram_freq' in result:
            final_trigram.update(result['trigram_freq'])
        
        # 合并上下文
        for char, counter in result['left_context'].items():
            if char not in final_left:
                final_left[char] = Counter()
            final_left[char].update(counter)
        
        for char, counter in result['right_context'].items():
            if char not in final_right:
                final_right[char] = Counter()
            final_right[char].update(counter)
    
    result = {
        'char_freq': final_char,
        'bigram_freq': final_bigram,
        'left_context': final_left,
        'right_context': final_right,
    }
    
    if use_trigram:
        result['trigram_freq'] = final_trigram
    
    return result


def load_texts_from_files(data_dir: str, max_texts: int = None):
    """从Excel文件加载文本"""
    data_path = Path(data_dir).expanduser()
    pattern = str(data_path / "all_data_match*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        raise ValueError(f"未找到Excel文件: {pattern}")
    
    print(f"找到 {len(files)} 个数据文件")
    
    all_texts = []
    for file_path in files:
        print(f"\n读取: {Path(file_path).name}")
        
        # 读取Excel
        df = pd.read_excel(file_path)
        
        for _, row in df.iterrows():
            parts = []
            for col in ['描述', '结论']:
                if col in row and pd.notna(row[col]):
                    text = str(row[col]).strip()
                    text = text.replace('_x000D_', ' ').replace('\r', ' ')
                    parts.append(text)
            
            if parts:
                all_texts.append(' '.join(parts))
        
        if max_texts and len(all_texts) >= max_texts:
            all_texts = all_texts[:max_texts]
            break
        
        print(f"  已收集: {len(all_texts)} 条")
    
    return all_texts


def train_with_multiprocessing(texts, num_workers=None, use_trigram: bool = True):
    """
    使用多进程训练
    
    这是推荐的训练方式，比GPU更适合此任务
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # 保留一个核心给系统
    
    print(f"\n使用 {num_workers} 个进程并行训练 (trigram={'启用' if use_trigram else '禁用'})")
    
    # 将文本分成多个块
    chunk_size = max(1, len(texts) // num_workers)
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    
    print(f"分割为 {len(chunks)} 个块，每块约 {chunk_size} 条")
    
    # 并行处理
    start_time = time.time()
    
    # 使用partial传递use_trigram参数
    from functools import partial
    process_func = partial(process_text_chunk, use_trigram=use_trigram)
    
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, chunks),
            total=len(chunks),
            desc="训练进度"
        ))
    
    elapsed = time.time() - start_time
    print(f"\n训练完成，耗时: {elapsed:.1f}秒 ({len(texts)/elapsed:.0f} 条/秒)")
    
    # 合并结果
    print("合并结果...")
    return merge_results(results, use_trigram)


def train_sequential(texts, use_trigram: bool = True):
    """单进程训练（用于小数据量测试）"""
    print(f"\n使用单进程训练 (trigram={'启用' if use_trigram else '禁用'})")
    start_time = time.time()
    
    result = process_text_chunk(texts, use_trigram)
    
    elapsed = time.time() - start_time
    print(f"训练完成，耗时: {elapsed:.1f}秒 ({len(texts)/elapsed:.0f} 条/秒)")
    
    return result


def save_model(result, output_dir, use_trigram: bool = True):
    """保存模型"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 转换为普通dict以便pickle
    left_ctx = {k: dict(v) for k, v in result['left_context'].items()}
    right_ctx = {k: dict(v) for k, v in result['right_context'].items()}
    
    # 保存字符模型
    char_model = {
        'char_freq': dict(result['char_freq']),
        'bigram_freq': dict(result['bigram_freq']),
        'total_chars': sum(result['char_freq'].values()),
        'total_bigrams': sum(result['bigram_freq'].values()),
        'use_trigram': use_trigram,
    }
    
    if use_trigram and 'trigram_freq' in result:
        char_model['trigram_freq'] = dict(result['trigram_freq'])
        char_model['total_trigrams'] = sum(result['trigram_freq'].values())
    
    with open(f"{output_dir}/char_anomaly.pkl", 'wb') as f:
        pickle.dump(char_model, f)
    
    # 保存熵模型
    entropy_model = {
        'left_context': left_ctx,
        'right_context': right_ctx,
    }
    with open(f"{output_dir}/entropy.pkl", 'wb') as f:
        pickle.dump(entropy_model, f)
    
    # 保存统计信息
    stats = {
        'unique_chars': len(result['char_freq']),
        'unique_bigrams': len(result['bigram_freq']),
        'total_chars': sum(result['char_freq'].values()),
        'top_chars': result['char_freq'].most_common(20),
        'top_bigrams': result['bigram_freq'].most_common(20),
        'use_trigram': use_trigram,
    }
    
    if use_trigram and 'trigram_freq' in result:
        stats['unique_trigrams'] = len(result['trigram_freq'])
        stats['total_trigrams'] = sum(result['trigram_freq'].values())
        stats['top_trigrams'] = result['trigram_freq'].most_common(20)
    
    import json
    with open(f"{output_dir}/stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n模型已保存到: {output_dir}")
    print(f"  - 唯一字符: {stats['unique_chars']}")
    print(f"  - 唯一bigram: {stats['unique_bigrams']}")
    if use_trigram and 'unique_trigrams' in stats:
        print(f"  - 唯一trigram: {stats['unique_trigrams']}")
    print(f"  - 总字符数: {stats['total_chars']:,}")


def main():
    parser = argparse.ArgumentParser(description='训练快速召回层模型（支持Trigram）')
    parser.add_argument('--data-dir',
                       default='~/work/python/Radiology_Entities/radiology_data',
                       help='Excel数据目录')
    parser.add_argument('--output', default='grammer/models',
                       help='模型输出目录')
    parser.add_argument('--workers', type=int, default=None,
                       help='进程数（默认CPU核心数-1）')
    parser.add_argument('--use-gpu', action='store_true',
                       help='使用GPU（注意：此任务GPU收益有限）')
    parser.add_argument('--use-trigram', action='store_true', default=True,
                       help='使用Trigram（推荐，对医学文本更可靠）')
    parser.add_argument('--no-trigram', action='store_true',
                       help='禁用Trigram（使用bigram）')
    parser.add_argument('--max-texts', type=int, default=None,
                       help='最大训练样本数（用于测试）')
    
    args = parser.parse_args()
    
    # 处理--no-trigram选项
    if args.no_trigram:
        args.use_trigram = False
    
    print("=" * 60)
    print("快速召回层模型训练")
    print(f"Trigram: {'启用' if args.use_trigram else '禁用'}")
    print("=" * 60)
    
    # 检查GPU
    if args.use_gpu:
        if CUDA_AVAILABLE:
            print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA不可用，将使用CPU")
            args.use_gpu = False
    
    # 加载数据
    print("\n加载数据...")
    texts = load_texts_from_files(args.data_dir, args.max_texts)
    print(f"总样本数: {len(texts):,}")
    
    # 训练
    if args.workers == 1 or len(texts) < 10000:
        result = train_sequential(texts, args.use_trigram)
    else:
        result = train_with_multiprocessing(texts, args.workers, args.use_trigram)
    
    # 保存模型
    save_model(result, args.output, args.use_trigram)
    
    # 测试
    print("\n快速测试:")
    from grammer.layered_grammar_detector import LayeredGrammarDetector
    
    detector = LayeredGrammarDetector(model_dir=args.output)
    test_cases = [
        "双肺纹理增粗",
        "双肺文里增粗",
        "胸部CT正常",
    ]
    
    for text in test_cases:
        suspicious = detector.fast_detect(text)
        trigrams = [f for f in suspicious if f.strategy == 'trigram_rarity']
        print(f"  '{text}': {len(suspicious)}个可疑片段（含{len(trigrams)}个trigram）")


if __name__ == '__main__':
    main()
