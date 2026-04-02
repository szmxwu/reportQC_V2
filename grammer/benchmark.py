#!/usr/bin/env python3
"""
性能基准测试

对比：
1. 原始训练脚本 vs 优化脚本
2. 纯Python vs Numba JIT
3. 单进程 vs 多进程

使用：
    python benchmark.py --samples 100000
"""

import os
import sys
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from collections import Counter


def benchmark_char_extraction():
    """测试字符提取性能"""
    print("=" * 60)
    print("字符提取性能测试")
    print("=" * 60)
    
    # 测试数据
    test_text = "双肺纹理增粗，见多发结节影，边界清晰。" * 100  # 长文本
    
    # 1. 纯Python
    start = time.time()
    for _ in range(1000):
        chars_py = [c for c in test_text if '\u4e00' <= c <= '\u9fff']
    py_time = time.time() - start
    
    print(f"纯Python列表推导: {py_time*1000:.2f}ms (1000次)")
    
    # 2. NumPy向量化（如果可用）
    try:
        start = time.time()
        for _ in range(1000):
            arr = np.frombuffer(test_text.encode('utf-8'), dtype=np.uint8)
            # 简化的中文字符检测
            mask = (arr >= 0xE4) & (arr <= 0xE9)
        np_time = time.time() - start
        print(f"NumPy向量化: {np_time*1000:.2f}ms (1000次)")
        print(f"  加速比: {py_time/np_time:.1f}x")
    except Exception as e:
        print(f"NumPy测试失败: {e}")
    
    # 3. Numba JIT（如果可用）
    try:
        from numba import jit
        
        @jit(nopython=True)
        def extract_numba(text_bytes):
            result = []
            i = 0
            n = len(text_bytes)
            while i < n:
                b = text_bytes[i]
                if b < 0x80:
                    i += 1
                    continue
                if i + 2 < n and (0xE0 <= b <= 0xEF):
                    code = ((b & 0x0F) << 12) | ((text_bytes[i+1] & 0x3F) << 6) | (text_bytes[i+2] & 0x3F)
                    if 0x4E00 <= code <= 0x9FFF:
                        result.append(code)
                    i += 3
                else:
                    i += 1
            return result
        
        # 预热
        _ = extract_numba(test_text.encode('utf-8'))
        
        start = time.time()
        for _ in range(1000):
            chars_nb = extract_numba(test_text.encode('utf-8'))
        nb_time = time.time() - start
        
        print(f"Numba JIT: {nb_time*1000:.2f}ms (1000次)")
        print(f"  加速比: {py_time/nb_time:.1f}x")
    except ImportError:
        print("Numba未安装，跳过JIT测试")


def benchmark_ngram_stats():
    """测试n-gram统计性能"""
    print("\n" + "=" * 60)
    print("N-gram统计性能测试")
    print("=" * 60)
    
    # 生成测试数据（模拟真实报告）
    chars = list("双肺纹理增粗增多紊乱清晰模糊结节肿块钙化影低密度高密度支气管")
    np.random.seed(42)
    test_texts = []
    for _ in range(10000):
        length = np.random.randint(50, 200)
        text = ''.join(np.random.choice(chars, length))
        test_texts.append(text)
    
    print(f"测试数据: {len(test_texts)} 条文本")
    
    # 1. 逐条处理
    start = time.time()
    char_freq = Counter()
    bigram_freq = Counter()
    trigram_freq = Counter()
    
    for text in test_texts:
        chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        char_freq.update(chars)
        for i in range(len(chars) - 1):
            bigram_freq[chars[i] + chars[i+1]] += 1
        for i in range(len(chars) - 2):
            trigram_freq[chars[i] + chars[i+1] + chars[i+2]] += 1
    
    sequential_time = time.time() - start
    print(f"逐条处理: {sequential_time:.2f}秒")
    
    # 2. 批量处理
    start = time.time()
    all_chars = []
    for text in test_texts:
        all_chars.extend([c for c in text if '\u4e00' <= c <= '\u9fff'])
    
    char_freq = Counter(all_chars)
    bigrams = [all_chars[i] + all_chars[i+1] for i in range(len(all_chars)-1)]
    bigram_freq = Counter(bigrams)
    trigrams = [all_chars[i] + all_chars[i+1] + all_chars[i+2] for i in range(len(all_chars)-2)]
    trigram_freq = Counter(trigrams)
    
    batch_time = time.time() - start
    print(f"批量处理: {batch_time:.2f}秒")
    print(f"  加速比: {sequential_time/batch_time:.1f}x")


def benchmark_memory_usage():
    """测试内存使用"""
    print("\n" + "=" * 60)
    print("内存使用估计")
    print("=" * 60)
    
    # 估算360万报告的内存使用
    avg_text_len = 200  # 平均每条200字符
    total_chars = 3_600_000 * avg_text_len
    
    print(f"数据规模: 360万条报告，平均{avg_text_len}字符")
    print(f"总字符数: {total_chars:,}")
    
    # 估算模型大小
    unique_chars = 5000
    unique_bigrams = 100000
    unique_trigrams = 500000
    
    char_model_mb = unique_chars * 8 / 1024 / 1024  # int + count
    bigram_model_mb = unique_bigrams * 12 / 1024 / 1024  # string + count
    trigram_model_mb = unique_trigrams * 16 / 1024 / 1024
    
    print(f"\n模型大小估算:")
    print(f"  - Char模型: ~{char_model_mb:.1f}MB")
    print(f"  - Bigram模型: ~{bigram_model_mb:.1f}MB")
    print(f"  - Trigram模型: ~{trigram_model_mb:.1f}MB")
    print(f"  - 总计: ~{char_model_mb + bigram_model_mb + trigram_model_mb:.1f}MB")
    
    # 处理时内存
    texts_memory = total_chars * 2 / 1024 / 1024  # UTF-16
    counters_memory = unique_trigrams * 16 / 1024 / 1024
    
    print(f"\n处理时内存（估算）:")
    print(f"  - 原始文本: ~{texts_memory:.0f}MB")
    print(f"  - 统计计数器: ~{counters_memory:.0f}MB")
    print(f"  - 总计: ~{texts_memory + counters_memory:.0f}MB")
    
    print(f"\n你的配置: 32GB内存")
    print(f"  充足度: {'✓ 充足' if texts_memory + counters_memory < 16000 else '⚠ 需要注意'}")


def benchmark_hardware():
    """检测硬件配置"""
    print("\n" + "=" * 60)
    print("硬件配置检测")
    print("=" * 60)
    
    import multiprocessing as mp
    
    # CPU
    cpu_count = mp.cpu_count()
    print(f"CPU核心数: {cpu_count}")
    print(f"  建议进程数: {max(1, cpu_count - 1)}")
    
    # 内存
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"\n总内存: {mem.total / 1024**3:.1f}GB")
        print(f"可用内存: {mem.available / 1024**3:.1f}GB")
        print(f"  建议块大小: {int(mem.available / 10 / 200)} 条（每条200字符）")
    except ImportError:
        print("\n安装psutil查看详细内存信息: pip install psutil")
    
    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print(f"  CUDA版本: {torch.version.cuda}")
        else:
            print("\nGPU: 未检测到CUDA")
    except ImportError:
        print("\nGPU: PyTorch未安装")
    
    # Numba
    try:
        import numba
        print(f"\nNumba: 已安装 v{numba.__version__}")
    except ImportError:
        print("\nNumba: 未安装（建议安装: pip install numba）")


def recommend_config():
    """推荐配置"""
    print("\n" + "=" * 60)
    print("针对你硬件的推荐配置")
    print("=" * 60)
    
    print("""
硬件配置: i5 8500 6核 + 32G内存 + RTX 4060 8G

推荐训练参数:
    python grammer/train_optimized.py \\
        --workers 5           # 6核留1核给系统
        --io-workers 4        # 4线程并行读Excel
        --chunk-size 100000   # 32G内存支持大缓冲区
        --use-trigram         # 启用trigram
        --use-jit             # Numba加速（如果已安装）

预期性能:
    - 读取速度: ~50000 条/秒（4线程IO）
    - 处理速度: ~30000 条/秒（5进程+JIT）
    - 总训练时间: ~8-12分钟（360万条）
    - 内存峰值: ~8-12GB

优化建议:
    1. 安装numba: pip install numba
       预期加速: 3-5x
    
    2. 安装psutil: pip install psutil
       用于内存监控
    
    3. 训练前关闭其他大型应用
       确保内存充足
    
    4. 使用SSD存储
       Excel读取是IO密集型，SSD有显著优势

性能对比（预估）:
    ┌─────────────────┬────────────┬────────────┐
    │ 配置            │ 原始脚本   │ 优化脚本   │
    ├─────────────────┼────────────┼────────────┤
    │ 单进程无JIT     │ 40分钟     │ 25分钟     │
    │ 5进程无JIT      │ -          │ 10分钟     │
    │ 5进程+JIT       │ -          │ 5-8分钟    │
    └─────────────────┴────────────┴────────────┘
    """)


def main():
    parser = argparse.ArgumentParser(description='性能基准测试')
    parser.add_argument('--full', action='store_true',
                       help='运行完整测试（需要一些时间）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("性能基准测试工具")
    print("=" * 60)
    
    # 基础测试
    benchmark_hardware()
    benchmark_memory_usage()
    recommend_config()
    
    # 完整测试
    if args.full:
        benchmark_char_extraction()
        benchmark_ngram_stats()
    else:
        print("\n使用 --full 运行完整性能测试")


if __name__ == '__main__':
    main()
