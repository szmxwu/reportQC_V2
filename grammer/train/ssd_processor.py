"""
SSD 流式处理器模块

实现时间换空间方案：
- 分块处理大数据集
- 磁盘缓存中间结果
- 流式合并避免内存溢出
"""
import os
import json
import gc
from pathlib import Path
from collections import Counter
from typing import List, Iterator, Tuple, Dict
from tqdm import tqdm

import jieba

from utils.config import CHUNK_SIZE, TEMP_DIR, CHINESE_RANGE
from utils.utils import split_sentences, extract_chinese_chars, is_chinese_word


class SSDStreamingProcessor:
    """
    SSD 流式处理器
    
    用于在内存受限环境下处理大规模文本数据
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, temp_dir: Path = None):
        """
        初始化处理器
        
        Args:
            chunk_size: 每块处理的记录数
            temp_dir: 临时文件目录
        """
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir or TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 jieba 词典
        self._init_jieba()
    
    def _init_jieba(self):
        """初始化 jieba 分词器，加载医学词典和 huqie 高频词"""
        from config import MEDICAL_DICT_PATHS, HUQIE_PATH
        
        # 加载医学词典
        for path in MEDICAL_DICT_PATHS:
            if path.exists() and path.suffix == '.txt':
                jieba.load_userdict(str(path))
                print(f"加载医学词典: {path}")
        
        # 加载 huqie 高频词（频次>1000）
        if HUQIE_PATH.exists():
            high_freq_words = []
            with open(HUQIE_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            word = parts[0]
                            freq = int(parts[1])
                            if freq > 1000 and len(word) >= 2:
                                high_freq_words.append(word)
                        except ValueError:
                            continue
            
            # 将高频词写入临时词典文件
            if high_freq_words:
                temp_dict = self.temp_dir / 'huqie_high_freq.txt'
                with open(temp_dict, 'w', encoding='utf-8') as f:
                    for word in high_freq_words[:10000]:  # 最多1万条
                        f.write(f"{word}\n")
                jieba.load_userdict(str(temp_dict))
                print(f"加载 huqie 高频词: {len(high_freq_words[:10000])} 条")
    
    def process_texts_streaming(self, texts: List[str]) -> Tuple[str, str]:
        """
        流式处理文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            (corpus_file, vocab_file): 语料文件路径和词频文件路径
        """
        chunk_files = []
        
        print(f"开始流式处理 {len(texts):,} 条文本...")
        print(f"分块大小: {self.chunk_size}, 预计块数: {len(texts) // self.chunk_size + 1}")
        
        # 分块处理
        for i, chunk in enumerate(self._chunk_generator(texts)):
            chunk_file = self._process_and_save_chunk(chunk, i)
            chunk_files.append(chunk_file)
            
            # 显式释放内存
            del chunk
            gc.collect()
        
        print(f"处理完成，共 {len(chunk_files)} 个临时块")
        
        # 合并语料文件（字符轨）
        corpus_file = self._merge_corpus(chunk_files)
        
        # 合并词频文件（词语轨）
        vocab_file = self._merge_vocab(chunk_files)
        
        # 清理临时文件
        self._cleanup(chunk_files)
        
        return corpus_file, vocab_file
    
    def _chunk_generator(self, texts: List[str]) -> Iterator[List[str]]:
        """
        将文本列表分块生成
        
        Args:
            texts: 文本列表
            
        Yields:
            文本块
        """
        for i in range(0, len(texts), self.chunk_size):
            yield texts[i:i + self.chunk_size]
    
    def _process_and_save_chunk(self, chunk: List[str], chunk_id: int) -> Path:
        """
        处理单个块并保存到磁盘
        
        Args:
            chunk: 文本块
            chunk_id: 块ID
            
        Returns:
            临时文件路径
        """
        char_tokens = []  # 字符级 token（用于 KenLM）
        word_freq = Counter()  # 词频统计（用于策略 B）
        
        for text in chunk:
            if not text or not isinstance(text, str):
                continue
            
            # 拆句
            sentences = split_sentences(text)
            
            for sent in sentences:
                if not sent.strip():
                    continue
                
                # 字符轨：提取中文字符
                chars = extract_chinese_chars(sent)
                if chars:
                    char_tokens.extend(chars)
                
                # 词语轨：jieba 分词
                words = jieba.lcut(sent)
                words = [w.strip() for w in words if is_chinese_word(w)]
                if words:
                    word_freq.update(words)
        
        # 保存到临时文件
        chunk_file = self.temp_dir / f"chunk_{chunk_id:04d}.json"
        
        data = {
            'char_tokens': char_tokens,
            'word_freq': dict(word_freq)
        }
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        
        return chunk_file
    
    def _merge_corpus(self, chunk_files: List[Path]) -> str:
        """
        合并语料文件（字符轨）
        
        Args:
            chunk_files: 临时块文件列表
            
        Returns:
            语料文件路径
        """
        from config import RADIOLOGY_CORPUS
        
        corpus_file = str(RADIOLOGY_CORPUS)
        
        print(f"合并语料文件 -> {corpus_file}")
        
        with open(corpus_file, 'w', encoding='utf-8') as f_out:
            for chunk_file in tqdm(chunk_files, desc="合并语料"):
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    char_tokens = data.get('char_tokens', [])
                    
                    # 写入字符（KenLM 格式：空格分隔）
                    if char_tokens:
                        f_out.write(' '.join(char_tokens))
                        f_out.write('\n')
                
                # 释放内存
                del data
        
        print(f"语料文件大小: {Path(corpus_file).stat().st_size / 1024 / 1024:.1f} MB")
        return corpus_file
    
    def _merge_vocab(self, chunk_files: List[Path]) -> str:
        """
        合并词频文件（词语轨）
        
        Args:
            chunk_files: 临时块文件列表
            
        Returns:
            词频文件路径
        """
        from config import RADIOLOGY_VOCAB
        
        vocab_file = str(RADIOLOGY_VOCAB)
        
        print(f"合并词频文件 -> {vocab_file}")
        
        # 流式累加词频
        total_word_freq = Counter()
        
        for chunk_file in tqdm(chunk_files, desc="合并词频"):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                word_freq = data.get('word_freq', {})
                
                # 累加词频
                for word, count in word_freq.items():
                    total_word_freq[word] += count
            
            # 释放内存
            del data
        
        # 保存词频（按频次降序）
        vocab_data = {
            'total_words': sum(total_word_freq.values()),
            'unique_words': len(total_word_freq),
            'word_freq': dict(total_word_freq.most_common())
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"词表统计: {vocab_data['unique_words']:,} 个唯一词, "
              f"{vocab_data['total_words']:,} 个总词")
        
        return vocab_file
    
    def _cleanup(self, chunk_files: List[Path]):
        """清理临时文件"""
        print("清理临时文件...")
        for chunk_file in chunk_files:
            try:
                chunk_file.unlink()
            except Exception:
                pass
        print("清理完成")
    
    def process_from_excel_streaming(self, excel_files: List[Path], 
                                     text_columns: List[str] = None) -> Tuple[str, str]:
        """
        从 Excel 文件流式处理
        
        Args:
            excel_files: Excel 文件路径列表
            text_columns: 文本列名列表
            
        Returns:
            (corpus_file, vocab_file)
        """
        if text_columns is None:
            text_columns = ['描述', '结论']
        
        all_texts = []
        
        print(f"读取 {len(excel_files)} 个 Excel 文件...")
        
        for excel_file in tqdm(excel_files, desc="读取Excel"):
            try:
                import pandas as pd
                df = pd.read_excel(excel_file, usecols=text_columns)
                
                for col in text_columns:
                    if col in df.columns:
                        texts = df[col].dropna().astype(str).tolist()
                        all_texts.extend(texts)
                
                # 及时释放 DataFrame
                del df
                gc.collect()
                
            except Exception as e:
                print(f"读取 {excel_file} 失败: {e}")
                continue
        
        print(f"共读取 {len(all_texts):,} 条文本记录")
        
        return self.process_texts_streaming(all_texts)


# ==================== 便捷函数 ====================

def process_texts_with_ssd(texts: List[str], chunk_size: int = CHUNK_SIZE) -> Tuple[str, str]:
    """
    使用 SSD 流式处理文本
    
    Args:
        texts: 文本列表
        chunk_size: 分块大小
        
    Returns:
        (corpus_file, vocab_file)
    """
    processor = SSDStreamingProcessor(chunk_size=chunk_size)
    return processor.process_texts_streaming(texts)


def process_excel_files_with_ssd(excel_files: List[Path],
                                 text_columns: List[str] = None,
                                 chunk_size: int = CHUNK_SIZE) -> Tuple[str, str]:
    """
    使用 SSD 流式处理 Excel 文件
    
    Args:
        excel_files: Excel 文件路径列表
        text_columns: 文本列名列表
        chunk_size: 分块大小
        
    Returns:
        (corpus_file, vocab_file)
    """
    processor = SSDStreamingProcessor(chunk_size=chunk_size)
    return processor.process_from_excel_streaming(excel_files, text_columns)


# ==================== 测试 ====================
if __name__ == '__main__':
    # 测试流式处理
    test_texts = [
        "双肺纹理增粗，见多发结节影。",
        "肝实质内见低密度灶，边界清晰。",
        "心脏各房室大小正常，未见明显异常。",
    ] * 1000  # 3000 条测试数据
    
    processor = SSDStreamingProcessor(chunk_size=500)
    corpus_file, vocab_file = processor.process_texts_streaming(test_texts)
    
    print(f"\n语料文件: {corpus_file}")
    print(f"词频文件: {vocab_file}")
    
    # 验证词频
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        print(f"\n词频统计:")
        for word, count in list(vocab_data['word_freq'].items())[:10]:
            print(f"  {word}: {count}")
