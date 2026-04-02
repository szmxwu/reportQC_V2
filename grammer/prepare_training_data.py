"""
训练数据预处理 - 为Confusion Set模型准备数据

功能：
1. 分批读取大Excel文件（避免内存溢出）
2. 使用jieba分词（加载用户词典）
3. 统计词频和混淆字的上下文n-gram
4. 生成白名单和可疑模式

使用：
    python prepare_training_data.py \
        --data-dir ~/work/python/Radiology_Entities/radiology_data \
        --output ./models \
        --chunk-size 50000
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

import pandas as pd
import jieba


# 添加项目根目录到路径
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# 混淆字集合（来自typo_database.py的核心混淆对）
CONFUSION_CHARS = {
    '文', '纹', '渡', '度', '象', '像', '曾', '增', '末', '未',
    '也', '野', '追', '椎', '干', '肝', '胜', '脏', '记', '剂',
    '岩', '炎', '买', '脉', '伟', '位', '苗', '描', '绿', '虑',
    '处', '除', '建', '见', '并', '病', '古', '股', '骨', '种',
    '肿', '正', '症', '话', '化', '排', '牌', '条', '索',
}


class TrainingDataProcessor:
    """训练数据处理器"""
    
    def __init__(self, user_dict_path: str = None):
        """
        Args:
            user_dict_path: 用户词典路径
        """
        # 加载jieba用户词典
        if user_dict_path and os.path.exists(user_dict_path):
            print(f"加载用户词典: {user_dict_path}")
            jieba.load_userdict(user_dict_path)
            # 确保词典加载完成
            jieba.initialize()
        else:
            print(f"警告: 用户词典未找到: {user_dict_path}")
        
        # 统计器
        self.word_freq = Counter()  # 全局词频
        self.confusion_context = defaultdict(Counter)  # 混淆字的上下文
        self.char_bigram = Counter()  # 字符级bigram（用于检测罕见组合）
        self.total_reports = 0
        
    def extract_text(self, row: pd.Series) -> str:
        """从数据行提取要分析的文本"""
        parts = []
        for col in ['描述', '结论']:
            if col in row and pd.notna(row[col]):
                text = str(row[col]).strip()
                # 清理Windows换行符
                text = text.replace('_x000D_', ' ').replace('\r', ' ')
                parts.append(text)
        return ' '.join(parts)
    
    def process_text(self, text: str):
        """处理单条文本，更新统计"""
        if not text or len(text) < 5:
            return
        
        self.total_reports += 1
        
        # 1. jieba分词统计
        words = list(jieba.cut(text))
        self.word_freq.update(words)
        
        # 2. 字符级bigram统计（用于检测罕见字组合）
        # 只保留中文字符
        chinese_text = ''.join(c for c in text if '\u4e00' <= c <= '\u9fff')
        for i in range(len(chinese_text) - 1):
            bigram = chinese_text[i:i+2]
            self.char_bigram[bigram] += 1
        
        # 3. 混淆字的上下文统计
        for i, char in enumerate(chinese_text):
            if char in CONFUSION_CHARS:
                # 提取前后各2个字的上下文
                context_start = max(0, i - 2)
                context_end = min(len(chinese_text), i + 3)
                context = chinese_text[context_start:context_end]
                self.confusion_context[char][context] += 1
    
    def process_dataframe(self, df: pd.DataFrame):
        """处理DataFrame"""
        for _, row in tqdm(df.iterrows(), total=len(df), desc="处理文本"):
            text = self.extract_text(row)
            self.process_text(text)
    
    def get_whitelist(self, min_freq: int = 5) -> Dict[str, Set[str]]:
        """
        生成白名单
        
        Args:
            min_freq: 最小频率阈值，低于此值的视为可疑
        
        Returns:
            {混淆字: {白名单上下文集合}}
        """
        whitelist = {}
        for char, contexts in self.confusion_context.items():
            whitelist[char] = {ctx for ctx, freq in contexts.items() if freq >= min_freq}
        return whitelist
    
    def get_suspicious_patterns(self, max_freq: int = 3) -> Dict[str, List[Tuple[str, int]]]:
        """
        获取可疑模式（低频率的上下文）
        
        Returns:
            {混淆字: [(上下文, 频率), ...]}
        """
        suspicious = {}
        for char, contexts in self.confusion_context.items():
            low_freq = [(ctx, freq) for ctx, freq in contexts.items() if freq <= max_freq]
            if low_freq:
                suspicious[char] = sorted(low_freq, key=lambda x: x[1])
        return suspicious
    
    def save(self, output_dir: str):
        """保存统计结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存词频
        with open(output_path / 'word_frequency.pkl', 'wb') as f:
            pickle.dump(dict(self.word_freq), f)
        
        # 2. 保存混淆字上下文
        confusion_data = {
            char: dict(contexts) 
            for char, contexts in self.confusion_context.items()
        }
        with open(output_path / 'confusion_context.pkl', 'wb') as f:
            pickle.dump(confusion_data, f)
        
        # 3. 保存白名单
        whitelist = self.get_whitelist(min_freq=5)
        with open(output_path / 'confusion_whitelist.pkl', 'wb') as f:
            pickle.dump(whitelist, f)
        
        # 4. 保存字符bigram
        with open(output_path / 'char_bigram.pkl', 'wb') as f:
            pickle.dump(dict(self.char_bigram), f)
        
        # 5. 保存统计摘要（JSON可读）
        summary = {
            'total_reports': self.total_reports,
            'unique_words': len(self.word_freq),
            'unique_bigrams': len(self.char_bigram),
            'confusion_chars': list(self.confusion_context.keys()),
            'top_words': self.word_freq.most_common(100),
            'whitelist_stats': {
                char: len(contexts) 
                for char, contexts in whitelist.items()
            }
        }
        with open(output_path / 'training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n统计结果已保存到: {output_dir}")
        print(f"  - 总报告数: {self.total_reports}")
        print(f"  - 唯一词汇: {len(self.word_freq)}")
        print(f"  - 唯一bigram: {len(self.char_bigram)}")
        print(f"  - 混淆字数量: {len(self.confusion_context)}")


def process_excel_files(data_dir: str, processor: TrainingDataProcessor, 
                        chunk_size: int = 50000):
    """
    分批处理Excel文件
    
    Args:
        data_dir: 数据目录
        processor: 数据处理器
        chunk_size: 每次处理的行数
    """
    data_path = Path(data_dir).expanduser()
    excel_files = sorted(data_path.glob('all_data_match*.xlsx'))
    
    if not excel_files:
        raise ValueError(f"未找到Excel文件: {data_dir}")
    
    print(f"找到 {len(excel_files)} 个Excel文件")
    
    for file_path in excel_files:
        print(f"\n处理文件: {file_path.name}")
        
        # 分批读取
        chunk_iter = pd.read_excel(file_path, chunksize=chunk_size)
        
        for chunk_num, chunk in enumerate(chunk_iter):
            print(f"  处理第 {chunk_num + 1} 批 ({len(chunk)} 行)...")
            processor.process_dataframe(chunk)


def main():
    parser = argparse.ArgumentParser(description='准备语法检测训练数据')
    parser.add_argument('--data-dir', 
                       default='~/work/python/Radiology_Entities/radiology_data',
                       help='数据目录')
    parser.add_argument('--user-dict',
                       default='config/user_dic_expand.txt',
                       help='用户词典路径')
    parser.add_argument('--output', default='./models',
                       help='输出目录')
    parser.add_argument('--chunk-size', type=int, default=50000,
                       help='每批处理行数')
    
    args = parser.parse_args()
    
    # 初始化处理器
    user_dict_path = str(REPO_ROOT / args.user_dict)
    processor = TrainingDataProcessor(user_dict_path=user_dict_path)
    
    # 处理数据
    process_excel_files(args.data_dir, processor, args.chunk_size)
    
    # 保存结果
    processor.save(args.output)
    
    # 显示可疑模式示例
    print("\n可疑模式示例:")
    suspicious = processor.get_suspicious_patterns(max_freq=3)
    for char, patterns in list(suspicious.items())[:5]:
        print(f"  {char}: {patterns[:3]}")


if __name__ == '__main__':
    main()
