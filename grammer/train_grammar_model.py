"""
训练语法/错别字检测模型

使用历史报告数据训练N-gram统计模型
支持两种模式：
1. 从数据库加载（production）
2. 从目录加载文本文件（development）
"""

import os
import re
import pickle
import argparse
from pathlib import Path
from collections import Counter
from typing import Iterator, Optional
import sqlite3
from tqdm import tqdm


class NgramTrainer:
    """N-gram模型训练器"""
    
    def __init__(self, n: int = 2):
        self.n = n
        self.ngrams = Counter()
        self.total = 0
    
    def extract_chinese_text(self, text: str) -> str:
        """提取中文字符"""
        return re.sub(r'[^\u4e00-\u9fa5]', '', str(text))
    
    def process_text(self, text: str):
        """处理单条文本，提取N-gram"""
        text = self.extract_chinese_text(text)
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i+self.n]
            self.ngrams[ngram] += 1
            self.total += 1
    
    def process_file(self, filepath: str):
        """处理单个文件"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                self.process_text(content)
        except Exception as e:
            print(f"处理文件 {filepath} 失败: {e}")
    
    def process_directory(self, directory: str, pattern: str = "*.txt"):
        """递归处理目录中的所有文本文件"""
        path = Path(directory)
        files = list(path.rglob(pattern))
        
        print(f"找到 {len(files)} 个文件...")
        
        for filepath in tqdm(files, desc="处理文件"):
            self.process_file(str(filepath))
    
    def save(self, output_path: str):
        """保存模型"""
        model = {
            'n': self.n,
            'ngrams': dict(self.ngrams),
            'total': self.total
        }
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"模型已保存到: {output_path}")
        print(f"总N-gram数: {self.total}")
        print(f"唯一N-gram数: {len(self.ngrams)}")


def load_reports_from_db(db_path: str, table: str = "reports", 
                         column: str = "content") -> Iterator[str]:
    """从SQLite数据库加载报告"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(f"SELECT {column} FROM {table}")
    while True:
        row = cursor.fetchone()
        if row is None:
            break
        yield row[0]
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='训练语法检测N-gram模型')
    parser.add_argument('--source', choices=['dir', 'db'], default='dir',
                       help='数据源类型: dir=目录, db=数据库')
    parser.add_argument('--path', required=True,
                       help='数据路径(目录或数据库文件)')
    parser.add_argument('--n', type=int, default=2,
                       help='N-gram大小 (默认2)')
    parser.add_argument('--output', default='grammar_model.pkl',
                       help='输出模型文件路径')
    parser.add_argument('--pattern', default='*.txt',
                       help='文件匹配模式(仅目录模式)')
    
    args = parser.parse_args()
    
    trainer = NgramTrainer(n=args.n)
    
    if args.source == 'dir':
        trainer.process_directory(args.path, args.pattern)
    else:
        print("数据库模式: 加载报告中...")
        for report in tqdm(load_reports_from_db(args.path), desc="处理报告"):
            trainer.process_text(report)
    
    trainer.save(args.output)
    
    # 显示最常见和最罕见的N-gram
    print("\n最常见的10个N-gram:")
    for ngram, count in trainer.ngrams.most_common(10):
        print(f"  {ngram}: {count}")
    
    print("\n最罕见的10个N-gram:")
    for ngram, count in trainer.ngrams.most_common()[-10:]:
        print(f"  {ngram}: {count}")


if __name__ == '__main__':
    main()
