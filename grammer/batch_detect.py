#!/usr/bin/env python3
"""
批量语法错误检测脚本 - 支持断点续传和实时写入

功能：
1. 读取所有训练样本（360万条报告）
2. 使用已训练的模型检测语法错误
3. 输出JSONL格式，包含错误定位信息
4. 支持断点续传（从上次停止处继续）
5. 实时写入（独立线程，可中途查看结果）

输出字段：
- error_phrase: 出现错误的短语
- sentence: 短语所在的完整句子
- suggestion: 建议修正
- error_type: 错误类型
- confidence: 置信度
- report_text: 完整报告文本（可选，用于调试）
- position: 短语在报告中的位置
- llm_verified: 是否LLM验证
- detected_by: 检测策略

使用：
    # 基础用法（自动支持断点续传）
    python batch_detect.py --output errors.jsonl
    
    # 强制重新开始（忽略之前的进度）
    python batch_detect.py --output errors.jsonl --restart
    
    # 限制样本数（测试）
    python batch_detect.py --max-samples 1000 --output errors_test.jsonl
    
    # 不包含完整报告文本（减小文件大小）
    python batch_detect.py --no-full-text --output errors.jsonl
    
    # 使用启发式规则（不调用LLM，更快）
    python batch_detect.py --no-llm --output errors.jsonl
"""

import os
import sys
import json
import argparse
import threading
import queue
import time
from pathlib import Path
from glob import glob
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from grammer import LayeredGrammarDetector


# ============== 实时写入管理器 ==============

class AsyncWriter:
    """
    异步写入管理器
    
    使用独立线程将结果实时写入文件
    """
    
    def __init__(self, output_file: str, stats_file: str):
        self.output_file = output_file
        self.stats_file = stats_file
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.writer_thread = None
        self.written_count = 0
        self.lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_reports': 0,
            'reports_with_errors': 0,
            'total_errors': 0,
            'error_types': {},
            'processed_ids': set(),  # 已处理的报告ID
        }
    
    def start(self):
        """启动写入线程"""
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        print(f"[AsyncWriter] 写入线程已启动: {self.output_file}")
    
    def _writer_loop(self):
        """写入线程主循环"""
        # 以追加模式打开文件
        with open(self.output_file, 'a', encoding='utf-8') as f:
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    # 批量写入，每次最多取100条或等待1秒
                    batch = []
                    try:
                        batch.append(self.queue.get(timeout=1))
                        # 尝试获取更多
                        for _ in range(99):
                            batch.append(self.queue.get_nowait())
                    except queue.Empty:
                        pass
                    
                    if batch:
                        for item in batch:
                            if item is None:  # 结束信号
                                continue
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                            with self.lock:
                                self.written_count += 1
                        f.flush()  # 强制刷新到磁盘
                        
                except Exception as e:
                    print(f"[AsyncWriter] 写入错误: {e}")
    
    def write(self, error_record: Dict):
        """将错误记录加入写入队列"""
        self.queue.put(error_record)
    
    def update_stats(self, report_id: int, has_error: bool, error_types: List[str]):
        """更新统计信息"""
        with self.lock:
            self.stats['total_reports'] += 1
            if has_error:
                self.stats['reports_with_errors'] += 1
                for err_type in error_types:
                    self.stats['error_types'][err_type] = self.stats['error_types'].get(err_type, 0) + 1
            self.stats['processed_ids'].add(report_id)
    
    def save_stats(self):
        """保存统计信息到文件"""
        with self.lock:
            stats_copy = self.stats.copy()
            # 将set转换为list以便JSON序列化
            stats_copy['processed_ids'] = list(stats_copy['processed_ids'])
            stats_copy['end_time'] = datetime.now().isoformat()
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_copy, f, ensure_ascii=False, indent=2)
    
    def stop(self):
        """停止写入线程"""
        self.stop_event.set()
        self.queue.put(None)  # 发送结束信号
        if self.writer_thread:
            self.writer_thread.join(timeout=5)
        self.save_stats()
        print(f"[AsyncWriter] 总计写入: {self.written_count} 条记录")


# ============== 断点管理 ==============

class CheckpointManager:
    """
    断点管理器
    
    管理检测进度，支持断点续传
    """
    
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.checkpoint_file = self.output_file.with_suffix('.checkpoint.json')
        self.stats_file = self.output_file.with_suffix('.stats.json')
    
    def load_checkpoint(self) -> Dict:
        """加载断点信息"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'processed_ids': [], 'last_update': None}
    
    def save_checkpoint(self, processed_ids: set, total_reports: int):
        """保存断点信息"""
        checkpoint = {
            'processed_ids': list(processed_ids),
            'total_reports': total_reports,
            'last_update': datetime.now().isoformat(),
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def clear_checkpoint(self):
        """清除断点（重新开始）"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.output_file.exists():
            self.output_file.unlink()
        if self.stats_file.exists():
            self.stats_file.unlink()
        print(f"[Checkpoint] 已清除断点，重新开始")
    
    def get_processed_ids(self) -> set:
        """获取已处理的报告ID集合"""
        checkpoint = self.load_checkpoint()
        return set(checkpoint.get('processed_ids', []))


# ============== 句子拆分和过滤工具 ==============

def split_sentences_rigid(text: str) -> List[Dict]:
    """
    使用刚性断句（。
\r）分割句子
    
    比实体感知拆分更保守，保留更多上下文，降低LLM误报
    """
    import re
    
    # 使用句号、换行符等硬性标点分割
    # 保留分隔符以便后续处理
    parts = re.split(r'([。！？\n\r]+)', text)
    
    sentences = []
    current_pos = 0
    
    for i in range(0, len(parts), 2):
        # 获取文本部分
        text_part = parts[i] if i < len(parts) else ''
        # 获取分隔符（如果有）
        sep = parts[i+1] if i+1 < len(parts) else ''
        
        # 合并文本和分隔符
        sentence = (text_part + sep).strip()
        
        if not sentence:
            continue
        
        # 计算位置
        start = text.find(sentence, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(sentence)
        current_pos = end
        
        sentences.append({
            'text': sentence,
            'start': start,
            'end': end,
        })
    
    return sentences


def is_phrase_in_single_sentence(phrase: str, text: str) -> bool:
    """
    验证短语是否完整存在于单个句子中（不跨越标点）
    
    这是n-gram检测后的保险层，用于过滤跨越标点的误报
    """
    import re
    
    if not phrase or not text:
        return False
    
    # 按标点分割句子
    sentence_boundaries = []
    last_end = 0
    
    for match in re.finditer(r'[。！？；\.\!\?\n\r]+', text):
        sent_start = last_end
        sent_end = match.end()
        sentence_boundaries.append((sent_start, sent_end))
        last_end = match.end()
    
    # 最后一段
    if last_end < len(text):
        sentence_boundaries.append((last_end, len(text)))
    
    # 在每个句子中查找短语
    phrase_clean = phrase.strip()
    
    for sent_start, sent_end in sentence_boundaries:
        sentence = text[sent_start:sent_end]
        if phrase_clean in sentence:
            phrase_pos_in_sent = sentence.find(phrase_clean)
            phrase_start_in_text = sent_start + phrase_pos_in_sent
            phrase_end_in_text = phrase_start_in_text + len(phrase_clean)
            
            if phrase_start_in_text >= sent_start and phrase_end_in_text <= sent_end:
                return True
    
    return False


def is_english_or_number(text: str) -> bool:
    """
    检查文本是否主要是英文或数字
    
    这些短语误报率太高，应该跳过
    """
    if not text:
        return False
    
    total_chars = len(text)
    if total_chars == 0:
        return False
    
    en_count = sum(1 for c in text if c.isascii() and c.isalpha())
    num_count = sum(1 for c in text if c.isdigit())
    
    ratio = (en_count + num_count) / total_chars
    return ratio > 0.5


def is_pure_whitespace(text: str) -> bool:
    """
    检查文本是否纯空格/空白字符
    
    这种片段完全无意义，应该跳过
    """
    if not text:
        return True
    return text.strip() == ''


def is_meaningless_fragment(text: str) -> bool:
    """
    检查片段是否无意义
    
    过滤：
    1. 纯空格
    2. 单个字符（除非是常见错别字模式）
    3. 纯符号
    """
    if not text:
        return True
    
    stripped = text.strip()
    if not stripped:
        return True
    
    # 单个字符通常无意义（除非是已知的常见错别字）
    if len(stripped) == 1:
        # 单字错误需要上下文，单独检测无意义
        return True
    
    # 纯符号（非中英文、非数字）
    has_meaningful_char = False
    for c in stripped:
        if '\u4e00' <= c <= '\u9fff' or c.isalpha() or c.isdigit():
            has_meaningful_char = True
            break
    
    return not has_meaningful_char


def is_valid_suggestion(error_text: str, suggestion: str, sentence: str) -> bool:
    """
    检查LLM的纠错建议是否合理
    
    注意：新版LLM只输出YES/NO，不输出fix，所以suggestion可能为空
    """
    # 如果suggestion为空，接受（由LLM的判断决定）
    if not suggestion or not suggestion.strip():
        return True
    
    suggestion_clean = suggestion.strip()
    error_clean = error_text.strip()
    
    # suggestion 等于错误片段本身（没有改正）
    if suggestion_clean == error_clean:
        return False
    
    # suggestion 只是重复了原句子（没有具体改正）
    if suggestion_clean == sentence.strip():
        return False
    
    return True


def has_chinese_at_boundaries(text: str) -> bool:
    """
    检查文本首尾是否都是中文字符
    
    过滤掉首尾是标点、英文、数字的情况（错误截断）
    """
    if not text or len(text.strip()) < 2:
        return False
    
    stripped = text.strip()
    first_char = stripped[0]
    last_char = stripped[-1]
    
    # 首字符必须是中文
    if not ('\u4e00' <= first_char <= '\u9fff'):
        return False
    
    # 尾字符必须是中文
    if not ('\u4e00' <= last_char <= '\u9fff'):
        return False
    
    return True


# ============== 数据加载 ==============

def load_reports_from_excel(data_dir: str, max_samples: Optional[int] = None,
                            skip_ids: Optional[set] = None) -> List[Dict]:
    """
    从Excel文件加载报告数据
    
    Args:
        skip_ids: 需要跳过的报告ID集合（已处理的）
    """
    data_path = Path(data_dir).expanduser()
    pattern = str(data_path / "all_data_match*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        raise ValueError(f"未找到Excel文件: {pattern}")
    
    print(f"找到 {len(files)} 个数据文件")
    if skip_ids:
        print(f"将跳过 {len(skip_ids)} 个已处理的报告")
    
    reports = []
    report_id = 0
    skipped_count = 0
    
    for file_path in files:
        print(f"\n读取: {Path(file_path).name}")
        
        df = pd.read_excel(file_path)
        
        for _, row in df.iterrows():
            # 检查是否需要跳过
            if skip_ids and report_id in skip_ids:
                skipped_count += 1
                report_id += 1
                continue
            
            parts = []
            description = ''
            conclusion = ''
            
            if '描述' in row and pd.notna(row['描述']):
                description = str(row['描述']).strip()
                description = description.replace('_x000D_', ' ').replace('\r', ' ')
                parts.append(description)
            
            if '结论' in row and pd.notna(row['结论']):
                conclusion = str(row['结论']).strip()
                conclusion = conclusion.replace('_x000D_', ' ').replace('\r', ' ')
                parts.append(conclusion)
            
            if parts:
                full_text = ' '.join(parts)
                
                reports.append({
                    'id': report_id,
                    'description': description,
                    'conclusion': conclusion,
                    'full_text': full_text,
                    'source_file': Path(file_path).name,
                })
                
                report_id += 1
                
                if max_samples and len(reports) >= max_samples:
                    break
        
        print(f"  已加载: {len(reports)} 条新报告 (跳过: {skipped_count})")
        
        if max_samples and len(reports) >= max_samples:
            break
    
    return reports


# ============== 错误检测 ==============

def detect_errors_in_report(report: Dict, detector: LayeredGrammarDetector,
                            writer: AsyncWriter, use_llm: bool = True,
                            include_full_text: bool = True) -> bool:
    """
    检测单条报告中的语法错误，并实时写入结果
    
    Returns:
        是否发现错误
    """
    full_text = report['full_text']
    
    if not full_text or len(full_text) < 5:
        writer.update_stats(report['id'], False, [])
        return False
    
    has_error = False
    error_types = []
    
    try:
        grammar_errors = detector.detect(full_text, use_llm=use_llm)
        
        if grammar_errors:
            sentences = split_sentences_rigid(full_text)
            
            for err in grammar_errors:
                # 跳过英文或数字短语
                if is_english_or_number(err.text):
                    continue
                
                # 跳过纯空格/无意义片段
                if is_meaningless_fragment(err.text):
                    continue
                
                # 跳过首尾非汉字的片段（错误截断）
                if not has_chinese_at_boundaries(err.text):
                    continue
                
                # 保险层：验证短语是否完整存在于单个句子中
                if not is_phrase_in_single_sentence(err.text, full_text):
                    continue
                
                # 找到包含错误的句子（通过文本匹配，而非位置）
                # 位置可能因分词/处理而有偏差，文本匹配更可靠
                sentence_text = ''
                phrase_clean = err.text.strip()
                
                for sent in sentences:
                    if phrase_clean in sent['text']:
                        sentence_text = sent['text']
                        break
                
                # 如果找不到包含该短语的句子，回退到按位置查找
                if not sentence_text:
                    for sent in sentences:
                        if sent['start'] <= err.position[0] < sent['end']:
                            sentence_text = sent['text']
                            break
                
                # 最后的fallback：提取位置周围的文本
                if not sentence_text:
                    start = max(0, err.position[0] - 30)
                    end = min(len(full_text), err.position[1] + 30)
                    sentence_text = full_text[start:end]
                
                # 验证LLM的纠错建议是否合理
                if err.llm_verified and not is_valid_suggestion(err.text, err.suggestion, sentence_text):
                    # LLM suggestion 无效，使用启发式建议或跳过
                    continue
                
                error_record = {
                    'report_id': report['id'],
                    'error_phrase': err.text,
                    'sentence': sentence_text,
                    'suggestion': err.suggestion,
                    'error_type': err.error_type,
                    'confidence': err.confidence,
                    'position': {
                        'start': err.position[0],
                        'end': err.position[1],
                    },
                    'llm_verified': err.llm_verified,
                    'detected_by': 'layered_detector',
                    'source_file': report['source_file'],
                    'timestamp': datetime.now().isoformat(),
                }
                
                if include_full_text:
                    error_record['report_text'] = full_text
                
                # 实时写入队列
                writer.write(error_record)
                
                has_error = True
                error_types.append(err.error_type)
        
        # 更新统计
        writer.update_stats(report['id'], has_error, error_types)
        
    except Exception as e:
        print(f"检测报告 {report['id']} 时出错: {e}")
        writer.update_stats(report['id'], False, [])
    
    return has_error


# ============== 主函数 ==============

def batch_detect(model_dir: str, data_dir: str, output_file: str,
                max_samples: Optional[int] = None,
                use_llm: bool = True,
                include_full_text: bool = True,
                restart: bool = False,
                save_interval: int = 100):
    """
    批量检测语法错误 - 支持断点续传和实时写入
    """
    print("=" * 70)
    print("批量语法错误检测 - 支持断点续传")
    print("=" * 70)
    
    # 初始化断点管理器
    checkpoint_mgr = CheckpointManager(output_file)
    
    # 如果需要重新开始
    if restart:
        checkpoint_mgr.clear_checkpoint()
    
    # 加载已处理的ID
    processed_ids = checkpoint_mgr.get_processed_ids()
    print(f"已处理报告数: {len(processed_ids)}")
    
    # 初始化异步写入器
    writer = AsyncWriter(output_file, checkpoint_mgr.stats_file)
    writer.start()
    
    # 加载报告数据
    print("\n加载报告数据...")
    reports = load_reports_from_excel(data_dir, max_samples, processed_ids)
    print(f"待处理报告数: {len(reports)}")
    
    if not reports:
        print("没有新的报告需要处理")
        writer.stop()
        return
    
    # 创建检测器
    print("\n初始化检测器...")
    detector = LayeredGrammarDetector(model_dir=model_dir, llm_workers=2)
    
    # 开始检测
    print(f"\n开始检测语法错误...")
    print(f"输出文件: {output_file}")
    print(f"按 Ctrl+C 可以中断，下次会自动从断点继续")
    print("=" * 70)
    
    try:
        with tqdm(total=len(reports), desc="检测进度") as pbar:
            for i, report in enumerate(reports):
                detect_errors_in_report(
                    report, detector, writer,
                    use_llm=use_llm,
                    include_full_text=include_full_text
                )
                
                pbar.update(1)
                
                # 定期保存断点
                if (i + 1) % save_interval == 0:
                    processed_so_far = processed_ids | writer.stats['processed_ids']
                    checkpoint_mgr.save_checkpoint(
                        processed_so_far,
                        len(processed_ids) + i + 1
                    )
                    writer.save_stats()
                    pbar.set_postfix({'errors': writer.written_count})
        
        # 最终保存
        processed_so_far = processed_ids | writer.stats['processed_ids']
        checkpoint_mgr.save_checkpoint(processed_so_far, len(processed_ids) + len(reports))
        writer.save_stats()
        
    except KeyboardInterrupt:
        print("\n\n检测到中断，保存断点...")
        processed_so_far = processed_ids | writer.stats['processed_ids']
        checkpoint_mgr.save_checkpoint(processed_so_far, len(processed_ids) + len(reports))
        writer.save_stats()
        print(f"已保存断点，已处理: {len(processed_so_far)} 条报告")
    
    finally:
        writer.stop()
    
    # 输出统计
    print("\n" + "=" * 70)
    print("检测完成")
    print("=" * 70)
    print(f"总报告数: {writer.stats['total_reports']}")
    print(f"含错误的报告: {writer.stats['reports_with_errors']}")
    print(f"总错误数: {writer.written_count}")
    print(f"输出文件: {output_file}")
    print(f"统计文件: {checkpoint_mgr.stats_file}")


def main():
    parser = argparse.ArgumentParser(description='批量语法错误检测 - 支持断点续传')
    parser.add_argument('--model-dir', default='grammer/models',
                       help='模型目录')
    parser.add_argument('--data-dir',
                       default='~/work/python/Radiology_Entities/radiology_data',
                       help='数据目录')
    parser.add_argument('--output', default='grammer/errors.jsonl',
                       help='输出JSONL文件')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数（用于测试）')
    parser.add_argument('--no-llm', action='store_true',
                       help='不使用LLM验证（仅使用启发式规则，更快）')
    parser.add_argument('--no-full-text', action='store_true',
                       help='不包含完整报告文本（减小输出文件大小）')
    parser.add_argument('--restart', action='store_true',
                       help='强制重新开始（忽略之前的进度）')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='每处理多少条报告保存一次断点（默认100）')
    
    args = parser.parse_args()
    
    batch_detect(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_file=args.output,
        max_samples=args.max_samples,
        use_llm=not args.no_llm,
        include_full_text=not args.no_full_text,
        restart=args.restart,
        save_interval=args.save_interval,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
