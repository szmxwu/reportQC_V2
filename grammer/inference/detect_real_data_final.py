#!/usr/bin/env python3
"""
真实数据错别字检测脚本 - 最终版

使用改进版引擎（分词后匹配），避免子串误报
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import jieba
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.phase3_build_engine_v2 import FastMedicalCorrectorV2
from inference.word_order_detector import WordOrderDetector
from train.phase2_mine_blacklist import contains_digit_or_english


# 句子分割正则
SENTENCE_PATTERN = re.compile(r'[。！？；，、：:;,.!?\n\r]+')


def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """将文本分割为短句列表"""
    if not text or not isinstance(text, str):
        return []
    
    sentences = []
    start = 0
    
    for match in SENTENCE_PATTERN.finditer(text):
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


def find_error_context(text: str, error_start: int, error_end: int) -> str:
    """找到错误所在的短句"""
    sentences = split_into_sentences(text)
    
    for sentence, sent_start, sent_end in sentences:
        if sent_start <= error_start < sent_end or sent_start < error_end <= sent_end:
            return sentence
    
    return text[error_start:error_end]


def is_sufficient_word_order_context(context: str, error_word: str) -> bool:
    """判断词序错误是否有足够上下文，避免孤立短语误报。"""
    if not context:
        return False

    normalized_context = re.sub(r'\s+', '', context)
    normalized_error = re.sub(r'\s+', '', error_word)

    # 句子仅等于错误词时，无法判断真实词序关系。
    if normalized_context == normalized_error:
        return False

    # 上下文仅比错误词多 1 个字符时，信息依然不足，保守跳过。
    if len(normalized_context) <= len(normalized_error) + 1:
        return False

    return True


class MedicalTypoDetectorFinal:
    """最终版检测器（拼音+词序）"""
    
    def __init__(self):
        self.corrector = None
        self.word_order_detector = None
        self.lm_model = None
        self.typo_lm_reject_delta = 0.30
        self._lm_score_cache = {}
    
    def load(self):
        """加载引擎"""
        # 加载拼音/高危词检测器
        self.corrector = FastMedicalCorrectorV2()
        
        engine_path = 'models/ac_automaton_v2.pkl'
        if Path(engine_path).exists():
            self.corrector.load(engine_path)
        else:
            self.corrector.load_blacklists(
                'models/medical_confusion.txt',
                'models/high_risk_general.txt'
            )
            self.corrector.save(engine_path)
        
        # 加载词序检测器
        word_order_path = 'models/word_order_templates.json'
        if Path(word_order_path).exists():
            self.word_order_detector = WordOrderDetector(word_order_path)

        # 加载 KenLM 语言模型（可选）
        lm_path = Path('models/radiology_ngram.klm')
        try:
            import kenlm
            if lm_path.exists():
                self.lm_model = kenlm.LanguageModel(str(lm_path))
                print(f"KenLM 上下文模型已加载: {lm_path}")
            else:
                print(f"KenLM 模型不存在，跳过上下文校验: {lm_path}")
        except Exception as e:
            print(f"KenLM 加载失败，跳过上下文校验: {e}")
        
        print("检测器加载完成")

    @staticmethod
    def _extract_chinese_chars(text: str) -> List[str]:
        return [c for c in text if '\u4e00' <= c <= '\u9fff']

    def _kenlm_score(self, text: str) -> float:
        if not self.lm_model:
            return 0.0

        chars = self._extract_chinese_chars(text)
        if not chars:
            return 0.0

        seq = ' '.join(chars)
        if seq in self._lm_score_cache:
            return self._lm_score_cache[seq]

        score = self.lm_model.score(seq, bos=False, eos=False)
        self._lm_score_cache[seq] = score
        return score

    @staticmethod
    def _tokenize_with_positions(text: str) -> List[Tuple[str, int, int]]:
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
            tokens.append((word, start, end))
            search_start = end
        return tokens

    def _get_prev_next_word(self, text: str, start: int, end: int) -> Tuple[str, str]:
        tokens = self._tokenize_with_positions(text)
        idx = -1
        for i, (_, s, e) in enumerate(tokens):
            if s <= start < e or s < end <= e or (start <= s and e <= end):
                idx = i
                break

        if idx == -1:
            return '', ''

        prev_word = tokens[idx - 1][0] if idx - 1 >= 0 else ''
        next_word = tokens[idx + 1][0] if idx + 1 < len(tokens) else ''
        return prev_word, next_word

    def _is_typo_plausible_by_context(
        self,
        text: str,
        error_word: str,
        suggestion_word: str,
        start: int,
        end: int,
    ) -> bool:
        """基于 KenLM 上下文概率判断 typo 是否可信。"""
        if not self.lm_model or not suggestion_word:
            return True

        prev_word, next_word = self._get_prev_next_word(text, start, end)
        if not prev_word and not next_word:
            return True

        original_context = f"{prev_word}{error_word}{next_word}"
        corrected_context = f"{prev_word}{suggestion_word}{next_word}"

        original_score = self._kenlm_score(original_context)
        corrected_score = self._kenlm_score(corrected_context)
        delta = corrected_score - original_score

        # 仅在“原词明显优于建议词”时拒绝，避免过度过滤造成漏报。
        if delta <= -self.typo_lm_reject_delta:
            return False
        return True
    
    def detect(self, text: str) -> List[Dict]:
        """检测文本（过滤掉包含数字或英文的错误）"""
        if not text or not isinstance(text, str):
            return []
        
        errors = []
        
        # 1. 拼音/高危词检测（分词后匹配）
        if self.corrector:
            matches = self.corrector.scan(text)
            for m in matches:
                # 过滤包含数字或英文的错误（只关注纯中文错误）
                if contains_digit_or_english(m.error):
                    continue
                if m.suggestion and contains_digit_or_english(m.suggestion):
                    continue

                if m.error_type == 'typo' and m.suggestion:
                    if not self._is_typo_plausible_by_context(
                        text=text,
                        error_word=m.error,
                        suggestion_word=m.suggestion,
                        start=m.position[0],
                        end=m.position[1],
                    ):
                        continue
                
                context = find_error_context(text, m.position[0], m.position[1])
                errors.append({
                    'error': m.error,
                    'suggestion': m.suggestion,
                    'type': m.error_type,
                    'position': {'start': m.position[0], 'end': m.position[1]},
                    'context': context,
                    'score': m.score
                })
        
        # 2. 词序错误检测
        if self.word_order_detector:
            word_order_errors = self.word_order_detector.detect(text)
            for e in word_order_errors:
                context = e.get('context') or find_error_context(text, e['position'][0], e['position'][1])

                if not is_sufficient_word_order_context(context, e['error']):
                    continue

                # 检查是否已存在（避免重复）
                pos = e['position']
                is_duplicate = any(
                    err['position']['start'] == pos[0] and err['position']['end'] == pos[1]
                    for err in errors
                )
                if not is_duplicate:
                    errors.append({
                        'error': e['error'],
                        'suggestion': e['suggestion'],
                        'type': 'word_order',
                        'position': {'start': pos[0], 'end': pos[1]},
                        'context': context,
                        'confidence': e.get('confidence')
                    })
        
        # 按位置排序
        errors.sort(key=lambda x: x['position']['start'])
        
        return errors


def process_excel_file(
    detector: MedicalTypoDetectorFinal,
    input_path: str,
    output_path: str,
    text_columns: List[str] = ['描述', '结论'],
    limit: int = None
) -> Dict:
    """处理Excel文件"""
    print(f"读取数据: {input_path}")
    start_time = time.perf_counter()
    df = pd.read_excel(input_path)
    
    if limit:
        df = df.head(limit)
    
    total_rows = len(df)
    print(f"总记录数: {total_rows}")
    
    results = []
    error_stats = {
        'total_reports': 0,
        'reports_with_errors': 0,
        'total_errors': 0,
        'by_type': {'typo': 0, 'word_order': 0, 'general_high_risk': 0},
        'by_column': {}
    }
    
    for col in text_columns:
        error_stats['by_column'][col] = 0
    
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="检测中"):
        report_id = str(row.get('影像号', f'row_{idx}'))
        
        for col in text_columns:
            if col not in df.columns:
                continue
            
            text = str(row.get(col, ''))
            if pd.isna(text) or not text.strip():
                continue
            
            error_stats['total_reports'] += 1
            
            errors = detector.detect(text)
            
            if errors:
                error_stats['reports_with_errors'] += 1
                error_stats['total_errors'] += len(errors)
                error_stats['by_column'][col] += len(errors)
                
                for err in errors:
                    error_stats['by_type'][err['type']] += 1

                sentence_groups = {}
                for err in errors:
                    sentence = err.get('context') or find_error_context(
                        text,
                        err['position']['start'],
                        err['position']['end'],
                    )
                    sentence_groups.setdefault(sentence, []).append(err)

                for sentence, sentence_errors in sentence_groups.items():
                    results.append({
                        'report_id': report_id,
                        'column': col,
                        'text': sentence,
                        'errors': sentence_errors,
                    })
    
    output = {
        'metadata': {
            'input_file': input_path,
            'total_rows': total_rows,
            'text_columns': text_columns,
            'statistics': error_stats
        },
        'results': results
    }

    total_runtime_seconds = time.perf_counter() - start_time
    total_reports = error_stats['total_reports']
    output['metadata']['performance'] = {
        'total_runtime_seconds': round(total_runtime_seconds, 3),
        'avg_runtime_per_report_ms': round(
            total_runtime_seconds * 1000 / max(total_reports, 1),
            3,
        ),
        'avg_runtime_per_row_ms': round(
            total_runtime_seconds * 1000 / max(total_rows, 1),
            3,
        ),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    return error_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='真实放射科报告错别字检测 - 最终版')
    parser.add_argument('input', nargs='?', help='输入Excel文件路径')
    parser.add_argument('-o', '--output', help='输出JSON路径', default=None)
    parser.add_argument('-c', '--columns', nargs='+', default=['描述', '结论'])
    parser.add_argument('-l', '--limit', type=int, help='限制处理条数')
    parser.add_argument('--sample', action='store_true', help='使用示例数据')
    
    args = parser.parse_args()
    
    if args.sample:
        input_path = '/home/wmx/work/python/Radiology_Entities/radiology_data/all_data_sample.xlsx'
    elif args.input:
        input_path = args.input
    else:
        parser.print_help()
        sys.exit(1)
    
    output_path = args.output or f'output/detect_results_{Path(input_path).stem}_final.json'
    
    print("="*70)
    print("加载检测器...")
    print("="*70)
    
    detector = MedicalTypoDetectorFinal()
    detector.load()
    
    print("\n" + "="*70)
    print("开始检测...")
    print("="*70)
    
    error_stats = process_excel_file(
        detector, input_path, output_path,
        text_columns=args.columns, limit=args.limit
    )
    
    print("\n" + "="*70)
    print("检测统计")
    print("="*70)
    print(f"总报告数: {error_stats['total_reports']}")
    print(f"有错误的报告: {error_stats['reports_with_errors']} ({error_stats['reports_with_errors']/max(error_stats['total_reports'],1)*100:.1f}%)")
    print(f"总错误数: {error_stats['total_errors']}")
    
    print(f"\n按类型分布:")
    for err_type, count in error_stats['by_type'].items():
        if count > 0:
            print(f"  - {err_type}: {count}")
    
    print("\n" + "="*70)
    print(f"完成！结果保存至: {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()
