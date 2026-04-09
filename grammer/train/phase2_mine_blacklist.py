"""
阶段二（增强版）：双策略黑名单挖掘

策略 A（增强）：多层次拼音混淆挖掘
- Level 1: 单字同音替换（已实现）
- Level 2: 双字组合替换（处理复杂混淆）
- Level 3: 医学术语锚点扩展（利用医学词典生成跨词混淆）
- Level 4: 上下文敏感混淆（如"平扫时"→"平扫示"）

策略 B：高危通用词筛选（保持不变）
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict
from itertools import product, combinations
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    HUQIE_PATH, MEDICAL_DICT_PATHS,
    RADIOLOGY_VOCAB, MEDICAL_CONFUSION, HIGH_RISK_GENERAL, EXPANDED_CANDIDATES,
    EXPAND_MIN_RADIO_FREQ, EXPAND_TWO_CHAR_THRESHOLD,
    EXPAND_MAX_VARIANT_FREQ, EXPAND_MAX_CANDIDATES_PER_WRONG,
    MIN_WORD_LEN, MAX_WORD_LEN, MIN_HUQIE_FREQ, MAX_RADIO_FREQ,
    RISK_THRESHOLD, PRIORITY_POS, POS_WEIGHT, MED_PROTECT_WEIGHT,
    ensure_dirs
)
from utils.utils import (
    load_huqie_vocab, load_medical_vocab, get_pinyin
)


def load_radiology_vocab(vocab_path: str) -> Dict[str, int]:
    """加载放射语料词频"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('word_freq', {})


def load_substring_vocab(substring_vocab_path: str = 'models/substring_vocab.json') -> Dict[str, int]:
    """
    加载子串频次表（基于原始未分词文本）
    
    用于解决分词粒度不一致导致的假阳性问题
    """
    if not Path(substring_vocab_path).exists():
        print(f"警告: 子串频次表不存在 {substring_vocab_path}")
        print("  提示: 运行 build_substring_vocab.py 生成")
        return {}
    
    with open(substring_vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('substring_freq', {})


def load_same_pinyin_dict(path: str = None) -> Dict[str, Set[str]]:
    """加载同音字字典"""
    if path is None:
        import pycorrector
        path = os.path.join(os.path.dirname(pycorrector.__file__), 'data', 'same_pinyin.txt')
    
    same_pinyin = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                char = parts[0]
                same_tone = parts[1] if len(parts) > 1 else ''
                diff_tone = parts[2] if len(parts) > 2 else ''
                homophones = set(same_tone + diff_tone)
                homophones.discard(char)
                if homophones:
                    same_pinyin[char] = homophones
    return dict(same_pinyin)


def contains_digit_or_english(text: str) -> bool:
    """
    检查文本是否包含数字或英文字母
    
    用于过滤非中文错误（如日期、测量值等）
    """
    if not text:
        return False
    # 检查是否包含数字
    if any(c.isdigit() for c in text):
        return True
    # 检查是否包含英文字母（不包括中文标点）
    if any(c.isalpha() and ord(c) < 128 for c in text):
        return True
    return False


def should_keep_confusion_pair(
    wrong_word: str,
    correct_word: str,
    wrong_freq: int,
    substring_vocab: Dict[str, int] = None,
    min_substring_freq: int = 1
) -> bool:
    """用原始文本子串频次过滤零证据候选。"""
    if contains_digit_or_english(wrong_word) or contains_digit_or_english(correct_word):
        return False

    if not substring_vocab:
        return True

    substring_freq = substring_vocab.get(wrong_word, 0)

    # 分词词表中的 0 频候选，如果在原始文本里也完全没出现，直接丢弃。
    if wrong_freq == 0 and substring_freq < min_substring_freq:
        return False

    return True


def pinyin_edit_distance(pinyin1: List[str], pinyin2: List[str]) -> int:
    m, n = len(pinyin1), len(pinyin2)
    if abs(m - n) > 2:
        return 100
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pinyin1[i-1] == pinyin2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def generate_single_replacements(
    word: str,
    same_pinyin: Dict[str, Set[str]]
) -> List[str]:
    """Level 1: 单字替换"""
    candidates = []
    for i, char in enumerate(word):
        if char not in same_pinyin:
            continue
        for homophone in same_pinyin[char]:
            variant = word[:i] + homophone + word[i+1:]
            if variant != word:
                candidates.append(variant)
    return candidates


def generate_double_replacements(
    word: str,
    same_pinyin: Dict[str, Set[str]],
    max_combinations: int = 200,  # 增加总组合数限制
    max_chars_per_position: int = 20  # 增加每个位置的选择数量
) -> List[str]:
    """
    Level 2: 双字组合替换
    
    处理复杂情况，如：
    - "纹理" (wen li) → "文里" (wen li) 需要同时替换两个字
    - "胆囊" (dan nang) → "但囊" (dan nang) 等
    
    Bug修复历史:
    - 原[:3]限制太严格，遗漏常用同音字
    - 原max_combinations=50太小，导致提前返回遗漏组合
    现在增加到200个组合、20个字符，确保覆盖常用混淆对
    """
    if len(word) < 2:
        return []
    
    candidates = []
    chars = list(word)
    
    # 找出所有可以替换的位置
    replaceable_positions = []
    for i, char in enumerate(chars):
        if char in same_pinyin:
            replaceable_positions.append((i, same_pinyin[char]))
    
    # 如果有至少2个可替换位置，生成双字组合
    if len(replaceable_positions) >= 2:
        count = 0
        for (i, chars_i), (j, chars_j) in combinations(replaceable_positions, 2):
            if abs(i - j) > 3:  # 限制距离，避免生成完全不相关的词
                continue
            # 增加每个位置的选择数量和总组合数限制
            for c1 in list(chars_i)[:max_chars_per_position]:
                for c2 in list(chars_j)[:max_chars_per_position]:
                    variant_chars = chars.copy()
                    variant_chars[i] = c1
                    variant_chars[j] = c2
                    variant = ''.join(variant_chars)
                    if variant != word:
                        candidates.append(variant)
                        count += 1
                        if count >= max_combinations:
                            return candidates
    
    return candidates


def generate_context_sensitive_errors(
    prefix: str,
    word: str,
    same_pinyin: Dict[str, Set[str]],
    radio_vocab: Dict[str, int]
) -> List[Tuple[str, str]]:
    """
    Level 4: 上下文敏感错误
    
    例如：
    - "平扫时" 中的 "时" 在 "平扫" 上下文中更可能是 "示"
    - 需要验证 "平扫示" 是否也是高频词
    
    Returns: [(错误词, 正确词), ...]
    """
    candidates = []
    
    # 组合前缀和当前词
    full_word = prefix + word
    if len(full_word) < 2:
        return candidates
    
    # 对整个组合词生成混淆
    single_vars = generate_single_replacements(full_word, same_pinyin)
    
    for variant in single_vars:
        # 确保变体在语料中存在（即使是低频）
        if variant in radio_vocab or radio_vocab.get(variant, 0) < 100:
            # 检查变体是否比原词低频
            if radio_vocab.get(variant, 0) < radio_vocab.get(full_word, 0) * 0.1:
                candidates.append((variant, full_word))
    
    return candidates


def expand_medical_terms(
    medical_vocab: Set[str],
    same_pinyin: Dict[str, Set[str]],
    radio_vocab: Dict[str, int],
    max_freq: int = 50
) -> List[Tuple[str, str, int, int]]:
    """
    Level 3: 医学术语锚点扩展
    
    对医学术语进行深度扩展，确保医学词汇的混淆覆盖
    """
    confusion_pairs = []
    
    for med_term in tqdm(medical_vocab, desc="医学术语扩展"):
        if len(med_term) < 2 or len(med_term) > 8:
            continue
        
        med_freq = radio_vocab.get(med_term, 0)
        if med_freq < 100:  # 只处理有一定频次的医学术语
            continue
        
        # 单字替换
        single_vars = generate_single_replacements(med_term, same_pinyin)
        
        # 双字替换
        double_vars = generate_double_replacements(med_term, same_pinyin, max_combinations=20)
        
        all_vars = set(single_vars + double_vars)
        
        # 检查每个变体
        for variant in all_vars:
            var_freq = radio_vocab.get(variant, 0)
            if var_freq > max_freq:
                continue
            
            # 拼音相似度检查
            med_pinyin = get_pinyin(med_term)
            var_pinyin = get_pinyin(variant)
            distance = pinyin_edit_distance(med_pinyin, var_pinyin)
            if distance > 2:
                continue
            
            confusion_pairs.append((variant, med_term, var_freq, med_freq))
    
    return confusion_pairs


def strategy_a_enhanced(
    radio_vocab: Dict[str, int],
    same_pinyin: Dict[str, Set[str]],
    medical_vocab: Set[str],
    min_radio_freq: int = 100,
    max_variant_freq: int = 50,
    max_pinyin_distance: int = 2,
    substring_vocab: Dict[str, int] = None,
    min_substring_freq: int = 1  # 子串在原始文本中最少出现次数
) -> List[Tuple[str, str, int, int]]:
    """
    策略 A（增强版）：多层次拼音混淆挖掘
    
    Args:
        substring_vocab: 子串频次表（基于原始未分词文本）
        min_substring_freq: 子串最小频次阈值，用于过滤假阳性
    """
    print("\n策略 A（增强版）：多层次拼音混淆挖掘...")
    print(f"Level 1: 单字替换")
    print(f"Level 2: 双字组合替换")
    print(f"Level 3: 医学术语锚点扩展")
    print(f"正确词阈值: >{min_radio_freq} 次")
    print(f"错误词阈值: <{max_variant_freq} 次")
    
    if substring_vocab:
        print(f"子串频次验证: 开启 (最小频次 {min_substring_freq})")
    
    confusion_pairs = []
    seen_pairs = set()
    
    # 选取候选正确词
    # 标准：3-10字，频次>=min_radio_freq
    standard_words = [
        (word, freq) for word, freq in radio_vocab.items()
        if MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN
        and freq >= min_radio_freq
    ]
    
    # 特殊处理：高频2字医学术语（如"纹理"、"胆囊"）
    # 2字词需要更高频次阈值（1000次），避免误报
    two_char_threshold = 1000
    two_char_words = [
        (word, freq) for word, freq in radio_vocab.items()
        if len(word) == 2
        and freq >= two_char_threshold
        and freq >= min_radio_freq  # 同时满足基本阈值
    ]
    
    candidate_words = standard_words + two_char_words
    # 去重（可能有重叠）
    seen = set()
    unique_candidates = []
    for word, freq in candidate_words:
        if word not in seen:
            seen.add(word)
            unique_candidates.append((word, freq))
    candidate_words = unique_candidates
    
    candidate_words.sort(key=lambda x: x[1], reverse=True)
    
    print(f"候选正确词数量: {len(candidate_words)}")
    print(f"  - 标准词(3-10字): {len(standard_words)}")
    print(f"  - 2字高频词: {len(two_char_words)}")
    
    # Level 1 & 2: 单字和双字替换
    for correct_word, correct_freq in tqdm(candidate_words, desc="Level 1&2"):
        # Level 1: 单字替换
        single_vars = generate_single_replacements(correct_word, same_pinyin)
        
        # Level 2: 双字替换
        double_vars = generate_double_replacements(correct_word, same_pinyin)
        
        all_vars = set(single_vars + double_vars)
        
        for wrong_word in all_vars:
            # 去重
            key = (wrong_word, correct_word)
            if key in seen_pairs:
                continue
            
            # 频次检查
            wrong_freq = radio_vocab.get(wrong_word, 0)
            if wrong_freq > max_variant_freq:
                continue
            
            # 医学术语保护
            if wrong_word in medical_vocab:
                continue
            
            # 拼音相似度检查
            correct_pinyin = get_pinyin(correct_word)
            wrong_pinyin = get_pinyin(wrong_word)
            distance = pinyin_edit_distance(correct_pinyin, wrong_pinyin)
            if distance > max_pinyin_distance:
                continue
            
            if not should_keep_confusion_pair(
                wrong_word,
                correct_word,
                wrong_freq,
                substring_vocab=substring_vocab,
                min_substring_freq=min_substring_freq,
            ):
                continue
            
            seen_pairs.add(key)
            confusion_pairs.append((wrong_word, correct_word, wrong_freq, correct_freq))
    
    print(f"Level 1&2 发现 {len(confusion_pairs)} 个混淆对")
    
    # Level 3: 医学术语深度扩展
    print("\nLevel 3: 医学术语锚点扩展...")
    medical_pairs = expand_medical_terms(medical_vocab, same_pinyin, radio_vocab)
    
    # 合并并去重
    for pair in medical_pairs:
        key = (pair[0], pair[1])
        if key not in seen_pairs:
            seen_pairs.add(key)
            confusion_pairs.append(pair)
    
    # 排序
    confusion_pairs.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n策略 A 总计发现 {len(confusion_pairs)} 个混淆对")
    
    # 显示示例
    if confusion_pairs:
        print("\n前30个混淆对（按正确词频次排序）:")
        print(f"{'错误词':<12} {'正确词':<12} {'错误频次':<10} {'正确频次':<10}")
        print("-" * 50)
        for wrong, correct, w_freq, c_freq in confusion_pairs[:30]:
            print(f"{wrong:<12} {correct:<12} {w_freq:<10} {c_freq:<10}")
        
        # 显示频次>0的真实错误
        real_errors = [p for p in confusion_pairs if p[2] > 0]
        print(f"\n其中真实出现过的错误（频次>0）: {len(real_errors)} 个")
        if real_errors:
            print("\n真实错误示例（前20个）:")
            for wrong, correct, w_freq, c_freq in real_errors[:20]:
                print(f"  {wrong} -> {correct} (发生 {w_freq} 次)")
    
    return confusion_pairs


def strategy_b_high_risk_general(
    huqie_vocab: Dict[str, Dict],
    radio_vocab: Dict[str, int],
    medical_vocab: Set[str],
    confusion_pairs: List[Tuple] = None
) -> List[Dict]:
    """策略 B：高危通用词筛选"""
    print("\n策略 B：高危通用词筛选...")
    
    confusion_error_words = set()
    if confusion_pairs:
        for wrong, correct, w_freq, c_freq in confusion_pairs:
            confusion_error_words.add(wrong)
        print(f"  排除 {len(confusion_error_words)} 个已知拼音错误词")
    
    high_risk_words = []
    
    for word, info in tqdm(huqie_vocab.items(), desc="高危词筛选"):
        huqie_freq = info['freq']
        pos = info.get('pos', 'n')
        
        # 过滤包含数字或英文的词语（只关注纯中文错误）
        if contains_digit_or_english(word):
            continue
        
        if word in confusion_error_words:
            continue
        if len(word) < MIN_WORD_LEN or len(word) > MAX_WORD_LEN:
            continue
        if huqie_freq < MIN_HUQIE_FREQ:
            continue
        
        radio_count = radio_vocab.get(word, 0)
        if radio_count > MAX_RADIO_FREQ:
            continue
        
        general_score = math.log(huqie_freq + 1)
        radio_score = math.log(radio_count + 2)
        risk_ratio = general_score / radio_score
        
        pos_weight = POS_WEIGHT if pos in PRIORITY_POS else 1.0
        med_weight = MED_PROTECT_WEIGHT if word in medical_vocab else 1.0
        
        final_score = risk_ratio * pos_weight * med_weight
        
        if final_score > RISK_THRESHOLD:
            high_risk_words.append({
                'word': word,
                'score': final_score,
                'huqie_freq': huqie_freq,
                'radio_freq': radio_count,
                'pos': pos
            })
    
    high_risk_words.sort(key=lambda x: x['score'], reverse=True)
    print(f"策略 B 发现 {len(high_risk_words)} 个高危通用词")
    
    return high_risk_words


def save_confusion_pairs(confusion_pairs: List[Tuple], output_path: str):
    """保存混淆对"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 拼音混淆对（增强版：单字+双字+医学术语扩展）\n")
        f.write("# 格式: 错误词 正确词 错误频次 正确频次\n\n")
        for wrong, correct, w_freq, c_freq in confusion_pairs:
            f.write(f"{wrong} {correct} {w_freq} {c_freq}\n")
    print(f"\n混淆对已保存: {output_path} ({len(confusion_pairs)} 对)")


def save_high_risk_words(high_risk_words: List[Dict], output_path: str):
    """保存高危词"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 高危通用词列表\n")
        f.write("# 格式: 词 词性 通用频次 放射频次 风险分数\n\n")
        for item in high_risk_words:
            line = f"{item['word']} {item['pos']} {item['huqie_freq']} " \
                   f"{item['radio_freq']} {item['score']:.2f}\n"
            f.write(line)
    print(f"高危词已保存: {output_path} ({len(high_risk_words)} 个)")


def build_expanded_candidate_lexicon(
    radio_vocab: Dict[str, int],
    same_pinyin: Dict[str, Set[str]],
    medical_vocab: Set[str],
    min_radio_freq: int = EXPAND_MIN_RADIO_FREQ,
    two_char_threshold: int = EXPAND_TWO_CHAR_THRESHOLD,
    max_variant_freq: int = EXPAND_MAX_VARIANT_FREQ,
    max_pinyin_distance: int = 2,
    max_candidates_per_wrong: int = EXPAND_MAX_CANDIDATES_PER_WRONG,
) -> Dict[str, List[Dict]]:
    """构建离线候选词库，用于运行时 KenLM 精筛。

    该词库与 medical_confusion.txt 分工不同：
    - medical_confusion.txt 保留高置信、可直接命中的显式混淆对
    - expanded_candidates.json 保留低证据/零证据但医学上合理的候选，交由运行时 KenLM 再筛
    """
    print("\n构建离线候选词库（供运行时 KenLM 精筛）...")

    standard_words = [
        (word, freq) for word, freq in radio_vocab.items()
        if MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN and freq >= min_radio_freq
    ]
    two_char_words = [
        (word, freq) for word, freq in radio_vocab.items()
        if len(word) == 2 and freq >= two_char_threshold and freq >= min_radio_freq
    ]

    seen = set()
    candidate_words = []
    for word, freq in standard_words + two_char_words:
        if word in seen:
            continue
        seen.add(word)
        candidate_words.append((word, freq))

    candidate_words.sort(key=lambda x: x[1], reverse=True)

    lexicon = defaultdict(list)
    seen_pairs = set()

    for correct_word, correct_freq in tqdm(candidate_words, desc="扩展候选"):
        single_vars = generate_single_replacements(correct_word, same_pinyin)

        for wrong_word in set(single_vars):
            if wrong_word == correct_word:
                continue
            if contains_digit_or_english(wrong_word) or contains_digit_or_english(correct_word):
                continue
            if wrong_word in medical_vocab:
                continue

            wrong_freq = radio_vocab.get(wrong_word, 0)
            if wrong_freq > max_variant_freq:
                continue

            correct_pinyin = get_pinyin(correct_word)
            wrong_pinyin = get_pinyin(wrong_word)
            distance = pinyin_edit_distance(correct_pinyin, wrong_pinyin)
            if distance > max_pinyin_distance:
                continue

            key = (wrong_word, correct_word)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            offline_score = math.log((correct_freq + 1) / (wrong_freq + 1))

            lexicon[wrong_word].append({
                'suggestion': correct_word,
                'wrong_freq': wrong_freq,
                'correct_freq': correct_freq,
                'offline_score': round(offline_score, 6),
            })

    trimmed_lexicon = {}
    for wrong_word, candidates in lexicon.items():
        candidates.sort(
            key=lambda x: (x.get('offline_score', 0.0), x.get('correct_freq', 0)),
            reverse=True,
        )
        trimmed_lexicon[wrong_word] = candidates[:max_candidates_per_wrong]

    print(f"离线候选词条数: {len(trimmed_lexicon)}")
    return trimmed_lexicon


def save_expanded_candidates(candidate_lexicon: Dict[str, List[Dict]], output_path: str):
    payload = {
        'metadata': {
            'version': 'v1',
            'description': '供运行时 KenLM 精筛使用的离线候选词库',
        },
        'candidates': candidate_lexicon,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"离线候选词库已保存: {output_path} ({len(candidate_lexicon)} 个错误词)")


def mine_blacklist_v4(
    radio_vocab_path: str = None,
    huqie_path: str = None,
    medical_vocab_paths: List[Path] = None,
    output_confusion: str = None,
    output_high_risk: str = None,
    output_candidates: str = None,
    min_radio_freq: int = EXPAND_MIN_RADIO_FREQ,
    two_char_threshold: int = EXPAND_TWO_CHAR_THRESHOLD,
    max_variant_freq: int = EXPAND_MAX_VARIANT_FREQ,
    max_candidates_per_wrong: int = EXPAND_MAX_CANDIDATES_PER_WRONG,
) -> Tuple[str, str, str]:
    """主函数"""
    radio_vocab_path = radio_vocab_path or str(RADIOLOGY_VOCAB)
    huqie_path = huqie_path or str(HUQIE_PATH)
    medical_vocab_paths = medical_vocab_paths or MEDICAL_DICT_PATHS
    output_confusion = output_confusion or str(MEDICAL_CONFUSION)
    output_high_risk = output_high_risk or str(HIGH_RISK_GENERAL)
    output_candidates = output_candidates or str(EXPANDED_CANDIDATES)
    
    ensure_dirs()
    
    print("=" * 70)
    print("阶段二（增强版）：多层次拼音混淆挖掘")
    print("=" * 70)
    
    # 加载数据
    print("\n加载数据...")
    radio_vocab = load_radiology_vocab(radio_vocab_path)
    print(f"  放射语料: {len(radio_vocab):,} 个唯一词")
    
    same_pinyin = load_same_pinyin_dict()
    print(f"  同音字字典: {len(same_pinyin):,} 个字")
    
    huqie_vocab = load_huqie_vocab(huqie_path)
    print(f"  通用词表: {len(huqie_vocab):,} 个词")
    
    medical_vocab = load_medical_vocab(medical_vocab_paths)
    print(f"  医学词典: {len(medical_vocab):,} 个术语")
    
    # 加载子串频次表（用于过滤假阳性）
    substring_vocab = load_substring_vocab()
    if substring_vocab:
        print(f"  子串频次表: {len(substring_vocab):,} 个子串")
    else:
        print("  警告: 未加载子串频次表，假阳性过滤可能不完整")
    
    # 策略 A（增强版）
    confusion_pairs = strategy_a_enhanced(
        radio_vocab, same_pinyin, medical_vocab,
        min_radio_freq=min_radio_freq,
        max_variant_freq=max_variant_freq,
        substring_vocab=substring_vocab,
        min_substring_freq=1
    )
    
    # 策略 B
    high_risk_words = strategy_b_high_risk_general(
        huqie_vocab, radio_vocab, medical_vocab, confusion_pairs
    )

    candidate_lexicon = build_expanded_candidate_lexicon(
        radio_vocab=radio_vocab,
        same_pinyin=same_pinyin,
        medical_vocab=medical_vocab,
        min_radio_freq=min_radio_freq,
        two_char_threshold=two_char_threshold,
        max_variant_freq=max_variant_freq,
        max_candidates_per_wrong=max_candidates_per_wrong,
    )
    
    # 保存
    save_confusion_pairs(confusion_pairs, output_confusion)
    save_high_risk_words(high_risk_words, output_high_risk)
    save_expanded_candidates(candidate_lexicon, output_candidates)
    
    print("\n" + "=" * 70)
    print("阶段二完成！")
    print("=" * 70)
    print(f"混淆对: {len(confusion_pairs)} 对")
    print(f"高危词: {len(high_risk_words)} 个")
    print(f"扩展候选: {len(candidate_lexicon)} 个错误词")
    
    return output_confusion, output_high_risk, output_candidates


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='阶段二（增强版）')
    parser.add_argument('--radio-vocab', default=str(RADIOLOGY_VOCAB))
    parser.add_argument('--huqie-path', default=str(HUQIE_PATH))
    parser.add_argument('--output-confusion', default=str(MEDICAL_CONFUSION))
    parser.add_argument('--output-high-risk', default=str(HIGH_RISK_GENERAL))
    parser.add_argument('--output-candidates', default=str(EXPANDED_CANDIDATES))
    parser.add_argument('--min-radio-freq', type=int, default=EXPAND_MIN_RADIO_FREQ)
    parser.add_argument('--two-char-threshold', type=int, default=EXPAND_TWO_CHAR_THRESHOLD)
    parser.add_argument('--max-variant-freq', type=int, default=EXPAND_MAX_VARIANT_FREQ)
    parser.add_argument('--max-candidates-per-wrong', type=int, default=EXPAND_MAX_CANDIDATES_PER_WRONG)
    args = parser.parse_args()
    
    confusion_file, high_risk_file, candidates_file = mine_blacklist_v4(
        radio_vocab_path=args.radio_vocab,
        huqie_path=args.huqie_path,
        output_confusion=args.output_confusion,
        output_high_risk=args.output_high_risk,
        output_candidates=args.output_candidates,
        min_radio_freq=args.min_radio_freq,
        two_char_threshold=args.two_char_threshold,
        max_variant_freq=args.max_variant_freq,
        max_candidates_per_wrong=args.max_candidates_per_wrong,
    )
    
    if confusion_file and high_risk_file and candidates_file:
        print("\n挖掘成功！")
        sys.exit(0)
    else:
        print("\n挖掘失败")
        sys.exit(1)
