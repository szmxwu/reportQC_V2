"""
工具函数模块
- 文本清洗、句子拆分
- 中文过滤
- 拼音混淆生成
"""
import re
import math
from typing import List, Set, Dict, Tuple
from pypinyin import lazy_pinyin, Style

try:
    from .config import (
        SENTENCE_DELIMITERS, CHINESE_RANGE,
        FUZZY_PINYIN_MAP, MAX_PINYIN_DISTANCE,
        MIN_WORD_LEN, MAX_WORD_LEN,
    )
except ImportError:
    from config import (
        SENTENCE_DELIMITERS, CHINESE_RANGE,
        FUZZY_PINYIN_MAP, MAX_PINYIN_DISTANCE,
        MIN_WORD_LEN, MAX_WORD_LEN,
    )


# ==================== 文本处理工具 ====================

def split_sentences(text: str) -> List[str]:
    """
    按标点符号拆分句子
    
    Args:
        text: 输入文本
        
    Returns:
        句子列表
    """
    if not text or not isinstance(text, str):
        return []
    
    # 按标点分割
    sentences = re.split(SENTENCE_DELIMITERS, text)
    # 过滤空句子和纯空白
    sentences = [s.strip() for s in sentences if s and s.strip()]
    return sentences


def is_chinese_char(char: str) -> bool:
    """判断字符是否为中文字符"""
    if len(char) != 1:
        return False
    code = ord(char)
    return CHINESE_RANGE[0] <= code <= CHINESE_RANGE[1]


def is_chinese_word(word: str) -> bool:
    """判断词是否包含中文字符"""
    if not word or not isinstance(word, str):
        return False
    return any(is_chinese_char(c) for c in word)


def extract_chinese_chars(text: str) -> List[str]:
    """
    提取文本中的中文字符序列
    
    Args:
        text: 输入文本
        
    Returns:
        中文字符列表
    """
    if not text:
        return []
    return [c for c in text if is_chinese_char(c)]


def filter_chinese_text(text: str) -> str:
    """
    过滤文本，只保留中文字符
    
    Args:
        text: 输入文本
        
    Returns:
        纯中文字符串
    """
    if not text:
        return ""
    return "".join(c for c in text if is_chinese_char(c))


# ==================== 拼音混淆工具 ====================

def get_pinyin(word: str) -> List[str]:
    """
    获取词的拼音列表
    
    Args:
        word: 中文词
        
    Returns:
        拼音列表
    """
    return lazy_pinyin(word, style=Style.NORMAL)


def generate_fuzzy_pinyin_variants(word: str) -> List[Tuple[str, ...]]:
    """
    生成词的拼音混淆变体
    
    基于 FUZZY_PINYIN_MAP 生成所有可能的拼音变体组合
    
    Args:
        word: 中文词
        
    Returns:
        拼音变体列表，每个变体是一个拼音元组
    """
    base_pinyin = get_pinyin(word)
    
    # 定义拼音混淆规则
    # 声母混淆
    initial_map = {
        'z': ['z', 'zh'], 'zh': ['z', 'zh'],
        'c': ['c', 'ch'], 'ch': ['c', 'ch'],
        's': ['s', 'sh'], 'sh': ['s', 'sh'],
        'n': ['n', 'l'], 'l': ['n', 'l'],
        'f': ['f', 'h'], 'h': ['f', 'h'],
        'r': ['r', 'l'], 'l': ['r', 'l', 'n'],
    }
    
    # 韵母混淆
    final_map = {
        'en': ['en', 'eng'], 'eng': ['en', 'eng'],
        'in': ['in', 'ing'], 'ing': ['in', 'ing'],
        'an': ['an', 'ang'], 'ang': ['an', 'ang'],
        'un': ['un', 'ong'], 'ong': ['un', 'ong'],
    }
    
    # 对每个拼音位置，找出可能的混淆音
    fuzzy_options = []
    for py in base_pinyin:
        options = {py}  # 包含原拼音
        
        # 解析拼音：声母 + 韵母
        # 常见声母列表（从长到短匹配）
        initials = ['zh', 'ch', 'sh', 'z', 'c', 's', 'b', 'p', 'm', 'f', 
                   'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 
                   'r', 'y', 'w']
        
        initial = ''
        final = py
        for ini in initials:
            if py.startswith(ini):
                initial = ini
                final = py[len(ini):]
                break
        
        # 应用声母混淆
        if initial in initial_map:
            for new_initial in initial_map[initial]:
                if new_initial != initial:
                    options.add(new_initial + final)
        
        # 应用韵母混淆
        for old_final, new_finals in final_map.items():
            if final == old_final or final.endswith(old_final):
                for new_final in new_finals:
                    if new_final != old_final:
                        # 替换韵母
                        if final == old_final:
                            options.add(initial + new_final)
                        else:
                            # 处理复合韵母，如 "ian" -> "iang"
                            prefix = final[:-len(old_final)]
                            options.add(initial + prefix + new_final)
        
        fuzzy_options.append(list(options))
    
    # 生成所有组合
    from itertools import product
    variants = list(product(*fuzzy_options))
    
    # 过滤掉原拼音
    original = tuple(base_pinyin)
    variants = [v for v in variants if v != original]
    
    return variants


def pinyin_edit_distance(pinyin1: List[str], pinyin2: List[str]) -> int:
    """
    计算两个拼音列表的编辑距离
    
    Args:
        pinyin1: 第一个拼音列表
        pinyin2: 第二个拼音列表
        
    Returns:
        编辑距离
    """
    # 使用动态规划计算最小编辑距离
    m, n = len(pinyin1), len(pinyin2)
    
    # 如果长度差异太大，直接返回大值
    if abs(m - n) > MAX_PINYIN_DISTANCE:
        return MAX_PINYIN_DISTANCE + 1
    
    # 动态规划表
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
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # 删除
                    dp[i][j-1],      # 插入
                    dp[i-1][j-1]     # 替换
                )
    
    return dp[m][n]


def are_pinyin_similar(word1: str, word2: str, max_distance: int = MAX_PINYIN_DISTANCE) -> bool:
    """
    判断两个词的拼音是否相似
    
    Args:
        word1: 第一个词
        word2: 第二个词
        max_distance: 最大编辑距离
        
    Returns:
        是否相似
    """
    pinyin1 = get_pinyin(word1)
    pinyin2 = get_pinyin(word2)
    
    distance = pinyin_edit_distance(pinyin1, pinyin2)
    return distance <= max_distance


# ==================== 动态阈值计算工具 ====================

def calculate_risk_score(huqie_freq: int, radio_freq: int, 
                         pos: str = '', medical_dict: Set[str] = None,
                         word: str = '') -> float:
    """
    计算高危词风险分数
    
    公式：risk_ratio * pos_weight * med_weight
    - risk_ratio = log(huqie_freq + 1) / log(radio_freq + 2)
    - pos_weight: 专有名词 1.5，其他 1.0
    - med_weight: 医学词典中 0.1，其他 1.0
    
    Args:
        huqie_freq: 通用词频次
        radio_freq: 放射语料频次
        pos: 词性
        medical_dict: 医学词典集合
        word: 词本身
        
    Returns:
        风险分数（越高越危险）
    """
    try:
        from .config import PRIORITY_POS, POS_WEIGHT, MED_PROTECT_WEIGHT
    except ImportError:
        from config import PRIORITY_POS, POS_WEIGHT, MED_PROTECT_WEIGHT
    
    # 基础风险比（平滑处理）
    general_score = math.log(huqie_freq + 1)
    radio_score = math.log(radio_freq + 2)  # +2 避免 log(1)=0
    risk_ratio = general_score / radio_score
    
    # 词性权重
    pos_weight = POS_WEIGHT if pos in PRIORITY_POS else 1.0
    
    # 医学词典保护
    if medical_dict and word in medical_dict:
        med_weight = MED_PROTECT_WEIGHT
    else:
        med_weight = 1.0
    
    return risk_ratio * pos_weight * med_weight


# ==================== 文件加载工具 ====================

def load_medical_vocab(paths: List) -> Set[str]:
    """
    加载医学词典
    
    Args:
        paths: 医学词典文件路径列表
        
    Returns:
        医学词集合
    """
    medical_vocab = set()
    
    for path in paths:
        if not path.exists():
            continue
        
        # 处理 txt 文件（user_dic_expand.txt 格式：词 词频 词性）
        if path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        word = parts[0]
                        if MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN:
                            medical_vocab.add(word)
        
        # 处理 xlsx 文件（knowledgegraph.xlsx）
        elif path.suffix == '.xlsx':
            try:
                import pandas as pd
                df = pd.read_excel(path)
                # 假设包含中文词列
                for col in df.columns:
                    if '部位' in col or '名称' in col:
                        for word in df[col].dropna():
                            word = str(word).strip()
                            if MIN_WORD_LEN <= len(word) <= MAX_WORD_LEN:
                                medical_vocab.add(word)
            except Exception as e:
                print(f"加载 {path} 失败: {e}")
    
    return medical_vocab


def load_huqie_vocab(path) -> Dict[str, Dict]:
    """
    加载 huqie.txt 词表
    
    格式：词语 词频 词性
    
    Args:
        path: huqie.txt 路径
        
    Returns:
        {词: {'freq': 词频, 'pos': 词性}}
    """
    vocab = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                word = parts[0]
                try:
                    freq = int(parts[1])
                    pos = parts[2]
                    vocab[word] = {'freq': freq, 'pos': pos}
                except ValueError:
                    continue
            elif len(parts) == 2:
                word = parts[0]
                try:
                    freq = int(parts[1])
                    vocab[word] = {'freq': freq, 'pos': 'n'}
                except ValueError:
                    continue
    
    return vocab


# ==================== 测试 ====================
if __name__ == '__main__':
    # 测试句子拆分
    text = "双肺纹理增粗，见多发结节影。边界清晰，建议随访。"
    sentences = split_sentences(text)
    print(f"句子拆分: {sentences}")
    
    # 测试中文提取
    chinese = extract_chinese_chars(text)
    print(f"中文字符: {''.join(chinese)}")
    
    # 测试拼音混淆
    word = "纹理"
    variants = generate_fuzzy_pinyin_variants(word)
    print(f"'{word}' 的拼音变体: {variants[:5]}...")
    
    # 测试拼音相似度
    word1, word2 = "文里", "纹理"
    similar = are_pinyin_similar(word1, word2)
    print(f"'{word1}' 和 '{word2}' 拼音相似: {similar}")
    
    # 测试风险分数
    score = calculate_risk_score(huqie_freq=1560, radio_freq=0, pos='nt', word='沙钢股份')
    print(f"'沙钢股份' 风险分数: {score:.2f}")
