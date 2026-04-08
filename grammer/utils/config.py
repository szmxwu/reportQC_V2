"""
配置管理模块
"""
import os
from pathlib import Path

# 项目根目录（grammer/目录）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 确保是grammer目录（处理从utils导入的情况）
if PROJECT_ROOT.name == 'reportQC_v2':
    PROJECT_ROOT = PROJECT_ROOT / 'grammer'

REPO_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == 'grammer' else PROJECT_ROOT
CONFIG_ROOT = PROJECT_ROOT / 'config'
if not CONFIG_ROOT.exists():
    CONFIG_ROOT = REPO_ROOT / 'config'

# ==================== 路径配置 ====================
# 输入数据路径
DATA_DIR = os.path.expanduser("~/work/python/Radiology_Entities/radiology_data")
HUQIE_PATH = PROJECT_ROOT / "huqie.txt"

# 医学词典路径
MEDICAL_DICT_PATHS = [
    CONFIG_ROOT / "user_dic_expand.txt",
    CONFIG_ROOT / "knowledgegraph.xlsx",
]

# 输出路径（相对于项目根目录）
OUTPUT_DIR = PROJECT_ROOT / "models"
TEMP_DIR = PROJECT_ROOT / "temp_chunks"

# 输出文件
RADIOLOGY_CORPUS = OUTPUT_DIR / "radiology_corpus.txt"
RADIOLOGY_NGRAM = OUTPUT_DIR / "radiology_ngram.klm"
RADIOLOGY_VOCAB = OUTPUT_DIR / "radiology_vocab.json"
MEDICAL_CONFUSION = OUTPUT_DIR / "medical_confusion.txt"
HIGH_RISK_GENERAL = OUTPUT_DIR / "high_risk_general.txt"
AC_AUTOMATON = OUTPUT_DIR / "ac_automaton.pkl"
WORD_ORDER_TEMPLATES = OUTPUT_DIR / "word_order_templates.json"

# ==================== SSD 流式处理配置 ====================
CHUNK_SIZE = 10000  # 每块处理 10000 条报告
MAX_WORKERS = 4  # 并行处理进程数

# ==================== 动态阈值配置 ====================
# 风险判定阈值：通用频次/放射频次 比值
RISK_THRESHOLD = 5.0  # 降低阈值，捕获更多低频专有名词

# 词长度限制
MIN_WORD_LEN = 3  # 至少3个字，减少短词误报
MAX_WORD_LEN = 10

# huqie.txt 频次过滤
MIN_HUQIE_FREQ = 1  # 通用词最小频次（降低以捕获更多专有名词）
MAX_RADIO_FREQ = 10   # 放射语料最大安全频次

# 词性权重（专有名词风险更高）
PRIORITY_POS = {'nr', 'nt', 'ns', 'nz', 'nrt', 'nis'}  # 人名、机构名、地名、专有名词
POS_WEIGHT = 1.0

# 医学词典保护权重
MED_PROTECT_WEIGHT = 0.1

# ==================== 拼音混淆配置 ====================
FUZZY_PINYIN_MAP = {
    # 平翘舌
    'z': ['z', 'zh'], 'zh': ['z', 'zh'],
    'c': ['c', 'ch'], 'ch': ['c', 'ch'],
    's': ['s', 'sh'], 'sh': ['s', 'sh'],
    # 前后鼻音
    'en': ['en', 'eng'], 'eng': ['en', 'eng'],
    'in': ['in', 'ing'], 'ing': ['in', 'ing'],
    'an': ['an', 'ang'], 'ang': ['an', 'ang'],
    # 边音鼻音
    'n': ['n', 'l'], 'l': ['n', 'l'],
    # 其他常见混淆
    'f': ['f', 'h'], 'h': ['f', 'h'],
}

# 拼音编辑距离阈值
MAX_PINYIN_DISTANCE = 2

# ==================== KenLM 配置 ====================
NGRAM_ORDER = 4  # 4-gram 模型

# ==================== 标点符号配置 ====================
# 句子分割标点
SENTENCE_DELIMITERS = r'[。！？；，、：:;,.!?\n\r]+'

# 中文范围
CHINESE_RANGE = (0x4E00, 0x9FFF)


def ensure_dirs():
    """确保输出目录存在"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_config():
    """获取完整配置字典"""
    return {
        'data_dir': DATA_DIR,
        'huqie_path': HUQIE_PATH,
        'medical_dict_paths': MEDICAL_DICT_PATHS,
        'output_dir': OUTPUT_DIR,
        'temp_dir': TEMP_DIR,
        'chunk_size': CHUNK_SIZE,
        'risk_threshold': RISK_THRESHOLD,
        'min_word_len': MIN_WORD_LEN,
        'max_word_len': MAX_WORD_LEN,
        'min_huqie_freq': MIN_HUQIE_FREQ,
        'max_radio_freq': MAX_RADIO_FREQ,
        'pos_weight': POS_WEIGHT,
        'med_protect_weight': MED_PROTECT_WEIGHT,
        'ngram_order': NGRAM_ORDER,
    }
