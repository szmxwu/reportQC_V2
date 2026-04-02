"""
常见中文错别字数据库 - 医学报告专用

数据来源：
1. 医学影像报告常见笔误
2. 中文输入法常见错误
3. 形近字、音近字混淆

使用：
    from grammer.typo_database import MEDICAL_TYPOS, CONFUSION_PAIRS
    
    # 检测文本中的错别字
    errors = detect_typos("肺文里增粗")
"""

from typing import Dict, List, Tuple, Optional
import re

# ==================== 医学影像专用错别字 ====================

# 格式: (错误写法, 正确写法, 错误类型, 置信度)
# 错误类型: typo=错别字, confusion=形近字混淆, pinyin=音近字
MEDICAL_TYPOS: List[Tuple[str, str, str, float]] = [
    # 肺相关
    ('肺文里', '肺纹理', 'typo', 0.95),
    ('肺致', '肺质', 'typo', 0.90),
    ('肺文', '肺纹', 'typo', 0.90),
    ('纹理曾粗', '纹理增粗', 'typo', 0.95),
    ('纹理曾多', '纹理增多', 'typo', 0.95),
    ('纹理曾生', '纹理增生', 'typo', 0.95),
    ('肺也', '肺野', 'confusion', 0.90),
    ('肺也内', '肺野内', 'confusion', 0.90),
    ('肺们', '肺门', 'typo', 0.90),
    ('肺文', '肺纹', 'typo', 0.90),
    ('索条', '条索', 'confusion', 0.85),  # 纤维条索影 vs 纤维索条影
    
    # 密度相关
    ('低密渡', '低密度', 'pinyin', 0.95),
    ('高密渡', '高密度', 'pinyin', 0.95),
    ('密谋度', '密度', 'pinyin', 0.95),
    ('密谋', '密度', 'pinyin', 0.95),
    ('曾强', '增强', 'typo', 0.95),
    ('曾高', '增高', 'typo', 0.95),
    ('曾加', '增加', 'typo', 0.95),
    ('曾大', '增大', 'typo', 0.95),
    ('曾宽', '增宽', 'typo', 0.95),
    ('曾厚', '增厚', 'typo', 0.95),
    ('曾强扫苗', '增强扫描', 'typo', 0.95),
    
    # 形态相关
    ('影象', '影像', 'typo', 0.90),
    ('形象', '影像', 'typo', 0.85),
    ('占伟', '占位', 'pinyin', 0.90),
    ('占们', '占位', 'pinyin', 0.90),
    ('占位行病变', '占位性病变', 'typo', 0.95),
    ('古鸽', '骨胳', 'pinyin', 0.90),
    ('古格', '骨骼', 'pinyin', 0.90),
    
    # 检查相关
    ('扫苗', '扫描', 'pinyin', 0.95),
    ('平扫苗', '平扫描', 'pinyin', 0.95),
    ('曾强扫苗', '增强扫描', 'pinyin', 0.95),
    ('断成', '断层', 'pinyin', 0.90),
    ('成象', '成像', 'typo', 0.90),
    ('重建影象', '重建影像', 'typo', 0.90),
    
    # 方位相关
    ('左册', '左侧', 'typo', 0.90),
    ('右册', '右侧', 'typo', 0.90),
    ('双测', '双侧', 'typo', 0.90),
    ('侧叶', '侧叶', 'none', 0.0),  # 占位符
    ('上夜', '上叶', 'typo', 0.90),
    ('下夜', '下叶', 'typo', 0.90),
    ('夜段', '叶段', 'typo', 0.90),
    
    # 器官相关
    ('干脏', '肝脏', 'typo', 0.95),
    ('工脏', '宫脏', 'pinyin', 0.85),  # 子宫 vs 工脏
    ('脾藏', '脾脏', 'typo', 0.90),
    ('夷腺', '胰腺', 'typo', 0.90),
    ('肾胜', '肾脏', 'confusion', 0.90),
    ('肾下及', '肾下级', 'typo', 0.90),
    ('付件', '附件', 'typo', 0.90),
    ('前列县', '前列腺', 'pinyin', 0.95),
    ('前列线', '前列腺', 'typo', 0.95),
    ('月土', '臀部', 'confusion', 0.90),
    ('月国', '腘窝', 'confusion', 0.85),
    
    # 结构相关
    ('追体', '椎体', 'typo', 0.90),
    ('追间盘', '椎间盘', 'typo', 0.90),
    ('追管', '椎管', 'typo', 0.90),
    ('追弓', '椎弓', 'typo', 0.90),
    ('追间孔', '椎间孔', 'typo', 0.90),
    ('追间隙', '椎间隙', 'typo', 0.90),
    ('追体滑脱', '椎体滑脱', 'typo', 0.90),
    ('追体终板', '椎体终板', 'typo', 0.90),
    
    # 血管相关
    ('动买', '动脉', 'typo', 0.90),
    ('静买', '静脉', 'typo', 0.90),
    ('买状动脉', '冠状动脉', 'typo', 0.90),
    ('雪管', '血管', 'confusion', 0.90),
    ('雪栓', '血栓', 'confusion', 0.90),
    
    # 病变相关
    ('并变', '病变', 'typo', 0.90),
    ('并灶', '病灶', 'typo', 0.90),
    ('冰变', '病变', 'pinyin', 0.90),
    ('种块', '肿块', 'typo', 0.90),
    ('种瘤', '肿瘤', 'typo', 0.90),
    ('良性种瘤', '良性肿瘤', 'typo', 0.95),
    ('恶性种瘤', '恶性肿瘤', 'typo', 0.95),
    ('转一', '转移', 'typo', 0.90),
    ('播散', '播散', 'none', 0.0),
    ('水种', '水肿', 'typo', 0.90),
    ('种胀', '肿胀', 'typo', 0.90),
    ('炎正', '炎症', 'typo', 0.90),
    ('岩症', '炎症', 'pinyin', 0.90),
    
    # 组织相关
    ('纤微化', '纤维化', 'typo', 0.95),
    ('纤维画', '纤维化', 'typo', 0.95),
    ('硬化性', '硬化性', 'none', 0.0),
    ('机话', '机化', 'typo', 0.85),
    
    # 对比剂相关
    ('造影记', '造影剂', 'typo', 0.90),
    ('造影计', '造影剂', 'typo', 0.90),
    ('强化', '强化', 'none', 0.0),
    ('造影', '造影', 'none', 0.0),
    
    # 常见表述错误
    ('末见', '未见', 'confusion', 0.95),
    ('未件', '未见', 'pinyin', 0.90),
    ('末见明显', '未见明显', 'confusion', 0.95),
    ('可建', '可见', 'typo', 0.90),
    ('考绿', '考虑', 'pinyin', 0.90),
    ('排处', '排除', 'pinyin', 0.90),
    ('不处外', '不除外', 'typo', 0.95),
    ('符合', '符合', 'none', 0.0),
    ('建议', '建议', 'none', 0.0),
    ('请结和', '请结合', 'pinyin', 0.90),
    
    # 数量/大小
    ('约', '约', 'none', 0.0),
    ('大小约', '大小约', 'none', 0.0),
    ('经线约', '径线约', 'typo', 0.90),
    ('经线', '径线', 'typo', 0.90),
    ('责', '责任', 'confusion', 0.85),  # 负责 vs 责
]

# ==================== 形近字/音近字混淆对 ====================
# 用于模糊匹配和推荐
CONFUSION_PAIRS: Dict[str, List[str]] = {
    # 形近字
    '末': ['未'],
    '未': ['末'],
    '文': ['纹'],
    '纹': ['文'],
    '渡': ['度'],
    '度': ['渡'],
    '象': ['像', '相'],
    '像': ['象', '相'],
    '曾': ['增', '层'],
    '增': ['曾', '层'],
    '密': ['蜜'],
    '也': ['野'],
    '野': ['也'],
    '他们': ['它们'],
    '追': ['椎'],
    '椎': ['追'],
    '干': ['肝', '竿'],
    '胜': ['脏'],
    '记': ['剂', '计'],
    '岩': ['炎'],
    '买': ['脉'],
    '文': ['纹', '紊'],
    '伟': ['位'],
    '苗': ['描', '瞄'],
    
    # 音近字（拼音相似）
    '绿': ['虑', '率'],
    '处': ['除'],
    '建': ['见'],
    '并': ['病'],
    '追': ['椎'],
    '古': ['股', '骨'],
    '种': ['肿'],
    '正': ['症', '征'],
    '转': ['传', '专'],
    '话': ['化'],
    '排': ['牌'],
    
    # 医学专用混淆
    '条索': ['索条'],  # 纤维条索影 vs 纤维索条影
    '上述': ['以上'],  # 虽然都算对，但医学报告常用"上述"
}

# ==================== 高频短语模式 ====================
# 用于正则匹配
PHRASE_PATTERNS: List[Tuple[str, str, str, float]] = [
    # 模式, 建议修正, 说明, 置信度
    (r'肺文里\w{0,2}曾[粗多生]', '肺纹理增粗/增多', '常见错别字组合', 0.95),
    (r'低密渡影', '低密度影', '音近错误', 0.95),
    (r'高密渡影', '高密度影', '音近错误', 0.95),
    (r'\w密谋度', '密度', '音近错误', 0.95),
    (r'末见\w{0,3}异常', '未见明显异常', '形近错误', 0.90),
    (r'\w也内', '野内', '形近错误', 0.90),
    (r'增强扫苗', '增强扫描', '音近错误', 0.95),
    (r'纤维索条', '纤维条索', '词序错误', 0.85),
]


def detect_typos(text: str, min_confidence: float = 0.85) -> List[Dict]:
    """
    检测文本中的错别字
    
    Args:
        text: 待检测文本
        min_confidence: 最小置信度阈值
    
    Returns:
        检测到的错误列表
    """
    errors = []
    
    # 精确匹配
    for wrong, correct, err_type, conf in MEDICAL_TYPOS:
        if conf < min_confidence:
            continue
        for match in re.finditer(re.escape(wrong), text):
            errors.append({
                'position': match.start(),
                'text': wrong,
                'suggestion': correct,
                'type': err_type,
                'confidence': conf,
                'context': text[max(0, match.start()-5):match.end()+5]
            })
    
    # 短语模式匹配
    for pattern, suggestion, desc, conf in PHRASE_PATTERNS:
        if conf < min_confidence:
            continue
        for match in re.finditer(pattern, text):
            errors.append({
                'position': match.start(),
                'text': match.group(),
                'suggestion': suggestion,
                'type': 'pattern',
                'confidence': conf,
                'context': desc
            })
    
    # 去重：重叠的错误只保留置信度最高的
    errors = sorted(errors, key=lambda x: (x['position'], -x['confidence']))
    filtered = []
    last_end = -1
    for err in errors:
        if err['position'] >= last_end:
            filtered.append(err)
            last_end = err['position'] + len(err['text'])
    
    return filtered


# ==================== 从文件扩展 ====================

def load_custom_typos(filepath: str) -> List[Tuple[str, str, str, float]]:
    """从文件加载自定义错别字库"""
    custom = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    wrong, correct = parts[0], parts[1]
                    err_type = parts[2] if len(parts) > 2 else 'custom'
                    conf = float(parts[3]) if len(parts) > 3 else 0.85
                    custom.append((wrong, correct, err_type, conf))
    except FileNotFoundError:
        pass
    return custom


if __name__ == '__main__':
    # 测试
    test_cases = [
        "双肺文里增粗、增多",
        "见低密渡影，边界尚清",
        "肝内末见明显异常",
        "胸部影象显示正常",
        "考虑肝占位行病变",
    ]
    
    for text in test_cases:
        print(f"\n原文: {text}")
        errors = detect_typos(text)
        if errors:
            for err in errors:
                print(f"  -> {err['text']} 应为 {err['suggestion']} ({err['type']}, conf={err['confidence']})")
        else:
            print("  未检测到错别字")
