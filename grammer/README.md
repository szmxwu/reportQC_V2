# 医学报告错别字检测引擎 v3.0

基于统计语料和拼音混淆的医学影像报告错别字自动检测系统。

## 项目结构

```
grammer/
├── train/                      # 训练脚本
│   ├── phase1_train_kenlm.py      # Phase 1: KenLM训练 + 词频统计
│   ├── phase2_mine_blacklist.py   # Phase 2: 拼音混淆/高危词挖掘
│   ├── phase3_build_engine_v2.py  # Phase 3: AC自动机引擎构建
│   ├── extract_word_order_templates.py  # 词序模板提取
│   ├── build_substring_vocab.py     # 子串频次表构建
│   └── ssd_processor.py            # SSD流式处理器
├── inference/                  # 推理脚本
│   ├── detect_real_data_final.py   # 真实数据检测（主入口）
│   ├── medical_typo_detector.py    # 统一检测接口
│   └── word_order_detector.py      # 词序错误检测器
├── utils/                      # 工具模块
│   ├── config.py                   # 配置管理
│   └── utils.py                    # 工具函数
├── models/                     # 模型文件（生成）
│   ├── radiology_vocab.json        # 放射语料词频
│   ├── radiology_ngram.klm         # KenLM模型
│   ├── medical_confusion.txt       # 拼音混淆对
│   ├── high_risk_general.txt       # 高危通用词
│   ├── word_order_templates.json   # 词序错误模板
│   └── ac_automaton_v2.pkl         # AC自动机引擎
├── output/                     # 检测结果输出
├── huqie.txt                   # 通用词表（中文分词词典）
├── requirements.txt            # 依赖列表
└── README.md                   # 本文档
```

## 核心算法

### 1. 拼音混淆检测（策略A）

**问题**：医学报告中常见的拼音输入错误，如 "纹理" 误写为 "文里"。

**算法**：多层次拼音混淆挖掘

#### Level 1: 单字同音替换
基于同音字字典，对正确词的每个字符进行替换：
```python
"纹理" (wen li)
  ↓ 替换"纹"→"文"
"文理" - 候选
  ↓ 替换"理"→"里"  
"文里" - 有效混淆对
```

#### Level 2: 双字组合替换
处理需要同时替换两个字的复杂情况：
```python
"胆囊" (dan nang)
  ↓ 同时替换"胆"→"但", "囊"→"?
"但囊" - 有效混淆对
```

**关键修复**：
- 原`[:3]`限制导致常用同音字被遗漏 → 扩大到`[:20]`
- `max_combinations=50`导致提前返回 → 增加到200

#### Level 3: 医学术语锚点扩展
对医学词典中的高频术语进行深度扩展：
- 扫描医学术语（如"椎间盘"、"肺气肿"）
- 生成单字+双字混淆变体
- 基于分词词频表过滤（错误词频次<50）

**过滤条件**：
1. 错误词频次 < 50（分词后）
2. 正确词频次 > 100
3. 拼音编辑距离 <= 2
4. **不包含数字/英文**（纯中文）
5. 不在医学保护词典中

### 2. 高危通用词检测（策略B）

**问题**：报告混入通用专有名词（地名、机构名、人名），如 "阿里巴巴"。

**算法**：频反差异常检测

```
风险分 = log(通用频次) / log(放射频次 + 2) × 词性权重 × 医学保护权重

判定条件：
- 通用词频次 > 1
- 放射语料频次 < 10
- 风险分 > 2.0
```

**示例**：
| 词语 | 通用频次 | 放射频次 | 风险分 | 判定 |
|------|----------|----------|--------|------|
| 阿里巴巴 | 高 | 极低 | 9.6 | ✅ 高危 |
| 马化腾 | 高 | 极低 | 8.2 | ✅ 高危 |
| 纹理 | 中 | 高 | 0.3 | ❌ 正常 |

### 3. 词序错误检测（策略C）

**问题**：词语顺序错误，如 "异常未见" 应为 "未见异常"。

**算法**：高频bigram统计 + 保守策略

```python
提取条件（同时满足）：
1. 正序搭配频次 > 100（确保是高频固定搭配）
2. 反序搭配频次 < 5（确保反向几乎不出现）
3. 正序/反序比值 > 100倍（显著差异）
```

**典型检测**：
| 错误 | 正确 | 正序频次 | 反序频次 | 差异倍数 |
|------|------|----------|----------|----------|
| 异常未见 | 未见异常 | 414万 | 0 | ∞ |
| 扫描增强 | 增强扫描 | 46万 | 0 | ∞ |
| 增生骨质 | 骨质增生 | 51万 | 0 | ∞ |

### 4. 子串误报问题解决

**原始问题**：AC自动机做子串匹配导致误报
```
原文: "左肺上叶前段胸膜下见一微小结节"
误报: "见一" -> "建议" （匹配了"下见一"中的子串）
```

**解决方案**：分词后匹配

```python
def scan(text):
    words = jieba.cut(text)  # 先分词
    for word in words:
        if word in confusion_pairs:  # 只在分词边界匹配
            report_error()
        # 连续单字组合检测（如"扫"+"瞄"="扫瞄"）
        if is_consecutive_single_chars(words):
            combined = combine_chars()
            if combined in confusion_pairs:
                report_error()
```

**效果**：
- ✅ "见一" 不再误报（jieba分词为["下见", "一"]）
- ✅ "扫瞄" 正确检测（["扫", "瞄"]组合）

## 训练流程

### Phase 1: KenLM训练 + 词频统计

```bash
cd train
python3 phase1_train_kenlm.py --data /path/to/radiology_data.xlsx
```

**输出**：
- `models/radiology_corpus.txt` - 分词语料
- `models/radiology_vocab.json` - 词频统计
- `models/radiology_ngram.klm` - KenLM模型

**技术**：SSD流式处理（时间换空间），支持亿级语料

### Phase 2: 黑名单挖掘

```bash
python3 phase2_mine_blacklist.py
```

**输出**：
- `models/medical_confusion.txt` - 拼音混淆对（约90万对）
- `models/high_risk_general.txt` - 高危通用词（约27万个）

**算法细节**：
- 使用pycorrector的same_pinyin.txt作为同音字字典
- 拼音编辑距离计算（动态规划）
- 双重过滤：分词频次 + 数字/英文过滤

### Phase 3: AC自动机引擎构建

```bash
python3 phase3_build_engine_v2.py
```

**输出**：
- `models/ac_automaton_v2.pkl` - 序列化引擎

**技术**：pyahocorasick实现O(N)多模式匹配

### 词序模板提取（可选）

```bash
python3 extract_word_order_templates.py
```

**输出**：
- `models/word_order_templates.json` - 374个高频词序模式

## 推理使用

### 快速检测

```bash
cd inference
python3 detect_real_data_final.py --sample --limit 100
```

**参数**：
- `--sample` - 使用示例数据
- `--limit N` - 只检测前N条
- `-c 描述 结论` - 指定检测列

**输出**：`output/detect_results_*.json`

```json
{
  "report_id": "CT1808030005",
  "column": "描述",
  "text": "双肺文里增粗...",
  "errors": [
    {
      "error": "文里",
      "suggestion": "纹理",
      "type": "typo",
      "context": "双肺文里增粗"
    }
  ]
}
```

### Python API

```python
from inference.medical_typo_detector import MedicalTypoDetector

detector = MedicalTypoDetector()
detector.load()

# 检测文本
errors = detector.detect("双肺文里增粗，异常未见")
for err in errors:
    print(f"[{err['type']}] {err['error']} -> {err['suggestion']}")
    print(f"  上下文: {err['context']}")

# 输出：
# [typo] 文里 -> 纹理
#   上下文: 双肺文里增粗
# [word_order] 异常未见 -> 未见异常
#   上下文: 异常未见
```

## 性能指标

| 指标 | 数值 |
|------|------|
| 拼音混淆对 | 908,499对 |
| 高危通用词 | 271,591个 |
| 词序检测模式 | 374个 |
| **总检测能力** | **1,180,464** |
| 检测速度 | ~130条/秒 |
| 假阳性率 | <5%（分词后匹配） |

## 关键参数配置

编辑 `utils/config.py`：

```python
# 策略A：拼音混淆
MIN_RADIO_FREQ = 100          # 正确词最小频次
MAX_VARIANT_FREQ = 50         # 错误词最大频次
MAX_PINYIN_DISTANCE = 2       # 最大拼音编辑距离

# 策略B：高危通用词
RISK_THRESHOLD = 2.0          # 风险判定阈值
MAX_RADIO_FREQ = 10           # 放射语料最大安全频次

# 策略C：词序错误
MIN_FREQ = 100                # 正确搭配最小频次
MAX_WRONG_FREQ = 5            # 错误搭配最大频次
```

## 依赖安装

```bash
pip install -r requirements.txt
```

**核心依赖**：
- pypinyin - 拼音转换
- pyahocorasick - AC自动机
- pycorrector - 同音字/形似字字典
- jieba - 中文分词
- pandas - Excel数据处理
- kenlm - N-gram语言模型（可选）

## 算法优势

1. **低假阳性**：分词后匹配避免子串误报
2. **高召回率**：90万+混淆对覆盖常见拼音错误
3. **纯中文**：过滤数字/英文，专注语法错误
4. **可解释**：基于真实语料统计，非黑盒
5. **可增量**：支持随时添加新的混淆对

## 版本历史

- **v3.0** (当前)：项目重构，分离训练/推理，添加分词后匹配
- **v2.1**：新增词序错误检测，374个高频搭配
- **v2.0**：双策略检测（拼音混淆 + 高危通用词）
- **v1.0**：基础AC自动机实现

## 许可

MIT License
