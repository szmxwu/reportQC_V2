# 医学报告语法/错别字检测模块

## 项目结构

```
grammer/
├── README.md                    # 本文件
├── grammar_detector.py          # 主检测器（v2.0）
├── confusion_checker.py         # Confusion Set检测器
├── typo_database.py             # 常见错别字库
├── prepare_training_data.py     # 训练数据预处理
├── train_grammar_model.py       # 原始N-gram训练（备用）
├── load_data.py                 # 数据加载（用户提供）
├── training_plan.md             # 训练计划文档
└── models/                      # 训练输出目录（待生成）
    ├── word_frequency.pkl
    ├── confusion_context.pkl
    ├── confusion_whitelist.pkl
    ├── char_bigram.pkl
    └── training_summary.json
```

## 核心优势：Confusion Set vs N-gram

| 特性 | 纯N-gram | Confusion Set (本项目) |
|------|----------|----------------------|
| 检测范围 | 所有罕见组合 | 仅已知混淆字 |
| 假阳性率 | **高**（医学术语罕见） | **低**（白名单过滤） |
| 理解术语 | 否 | 是（jieba+用户词典） |
| 需要负样本 | 是 | 否 |
| 训练数据利用 | 简单统计 | 构建白名单 |
| 维护成本 | 低 | 中（需维护混淆集） |

## 快速开始

### 1. 快速检测（无需训练）

```python
from grammer.grammar_detector import quick_check

# 检测文本
errors = quick_check("双肺文里增粗")
# -> [{'text': '肺文里', 'suggestion': '肺纹理', 'confidence': 0.95}]
```

### 2. 加载白名单检测（推荐）

```python
from grammer.grammar_detector import GrammarDetector

detector = GrammarDetector(
    whitelist_path='grammer/models/confusion_whitelist.pkl',
    user_dict_path='config/user_dic_expand.txt'
)

errors = detector.detect("肺文里增粗")
for e in errors:
    print(f"{e.text} -> {e.suggestion}")
```

### 3. 训练白名单模型

```bash
# 预处理360万条报告数据（约2小时）
python grammer/prepare_training_data.py \
    --data-dir ~/work/python/Radiology_Entities/radiology_data \
    --user-dict config/user_dic_expand.txt \
    --output grammer/models \
    --chunk-size 50000

# 输出：
# - word_frequency.pkl: 词频表
# - confusion_context.pkl: 混淆字上下文
# - confusion_whitelist.pkl: 白名单（核心）
# - training_summary.json: 统计摘要
```

## 已覆盖的错别字（50+）

### 高频错误
| 错误 | 正确 | 类型 |
|------|------|------|
| 肺文里 | 肺纹理 | 形近字 |
| 低密渡 | 低密度 | 音近字 |
| 末见 | 未见 | 形近字 |
| 曾强 | 增强 | 形近字 |
| 影象 | 影像 | 习惯误用 |

### 完整列表见
`grammer/typo_database.py` 中的 `MEDICAL_TYPOS`

## 训练计划（4步）

### 第1步：数据预处理 ✓（脚本已创建）
```bash
python grammer/prepare_training_data.py
```
- 分批读取4个Excel文件
- 使用jieba+用户词典分词
- 统计混淆字上下文

### 第2步：构建白名单（自动）
- 频率>5的上下文加入白名单
- 输出 `confusion_whitelist.pkl`

### 第3步：集成到检测器 ✓（已完成）
```python
from grammer.confusion_checker import ConfusionChecker

checker = ConfusionChecker(whitelist_path='...')
errors = checker.check(text)
```

### 第4步：API集成
```python
# api_server.py
@app.post("/api/v1/quality/grammar")
async def check_grammar(input: GrammarInput):
    detector = GrammarDetector(whitelist_path='...')
    return {"errors": detector.detect(input.text)}
```

## 为什么N-gram假阳性高？

### 问题示例
```python
# 正常医学术语
"钙化影"  # 在通用语料中罕见，N-gram概率低 -> 误报
"条索影"  # 同上
"磨玻璃"  # 同上

# 真实错误
"肺文里"  # 文里组合罕见 -> 正确检测
```

### 解决方案：Confusion Set
```python
# 只检测已知混淆字
CONFUSION_CHARS = {'文', '渡', '末', '曾', '象', ...}

# "钙化影" -> '钙'、'化'、'影' 都不在混淆集 -> 跳过
# "肺文里" -> '文' 在混淆集 -> 检测
```

## 扩展错别字库

### 方法1：直接修改代码
```python
# grammer/typo_database.py
MEDICAL_TYPOS += [
    ('新错字', '正确字', 'typo', 0.95),
]
```

### 方法2：从训练数据自动发现
```python
# 训练后会生成可疑模式列表
# 人工审核后添加到规则库
```

## 性能预期

| 配置 | 检测耗时 | 召回率 | 假阳性率 |
|------|----------|--------|----------|
| 仅规则 | ~5ms | ~80% | <1% |
| 规则+白名单 | ~20ms | ~90% | <5% |
| 规则+白名单+LLM | ~200ms | ~95% | <1% |

## 下一步行动

1. **立即使用**：`quick_check()` 已可用，覆盖50+常见错别字
2. **训练模型**：运行 `prepare_training_data.py` 生成白名单
3. **扩展规则**：从训练结果中发现新错误模式
4. **API集成**：添加到 `api_server.py`
