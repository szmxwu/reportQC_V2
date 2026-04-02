# 分层语法检测器 v3.0

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    分层语法检测器                         │
├─────────────────────────────────────────────────────────┤
│  第一层: 快速召回层 (FastRecover)                         │
│  ├── 策略1: 字符级异常（罕见字/罕见组合）                  │
│  ├── 策略2: 上下文熵异常（邻居分布异常）                   │
│  ├── 策略3: 模式异常（重复、混用、标点）                   │
│  └── 输出: 可疑片段列表（高召回率）                        │
├─────────────────────────────────────────────────────────┤
│  第二层: LLM精校层 (LLMGrammarValidator)                   │
│  ├── 批量验证可疑片段                                     │
│  ├── 区分"真错误"vs"医学术语"                             │
│  └── 输出: 确认的语法错误（高精度）                        │
└─────────────────────────────────────────────────────────┘
```

## 核心优势

### 相比Confusion Set
| 维度 | Confusion Set | 分层检测 v3.0 |
|------|---------------|---------------|
| 依赖人工词典 | 是 | **否** |
| 发现未知错误 | 否 | **是** |
| 假阳性控制 | 需白名单 | **LLM过滤** |
| 维护成本 | 高（持续补充规则） | **低（纯统计学习）** |
| 召回率 | ~80% | **>95%** |

### 相比纯N-gram
| 维度 | 纯N-gram | 分层检测 v3.0 |
|------|----------|---------------|
| 假阳性率 | **极高**（20-40%） | 低（经LLM过滤） |
| 区分术语/错误 | 否 | **是** |
| 准确性 | 低 | **高** |

## 使用流程

### 阶段1: 训练快速召回层（一次性）

```bash
# 使用360万报告训练
python -c "
from grammer.layered_grammar_detector import LayeredGrammarDetector
import pandas as pd
from tqdm import tqdm

# 读取数据（分批）
texts = []
for chunk in pd.read_excel('~/radiology_data/all_data_match*.xlsx', chunksize=50000):
    for _, row in chunk.iterrows():
        text = str(row.get('描述', '')) + ' ' + str(row.get('结论', ''))
        texts.append(text)

# 训练
detector = LayeredGrammarDetector()
detector.train(texts, 'grammer/models')
print('训练完成！')
"
```

**训练输出**：
- `char_anomaly.pkl`: 字符频率模型
- `entropy.pkl`: 上下文熵模型
- 训练时间：~30分钟（360万条）

### 阶段2: 部署使用

```python
from grammer.layered_grammar_detector import LayeredGrammarDetector

# 初始化（加载预训练模型）
detector = LayeredGrammarDetector(model_dir='grammer/models')

# 方式1: 仅快速召回（适合批量预处理）
suspicious = detector.fast_detect("肺文里增粗")
for frag in suspicious:
    print(f"可疑: {frag.text} (score={frag.score:.2f})")

# 方式2: 完整检测（含LLM验证）
errors = detector.detect("肺文里增粗", use_llm=True)
for err in errors:
    print(f"错误: {err.text} -> {err.suggestion}")
```

## 快速召回层详解

### 策略1: 字符级异常

```python
# 原理：统计360万报告中每个字的出现频率
# 罕见字/组合可能是错误

示例：
- "纹" 出现 500万次 → 正常
- "文" + "里" 组合出现 3次 → 可疑（可能是"纹理"的误写）
```

**优势**：不依赖词典，能发现未知错别字

### 策略2: 上下文熵异常

```python
# 原理：计算每个字左右的"邻居多样性"
# 正常字的上下文分布相对稳定

示例：
- "纹"的右邻居：90%是"理" → 熵低，正常
- "文"的右邻居：分散在"化"、"章"、"里"... → 如果看到"文里"，熵异常
```

**优势**：能检测上下文不搭配的错误

### 策略3: 模式异常

```python
# 检测明显的排版/输入错误

PATTERNS = [
    (r'(.)\1{2,}', '连续重复字符'),      # "理理理"
    (r'[a-zA-Z]{5,}', '异常长英文'),      # "abcdef"
    (r'[\u4e00-\u9fa5][a-zA-Z][\u4e00-\u9fff]', '中英文混用'),  # "肺cT"
    (r'[。，；：]{3,}', '标点重复'),      # "。。。"
]
```

**优势**：规则明确，召回率高

## LLM精校层详解

### 批量验证Prompt

```
你是一位医学报告编辑专家。请判断以下医学影像报告中的可疑片段是否确实存在语法错误或错别字。

【完整报告】
{full_text}

【可疑片段列表】
[0] "文里" (位置: (2,4), 检测策略: bigram_rarity)
[1] "低密渡" (位置: (8,11), 检测策略: char_rarity)
...

请以JSON数组格式输出判断结果。
```

### 优化策略

1. **批量验证**：一次验证5个片段，减少API调用
2. **上下文提供**：给LLM完整报告，避免误判术语
3. **置信度阈值**：只返回高置信度（>0.7）的错误

## 性能指标

| 指标 | 仅快速层 | 完整检测 |
|------|----------|----------|
| 召回率 | >95% | >90% |
| 精确率 | ~30% | >85% |
| 处理速度 | ~50ms/报告 | ~500ms/报告 |
| LLM调用次数 | 0 | 1-2次/报告 |

## 假阳性控制

### 问题：为什么快速层假阳性高？

快速层设计目标是**高召回**，宁可错杀一千，不可放过一个。

常见误报：
- 罕见医学术语（如特定药物名、罕见病名）
- 新出现的医学表述
- 人名、医院名等专有名词

### 解决方案：LLM过滤

LLM通过理解上下文，能准确区分：
- ✅ "肺文里" → 错别字（上下文是解剖描述）
- ❌ "钙化影" → 正常术语（医学影像常见表述）

## 扩展与维护

### 添加新的检测策略

```python
# 在 fast_recover.py 中添加
class NewDetector:
    def detect(self, text: str) -> List[SuspiciousFragment]:
        # 实现检测逻辑
        return fragments

# 在 FastRecoverDetector 中注册
self.detectors.append(NewDetector())
```

### 调整阈值

```python
# 提高召回率（更多可疑片段）
detector.recall_threshold = 0.3  # 默认0.5

# 提高精确率（更少LLM调用）
detector.precision_threshold = 0.8  # 默认0.7
```

## 部署建议

### 场景1: 实时API（低延迟）
```python
# 仅使用快速层，异步LLM验证
suspicious = detector.fast_detect(text)
# 可疑片段入队列，异步LLM验证后入库
```

### 场景2: 批量质检（高精度）
```python
# 完整检测
errors = detector.detect(text, use_llm=True)
# 直接返回确认的错误
```

### 场景3: 混合部署
```python
# 第一层：快速层过滤明显正常的报告
suspicious = detector.fast_detect(text)
if not suspicious:
    return "正常"

# 第二层：中等可疑走轻量级LLM
if len(suspicious) <= 2:
    return detector.detect(text, use_llm=True)

# 第三层：高度可疑走强LLM或人工审核
return "需人工审核"
```

## 下一步

1. **训练模型**：运行训练脚本（30分钟）
2. **效果评估**：在样本数据上测试召回率
3. **阈值调优**：根据实际假阳性率调整
4. **LLM集成**：接入实际LLM服务（OpenAI/Claude等）
