# 分层语法检测 - 快速开始（Trigram增强版）

## 核心改进：Trigram检测

Trigram（3字组合）对医学文本更可靠：

| 特性 | Bigram | Trigram |
|------|--------|---------|
| 上下文 | 2字 | 3字 |
| 医学术语匹配 | 部分 | 完整（如"肺纹理"） |
| 区分相似错误 | 困难 | 容易 |
| 假阳性 | 较高 | 较低 |

示例：
```
文本: "肺文里增多"
Bigram检测: "肺文"可疑，"文里"可疑（无法确定位置）
Trigram检测: "肺文里"可疑（精确定位中间字错误）
```

## 架构

```
文本输入
    ↓
┌───────────────────────────────┐
│  快速召回层 (Trigram优先)       │
│  ├── Trigram检测（3字组合）     │ ← 最可靠，优先
│  ├── Bigram检测（2字组合）      │ ← 补充
│  ├── 单字检测                  │ ← 最后
│  └── 模式检测（重复/标点）      │
└────────┬──────────────────────┘
         ↓
    可疑片段列表
         ↓
┌───────────────────────────────┐
│  LLM精校层（多线程）            │
│  ├── 批量验证（每批5个）        │
│  ├── 多线程并发（默认8线程）     │
│  └── 本地缓存                  │
└────────┬──────────────────────┘
         ↓
    确认的语法错误
```

## 3步使用

### 第1步：训练模型（支持多进程+Trigram）

```bash
cd /home/wmx/work/python/reportQC_v2

# 使用多进程训练（推荐，利用你的4060 GPU机器的多核CPU）
python grammer/train_fast_recover.py \
    --data-dir ~/work/python/Radiology_Entities/radiology_data \
    --output grammer/models \
    --workers 8 \
    --use-trigram

# 输出：
# - char_anomaly.pkl: 包含unigram+bigram+trigram
# - entropy.pkl: 上下文熵模型
# - stats.json: 训练统计
```

**Trigram统计示例**：
```json
{
  "unique_chars": 3500,
  "unique_bigrams": 45000,
  "unique_trigrams": 120000,  // 新增
  "total_trigrams": 85000000
}
```

### 第2步：检测文本

```python
from grammer import LayeredGrammarDetector

# 加载模型（启用trigram）
detector = LayeredGrammarDetector(
    model_dir='grammer/models',
    llm_workers=8,      # 8线程LLM验证
    use_trigram=True    # 启用trigram（默认）
)

# 方式A: 仅快速召回（快）
suspicious = detector.fast_detect("双肺文里增粗")
for frag in suspicious:
    print(f"可疑: {frag.text} [{frag.strategy}] 分数:{frag.score:.2f}")
# 输出:
# 可疑: 肺文里 [trigram_rarity] 分数:0.95

# 方式B: 完整检测（准，多线程LLM）
errors = detector.detect("双肺文里增粗", use_llm=True)
for err in errors:
    print(f"错误: {err.text} -> {err.suggestion}")
```

### 第3步：批量异步处理（高效）

```python
import asyncio

# 异步批量检测（推荐用于API服务）
async def batch_check(texts):
    detector = LayeredGrammarDetector(
        model_dir='grammer/models',
        llm_workers=8
    )
    return await detector.detect_batch_async(texts)

texts = ["文本1...", "文本2...", "文本3..."]
results = asyncio.run(batch_check(texts))
```

## 性能优化

### 多进程训练
- `--workers 8`: 使用8个CPU核心并行
- 360万条报告约需10-15分钟
- 比单进程快5-8倍

### 多线程LLM
- `llm_workers=8`: 8个线程并发调用LLM
- 批量验证：每批5个片段
- 本地缓存：避免重复验证

### GPU支持
```bash
# 可选，但字符统计任务CPU更高效
python grammer/train_fast_recover.py --use-gpu
```

## 检测能力

### Trigram能检测的错误

| 错误类型 | 示例 | Trigram捕获 |
|----------|------|-------------|
| 错别字 | 肺文里→肺纹理 | ✅ "肺文里"罕见 |
| 音近字 | 低密渡→低密度 | ✅ "低密渡"罕见 |
| 形近字 | 末见→未见 | ✅ "末见异"罕见 |
| 中间错字 | 肺纹里→肺纹理 | ✅ "肺纹里"罕见 |
| 连续错误 | 肺文里增粗 | ✅ 多个trigram异常 |

### 模式检测
- 重复字符："理理理"
- 标点异常："。。。"
- 中英文混用："肺cT"
- 超长句子：缺少标点

## 完整API示例

```python
from grammer import LayeredGrammarDetector

detector = LayeredGrammarDetector(
    model_dir='grammer/models',
    llm_workers=8,
    use_trigram=True
)

# 获取详细统计
stats = detector.get_stats()
print(f"LLM缓存大小: {stats['llm_cache']['cache_size']}")

# 同步检测
errors = detector.detect(text, use_llm=True)

# 异步检测
errors = await detector.detect_async(text)

# 批量同步
results = detector.detect_batch(texts, use_llm=True)

# 批量异步
results = await detector.detect_batch_async(texts)
```

## 文件结构

```
grammer/
├── __init__.py                    # 模块入口
├── layered_grammar_detector.py    # 主检测器（多线程LLM）
├── fast_recover.py                # 快速召回层（Trigram）
├── train_fast_recover.py          # 训练脚本（多进程）
├── test_trigram.py                # Trigram对比测试
├── typo_database.py               # 规则库（备用）
├── QUICKSTART.md                  # 本文档
└── models/                        # 训练输出
    ├── char_anomaly.pkl           # Trigram+Bigram+Unigram
    ├── entropy.pkl                # 上下文模型
    └── stats.json                 # 统计信息
```

## 对比测试

运行Trigram对比测试：
```bash
python grammer/test_trigram.py
```

输出示例：
```
测试: '双肺文里增粗' (错别字：纹理->文里)
  Bigram模型: 5个可疑
  Trigram模型: 4个trigram可疑
    Trigram检测:
      - '双肺文': 罕见三字组合
      - '肺文里': 罕见三字组合  ← 精确定位
      - '文里增': 罕见三字组合
      - '里增粗': 罕见三字组合
```

## 常见问题

**Q: Trigram会漏掉短错误吗？**
A: 不会。检测器优先级：Trigram → Bigram → Unigram。短错误会被Bigram捕获。

**Q: 训练数据量要求？**
A: Trigram需要更多数据（建议>100万条）。360万条足够。

**Q: 内存占用？**
A: Trigram模型约500MB-1GB（120万唯一trigram）。

**Q: 如何禁用Trigram？**
A: `LayeredGrammarDetector(use_trigram=False)` 或训练时 `--no-trigram`。

**Q: 多线程LLM会触发限流吗？**
A: 可通过`llm_workers`控制并发数。建议8线程，如有限流可降低到4。
