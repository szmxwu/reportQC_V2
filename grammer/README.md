# 医学报告语法/错别字检测模块

基于分层架构的语法错误检测系统，专门针对医学影像报告优化。

## 架构

```
文本输入
    ↓
┌───────────────────────────────┐
│  第一层: 快速召回层            │
│  ├── Trigram检测（3字组合）    │ ← 最可靠，优先
│  ├── Bigram检测（2字组合）     │ ← 补充
│  ├── 单字检测                  │ ← 最后
│  └── 模式检测（重复/标点）      │
└────────┬──────────────────────┘
         ↓ 可疑片段列表
┌───────────────────────────────┐
│  第二层: LLM精校层             │
│  ├── 批量验证（每批5个）        │
│  ├── 多线程并发（默认8线程）     │
│  └── 本地缓存                  │
└────────┬──────────────────────┘
         ↓
    确认的语法错误
```

## 核心特点

- **Trigram检测**: 对医学固定搭配更可靠（如"肺纹理"vs"肺文里"）
- **高召回率**: 快速层>95%，宁可误报不可漏报
- **高精度**: LLM层过滤后>85%精确率
- **多线程优化**: 支持多进程训练、多线程LLM验证
- **不依赖人工词典**: 从数据自动学习正常模式

## 安装依赖

```bash
pip install pandas numpy tqdm

# 可选（推荐安装以获得3-5倍加速）
pip install numba psutil
```

## 快速开始

### 1. 训练模型

```bash
# 基础训练（单进程，适合测试）
python grammer/train_optimized.py --max-texts 10000

# 推荐训练（多进程+JIT加速，适合生产）
python grammer/train_optimized.py \
    --workers 5 \
    --io-workers 4 \
    --use-trigram \
    --use-jit

# 完整训练（360万条报告）
python grammer/train_optimized.py \
    --data-dir ~/work/python/Radiology_Entities/radiology_data \
    --output grammer/models \
    --workers 5 \
    --io-workers 4 \
    --chunk-size 100000 \
    --use-trigram \
    --use-jit
```

**训练参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--workers` | 5 | 处理进程数（建议CPU核心数-1） |
| `--io-workers` | 4 | Excel读取线程数 |
| `--chunk-size` | 100000 | 每批处理条数（内存越大可越大） |
| `--use-trigram` | True | 启用trigram检测 |
| `--use-jit` | True | 启用Numba JIT加速 |
| `--max-texts` | None | 最大训练样本数（测试用） |

**训练输出**:
```
grammer/models/
├── char_anomaly.pkl    # 字符统计模型
├── entropy.pkl         # 上下文熵模型
└── stats.json          # 训练统计
```

### 2. 检测文本

```python
from grammer import LayeredGrammarDetector

# 加载模型
detector = LayeredGrammarDetector(
    model_dir='grammer/models',
    llm_workers=8  # LLM并发线程数
)

# 方式1: 仅快速召回（快，可能有误报）
suspicious = detector.fast_detect("双肺文里增粗")
for frag in suspicious:
    print(f"可疑: {frag.text} [{frag.strategy}] 分数:{frag.score:.2f}")

# 方式2: 完整检测（准，含LLM验证）
errors = detector.detect("双肺文里增粗", use_llm=True)
for err in errors:
    print(f"错误: {err.text} -> {err.suggestion}")
```

### 3. 批量异步处理

```python
import asyncio
from grammer import LayeredGrammarDetector

async def batch_check(texts):
    detector = LayeredGrammarDetector(
        model_dir='grammer/models',
        llm_workers=8
    )
    return await detector.detect_batch_async(texts)

texts = ["文本1...", "文本2...", "文本3..."]
results = asyncio.run(batch_check(texts))
```

## 完整API

### LayeredGrammarDetector

```python
detector = LayeredGrammarDetector(
    model_dir='grammer/models',      # 模型目录
    llm_client=None,                  # LLM客户端（可选）
    llm_workers=8,                    # LLM并发线程数
    use_trigram=True                  # 是否使用trigram
)

# 快速召回
suspicious = detector.fast_detect(text: str) -> List[SuspiciousFragment]

# 完整检测（同步）
errors = detector.detect(text: str, use_llm: bool = True) -> List[GrammarError]

# 完整检测（异步）
errors = await detector.detect_async(text: str) -> List[GrammarError]

# 批量检测（同步）
results = detector.detect_batch(texts: List[str], use_llm: bool) -> List[List[GrammarError]]

# 批量检测（异步）
results = await detector.detect_batch_async(texts: List[str]) -> List[List[GrammarError]]

# 获取统计
stats = detector.get_stats()
```

### 检测结果

```python
# SuspiciousFragment（可疑片段）
{
    'text': '肺文里',           # 可疑文本
    'position': (2, 5),         # 在原文中的位置
    'strategy': 'trigram_rarity', # 检测策略
    'score': 0.95,              # 可疑分数(0-1)
    'context': '双肺文里增粗',   # 上下文
    'reason': '罕见三字组合'     # 判定理由
}

# GrammarError（确认错误）
{
    'text': '肺文里',
    'position': (2, 5),
    'suggestion': '肺纹理',      # 建议修正
    'error_type': '错别字',      # 错误类型
    'confidence': 0.95,          # 置信度
    'reason': '匹配常见错别字库',
    'llm_verified': False        # 是否LLM验证
}
```

## 性能优化

### 硬件配置建议

- **CPU**: 多核（训练用）
- **内存**: 16GB+（推荐32GB）
- **存储**: SSD（Excel读取是IO密集型）
- **GPU**: 可选（字符统计任务GPU收益有限）

### 针对6核32G配置的推荐参数

```bash
python grammer/train_optimized.py \
    --workers 5 \           # 6核留1核给系统
    --io-workers 4 \        # 4线程读Excel
    --chunk-size 100000 \   # 32G内存支持大缓冲区
    --use-trigram \         # 启用trigram
    --use-jit               # Numba JIT加速
```

**预期性能**:
- 读取速度: ~50000 条/秒（4线程IO）
- 处理速度: ~30000 条/秒（5进程+JIT）
- 总训练时间: ~6-8分钟（360万条）
- 内存峰值: ~8-12GB

## Trigram vs Bigram

Trigram（3字组合）对医学文本更可靠：

| 文本 | Bigram检测 | Trigram检测 |
|------|-----------|-------------|
| 肺纹理增粗 | "肺纹"常见 | "肺纹理"常见 ✓ |
| 肺文里增粗 | "肺文"可疑 | "肺文里"可疑 ✓ |

**优势**:
- 医学术语多为固定3字搭配
- 能精确定位中间字错误
- 上下文更丰富，假阳性更低

运行对比测试:
```bash
python grammer/test_trigram.py
```

## 基准测试

```bash
# 运行性能测试
python grammer/benchmark.py --full

# 预期输出:
# - 硬件配置检测
# - 内存使用估计
# - 推荐配置
# - 字符提取性能对比
# - N-gram统计性能
```

## 常见问题

### Q: 训练时内存不足？

```bash
# 降低块大小
python grammer/train_optimized.py --chunk-size 30000

# 减少进程数
python grammer/train_optimized.py --workers 3
```

### Q: Numba JIT报错？

```bash
# 禁用JIT回退到纯Python
python grammer/train_optimized.py --no-jit
```

### Q: 如何验证训练成功？

```bash
# 检查输出文件
ls -lh grammer/models/

# 运行测试
python -c "
from grammer import LayeredGrammarDetector
d = LayeredGrammarDetector('grammer/models')
errors = d.detect('肺文里增粗')
print(f'检测成功: {len(errors)}个错误')
"
```

## 文件结构

```
grammer/
├── __init__.py                    # 模块入口
├── README.md                      # 本文档
├── layered_grammar_detector.py    # 主检测器（分层架构）
├── fast_recover.py                # 快速召回层（Trigram）
├── train_optimized.py             # 优化训练脚本
├── typo_database.py               # 常见错别字库（备用）
├── benchmark.py                   # 性能测试工具
├── test_trigram.py                # Trigram对比测试
├── load_data.py                   # 数据加载（用户提供）
└── models/                        # 训练输出（生成）
    ├── char_anomaly.pkl
    ├── entropy.pkl
    └── stats.json
```

## 版本

v3.1.0 - Trigram增强版

- 支持Trigram检测
- 多进程训练优化
- 多线程LLM验证
- Numba JIT加速
