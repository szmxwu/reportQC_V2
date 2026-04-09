# Auto KnowledgeGraph

全自动知识图谱流水线：自动发现 → 自动归一化 → 自动验证 → 自动补丁发布

## 项目目标

在不依赖人工逐条扩充知识图谱的前提下，基于现有规则实体抽取层和本地 LLM，构建一条自动化流水线，用于：
1. 持续发现多院区新增解剖表达
2. 将其归一化到标准知识图谱节点
3. 以补丁方式增量发布

## 核心原则

- **主图谱是锚点**：不直接修改主图谱，通过补丁层增量发布
- **LLM 是增强器**：不替代规则抽取，而是辅助归一化
- **补丁层是发布方式**：所有自动扩图结果先进入补丁层，验证后才参与主流程
- **下游收益是最终验收标准**：通过下游任务回放证明实体层增强带来的收益

## 流水线架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Auto KnowledgeGraph Pipeline                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  阶段 A: 知识底座索引化                                          │
│  ┌─────────────────┐                                             │
│  │ build_graph_    │──→ graph_index.json                         │
│  │ index.py        │──→ graph_alias_index.json                   │
│  └─────────────────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  阶段 B: 新表达自动发现                                          │
│  ┌─────────────────┐                                             │
│  │ discover_       │──→ entity_candidate_discovery.jsonl         │
│  │ candidates.py   │    (n-gram挖掘, 上下文模式, 共现分析)        │
│  └─────────────────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  阶段 C: LLM 候选归一化                                          │
│  ┌─────────────────┐                                             │
│  │ normalize_      │──→ entity_candidate_normalized.jsonl        │
│  │ candidates_     │    (本地LLM候选排序与归一化)                 │
│  │ with_llm.py     │                                             │
│  └─────────────────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  阶段 D: 程序化验证                                              │
│  ┌─────────────────┐                                             │
│  │ validate_       │──→ entity_candidate_validated.jsonl         │
│  │ candidates.py   │    (A/B/C分级: 高/中/低可信)                │
│  └─────────────────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  阶段 E: 补丁发布                                                │
│  ┌─────────────────┐                                             │
│  │ publish_        │──→ config/patches/*.xlsx                    │
│  │ patches.py      │──→ reports/patch_release_report.md          │
│  └─────────────────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  阶段 F: 下游任务回放                                            │
│  ┌─────────────────┐                                             │
│  │ replay_         │──→ reports/downstream_replay_report.md      │
│  │ downstream_     │    (证明实体层增强的业务收益)               │
│  │ tasks.py        │                                             │
│  └─────────────────┘                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
V3/Auto_knowledgegraph/
├── AGENT_PLAN.md                    # 项目计划文档
├── README.md                        # 本文件
├── config.py                        # 配置管理
├── pipeline.py                      # 流水线主控
│
├── build_graph_index.py             # 阶段 A：知识底座索引化
├── discover_candidates.py           # 阶段 B：新表达自动发现
├── normalize_candidates_with_llm.py # 阶段 C：LLM 归一化
├── validate_candidates.py           # 阶段 D：程序化验证
├── publish_patches.py               # 阶段 E：补丁发布
├── replay_downstream_tasks.py       # 阶段 F：下游回放
│
├── prompts/
│   └── normalize_entity.prompt.md   # LLM 提示模板
│
├── schemas/
│   ├── entity_candidate.schema.json       # 候选实体 Schema
│   ├── normalized_candidate.schema.json   # 归一化结果 Schema
│   └── validation_result.schema.json      # 验证结果 Schema
│
├── data/                            # 数据目录（运行时生成）
│   ├── graph_index.json
│   ├── graph_alias_index.json
│   ├── entity_candidate_discovery.jsonl
│   ├── entity_candidate_normalized.jsonl
│   └── entity_candidate_validated.jsonl
│
└── reports/                         # 报告目录（运行时生成）
    ├── patch_release_report.md
    └── downstream_replay_report.md
```

## 快速开始

### 1. 安装依赖

```bash
# 确保已安装项目依赖
pip install pandas requests tqdm
```

### 2. 运行完整流水线

```bash
cd V3/Auto_knowledgegraph
python pipeline.py
```

### 3. 运行单个阶段

```bash
# 仅运行阶段 A
python pipeline.py --stage A

# 从阶段 B 运行到阶段 D
python pipeline.py --start B --end D

# 跳过候选发现（使用现有数据）
python pipeline.py --skip-discovery
```

### 4. 独立运行各模块

```bash
# 阶段 A: 构建图谱索引
python build_graph_index.py

# 阶段 B: 候选发现
python discover_candidates.py --input data/source_reports.jsonl

# 阶段 C: LLM 归一化
python normalize_candidates_with_llm.py

# 阶段 D: 验证
python validate_candidates.py

# 阶段 E: 补丁发布
python publish_patches.py

# 阶段 F: 下游回放
python replay_downstream_tasks.py
```

## 配置说明

流水线从 `.env` 文件读取配置：

```env
# LLM 配置（必需）
LLM_BASE_URL=http://192.0.0.193:9997/v1
LLM_MODEL=qwen3
LLM_API_KEY=
LLM_TIMEOUT=30
LLM_MAX_TOKENS=2048
LLM_BATCH_SIZE=5
LLM_CONFIDENCE_THRESHOLD=0.7
```

## 数据格式

### 输入：source_reports.jsonl

```jsonl
{"text": "胸部CT平扫：右肺上叶见结节影...", "hospital_id": "H001", "StudyPart": "胸部", "modality": "CT"}
{"text": "腹部MR：肝脏形态正常...", "hospital_id": "H002", "StudyPart": "腹部", "modality": "MR"}
```

### 输出：entity_candidate_discovery.jsonl

```jsonl
{
  "hospital_id": "H001",
  "report_id": "R12345",
  "sentence": "右肺上叶见结节影",
  "candidate_text": "右上叶",
  "StudyPart": "胸部",
  "modality": "CT",
  "matched_by_rule": false,
  "candidate_type": "unhit",
  "frequency_local": 5
}
```

## 验证分级标准

| 级别 | 标准 | 处理方式 |
|------|------|----------|
| A | 高可信 | 自动纳入补丁层 |
| B | 中可信 | 进入候选池，等待更多样本 |
| C | 低可信 | 仅保留样本，不发布 |

## 补丁文件格式

### config/patches/knowledgegraph_alias_patch.xlsx

| alias_text | target_node_id | target_path | validation_level | confidence |
|-----------|----------------|-------------|------------------|------------|
| 左上叶 | report:胸部/肺/肺上叶 | ["胸部","肺","肺上叶"] | A | 0.95 |

### config/patches/knowledgegraph_node_candidates.xlsx

| proposed_canonical_name | context_sentence | StudyPart | status | created_at |
|------------------------|------------------|-----------|--------|------------|
| 背段静脉属支 | 右肺下叶背段静脉属支显示 | 胸部 | pending_review | 2026-04-02 |

## 与主系统的关系

```
┌─────────────────────────────────────────┐
│           reportQC V3 系统               │
├─────────────────────────────────────────┤
│                                          │
│  ┌─────────────────────────────────┐    │
│  │   V3/Auto_knowledgegraph/       │    │
│  │   (全自动知识图谱流水线)         │    │
│  │                                 │    │
│  │   输出: config/patches/*.xlsx   │────┼──→ 被主系统加载
│  └─────────────────────────────────┘    │
│           │                              │
│           ▼                              │
│  ┌─────────────────────────────────┐    │
│  │   Extract_Entities.py           │    │
│  │   (规则实体抽取主入口)           │    │
│  │                                 │    │
│  │   加载: config/knowledgegraph.* │    │
│  │   加载: config/patches/*.xlsx   │    │
│  └─────────────────────────────────┘    │
│           │                              │
│           ▼                              │
│  ┌─────────────────────────────────┐    │
│  │   NLP_analyze.py / api_server   │    │
│  │   (质控任务主入口)               │    │
│  └─────────────────────────────────┘    │
│                                          │
└─────────────────────────────────────────┘
```

## 后续开发计划

1. **增强候选发现**：接入真实报告数据，优化 n-gram 和上下文模式
2. **优化 LLM 归一化**：基于真实反馈优化提示词和检索策略
3. **完善验证规则**：增加更多业务约束和跨院区一致性检查
4. **实现真实下游回放**：接入 NLP_analyze.py 进行真实任务收益评估
5. **可视化界面**：开发 Web 界面用于人工审核和干预

## 注意事项

1. 所有自动扩图结果必须先进入补丁层，不能直接修改主图谱
2. LLM 结果必须经过程序化验证后才能发布
3. 定期清理低质量的 C 级候选，避免数据膨胀
4. 建议在夜间低峰期运行完整流水线
