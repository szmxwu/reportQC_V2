# Auto KnowledgeGraph Agent Plan

## 1. 文件定位
- 本文件是为新的 agent 准备的执行计划文件。
- 假设新的 agent 不知道本项目历史上下文，只能通过阅读当前仓库文件来理解任务。
- 目标不是写一份概念说明，而是让 agent 能按本文档直接启动一个“全自动知识图谱自动发现 -> 自动归一化 -> 自动验证 -> 自动补丁发布”项目。
- 从现在开始，V3 的子计划遵循“一个主题一个独立文件”的原则；本文件就是 Auto KnowledgeGraph 方向的独立子计划文件。

## 2. 任务目标
- 在不依赖人工逐条扩充知识图谱的前提下，基于现有规则实体抽取层和本地 LLM，构建一条自动化流水线，用于持续发现多院区新增解剖表达、将其归一化到标准知识图谱节点，并以补丁方式增量发布。
- 最终目标是降低新院区实体适配成本，为 V3 的多院区泛化提供稳定的结构化实体层。

## 3. 绝对约束
- 不能把底层实体抽取直接改造成纯 LLM 自由生成系统。
- 不能丢弃现有的 partlist 结构，因为它是后续质控任务的统一锚点。
- 不能以人工逐条补知识图谱为主路径。
- 不能直接修改主图谱文件 [config/knowledgegraph.xlsx](config/knowledgegraph.xlsx) 和 [config/knowledgegraph_title.xlsx](config/knowledgegraph_title.xlsx) 作为默认发布方式。
- 所有自动扩图结果必须先进入补丁层，再经过验证后才能参与主流程。

## 4. 新 agent 必须首先阅读的文件

### 4.1 项目说明与总体背景
- [README.md](README.md)
- [V3/V3_PLAN.md](V3/V3_PLAN.md)
- [V3/Plan_Extract_Entities.md](V3/Plan_Extract_Entities.md)

### 4.2 实体抽取主流程代码
- [Extract_Entities.py](Extract_Entities.py)

重点关注以下函数：
- `build_unified_processor`：见 [Extract_Entities.py](Extract_Entities.py#L162)
- `Extract_Keywords`：见 [Extract_Entities.py](Extract_Entities.py#L470)
- `resplit_sentence_by_entities`：见 [Extract_Entities.py](Extract_Entities.py#L497)
- `text_extrac_process`：见 [Extract_Entities.py](Extract_Entities.py#L663)

### 4.3 知识图谱与规则资源
- [config/knowledgegraph.xlsx](config/knowledgegraph.xlsx)：报告内容知识图谱
- [config/knowledgegraph_title.xlsx](config/knowledgegraph_title.xlsx)：检查部位知识图谱
- [config/replace.xlsx](config/replace.xlsx)
- [config/replace_title.xlsx](config/replace_title.xlsx)
- [config/user_dic_expand.txt](config/user_dic_expand.txt)

### 4.4 LLM 与配置来源
- [.env](.env)：本地 LLM 地址和模型名称来自这里

当前关键参数位于：
- `LLM_BASE_URL`：见 [.env](.env)
- `LLM_MODEL`：见 [.env](.env)
- `LLM_TIMEOUT`：见 [.env](.env)
- `LLM_MAX_TOKENS`：见 [.env](.env)
- `LLM_BATCH_SIZE`：见 [.env](.env)
- `LLM_CONFIDENCE_THRESHOLD`：见 [.env](.env)

当前配置示例：
- `LLM_BASE_URL=http://192.0.0.193:9997/v1`
- `LLM_MODEL=qwen3`

### 4.5 现有 LLM 服务与验证代码
- [llm_service.py](llm_service.py)
- [report_analyze/llm_validator.py](report_analyze/llm_validator.py)

### 4.6 当前质控主流程入口
- [NLP_analyze.py](NLP_analyze.py)
- [api_server.py](api_server.py)

这些文件用于理解：实体层增强后，应如何验证对下游任务的收益。

## 5. 现有系统的关键事实

### 5.1 当前实体层是整个项目的基础层
- 当前项目的大部分质控任务，本质上都依赖结构化实体层。
- 现有规则实体层最重要的产物是 `partlist`。
- `partlist` 代表实体被锚定到知识图谱后的父子路径，是结论一致性、方位判断、部位缺失、申请单对照等任务的重要输入。

### 5.2 当前系统的优势
- 规则实体层能稳定输出标准化结果。
- 当前知识图谱已经沉淀了大量医疗实体和层级关系。
- 当前系统已经能完成：
  - 同义词归并
  - 父子路径表示
  - 方位推断
  - 歧义消解
  - 部位合并

### 5.3 当前系统的主要问题
- 新院区接入时，需要不断人工维护同义词、缩写和新增表达。
- 专科医院和综合医院在解剖粒度上差异较大。
- 不同院区对同一医学事实会采用差异极大的表达方式。
- 规则层可以锚定稳定，但难以低成本吸收多院区新增表达。

## 6. Auto KnowledgeGraph 项目的目标产物

### 6.1 总体产物
- 一个离线自动化流水线项目，能周期性运行并输出：
  1. 新实体表达候选
  2. 标准节点归一化候选
  3. 自动验证结果
  4. 知识图谱补丁候选
  5. 下游任务收益评估报告

### 6.2 推荐项目目录
建议在 [V3/Auto_knowledgegraph](V3/Auto_knowledgegraph) 下建立如下结构：

```text
V3/Auto_knowledgegraph/
├── AGENT_PLAN.md
├── README.md
├── config.py
├── build_graph_index.py
├── discover_candidates.py
├── normalize_candidates_with_llm.py
├── validate_candidates.py
├── publish_patches.py
├── replay_downstream_tasks.py
├── prompts/
│   └── normalize_entity.prompt.md
├── schemas/
│   ├── entity_candidate.schema.json
│   ├── normalized_candidate.schema.json
│   └── validation_result.schema.json
├── data/
│   ├── entity_candidate_discovery.jsonl
│   ├── entity_candidate_normalized.jsonl
│   ├── entity_candidate_validated.jsonl
│   └── entity_hard_cases.jsonl
└── reports/
    ├── patch_release_report.md
    └── downstream_replay_report.md
```

## 7. 要实现的流水线阶段

### 阶段 A：知识底座索引化

#### 目标
- 将 [config/knowledgegraph.xlsx](config/knowledgegraph.xlsx) 和 [config/knowledgegraph_title.xlsx](config/knowledgegraph_title.xlsx) 转为适合程序检索和 LLM 提示的结构化索引。

#### 任务
1. 读取两个 Excel 图谱。
2. 为每个节点生成：
   - 标准节点 ID
   - 层级路径
   - 主名称
   - 同义词列表
   - 坐标范围
   - 分类信息
3. 建立倒排索引：
   - 同义词 -> 标准节点
   - 主名称 -> 标准节点
   - 层级路径 -> 标准节点

#### 建议输出
- `graph_index.json`
- `graph_alias_index.json`

### 阶段 B：新表达自动发现

#### 目标
- 从多院区真实报告中发现规则未覆盖或覆盖不稳定的实体表达。

#### 任务输入
- 多院区历史报告数据
- 现有实体抽取结果（来自 [Extract_Entities.py](Extract_Entities.py#L663) 的 `text_extrac_process`）
- StudyPart、modality、上下文句子

#### 发现对象
1. 完全未命中的候选表达
2. 命中错误标准节点的表达
3. 粗粒度命中但应映射到更细粒度节点的表达
4. 高频别名、缩写、风格化表达
5. 隐式解剖表达

#### 推荐方法
- 高频 n-gram 挖掘
- 上下文模板挖掘
- 与已命中实体共现分析
- 下游任务失败样本反推

#### 输出文件
- `data/entity_candidate_discovery.jsonl`

#### 建议字段
- `hospital_id`
- `report_id`
- `sentence`
- `candidate_text`
- `context_before`
- `context_after`
- `StudyPart`
- `modality`
- `matched_by_rule`
- `current_partlist`
- `candidate_type`
- `frequency_local`
- `frequency_global`

### 阶段 C：LLM 候选归一化

#### 目标
- 使用本地 LLM 对候选表达做标准节点候选排序与归一化。

#### 模型来源
- 本地 LLM 参数来自 [.env](.env)
- 必须从 [.env](.env) 中读取：
  - `LLM_BASE_URL`
  - `LLM_MODEL`
  - `LLM_TIMEOUT`
  - `LLM_MAX_TOKENS`
  - `LLM_BATCH_SIZE`

#### 关键原则
- 不允许 LLM 直接自由输出最终知识图谱节点并立即落库。
- LLM 的默认职责是：
  1. 给出现有节点候选排序
  2. 判断是否可能只是现有节点别名
  3. 仅在现有候选都不合理时，输出“疑似新节点”

#### 推荐提示输入
- 候选短语
- 原始句子
- 前后上下文
- StudyPart
- modality
- 当前句中其他已抽出实体
- 从图谱索引中召回的候选节点列表
- 每个候选节点的路径和同义词

#### 推荐提示输出
- JSON 结构化输出，字段至少包括：
  - `candidate_text`
  - `top_candidates`
  - `candidate_node_id`
  - `candidate_node_path`
  - `score`
  - `reason`
  - `is_alias_of_existing_node`
  - `is_possible_new_node`

#### 输出文件
- `data/entity_candidate_normalized.jsonl`

### 阶段 D：程序化验证

#### 目标
- 对 LLM 归一化结果做自动验证，只保留高可信补丁候选。

#### 验证逻辑
1. 层级一致性验证
- 候选节点是否与 StudyPart、modality、句中其他实体相容。

2. 图谱约束验证
- 候选节点是否存在于主图谱或候选补丁层。
- 若是新节点候选，是否有足够父节点支撑。

3. 频次与稳定性验证
- 同一表达在单院区和跨院区是否稳定指向相同节点。

4. 语境一致性验证
- 同一表达在相似上下文中的归一化结果是否一致。

5. 下游收益验证
- 将增强后的实体层重新送入主质控链，观察 conclusion_missing、orient_error、apply_orient 等任务是否减少异常。

#### 分级标准
- A 级：高可信，可自动纳入补丁层
- B 级：中可信，进入候选池，等待更多样本
- C 级：低可信，仅保留样本，不发布

#### 输出文件
- `data/entity_candidate_validated.jsonl`

### 阶段 E：补丁发布

#### 目标
- 将验证通过的候选以“补丁层”方式发布，而不是直接修改主图谱。

#### 发布原则
- 不直接编辑 [config/knowledgegraph.xlsx](config/knowledgegraph.xlsx)
- 不直接编辑 [config/knowledgegraph_title.xlsx](config/knowledgegraph_title.xlsx)
- 先写入独立补丁文件，供下一版抽取器加载

#### 建议补丁文件
- `config/knowledgegraph_alias_patch.xlsx`
- `config/knowledgegraph_node_candidates.xlsx`
- `config/title_alias_patch.xlsx`
- `config/entity_normalization_cache.json`

#### 发布后动作
- 重新构建统一抽取器
- 重放一批多院区样本
- 输出补丁收益报告

### 阶段 F：下游任务回放验证

#### 目标
- 证明实体层增强对业务任务有真实收益，而不是只是在图谱层自嗨。

#### 回放入口
- 当前主质控入口：[NLP_analyze.py](NLP_analyze.py)
- 当前 API 入口：[api_server.py](api_server.py)

#### 重点验证任务
- `conclusion_missing`
- `orient_error`
- `apply_orient`
- `contradiction`
- 必要时验证 `partmissing`

#### 验证指标
- 实体召回率
- 标准化准确率
- partlist 稳定率
- 新院区新增表达吸收率
- 下游任务误报率/漏报率变化

#### 输出文件
- `reports/downstream_replay_report.md`

## 8. 建议 agent 的工作顺序
1. 阅读本文件和 [V3/Plan_Extract_Entities.md](V3/Plan_Extract_Entities.md)
2. 阅读 [README.md](README.md) 和 [Extract_Entities.py](Extract_Entities.py)
3. 先做图谱索引化，不要直接开始写 LLM 调用
4. 再做候选发现
5. 再做 LLM 归一化
6. 再做自动验证
7. 最后才做补丁发布和下游回放

## 9. 验收标准
- 能自动从多院区报告中发现规则未覆盖的实体表达
- 能基于本地 LLM 和图谱候选生成结构化归一化结果
- 能通过程序化验证筛出高可信补丁候选
- 能将补丁以独立文件方式发布，而不是直接改主图谱
- 能通过下游任务回放证明实体层增强带来的稳定收益

## 10. 失败标准
- 如果实现结果仍主要依赖人工逐条补图，则项目失败
- 如果 LLM 结果不可复现、不可约束、不可回写，则项目失败
- 如果补丁发布后不能改善下游任务，或显著破坏 partlist 稳定性，则项目失败

## 11. 最终判断
- 本项目不是要做一个“替代规则实体抽取的大模型”，而是要做一个“能自动增长知识图谱覆盖面、且仍保留规则锚定优势的离线知识增长系统”。
- 新 agent 必须始终记住：
  - 主图谱是锚点
  - LLM 是增强器
  - 补丁层是发布方式
  - 下游收益是最终验收标准