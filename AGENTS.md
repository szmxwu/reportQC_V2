# 医学影像报告智能质控系统 (Medical Imaging Report Quality Control System)

## 项目概述

本项目是一个基于自然语言处理（NLP）技术的医学影像报告质量控制系统，主要用于放射科医生撰写的CT、MR、DR、DX、MG等模态影像报告的自动质量检测。系统通过实体抽取、语义分析和规则匹配等技术，自动检测报告中可能存在的各类错误，辅助提升报告质量。

### 主要功能

1. **部位缺失检测** - 检查报告中是否遗漏了检查部位应描述的解剖结构
2. **描述与结论一致性检查** - 验证报告描述和结论部分在部位、方位上的一致性
3. **方位错误检测** - 检测左右方位描述是否矛盾或错误
4. **性别错误检测** - 检查报告中是否出现与患者性别不符的解剖部位术语
5. **测量值异常检测** - 检测测量单位（mm/cm/m）是否存在明显异常值
6. **术语规范性检查** - 识别非标准医学术语和解剖学术语错误
7. **危急值检测** - 根据危急值规则表自动识别需要紧急处理的异常发现
8. **特殊检查项目缺失检测** - 检查增强扫描、弥散成像(DWI)、磁敏感成像(SWI)等特殊序列是否描述
9. **RADS分类检查** - 检查乳腺(MG)和前列腺(MR)报告是否包含BI-RADS/PI-RADS分类
10. **语言矛盾检测** - 检测同一部位是否同时存在阳性和阴性描述
11. **申请单方位校验** - 验证报告描述与申请单部位方位的一致性

## 项目结构

```
reportQC_v2/
├── NLP_analyze.py              # 主分析模块，包含报告质控核心函数
├── keyword_extraction.py       # 关键词抽取和实体识别模块
├── llm_service.py              # LLM验证服务模块（Qwen3等大模型验证）
├── run_samples_test.py         # 批量样本测试脚本（支持效率统计）
├── api_server.py               # FastAPI服务接口
├── report_analyze/             # 模块化质检查器包
│   ├── __init__.py
│   ├── report_conclusion_checker.py  # 描述-结论匹配检查
│   ├── contradiction_checker.py      # 矛盾检测
│   ├── llm_validator.py              # LLM批量验证
│   └── config.py                     # 配置管理
├── flashtext/
│   └── keyword.py              # 修改版FlashText关键词匹配引擎
├── model/
│   ├── finetuned_word2vec.m    # 微调后的Word2Vec医学词向量模型（200维）
│   └── report_word2vec_processed_balance_med_jieba.m  # 预训练词向量模型
├── config/
│   ├── system_config.ini       # 系统核心配置参数
│   ├── user_config.ini         # 用户自定义配置参数
│   ├── knowledgegraph.xlsx     # 解剖部位知识图谱（6级层次结构）
│   ├── knowledgegraph_title.xlsx  # 检查条目知识图谱
│   ├── criticalvalue.xlsx      # 危急值规则表
│   ├── exam_special.xlsx       # 特殊检查项目配置
│   ├── replace.xlsx            # 报告文本替换规则
│   ├── replace_title.xlsx      # 检查条目替换规则
│   ├── replace_applytable.xlsx # 申请单文本替换规则
│   ├── Normal_measurement.xlsx # 正常测量值参考范围
│   ├── user_dic_expand.txt     # 结巴分词自定义词典
│   └── ignore_reports.json     # 忽略的设备和部位配置
├── .env                        # 环境变量配置（LLM开关、API地址等）
└── .vscode/settings.json       # VS Code编辑器配置
```

## 技术栈

### 核心依赖
- **Python 3.x** - 主要开发语言
- **pandas** - 数据处理和Excel文件读写
- **numpy** - 数值计算
- **jieba** - 中文分词
- **gensim** - Word2Vec词向量模型加载
- **scipy** - 余弦相似度计算
- **pydantic** - 数据模型验证
- **fastapi** - API服务框架
- **python-dotenv** - 环境变量管理

### 算法组件
- **FlashText** - Aho-Corasick算法实现的高效关键词匹配（本地化修改版）
- **Word2Vec** - 基于1亿放射文本语料+医学专业书籍预训练的200维词向量
- **知识图谱** - 6级层次结构的解剖部位本体库
- **规则引擎** - 基于正则表达式和配置文件的规则匹配系统
- **LLM验证** - 基于Qwen3等大语言模型的后置精筛验证

## 数据模型

### Report (报告医生数据结构)
```python
class Report(BaseModel):
    ConclusionStr: str  # 报告结论文本
    ReportStr: str      # 报告描述文本
    modality: str       # 设备类型 (CT/MR/DR/DX/MG等)
    StudyPart: str      # 检查条目名称
    Sex: str            # 患者性别
    applyTable: str     # 申请单信息（既往史+临床症状+主诉+现病史）
```

### AuditReport (审核医生数据结构)
```python
class AuditReport(BaseModel):
    beforeConclusionStr: str   # 报告医生的结论
    beforeReportStr: str       # 报告医生的描述
    afterConclusionStr: str    # 审核医生的结论
    afterReportStr: str        # 审核医生的描述
    modality: str              # 设备类型
    StudyPart: str             # 检查条目名称
    Sex: str                   # 性别
    report_doctor: str         # 报告医生姓名/工号
    audit_doctor: str          # 审核医生姓名/工号
    applyTable: str            # 申请单信息
```

## 核心配置说明

### .env 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| USE_LLM_VALIDATION | false | 是否启用LLM后置验证 |
| LLM_BASE_URL | http://192.0.0.193:9997/v1 | LLM API地址 |
| LLM_MODEL | qwen3 | 使用的模型名称 |
| LLM_TIMEOUT | 30 | API超时时间（秒）|
| LLM_BATCH_SIZE | 5 | 并发验证数量 |
| LLM_CONFIDENCE_THRESHOLD | 0.7 | 置信度阈值 |
| POSTOPERATIVE_THRESHOLD | 0.5 | 术后相关描述匹配阈值 |

### system_config.ini / config.ini

| 配置节 | 关键参数 | 说明 |
|--------|----------|------|
| sentence | stop_pattern, sentence_pattern | 分句规则正则表达式 |
| clean | stopwords, punctuation, ignore_keywords | 文本清洗规则 |
| positive | NormKeyWords, illness_words, deny_words | 阳性/阴性判定规则 |
| measure | mm_max, cm_max, m_max | 测量值阈值 |
| sex | MaleKeyWords, FemaleKeyWords | 性别关键词 |
| contradiction | key_part, aspects, exclud | 矛盾检测规则 |
| missing | enhance, dwi, swi, perfusion, MRS | 特殊检查关键词 |
| Critical | IgnoreWords | 危急值忽略词 |
| semantics | stopwords | 语义分析停用词 |

### user_config.ini

| 配置节 | 关键参数 | 说明 |
|--------|----------|------|
| application | host, port, workers | 服务运行配置 |
| report_score | A_level, B_level, C_level | 评分等级阈值 |
| report_score | defult_non_standard, MR_non_standard, CT_non_standard | 非标准术语正则 |
| Part_standard | Position_orientation, Exam_orientation | 方位词和体位词 |
| Complexity | CTcomplexity, MRcomplexity | 复杂度系数 |
| Check | Modality | 启用的设备类型 |

## 使用方式

### 命令行测试
```bash
cd /home/wmx/work/python/reportQC_v2
python3 NLP_analyze.py
```

### 批量样本测试（带效率统计）
```bash
# 运行所有样本
python3 run_samples_test.py

# 限制测试数量
python3 run_samples_test.py --limit 10

# 禁用LLM验证
python3 run_samples_test.py --no-llm

# 指定输出文件
python3 run_samples_test.py --output result.xlsx
```

### API调用示例
```python
from NLP_analyze import Report, Report_Quality

report = Report(
    ConclusionStr="1.双肺纹理增多。2.肝脏低密度灶。",
    ReportStr="双肺纹理增多、紊乱。肝实质内见低密度灶。",
    modality="CT",
    StudyPart="胸部/肺平扫,CT上腹部平扫",
    Sex="女",
    applyTable=""
)

result = Report_Quality(report)
print(result)
```

### 返回结果说明
```python
{
    "partmissing": "可能漏写部位: 胆囊",           # 部位缺失检测结果
    "partinverse": "未检查到检查项目方位错误",      # 方位错误检测
    "special_missing": "未检查到漏写特殊检查",      # 特殊检查缺失
    "conclusion_missing": "结论可能遗漏: 肝脏低密度灶",  # 描述与结论不符
    "orient_error": "未检查到描述与结论方位不符",   # 方位一致性
    "contradiction": "未检测到语言矛盾",            # 语言矛盾
    "sex_error": "未发现性别错误",                  # 性别错误
    "measure_unit_error": "未发现测量单位明显错误",   # 测量值异常
    "none_standard_term": "未检测到常见术语错误",     # 术语规范性
    "RADS": "",                                      # RADS分类检查
    "Critical_value": [],                            # 危急值列表
    "apply_orient": "",                              # 申请单位方位错误
    "_timers": {...}                                 # 性能统计（内部使用）
}
```

## 关键算法说明

### 1. 实体抽取流程
```
原始文本 -> 文本清洗(Str_replace) -> 脊柱/肋骨简写扩展 -> 
分句处理 -> FlashText关键词匹配 -> 方位词识别 -> 
歧义消解(clean_mean) -> 父子节点合并(merge_part) -> 实体列表
```

### 2. 歧义消解策略
- **Step 1**: 识别存在歧义的实体（同一位置匹配多个部位）
- **Step 2**: 利用检查部位信息辅助消歧（坐标区间匹配）
- **Step 3**: 利用上下文相邻实体进行消歧（最大公共子部位匹配）

### 3. 描述-结论匹配算法
- 基于部位知识图谱的层级匹配（父子节点关系）
- 方位词一致性校验（左/右/双）
- Word2Vec语义相似度计算（200维词向量，阈值0.3-0.5）
- 强制匹配逻辑（部位-数量兜底，但排除方位相反的情况）
- 排除模板化描述和特殊情况

### 4. LLM验证流程
- 规则引擎输出候选问题（结论缺失、方位错误、矛盾、性别错误）
- 批量提交给LLM进行精筛验证
- 高置信度（≥0.7）保留，中置信度（0.5-0.7）标记弱阳性，低置信度丢弃
- 缓存机制避免重复验证

## 模块化架构

### report_analyze 包

#### report_conclusion_checker.py
- `detect_missing_conclusions()` - 检测结论缺失（句子级分组）
- `detect_orientation_errors()` - 检测方位错误（概率阈值模型）
- `check_report_conclusion()` - 主入口函数

#### contradiction_checker.py
- `check_contradiction()` - 检测同一部位的阴阳性矛盾

#### llm_validator.py
- `batch_validate_with_llm()` - 批量LLM验证（统一入口）
- `validate_conclusion_missing()` - 结论缺失验证
- `validate_orient_error()` - 方位错误验证
- `validate_contradiction()` - 矛盾验证
- `validate_sex_error()` - 性别错误验证

#### config.py
- `SystemConfig` - 系统配置（只读）
- `UserConfig` - 用户配置（可调整）
- `LLMConfig` - LLM验证配置
- `PostoperativeConfig` - 术后相关配置

## 开发规范

### 代码风格
- 使用中文注释说明医学业务逻辑
- 函数命名采用下划线命名法（snake_case）
- 医学术语变量名保持中文拼音或英文缩写
- 关键正则表达式统一存储在配置文件中

### 配置文件管理
- `.env`: 环境变量配置（LLM开关、API地址等），便于不同环境部署
- `system_config.ini`: 系统级核心规则，一般不需要修改
- `user_config.ini`: 用户级配置，可根据医院需求调整阈值
- `*.xlsx`: 使用Excel管理规则便于非技术人员维护
- `user_dic_expand.txt`: 结巴分词自定义词典，格式为"词语 词频 词性"

### 知识图谱维护
- `knowledgegraph.xlsx`: 6级部位层次结构（一级~六级部位）
- 每行包含：各级部位（用\|分隔同义词）、起始坐标、终止坐标、分类
- `knowledgegraph_title.xlsx`: 检查条目名称知识图谱

### 模型文件
- Word2Vec模型使用gensim库训练保存
- 主模型文件为二进制格式，配套.npy文件存储词向量
- 模型输入：经jieba分词后的医学文本
- 向量维度：200维
- 词表大小：约10万医学术语

## 注意事项

1. **路径依赖**: 所有配置文件路径均为相对路径，要求从项目根目录运行
2. **编码格式**: 所有文本文件使用UTF-8编码
3. **内存占用**: 词向量模型加载后占用约500MB内存
4. **并发处理**: 使用多进程池(multiprocessing.Pool)进行批量处理
5. **缓存机制**: 实体抽取结果使用functools.lru_cache缓存，LLM验证结果使用内存缓存
6. **LLM依赖**: LLM验证需要外部大模型服务支持，可通过`.env`关闭

## 测试与验证

### 单元测试
各模块底部包含`if __name__ == "__main__"`测试代码，可直接运行验证：
```bash
python3 keyword_extraction.py  # 测试实体抽取
python3 NLP_analyze.py          # 测试完整质控流程
python3 llm_service.py          # 测试LLM服务
```

### 批量验证
```bash
# 带效率统计的批量测试
python3 run_samples_test.py --limit 100

# 输出包含：
# - 每条记录执行时间
# - 平均/最短/最长执行时间
# - LLM验证耗时统计
# - 吞吐量（条/秒）
```

## 性能统计

### 执行时间统计（run_samples_test.py）
```
效率统计
============================================================
总执行时间:      2.524s
记录数:          3条
平均执行时间:    841.28ms
最短执行时间:    21.27ms
最长执行时间:    1.463s
吞吐量:          1.19条/秒

LLM验证统计:
  LLM调用次数:   1次
  LLM总耗时:     1.438s
  LLM平均耗时:   1.438s
  LLM最短耗时:   1.438s
  LLM最长耗时:   1.438s
  LLM占比:       57.0%
============================================================
```

## 安全与隐私

1. 系统仅处理文本数据，不存储患者影像数据
2. 报告内容在内存中处理，不保留历史记录
3. 危急值检测结果需人工复核确认
4. LLM验证可通过环境变量完全禁用，确保无外发数据

---

*最后更新: 2026-04-05*
