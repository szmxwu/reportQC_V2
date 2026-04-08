# 医学影像报告智能质控系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CC--BY--NC%204.0-lightblue.svg)](LICENSE)

基于自然语言处理（NLP）技术的医学影像报告质量控制系统，支持 CT、MR、DR、DX、MG 等多种影像模态的自动质控检测。

## 目录

- [功能特性](#功能特性)
- [技术亮点](#技术亮点)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [语法错误检测（Grammar Detection）](#语法错误检测grammar-detection)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [性能表现](#性能表现)
- [贡献指南](#贡献指南)
- [许可证](#许可证)


## 功能特性

### 核心质控功能

| 功能 | 说明 |
|------|------|
| 🏥 **部位缺失检测** | 检查报告中是否遗漏了检查部位应描述的解剖结构 |
| 🔄 **描述-结论一致性** | 验证描述和结论在部位、方位上的一致性 |
| ↔️ **方位错误检测** | 检测左右方位描述是否矛盾 |
| ⚧️ **性别错误检测** | 检查与患者性别不符的解剖部位术语 |
| 📏 **测量值异常** | 检测 mm/cm/m 单位的明显异常值 |
| 📝 **术语规范性** | 识别非标准医学术语 |
| 🚨 **危急值检测** | 自动识别需要紧急处理的异常发现 |
| 🔬 **特殊检查缺失** | 检查增强、DWI、SWI 等特殊序列描述 |
| 📊 **RADS 分类** | 检查 BI-RADS/PI-RADS 分类 |
| ⚠️ **语言矛盾** | 检测同一部位的阴阳性矛盾 |
| 📋 **申请单校验** | 验证报告与申请单部位方位一致性 |
| 📚 **语法错误检测** | 自动检测拼写/语法类错误与医学术语混淆（集成 LLM 可选精筛） |

### 技术亮点

- **实体抽取**：基于 FlashText 的高效关键词匹配 + 6 级知识图谱
- **语义分析**：200 维 Word2Vec 医学词向量（10 万+术语）
- **LLM 精筛**：可选 Qwen3 大模型后置验证，降低假阳性
- **效率统计**：完整的性能监控和耗时分析

## 快速开始

### 环境要求

- Python 3.8+
- 内存：2GB+（词向量模型占用约 500MB）

### 安装依赖

```bash
pip install pandas numpy jieba gensim scipy pydantic fastapi python-dotenv openpyxl
```

### 配置文件

复制 `.env.example` 为 `.env`，根据需要调整：

```bash
# 功能开关
USE_LLM_VALIDATION=false

# LLM 配置（可选）
LLM_BASE_URL=http://localhost:9997/v1
LLM_MODEL=qwen3
```

### 运行测试

```bash
# 单条测试
python3 NLP_analyze.py

# 批量测试（带效率统计）
python3 run_samples_test.py --limit 10
```

## 使用示例

### Python API

```python
from NLP_analyze import Report, Report_Quality

# 创建报告
report = Report(
    ConclusionStr="1.双肺纹理增多。2.肝脏低密度灶。",
    ReportStr="双肺纹理增多、紊乱。肝实质内见低密度灶。",
    modality="CT",
    StudyPart="胸部/肺平扫,CT上腹部平扫",
    Sex="女",
    applyTable=""
)

# 质控检测
result = Report_Quality(report)

# 查看结果
print(result['conclusion_missing'])  # 结论缺失检测
print(result['orient_error'])        # 方位错误检测
print(result['sex_error'])           # 性别错误检测
```

### 批量测试

```bash
# 运行所有样本
python3 run_samples_test.py

# 限制数量
python3 run_samples_test.py --limit 100

# 禁用 LLM 验证
python3 run_samples_test.py --no-llm

# 指定输出文件
python3 run_samples_test.py --output result.xlsx
```

### 启动 API 服务

```bash
python3 api_server.py
```

服务将在 `http://localhost:8000` 启动，提供以下接口：

- `POST /api/v1/report/quality` - 单报告质控
- `POST /api/v1/report/batch_quality` - 批量质控
- `GET /api/v1/config` - 查看配置
- `GET /docs` - Swagger 文档

## 语法错误检测（Grammar Detection）

本项目包含一个独立的语法/拼写与医学术语混淆检测模块，位于 `grammer/` 目录，主要用途是发现报告中的拼写、用词或短语级别的医学术语混淆（例如“起腔” vs “气腔”）。主要要点：

- 模块位置：`grammer/`，推理脚本：`grammer/inference/detect_real_data_final.py`。
- 集成：已将语法检测器以懒加载方式集成到主入口 `NLP_analyze.Report_Quality()`，返回字段名为 `grammer_error`（列表），每项包含：错误短语、所在句子、错误类别、来源字段（描述/结论/申请单）及建议。
- 输出格式：批量推理输出包含 `metadata.performance`（性能统计）与按句子聚合的 `text` 字段；`grammer_error` 出现在单条或批量质控输出中。
- 独立运行示例：

```bash
# 对单条样例运行主质控（包含语法检测）
python3 NLP_analyze.py

# 运行语法检测独立推理（示例）
cd grammer && python3 inference/detect_real_data_final.py --sample --limit 100 -o ../output/detect_results_sample.json
```

在使用 API 时，`POST /api/v1/report/quality` 和批量接口 `POST /api/v1/report/batch_quality` 的返回 JSON 中会包含 `grammer_error` 字段（可能为空列表），Swagger 示例也已更新以展示该字段。

## 项目结构

```
reportQC_v2/
├── NLP_analyze.py          # 主分析模块
├── llm_service.py          # LLM 验证服务
├── run_samples_test.py     # 批量测试脚本
├── api_server.py           # API 服务
├── report_analyze/         # 模块化质检查器
│   ├── report_conclusion_checker.py
│   ├── contradiction_checker.py
│   ├── llm_validator.py
│   └── config.py
├── model/
│   └── finetuned_word2vec.m    # 医学词向量模型
├── config/                 # 配置文件
│   ├── system_config.ini
│   ├── user_config.ini
│   ├── knowledgegraph.xlsx
│   └── criticalvalue.xlsx
└── .env                    # 环境变量
```

## 配置说明

### 关键环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `USE_LLM_VALIDATION` | 启用 LLM 后置验证 | `false` |
| `LLM_BASE_URL` | LLM API 地址 | - |
| `LLM_MODEL` | 模型名称 | `qwen3` |
| `POSTOPERATIVE_THRESHOLD` | 术后描述匹配阈值 | `0.5` |

### 用户配置 (config/user_config.ini)

- `report_score`: 评分等级阈值
- `sex`: 性别关键词配置
- `Part_standard`: 方位词配置

## 性能表现

在标准测试集上的性能统计：

```
效率统计
============================================================
总执行时间:      2.524s
记录数:          3条
平均执行时间:    841ms
最短执行时间:    21ms
最长执行时间:    1.463s
吞吐量:          1.2条/秒

LLM验证统计:
  LLM调用次数:   1次
  LLM平均耗时:   1.438s
  LLM占比:       57%
============================================================
```

> 注：性能取决于是否启用 LLM 验证。纯规则引擎模式下吞吐量可达 50+ 条/秒。

## 算法原理

### 实体抽取流程

```
原始文本 → 文本清洗 → 分句处理 → FlashText匹配 → 
方位识别 → 歧义消解 → 实体列表
```

### 描述-结论匹配

1. **部位匹配**：基于 6 级知识图谱的层级匹配
2. **方位校验**：左/右/双方位一致性检查
3. **语义相似度**：Word2Vec 余弦相似度（阈值 0.3-0.5）
4. **LLM 精筛**：大模型验证降低假阳性（可选）

## 贡献指南

欢迎提交 Issue 和 PR！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 Creative Commons Attribution-NonCommercial 4.0 国际 (CC BY-NC 4.0) 许可证，允许学术研究与教学共享，但禁止商业用途。详见 [LICENSE](LICENSE) 文件。如需商业使用，请联系项目维护者以获取授权。

## 致谢

- 词向量模型基于 1 亿+ 放射文本语料和医学专业书籍训练
- FlashText 关键词匹配引擎（本地化修改）
- 结巴中文分词

## 联系方式

如有问题或建议，欢迎提交 Issue。

---

*本项目仅供医疗质控辅助使用，检测结果需经专业医师复核确认。*
