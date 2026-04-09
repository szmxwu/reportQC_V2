# 医学影像报告智能质控 API 服务

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于自然语言处理（NLP）和大型语言模型（LLM）的医学影像报告质量控制系统，提供高性能 RESTful API 服务，支持 CT、MR、DR、DX、MG 等多种影像模态的自动质控检测。

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.8+
# 方式1：使用 requirements.txt（推荐）
pip install -r requirements.txt

# 方式2：手动安装
pip install fastapi uvicorn pandas numpy jieba gensim scipy pydantic python-dotenv httpx jinja2
```

### 2. 解压模型文件

```bash
# 解压主模型（必需）
bash setup_model.sh

# 可选：解压 grammer 模块模型（语法错误检测）
bash grammer/setup_models.sh
```

### 3. 启动服务

```bash
# 方式1：直接启动
python api_server.py

# 方式2：使用 uvicorn（推荐生产环境）
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 📡 API 接口

### 接口概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态检查 |
| `/health` | GET | 健康检查 |
| `/api/v1/quality/check` | POST | **标准质控检查**（推荐） |
| `/api/v1/quality/check/fast` | POST | **快速质控检查**（纯规则，无 LLM） |
| `/api/v1/quality/check/batch` | POST | **批量质控检查** |
| `/api/v1/config` | GET | 获取服务配置 |

### 1. 标准质控检查

**请求示例：**

```bash
curl -X POST "http://localhost:8000/api/v1/quality/check" \
  -H "Content-Type: application/json" \
  -d '{
    "ConclusionStr": "1.双肺纹理增多。2.肝脏低密度灶。",
    "ReportStr": "双肺纹理增多、紊乱。肝实质内见低密度灶。",
    "modality": "CT",
    "StudyPart": "胸部/肺平扫,CT上腹部平扫",
    "Sex": "女",
    "applyTable": "",
    "use_llm": true,
    "generate_html": false
  }'
```

**请求字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| ConclusionStr | string | 是 | 报告结论 |
| ReportStr | string | 是 | 报告描述 |
| modality | string | 是 | 设备类型 (CT/MR/DR/DX/MG) |
| StudyPart | string | 是 | 检查条目名称 |
| Sex | string | 是 | 患者性别 (男/女) |
| applyTable | string | 否 | 申请单信息 |
| use_llm | boolean | 否 | 是否启用 LLM，默认 `true` |
| generate_html | boolean | 否 | 是否生成 HTML 预览，默认 `false` |

**响应示例：**

```json
{
  "partmissing": "可能漏写部位: 胆囊",
  "partinverse": "未检查到检查项目方位错误",
  "special_missing": "未检查到漏写特殊检查",
  "conclusion_missing": "结论可能遗漏: 肝脏低密度灶",
  "orient_error": "未检查到描述与结论方位不符",
  "contradiction": "未检测到语言矛盾",
  "sex_error": "未发现性别错误",
  "measure_unit_error": "未发现测量单位明显错误",
  "none_standard_term": "未检测到常见术语错误",
  "RADS": "",
  "Critical_value": [],
  "apply_orient": "",
  "grammer_error": [],
  "processing_time": 0.85,
  "llm_validated": true,
  "html_path": null
}
```

**生成 HTML 预览示例：**

```bash
curl -X POST "http://localhost:8000/api/v1/quality/check" \
  -H "Content-Type: application/json" \
  -d '{
    "ConclusionStr": "右肺上叶占位。",
    "ReportStr": "左肺上叶可见占位。",
    "modality": "CT",
    "StudyPart": "胸部/肺平扫",
    "Sex": "男",
    "generate_html": true
  }'
```

响应中会包含 `html_path` 字段，指向生成的 HTML 文件路径。

### 2. HTML 预览功能

当 `generate_html=true` 时，系统会生成一个独立的 HTML 预览文件：

- **独立文件**：所有 CSS 样式内联，无需外部依赖，可离线打开
- **颜色标注**：13 种错误类型用不同颜色高亮显示
- **错误汇总**：底部显示所有检测到的质量问题
- **文件位置**：`output/report_preview_{timestamp}.html`

**高亮颜色对照：**

| 颜色 | 错误类型 |
|------|----------|
| 🟡 黄色 | 部位缺失 |
| 🟠 橙色 | 方位错误 |
| 🔴 红色 | 结论缺失 |
| 🟣 紫色 | 方位不符-描述 |
| 🩷 粉色 | 方位不符-结论 |
| 🔵 青色 | 语言矛盾 |
| 💗 深粉 | 性别错误 |
| 🟤 棕色 | 测量值错误 |
| ⚪ 灰色 | 术语不规范 |
| ⚫ 深红+闪烁 | 危急值 |
| 🟢 绿色 | 申请单方位错误 |
| 💛 淡黄+波浪线 | 语法错误 |

### 3. 批量质控检查

**请求示例：**

```bash
curl -X POST "http://localhost:8000/api/v1/quality/check/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "reports": [
      {
        "id": "report_001",
        "ConclusionStr": "双肺纹理增多。",
        "ReportStr": "双肺纹理增多、紊乱。",
        "modality": "CT",
        "StudyPart": "胸部/肺平扫",
        "Sex": "女",
        "applyTable": ""
      }
    ],
    "use_llm": true
  }'
```

### 4. 快速质控检查（无 LLM）

适合大批量快速筛查，响应更快（约 200-500ms），但假阳性率较高。

```bash
curl -X POST "http://localhost:8000/api/v1/quality/check/fast" \
  -H "Content-Type: application/json" \
  -d '{
    "ConclusionStr": "双肺未见异常。",
    "ReportStr": "双肺纹理增多、紊乱。",
    "modality": "DR",
    "StudyPart": "胸部正侧位",
    "Sex": "男",
    "generate_html": false
  }'
```

## 🐍 Python API 直接使用

```python
from NLP_analyze import Report, Report_Quality

# 创建报告
report = Report(
    ConclusionStr="右肺上叶占位性病变。",
    ReportStr="左肺上叶可见占位性病变，大小约3cm。",
    modality="CT",
    StudyPart="胸部/肺平扫",
    Sex="男",
    applyTable=""
)

# 方式1: 使用环境变量控制 LLM（.env 中的 USE_LLM_VALIDATION）
result = Report_Quality(report)

# 方式2: 强制启用 LLM（忽略环境变量）
result = Report_Quality(report, llm=True)

# 方式3: 强制禁用 LLM（忽略环境变量）
result = Report_Quality(report, llm=False)

# 方式4: 生成 HTML 预览
result = Report_Quality(report, llm=False, html=True)
print(f"HTML 文件: {result['_html_path']}")

# 检查结果
if result['orient_error']:
    print(f"方位错误: {result['orient_error']}")
```

### Report_Quality 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ReportTxt` | Report | - | 报告数据对象（必填） |
| `debug` | bool | False | 是否打印详细耗时统计 |
| `llm` | bool/None | None | LLM 开关，None=使用环境变量，True=强制开启，False=强制关闭 |
| `html` | bool | False | 是否生成 HTML 预览文件 |

### 返回值

返回字典包含质控结果，当 `html=True` 时额外包含：
- `_html_path`: 生成的 HTML 文件路径
- `_timers`: 内部性能统计（调试用，仅 `debug=True` 时返回）

## 📊 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `partmissing` | list | 部位缺失检测 |
| `partinverse` | list | 检查项目方位错误 |
| `conclusion_missing` | list | 结论与描述不一致 |
| `orient_error` | list | 方位错误（左右矛盾） |
| `sex_error` | string | 性别错误（男女部位不符） |
| `contradiction` | list | 语言矛盾（同一部位阴阳性矛盾） |
| `measure_unit_error` | string | 测量值异常（mm/cm/m） |
| `none_standard_term` | list | 术语不规范 |
| `RADS` | string | RADS 分类检查（BI-RADS/PI-RADS） |
| `Critical_value` | array | 危急值列表 |
| `grammer_error` | array | 语法错误列表 |
| `processing_time` | float | 处理耗时（秒） |
| `llm_validated` | boolean | 是否经过 LLM 验证 |
| `html_path` | string/null | HTML 预览文件路径（如果 generate_html=true） |

### 字段值说明

- **空字符串 `""`** 或 **空列表 `[]`**：未检测到问题
- **有内容**：检测到问题，内容为具体描述
- **带 `[弱阳性]` 前缀**：LLM 认为置信度较低，建议人工复核

## 🔧 配置说明

### 环境变量 (.env)

```bash
# API 服务配置
API_HOST=0.0.0.0
API_PORT=8000

# LLM 精筛配置（可选）
USE_LLM_VALIDATION=true
LLM_BASE_URL=http://localhost:9997/v1
LLM_MODEL=qwen3
LLM_TIMEOUT=30
LLM_CONFIDENCE_THRESHOLD=0.7
```

### LLM 控制优先级

1. **函数参数 `llm=True/False`**：最高优先级，强制覆盖环境变量
2. **环境变量 `USE_LLM_VALIDATION`**：默认配置
3. **不传递参数**：使用环境变量设置

### 配置文件

- `config/system_config.ini` - 系统核心规则
- `config/user_config.ini` - 用户自定义配置
- `.env` - 环境变量配置

## 🏥 功能特性

### 核心质控能力

| 功能 | 说明 | 规则引擎 | LLM 精筛 |
|------|------|----------|----------|
| 部位缺失检测 | 检查应描述的解剖结构是否遗漏 | ✅ | ✅ |
| 描述-结论一致性 | 验证描述和结论的一致性 | ✅ | ✅ |
| 方位错误检测 | 检测左右方位矛盾 | ✅ | ✅ |
| 性别错误检测 | 检查与患者性别不符的术语 | ✅ | ✅ |
| 测量值异常 | 检测明显异常的测量单位 | ✅ | ❌ |
| 术语规范性 | 识别非标准医学术语 | ✅ | ❌ |
| 危急值检测 | 自动识别紧急异常发现 | ✅ | ❌ |
| 语法错误检测 | 错别字、词序错误等 | ✅ | ✅ |

## 🧠 技术架构详解

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           API 服务层                                     │
│              (FastAPI + Pydantic + Uvicorn)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                          HTML 预览生成                                   │
│              (Jinja2 模板 + 内联 CSS 样式)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                          质控引擎层                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   实体抽取  │  │  规则引擎   │  │  Word2Vec  │  │  LLM 精筛   │    │
│  │  (FlashText)│  │ (多维度检查)│  │  (语义分析) │  │  (Qwen3)   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
├─────────────────────────────────────────────────────────────────────────┤
│                           数据层                                         │
│        (医学知识图谱 + 词向量模型 + 配置文件 + 语法模型)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心技术流程

#### 1. 实体抽取流程 (Entity Extraction)

实体抽取是整个质控系统的基石，采用"预处理 → 关键词匹配 → 消歧 → 属性提取"的四阶段流水线：

```
输入文本 (原始医学报告)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  第一阶段：文本预处理 (medical_preprocessor.py)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  硬分句      │  │  文本清洗    │  │  术语扩展    │       │
│  │ (。；！？\n)│  │ (替换/删除)  │  │ (范围展开)   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                              │
│  核心任务：                                                  │
│  • 将文本切分为长句（按。；！？等标点）                      │
│  • 清洗无关字符、标准化标点                                  │
│  • 扩展医学缩写（如"L1-L5"→"L1、L2、L3、L4、L5"）            │
│  • 替换同义词（如"左侧"→"左"）                              │
│  • 为 FlashText 匹配提供清晰的词语边界                       │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  第二阶段：关键词提取 (FlashText + 知识图谱)                 │
│                                                              │
│  build_unified_processor()                                   │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────┐                   │
│  │  从 knowledgegraph.xlsx 构建         │                   │
│  │  统一的 FlashText KeywordProcessor   │                   │
│  │                                      │                   │
│  │  数据格式：一级部位|二级|...|六级    │                   │
│  │  例：胸部→肺→肺上叶→尖段→支气管    │                   │
│  │  坐标：人体解剖坐标 (start, end)     │                   │
│  └──────────────────────────────────────┘                   │
         │                                                       │
         ▼                                                       │
  Extract_Keywords(text)                                         │
      │                                                           │
      ▼                                                           │
  ┌──────────────────────────────────────┐                     │
  │  FlashText.extract_keywords()        │                     │
  │  • O(N) 复杂度，比正则快 10-100 倍   │                     │
  │  • 返回：关键词 + 位置坐标 + 层级链  │                     │
  └──────────────────────────────────────┘                     │
         │                                                       │
         ▼                                                       │
  resplit_sentence_by_entities()                                 │
      │                                                           │
      ▼                                                           │
  ┌──────────────────────────────────────┐                     │
  │  基于实体位置重新分句（软分句）      │                     │
  │                                      │                     │
  │  规则：                              │                     │
  │  • 按逗号、分号切分                  │                     │
  │  • 无实体子句与前后合并              │                     │
  │  • 强依赖词（"压迫"/"位于"）触发合并 │                     │
  │  • 生成虚拟短句 (virtual sentence)   │                     │
  └──────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  第三阶段：属性提取 (Get_Attributes.py)                      │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐   │
│  │ extract_       │  │ get_all_       │  │ find_measure │   │
│  │ orientation()  │  │ illness_       │  │ ()           │   │
│  │                │  │ descriptions() │  │              │   │
│  └────────────────┘  └────────────────┘  └──────────────┘   │
│                                                              │
│  提取内容：                                                  │
│  • orientation: 方位（左/右/双）                             │
│  • illness: 病理描述（如"增大"、"低密度影"）                │
│  • positive: 阴阳性（正常/异常）                             │
│  • measure/percent/volume: 测量值                            │
│  • attribute: 病理属性（形态/位置/大小/强化等）              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  第四阶段：实体消歧与合并                                    │
│                                                              │
│  ┌────────────────────────────────────────────┐            │
│  │ disambiguate_entities()                    │            │
│  │                                            │            │
│  │ 解决一词多义问题：                         │            │
│  │ 例："头" → 头部？骨头？头端？              │            │
│  │                                            │            │
│  │ 消歧策略（按优先级）：                     │            │
│  │ 1. 先验知识过滤（脊柱相关排除女性附件）    │            │
│  │ 2. 检查部位匹配（与 StudyPart 坐标交叉）   │            │
│  │ 3. 上下文 partlist 匹配                    │            │
│  │ 4. 上下文 axis 坐标匹配                    │            │
│  │ 5. 扩展坐标匹配                            │            │
│  │ 6. 兜底：层级最深/second_root 优先         │            │
│  └────────────────────────────────────────────┘            │
         │                                                       │
         ▼                                                       │
  ┌────────────────────────────────────────────┐              │
  │ merge_part()                               │              │
  │                                            │              │
  │ 合并父子节点：                             │              │
  │ • 标题模式：优先保留父节点                 │              │
  │ • 报告模式：优先保留子节点（更具体）       │              │
  │ • 方位/阴阳性不一致时不合并                │              │
  └────────────────────────────────────────────┘              │
         │                                                       │
         ▼                                                       │
┌─────────────────────────────────────────────────────────────┐
│  第五阶段：短句匹配 (short_sentence_match.py)                │
│                                                              │
│  将处理后的虚拟短句映射回原始文本中的对应子句                │
│  使用 difflib.SequenceMatcher 计算相似度                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    结构化实体列表（含完整上下文信息）
```

**关键技术点：**

| 模块 | 功能 | 技术实现 |
|------|------|----------|
| **medical_preprocessor.py** | 文本预处理 | 正则表达式清洗、术语扩展（脊柱/肋骨范围展开）、同义词替换 |
| **medical_expander.py** | 医学缩写扩展 | L1-L5→L1、L2...L5；第1-3肋→第1肋、第2肋、第3肋 |
| **FlashText** | 关键词匹配 | Aho-Corasick 自动机，O(N) 复杂度，构建自知识图谱 |
| **disambiguation.py** | 实体消歧 | 6 层策略：先验知识→检查部位→上下文匹配→坐标匹配→扩展匹配→兜底 |
| **entity_merge.py** | 实体合并 | 父子节点合并，报告模式保留子节点，标题模式保留父节点 |
| **Get_Attributes.py** | 属性提取 | 方位提取（左/右）、病理描述提取、阴阳性判断、测量值提取 |

#### 1.1 Tools 模块详解

**tools/medical_preprocessor.py**
文本预处理的核心模块，不同于常规NLP的jieba分词，而是通过正则表达式和规则替换来优化文本，使其更适合FlashText匹配：

| 处理阶段 | 功能说明 | 示例 |
|----------|----------|------|
| 硬分句 | 按句号、分号、感叹号等切分长句 | `"肝脏增大。胆囊未见异常。"` → 2个长句 |
| 基础清洗 | 移除多余空格、标准化标点 | `"肝脏 增大"` → `"肝脏增大"` |
| 高优先级替换 | 高频短词优先替换 | `"左侧"` → `"左"`，`"增强扫描"` → `"增强"` |
| 医学扩展 | 展开缩写和范围表示 | `"L1-L3"` → `"L1、L2、L3"` |
| 脊柱标识符转换 | 英文缩写转中文 | `"C5"` → `"颈5"`，`"T12"` → `"胸12"` |
| 一般替换 | 低频长词替换 | `"未见明显"` → `"未见"` |
| 正则替换 | 模式匹配替换 | 删除括号内容、移除序号前缀 |

规则来源：`config/replace.xlsx`（报告）、`config/replace_title.xlsx`（检查标题）

**tools/medical_expander.py**
专门处理医学文本中的范围表示和缩写扩展：

| 扩展类型 | 输入示例 | 输出结果 |
|----------|----------|----------|
| 脊柱范围 | `"L1-L5椎体"` | `"L1椎体、L2椎体、L3椎体、L4椎体、L5椎体"` |
| 椎间盘范围 | `"L1/2-L5/S1椎间盘"` | `"L1/2、L2/3、L3/4、L4/5、L5/S1椎间盘"` |
| 顿号形式 | `"T3、4、5椎体"` | `"T3椎体、T4椎体、T5椎体"` |
| 肋骨范围 | `"第1-3肋骨骨折"` | `"第1肋骨骨折、第2肋骨骨折、第3肋骨骨折"` |
| 多肋列举 | `"左侧第5、6前肋骨折"` | `"左侧第5前肋骨折、左侧第6前肋骨折"` |

核心方法：`expand_all()` → 依次调用 `_normalize_spine_abbreviations()` → `_expand_spine_ranges()` → `_expand_disk_ranges()` → `_expand_spine_dots()` → `_expand_rib_abbreviations()`

**tools/Get_Attributes.py**
从已分句的实体中提取医学属性：

```python
# 提取方位（左/右/双）
extract_orientation(entities)
# 规则：
# 1. 从实体前方查找方位词（逆序匹配）
# 2. 向后查找括号内方位（如"肝脏(左叶)"）
# 3. 上下文推理：同句前后实体方位继承

# 提取疾病描述
get_all_illness_descriptions(entities)
# 三种匹配模式：
# - 否定前置："未见肝脏异常" → illness="未见异常"
# - 倒置句："血肿位于肝脏" → illness="血肿"
# - 默认后置："肝脏增大" → illness="增大"

# 提取测量值 + 阴阳性判断
find_measure(text) → (max_val, percent, volume, is_positive)
```

**tools/disambiguation.py**
解决FlashText匹配后的一词多义问题。核心类 `AmbiguityResolver`：

| 消歧策略 | 触发条件 | 说明 |
|----------|----------|------|
| 先验知识过滤 | 句子含脊柱关键词 | 排除"女性附件"等不相关候选 |
| 检查部位匹配 | 实体坐标与 StudyPart 交叉 | 最可靠的消歧依据 |
| 上下文 partlist 匹配 | 与相邻实体共享部位层级 | 计算集合交集大小 |
| 上下文 axis 匹配 | 与相邻实体坐标区间交叉 | 基于人体坐标位置 |
| 扩展坐标匹配 | 上述策略失败 | 坐标取模100后比较 |
| 兜底策略 | 仍有多候选 | 优先选层级深的、second_root中的 |

**关键概念：**
- `partlist`: 实体的层级链，如 `["胸部", "肺", "肺上叶", "尖段"]`
- `axis`: 人体坐标 `(start, end)`，基于解剖位置定义
- `anchor`: 消歧依据的锚定实体，用于追溯消歧逻辑

**tools/entity_merge.py**
合并同一解剖分支上的父子节点实体：

```python
# 标题模式 vs 报告模式
_merge_title_mode(): 优先保留父节点（如保留"肝脏"而非"肝左叶"）
_merge_report_mode(): 优先保留子节点（如保留"肝左叶"而非"肝脏"）

# 合并条件（报告模式）
_can_merge_report(entity1, entity2):
    1. 是父子关系（partlist 包含）
    2. 方位兼容（相同或其一为空）
    3. 阳性状态相同
```

**tools/short_sentence_match.py**
将处理后的虚拟短句映射回原始句子中的对应子句：

| 匹配层级 | 策略 | 说明 |
|----------|------|------|
| L0 | 完全匹配 | 短句在原始句中直接存在 |
| L1 | 子句切分 | 按标点切分后逐子句匹配 |
| L2 | 字符重合度筛选 | 快速过滤明显不匹配的子句 |
| L3 | difflib 相似度 | `SequenceMatcher.quick_ratio()` → `ratio()` |
| L4 | 前缀加权 | 医学报告开头词汇更重要，前缀相似>0.8则加权 |

**tools/text_utils.py**
通用文本处理工具函数：

| 函数 | 功能 |
|------|------|
| `clean_whitespace()` | 清理各种空白字符（\xa0、\u3000等） |
| `normalize_punctuation()` | 中文标点转英文标点 |
| `remove_numbered_prefix()` | 移除句首序号（如"1."、"2、"） |
| `is_measurement_paragraph()` | 检测"测量型报告"段落（心功能分析等） |
| `extract_measurements()` | 提取测量值（数字+单位） |

#### 2. 质控检查流程 (Quality Control Pipeline)

```
结构化实体
    │
    ├────────────┬────────────┬────────────┬────────────┐
    ▼            ▼            ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│部位缺失 │  │结论-描述│  │方位矛盾 │  │语言矛盾 │  │危急值  │
│检查     │  │一致性  │  │检测     │  │检测     │  │检测    │
└────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘
     │           │           │           │           │
     └───────────┴───────────┴───────────┴───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   LLM 精筛验证      │  ← 对规则检出结果进行二次确认
              │  (可选，降低假阳性)  │
              └──────────┬──────────┘
                         │
                         ▼
                   质控结果输出
```

**各检测模块说明：**

| 模块 | 技术原理 | 实现细节 |
|------|----------|----------|
| **部位缺失** | 检查项目-部位映射 + 实体覆盖分析 | 基于 `config/system_config.ini` 中定义的检查项目应描述部位，对比实际抽取的实体部位 |
| **结论-描述一致性** | 语义相似度计算 (Word2Vec) | 计算描述实体和结论实体的余弦相似度，低于阈值则标记为缺失 |
| **方位矛盾** | 左右方位词匹配 + 实体关联 | 检测同一解剖部位在描述和结论中的左右方位是否矛盾 |
| **语言矛盾** | 阴阳性规则 + 方面词检测 | 识别同一部位的矛盾描述（如"规则"vs"不规则"、"清晰"vs"不清"） |
| **危急值** | 关键词匹配 + 程度修饰词过滤 | 匹配危急值词库，同时过滤"未见"、"无"等否定修饰词 |

#### 3. 语法错误检测流程

```
输入文本
    │
    ├───────────────┬─────────────────┐
    ▼               ▼                 ▼
┌─────────┐   ┌──────────┐    ┌──────────┐
│医学错别字│   │ 词序错误  │    │ KenLM    │
│检测     │   │ 检测     │    │ 语言模型 │
│(138K+对)│   │(3.7K模板)│    │困惑度检测│
└────┬────┘   └────┬─────┘    └────┬─────┘
     │             │               │
     └─────────────┴───────────────┘
                     │
                     ▼
            ┌────────────────┐
            │  错误聚合与去重 │
            └───────┬────────┘
                    │
                    ▼
               语法错误列表
```

**技术实现：**
- **医学错别字检测**：基于 AC 自动机的高效模糊匹配，支持 138,604 对医学混淆词
- **词序错误检测**：使用 3,707 个医学 bigram 模板检测词序异常
- **KenLM 语言模型**：基于 5-gram 模型的困惑度计算，识别不自然的语言表达

#### 4. LLM 精筛机制

```
规则引擎检出可疑问题
           │
           ▼
┌──────────────────────┐
│   构建验证 Prompt     │
│ (上下文+问题描述)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   批量 LLM 调用       │  ← 并发请求，超时控制
│  (OpenAI API 格式)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  置信度解析与阈值过滤  │  ← 阈值：0.7
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  置信度高     置信度低
  (保留)      (标记弱阳性)
```

**设计目的：**
- 规则引擎负责**召回率**（不漏检）
- LLM 精筛负责**精确率**（降低误报）
- 对于置信度低的结果，标记 `[弱阳性]` 提示人工复核

### 数据模型设计

#### 核心实体结构

```python
class Entity(BaseModel):
    """医学实体数据结构"""
    word: str           # 实体名称（如"肝脏"）
    position: str       # 解剖部位
    positive: bool      # 阴阳性（True=阳性/异常）
    percent: float      # 百分比数值
    measure: float      # 测量值
    unit: str           # 单位（mm/cm/m）
    orientation: str    # 方位（左/右/双侧）
    source: str         # 来源（描述/结论/申请单）
```

#### 质控结果结构

```python
class QCResult(BaseModel):
    # 结构性错误
    partmissing: List[str]        # 部位缺失
    partinverse: List[str]        # 项目方位错误
    
    # 内容一致性错误
    conclusion_missing: List[dict] # 结论遗漏
    orient_error: List[dict]      # 方位错误
    contradiction: List[dict]     # 语言矛盾
    
    # 医学逻辑错误
    sex_error: str                # 性别错误
    measure_unit_error: str       # 测量单位错误
    Critical_value: List[str]     # 危急值
    
    # 规范性错误
    none_standard_term: List[str] # 非标准术语
    grammer_error: List[dict]     # 语法错误
    RADS: str                     # RADS分类
```

### 配置文件体系

```
config/
├── config.ini           # 核心规则配置
├── user_config.ini      # 用户自定义配置
└── user_dic_expand.txt  # 扩展医学词典

grammer/
├── models/
│   ├── radiology_ngram.klm     # KenLM 5-gram 语言模型
│   └── word2vec/               # Word2Vec 词向量模型
└── data/
    ├── typo_pairs.json         # 医学错别字对照表
    └── word_order_templates.txt # 词序检测模板

model/
└── finetuned_word2vec.m        # 微调后的词向量模型
```

**配置层级（优先级从高到低）：**
1. 运行时参数（`llm`, `html`, `debug`）
2. 环境变量（`.env`）
3. 用户配置（`user_config.ini`）
4. 系统默认配置（`config.ini`）

## 🚀 部署建议

### 开发环境

```bash
# 热重载模式
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### 生产环境

```bash
# 多进程模式（根据 CPU 核心数调整 workers）
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用 gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker 部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

# 解压模型
RUN bash setup_model.sh

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📈 性能指标

| 场景 | 平均耗时 | 吞吐量 |
|------|----------|--------|
| 标准检查（含 LLM） | 0.8-1.5s | 1-2 条/秒 |
| 快速检查（无 LLM） | 0.2-0.5s | 5-10 条/秒 |
| 批量检查（10条） | 8-15s | - |
| 生成 HTML 预览 | +50-100ms | - |

## 🔒 安全与隐私

- 系统仅处理文本数据，不存储患者影像
- 报告内容在内存中处理，不保留历史记录
- HTML 预览文件生成在本地，不上传到外部服务器
- 支持离线部署，无外部数据传输
- 危急值检测结果需人工复核确认

## 📄 许可证

本项目基于 MIT 许可证开源。

---

*本项目仅供医疗质控辅助使用，检测结果需经专业医师复核确认。*
