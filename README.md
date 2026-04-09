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
- `_timers`: 内部性能统计（调试用）

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

### 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    API 服务层                        │
│         (FastAPI + Pydantic + Uvicorn)              │
├─────────────────────────────────────────────────────┤
│                 HTML 预览生成                        │
│         (Jinja2 模板 + 内联 CSS 样式)                │
├─────────────────────────────────────────────────────┤
│                    质控引擎层                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  规则引擎   │  │  Word2Vec  │  │  LLM 精筛   │ │
│  │ (FlashText)│  │  (语义分析) │  │  (Qwen3)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────┤
│                    数据层                           │
│       (知识图谱 + 词向量模型 + 配置文件)              │
└─────────────────────────────────────────────────────┘
```

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
