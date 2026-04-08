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
pip install fastapi uvicorn pandas numpy jieba gensim scipy pydantic python-dotenv httpx
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
    "use_llm": true
  }'
```

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
  "llm_validated": true
}
```

### 2. 批量质控检查

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

**响应示例：**

```json
{
  "results": [
    {
      "id": "report_001",
      "status": "success",
      "partmissing": "可能漏写部位: 胆囊",
      "conclusion_missing": "",
      ...
    }
  ],
  "summary": {
    "total": 1,
    "success": 1,
    "failed": 0,
    "avg_processing_time": 0.85
  }
}
```

### 3. 快速质控检查（无 LLM）

适合大批量快速筛查，响应更快（约 200-500ms），但假阳性率较高。

```bash
curl -X POST "http://localhost:8000/api/v1/quality/check/fast" \
  -H "Content-Type: application/json" \
  -d '{
    "ConclusionStr": "双肺未见异常。",
    "ReportStr": "双肺纹理增多、紊乱。",
    "modality": "DR",
    "StudyPart": "胸部正侧位",
    "Sex": "男"
  }'
```

### 前端对接示例

#### JavaScript/TypeScript

```typescript
// 单条检查
async function checkReport(reportData: ReportData) {
  const response = await fetch('http://localhost:8000/api/v1/quality/check', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reportData)
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  const result = await response.json();
  return result;
}

// 批量检查
async function batchCheckReports(reports: ReportData[]) {
  const response = await fetch('http://localhost:8000/api/v1/quality/check/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reports, use_llm: true })
  });
  
  return await response.json();
}
```

#### Python

```python
import requests

def check_report(report_data):
    """单条报告质控检查"""
    response = requests.post(
        'http://localhost:8000/api/v1/quality/check',
        json=report_data
    )
    response.raise_for_status()
    return response.json()

# 使用示例
result = check_report({
    'ConclusionStr': '双肺未见异常。',
    'ReportStr': '双肺纹理增多。',
    'modality': 'DR',
    'StudyPart': '胸部',
    'Sex': '男'
})

if result['conclusion_missing']:
    print(f"发现问题: {result['conclusion_missing']}")
```

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

### 配置文件

- `config/system_config.ini` - 系统核心规则
- `config/user_config.ini` - 用户自定义配置
- `.env` - 环境变量配置

## 📊 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `partmissing` | string | 部位缺失检测 |
| `conclusion_missing` | string | 描述与结论不一致 |
| `orient_error` | string | 方位错误（左右矛盾） |
| `sex_error` | string | 性别错误（男女部位不符） |
| `contradiction` | string | 语言矛盾（同一部位阴阳性矛盾） |
| `measure_unit_error` | string | 测量值异常（mm/cm/m） |
| `none_standard_term` | string | 术语不规范 |
| `RADS` | string | RADS 分类检查（BI-RADS/PI-RADS） |
| `Critical_value` | array | 危急值列表 |
| `grammer_error` | array | 语法错误列表 |
| `processing_time` | float | 处理耗时（秒） |
| `llm_validated` | boolean | 是否经过 LLM 验证 |

### 字段值说明

- **空字符串 `""`**：未检测到问题
- **有内容**：检测到问题，内容为具体描述
- **带 `[弱阳性]` 前缀**：LLM 认为置信度较低，建议人工复核

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
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────┐
│                    质控引擎层                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  规则引擎   │  │  Word2Vec  │  │  LLM 精筛   │ │
│  │ (FlashText)│  │  (语义分析) │  │  (Qwen3)   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────┐
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

## 🔒 安全与隐私

- 系统仅处理文本数据，不存储患者影像
- 报告内容在内存中处理，不保留历史记录
- 支持离线部署，无外部数据传输
- 危急值检测结果需人工复核确认

## 📄 许可证

本项目基于 MIT 许可证开源。

---

*本项目仅供医疗质控辅助使用，检测结果需经专业医师复核确认。*
