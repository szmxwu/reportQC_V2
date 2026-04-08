# API 接口指南

## 接口概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态检查 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger API 文档（离线/在线）|
| `/api/v1/quality/check` | POST | 标准质控检查 |
| `/api/v1/quality/check/fast` | POST | 快速质控检查（无 LLM）|
| `/api/v1/quality/check/batch` | POST | 批量质控检查 |
| `/api/v1/config` | GET | 获取服务配置 |

## 详细接口说明

### 1. 服务状态检查

**请求：**
```bash
GET /
```

**响应：**
```json
{
  "status": "running",
  "version": "2.0.0",
  "llm_available": true,
  "offline_docs": true
}
```

### 2. 标准质控检查

**请求：**
```bash
POST /api/v1/quality/check
Content-Type: application/json

{
  "ConclusionStr": "1.双肺纹理增多。2.肝脏低密度灶。",
  "ReportStr": "双肺纹理增多、紊乱。肝实质内见低密度灶。",
  "modality": "CT",
  "StudyPart": "胸部/肺平扫,CT上腹部平扫",
  "Sex": "女",
  "applyTable": "",
  "use_llm": true
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| ConclusionStr | string | 是 | 报告结论 |
| ReportStr | string | 是 | 报告描述 |
| modality | string | 是 | 设备类型 (CT/MR/DR/DX/MG) |
| StudyPart | string | 是 | 检查条目名称 |
| Sex | string | 是 | 患者性别 (男/女) |
| applyTable | string | 否 | 申请单信息 |
| use_llm | boolean | 否 | 是否启用 LLM，默认 true |

**响应：**
```json
{
  "partmissing": "可能漏写部位: 胆囊",
  "conclusion_missing": "结论可能遗漏: 肝脏低密度灶",
  "orient_error": "",
  "sex_error": "",
  "contradiction": "",
  "measure_unit_error": "",
  "none_standard_term": "",
  "RADS": "",
  "Critical_value": [],
  "apply_orient": "",
  "grammer_error": [],
  "processing_time": 0.85,
  "llm_validated": true
}
```

### 3. 批量质控检查

**请求：**
```bash
POST /api/v1/quality/check/batch
Content-Type: application/json

{
  "reports": [
    {
      "ConclusionStr": "双肺未见异常。",
      "ReportStr": "双肺纹理增多。",
      "modality": "DR",
      "StudyPart": "胸部",
      "Sex": "男"
    },
    {
      "ConclusionStr": "肝脏低密度灶。",
      "ReportStr": "肝实质内见低密度影。",
      "modality": "CT",
      "StudyPart": "上腹部",
      "Sex": "女"
    }
  ],
  "use_llm": true
}
```

**响应：**
```json
{
  "results": [
    {
      "status": "success",
      "conclusion_missing": "结论可能遗漏: 双肺纹理增多",
      "processing_time": 0.82
    },
    {
      "status": "success",
      "conclusion_missing": "",
      "processing_time": 0.76
    }
  ],
  "summary": {
    "total": 2,
    "success": 2,
    "failed": 0,
    "avg_processing_time": 0.79,
    "total_processing_time": 1.58
  }
}
```

## 错误处理

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 422 | 请求参数验证失败 |
| 500 | 服务器内部错误 |

### 错误响应示例

```json
{
  "detail": "质控处理失败: 报告内容为空"
}
```

## 前端对接示例

### Vue.js / React

```javascript
// API 封装
const API_BASE = 'http://localhost:8000';

async function checkReport(reportData) {
  const response = await fetch(`${API_BASE}/api/v1/quality/check`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(reportData),
  });
  
  if (!response.ok) {
    throw new Error(`请求失败: ${response.status}`);
  }
  
  return await response.json();
}

// 使用示例
const result = await checkReport({
  ConclusionStr: '双肺未见异常。',
  ReportStr: '双肺纹理增多。',
  modality: 'DR',
  StudyPart: '胸部',
  Sex: '男'
});

// 检查结果
if (result.conclusion_missing) {
  console.warn('结论缺失:', result.conclusion_missing);
}

if (result.Critical_value.length > 0) {
  console.error('危急值:', result.Critical_value);
}
```

### 错误处理

```javascript
async function safeCheckReport(reportData) {
  try {
    const result = await checkReport(reportData);
    return { success: true, data: result };
  } catch (error) {
    console.error('质控检查失败:', error);
    return { success: false, error: error.message };
  }
}
```

## 性能优化建议

### 1. 大批量检查

对于大批量报告检查，建议使用 `/api/v1/quality/check/batch` 端点，而不是并发调用单条接口。

### 2. LLM 开关

- **标准模式** (`use_llm: true`)：质量高，适合最终审核
- **快速模式** (`use_llm: false` 或 `/fast` 端点)：速度快，适合初筛

### 3. 超时设置

建议前端设置合理的超时时间：
- 单条检查：5-10 秒
- 批量检查（10条）：30-60 秒

## 常见问题

### Q: 如何处理跨域问题？

A: API 服务已默认启用 CORS，允许所有来源访问。如需限制，修改 `api_server.py`：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # 限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Q: 如何关闭 LLM 验证？

A: 两种方式：
1. 请求参数：`"use_llm": false`
2. 使用 `/api/v1/quality/check/fast` 端点
3. 环境变量：`USE_LLM_VALIDATION=false`

### Q: 响应中的空字符串表示什么？

A: 空字符串 `""` 表示该项未检测到问题，有内容时表示检测到的具体问题。
