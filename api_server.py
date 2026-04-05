"""
医学影像报告质控 API 服务（支持离线部署）
基于 FastAPI 封装 Report_Quality 功能
"""
import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 导入质控核心
from NLP_analyze import Report, Report_Quality

# 静态文件路径
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
SWAGGER_UI_DIR = os.path.join(STATIC_DIR, "swagger-ui")

# 检查离线资源是否可用
OFFLINE_DOCS_AVAILABLE = os.path.exists(os.path.join(SWAGGER_UI_DIR, "swagger-ui-bundle.js"))

def create_app() -> FastAPI:
    """创建 FastAPI 应用（支持离线文档）"""
    
    if OFFLINE_DOCS_AVAILABLE:
        # 使用离线文档
        app = FastAPI(
            title="医学影像报告质控 API",
            description="基于 NLP + LLM 的医学影像报告智能质控系统",
            version="2.0.0",
            docs_url=None,  # 禁用默认文档，使用自定义
            redoc_url=None
        )
        
        # 挂载静态文件
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        
        # 自定义文档端点
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            from fastapi.openapi.docs import get_swagger_ui_html
            return get_swagger_ui_html(
                openapi_url="/openapi.json",
                title="医学影像报告质控 API - 文档",
                swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
                swagger_css_url="/static/swagger-ui/swagger-ui.css",
                swagger_favicon_url="/static/swagger-ui/favicon-32x32.png"
            )
        
        @app.get("/redoc", include_in_schema=False)
        async def redoc_html():
            from fastapi.openapi.docs import get_redoc_html
            # Redoc 仍然需要在线资源，提供备用方案
            return get_redoc_html(
                openapi_url="/openapi.json",
                title="医学影像报告质控 API - ReDoc"
            )
    else:
        # 在线模式（依赖CDN）
        app = FastAPI(
            title="医学影像报告质控 API",
            description="基于 NLP + LLM 的医学影像报告智能质控系统",
            version="2.0.0"
        )
    
    return app

# 创建应用
app = create_app()

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 数据模型
class QualityCheckRequest(BaseModel):
    """质控检查请求"""
    ConclusionStr: str = Field(
        ..., 
        description="报告结论",
        example="1.双肺纹理增多。2.肝脏低密度灶。"
    )
    ReportStr: str = Field(
        ..., 
        description="报告描述",
        example="双肺纹理增多、紊乱。肝实质内见低密度灶。"
    )
    modality: str = Field(
        ..., 
        description="设备类型 (CT/MR/DR/DX/MG)",
        example="CT"
    )
    StudyPart: str = Field(
        ..., 
        description="检查条目名称",
        example="胸部/肺平扫,CT上腹部平扫"
    )
    Sex: str = Field(
        ..., 
        description="患者性别 (男/女)",
        example="女"
    )
    applyTable: Optional[str] = Field(
        default="",
        description="申请单信息（既往史+临床症状+主诉+现病史）",
        example=""
    )
    use_llm: bool = Field(
        default=True,
        description="是否启用 LLM 后置精筛"
    )


class QualityCheckResponse(BaseModel):
    """质控检查响应"""
    partmissing: str = Field(description="部位缺失检测结果")
    partinverse: str = Field(description="检查项目方位错误")
    special_missing: str = Field(description="特殊检查缺失")
    conclusion_missing: str = Field(description="结论与描述不符")
    orient_error: str = Field(description="结论与描述方位不符")
    contradiction: str = Field(description="语言矛盾检测")
    sex_error: str = Field(description="性别错误检测")
    measure_unit_error: str = Field(description="测量单位错误")
    none_standard_term: str = Field(description="术语规范性检查")
    RADS: str = Field(description="RADS分类检查")
    Critical_value: List[dict] = Field(description="危急值列表")
    apply_orient: str = Field(description="申请单位方位错误")
    
    # 元数据
    processing_time: Optional[float] = Field(
        default=None, 
        description="处理耗时（秒）"
    )
    llm_validated: bool = Field(
        default=False,
        description="是否经过 LLM 验证"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    llm_available: bool
    offline_docs: bool


# API 端点
@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 服务状态检查"""
    # 检查 LLM 可用性
    llm_available = False
    try:
        from llm_service import get_llm_validator
        validator = get_llm_validator()
        llm_available = validator.available()
    except:
        pass
    
    return HealthResponse(
        status="running",
        version="2.0.0",
        llm_available=llm_available,
        offline_docs=OFFLINE_DOCS_AVAILABLE
    )


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/v1/quality/check", response_model=QualityCheckResponse)
async def check_quality(request: QualityCheckRequest):
    """
    执行报告质量检查
    
    - 使用规则引擎快速检测
    - 可选 LLM 后置精筛（降低假阳性）
    - 返回完整的质控结果
    """
    import time
    start_time = time.time()
    
    try:
        # 构建 Report 对象
        report = Report(
            ConclusionStr=request.ConclusionStr,
            ReportStr=request.ReportStr,
            modality=request.modality,
            StudyPart=request.StudyPart,
            Sex=request.Sex,
            applyTable=request.applyTable
        )
        
        # 执行质控（同步调用）
        result = Report_Quality(report, debug=False)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 检查是否使用了 LLM
        llm_validated = (
            request.use_llm and 
            os.getenv('USE_LLM_VALIDATION', 'true').lower() == 'true'
        )
        
        # 构建响应
        return QualityCheckResponse(
            **result,
            processing_time=processing_time,
            llm_validated=llm_validated
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"质控处理失败: {str(e)}"
        )


@app.post("/api/v1/quality/check/fast")
async def check_quality_fast(request: QualityCheckRequest):
    """
    快速质量检查（禁用 LLM，纯规则引擎）
    
    - 响应更快（约 1 秒）
    - 假阳性率较高
    - 适合大批量快速筛查
    """
    import time
    start_time = time.time()
    
    try:
        # 临时禁用 LLM
        original_use_llm = os.getenv('USE_LLM_VALIDATION')
        os.environ['USE_LLM_VALIDATION'] = 'false'
        
        report = Report(
            ConclusionStr=request.ConclusionStr,
            ReportStr=request.ReportStr,
            modality=request.modality,
            StudyPart=request.StudyPart,
            Sex=request.Sex,
            applyTable=request.applyTable
        )
        
        result = Report_Quality(report, debug=False)
        processing_time = time.time() - start_time
        
        # 恢复 LLM 设置
        if original_use_llm:
            os.environ['USE_LLM_VALIDATION'] = original_use_llm
        
        return QualityCheckResponse(
            **result,
            processing_time=processing_time,
            llm_validated=False
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"质控处理失败: {str(e)}"
        )


@app.get("/api/v1/config")
async def get_config():
    """获取当前配置信息"""
    return {
        "llm_validation": os.getenv('USE_LLM_VALIDATION', 'true').lower() == 'true',
        "llm_model": os.getenv('LLM_MODEL', 'qwen3'),
        "llm_base_url": os.getenv('LLM_BASE_URL'),
        "confidence_threshold": float(os.getenv('LLM_CONFIDENCE_THRESHOLD', '0.7')),
        "offline_docs": OFFLINE_DOCS_AVAILABLE
    }


# 离线资源下载说明
if not OFFLINE_DOCS_AVAILABLE:
    print("警告: 离线文档资源不存在，将使用 CDN 资源")
    print(f"如需离线部署，请下载以下文件到 {SWAGGER_UI_DIR}:")
    print("  - swagger-ui-bundle.js")
    print("  - swagger-ui.css")
    print("  - favicon-32x32.png")
    print("")
    print("下载命令:")
    print("  curl -L 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js' -o static/swagger-ui/swagger-ui-bundle.js")
    print("  curl -L 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css' -o static/swagger-ui/swagger-ui.css")
    print("  curl -L 'https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/favicon-32x32.png' -o static/swagger-ui/favicon-32x32.png")


# 启动服务
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    print(f"启动医学影像报告质控 API 服务...")
    print(f"访问地址: http://{host}:{port}")
    if OFFLINE_DOCS_AVAILABLE:
        print(f"离线文档: http://{host}:{port}/docs")
    else:
        print(f"在线文档: http://{host}:{port}/docs (依赖CDN)")
    
    uvicorn.run(app, host=host, port=port)
