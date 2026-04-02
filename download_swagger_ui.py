#!/usr/bin/env python3
"""
下载 Swagger UI 静态文件（用于离线部署）
"""
import os
import urllib.request
import ssl

# 忽略 SSL 证书验证（某些环境需要）
ssl._create_default_https_context = ssl._create_unverified_context

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "swagger-ui")
BASE_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5"

FILES = [
    "swagger-ui-bundle.js",
    "swagger-ui.css",
    "favicon-32x32.png"
]

os.makedirs(STATIC_DIR, exist_ok=True)

print("下载 Swagger UI 静态文件...")
print(f"目标目录: {STATIC_DIR}")
print("")

for filename in FILES:
    url = f"{BASE_URL}/{filename}"
    filepath = os.path.join(STATIC_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"  ✓ {filename} 已存在")
        continue
    
    try:
        print(f"  ↓ 下载 {filename}...", end=" ")
        urllib.request.urlretrieve(url, filepath)
        print("完成")
    except Exception as e:
        print(f"失败: {e}")

print("")
print("下载完成！")
print(f"现在可以离线访问 http://localhost:8000/docs")
