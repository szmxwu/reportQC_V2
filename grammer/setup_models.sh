#!/bin/bash
# grammer 模块模型文件解压脚本

cd "$(dirname "$0")/models" || exit 1

echo "解压 grammer 模块模型文件..."

if [ -f "radiology_ngram.klm.gz" ]; then
    if [ ! -f "radiology_ngram.klm" ]; then
        echo "解压 radiology_ngram.klm.gz..."
        gunzip radiology_ngram.klm.gz
        echo "✓ 解压完成"
    else
        echo "✓ radiology_ngram.klm 已存在，跳过解压"
    fi
else
    echo "✗ radiology_ngram.klm.gz 不存在，跳过"
fi

echo ""
echo "模型文件状态:"
ls -lh radiology_ngram.klm 2>/dev/null || echo "  radiology_ngram.klm: 未找到（可选依赖）"
