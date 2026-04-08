#!/bin/bash
# 模型文件解压脚本
# 使用前请先下载 model/*.npy.gz 文件，然后运行此脚本解压

echo "解压模型文件..."

cd "$(dirname "$0")/model" || exit 1

if [ -f "finetuned_word2vec.m.wv.vectors.npy.gz" ]; then
    echo "解压 finetuned_word2vec.m.wv.vectors.npy.gz..."
    gunzip -k finetuned_word2vec.m.wv.vectors.npy.gz
fi

if [ -f "finetuned_word2vec.m.syn1neg.npy.gz" ]; then
    echo "解压 finetuned_word2vec.m.syn1neg.npy.gz..."
    gunzip -k finetuned_word2vec.m.syn1neg.npy.gz
fi

echo "模型文件解压完成！"
echo ""
echo "解压后的文件列表:"
ls -lh *.npy 2>/dev/null || echo "没有找到 .npy 文件"
