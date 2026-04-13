"""
阶段一：KenLM N-gram 训练 + 放射语料分词词频统计

SSD 时间换空间方案：
- 分块处理 Excel 数据
- 字符轨：生成语料文件（KenLM 训练）
- 词语轨：生成分词词频（策略 B 使用）
"""
import os
import sys
import subprocess
import gc
from pathlib import Path
from glob import glob
from typing import List, Tuple

# 添加 grammer 目录，确保可导入 utils/ssd_processor 等同级模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    DATA_DIR, OUTPUT_DIR, TEMP_DIR,
    RADIOLOGY_CORPUS, RADIOLOGY_NGRAM, RADIOLOGY_VOCAB,
    NGRAM_ORDER, CHUNK_SIZE, ensure_dirs
)
from train.ssd_processor import SSDStreamingProcessor


def find_excel_files(data_dir: str) -> List[Path]:
    """
    查找 Excel 数据文件
    
    Args:
        data_dir: 数据目录
        
    Returns:
        Excel 文件路径列表
    """
    pattern = os.path.join(os.path.expanduser(data_dir), "all_data_match*.xlsx")
    files = sorted(glob(pattern))
    return [Path(f) for f in files]


def train_kenlm_model(corpus_file: str, output_model: str, order: int = NGRAM_ORDER) -> bool:
    """
    训练 KenLM 模型
    
    Args:
        corpus_file: 语料文件路径
        output_model: 输出模型路径
        order: N-gram 阶数
        
    Returns:
        是否成功
    """
    print(f"\n开始训练 KenLM {order}-gram 模型...")
    print(f"输入语料: {corpus_file}")
    print(f"输出模型: {output_model}")
    
    # 检查 kenlm 是否安装
    try:
        import kenlm
    except ImportError:
        print("错误: 未安装 kenlm。请运行: pip install kenlm")
        return False
    
    # 步骤 1: 生成 ARPA 格式模型
    arpa_file = output_model.replace('.klm', '.arpa')
    
    def _build_lmplz_cmd(prune: List[str], use_discount_fallback: bool) -> List[str]:
        cmd = [
            'lmplz',
            '-o', str(order),
            '--prune', *prune,
            '--text', corpus_file,
            '--arpa', arpa_file
        ]
        if use_discount_fallback:
            cmd.append('--discount_fallback')
        return cmd

    # 构建 lmplz 命令
    # --prune: 剪枝，减少模型大小
    # 0 0 1 表示保留所有 1-gram 和 2-gram，对 3-gram 及以上进行剪枝
    prune_args = ['0', '0', '1']
    cmd = _build_lmplz_cmd(prune_args, use_discount_fallback=False)

    print(f"运行命令: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("ARPA 模型生成成功")
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr or ''
        is_bad_discount = 'BadDiscountException' in stderr_text or '--discount_fallback' in stderr_text
        if is_bad_discount:
            retry_cmd = _build_lmplz_cmd(prune_args, use_discount_fallback=True)
            print("检测到 Kneser-Ney 折扣异常，自动重试并启用 --discount_fallback")
            print(f"重试命令: {' '.join(retry_cmd)}")
            try:
                subprocess.run(
                    retry_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                print("ARPA 模型生成成功（使用 --discount_fallback）")
            except subprocess.CalledProcessError as retry_error:
                print(f"ARPA 模型生成失败: {retry_error}")
                print(f"错误输出: {retry_error.stderr}")
                return False
        else:
            print(f"ARPA 模型生成失败: {e}")
            print(f"错误输出: {stderr_text}")
            return False
    except FileNotFoundError:
        print("错误: 未找到 lmplz 命令。请确保 kenlm 已正确安装")
        print("安装方法: pip install kenlm")
        return False
    
    # 步骤 2: 转换为二进制格式（更快加载）
    cmd = [
        'build_binary',
        arpa_file,
        output_model
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print("二进制模型生成成功")
    except subprocess.CalledProcessError as e:
        print(f"二进制模型生成失败: {e}")
        print(f"错误输出: {e.stderr}")
        # ARPA 格式也可以用，只是加载慢一些
        print("将使用 ARPA 格式模型")
        output_model = arpa_file
    except FileNotFoundError:
        print("警告: 未找到 build_binary 命令，将使用 ARPA 格式")
        output_model = arpa_file
    
    # 检查输出文件
    if Path(output_model).exists():
        size_mb = Path(output_model).stat().st_size / 1024 / 1024
        print(f"模型文件大小: {size_mb:.1f} MB")
        return True
    else:
        print(f"错误: 模型文件未生成")
        return False


def train_kenlm(
    data_dir: str = None,
    output_model: str = None,
    output_vocab: str = None,
    prepared_corpus: str = None,
    prepared_vocab: str = None,
    text_columns: List[str] = None,
    ngram_order: int = None,
    chunk_size: int = None
) -> Tuple[str, str]:
    """
    主训练函数
    
    Args:
        data_dir: 数据目录
        output_model: 输出模型路径
        output_vocab: 输出词频路径
        text_columns: 文本列名列表
        ngram_order: N-gram 阶数
        chunk_size: 分块大小
        
    Returns:
        (model_file, vocab_file)
    """
    # 默认参数
    data_dir = data_dir or DATA_DIR
    output_model = output_model or str(RADIOLOGY_NGRAM)
    output_vocab = output_vocab or str(RADIOLOGY_VOCAB)
    text_columns = text_columns or ['描述', '结论']
    ngram_order = ngram_order or NGRAM_ORDER
    chunk_size = chunk_size or CHUNK_SIZE
    
    # 确保输出目录存在
    ensure_dirs()
    
    print("=" * 70)
    print("阶段一：KenLM N-gram 训练 + 分词词频统计")
    print("=" * 70)

    if prepared_corpus:
        print(f"使用准备好的语料: {prepared_corpus}")
        print(f"N-gram 阶数: {ngram_order}")
        if prepared_vocab:
            print(f"使用准备好的词频文件: {prepared_vocab}")
        success = train_kenlm_model(prepared_corpus, output_model, ngram_order)
        if success:
            print("\n" + "=" * 70)
            print("阶段一完成！")
            print("=" * 70)
            print(f"KenLM 模型: {output_model}")
            if prepared_vocab:
                print(f"词频统计: {prepared_vocab}")
            return output_model, prepared_vocab

        print("\n" + "=" * 70)
        print("阶段一失败！")
        print("=" * 70)
        return None, prepared_vocab

    print(f"数据目录: {data_dir}")
    print(f"N-gram 阶数: {ngram_order}")
    print(f"分块大小: {chunk_size}")
    print(f"内存模式: SSD 时间换空间（低内存占用）")
    print("=" * 70)
    
    # 查找 Excel 文件
    excel_files = find_excel_files(data_dir)
    
    if not excel_files:
        print(f"错误: 未找到 Excel 文件: {data_dir}/all_data_match*.xlsx")
        return None, None
    
    print(f"找到 {len(excel_files)} 个 Excel 文件")
    
    # 使用 SSD 流式处理
    print("\n使用 SSD 流式处理器（内存友好）...")
    processor = SSDStreamingProcessor(chunk_size=chunk_size)
    
    corpus_file, vocab_file = processor.process_from_excel_streaming(
        excel_files,
        text_columns=text_columns
    )
    
    if not corpus_file or not vocab_file:
        print("错误: 语料生成失败")
        return None, None
    
    print("\n语料和词频生成完成！")
    print(f"  语料文件: {corpus_file}")
    print(f"  词频文件: {vocab_file}")
    
    # 训练 KenLM
    success = train_kenlm_model(corpus_file, output_model, ngram_order)
    
    if success:
        print("\n" + "=" * 70)
        print("阶段一完成！")
        print("=" * 70)
        print(f"KenLM 模型: {output_model}")
        print(f"词频统计: {vocab_file}")
        return output_model, vocab_file
    else:
        print("\n" + "=" * 70)
        print("阶段一部分完成！")
        print("=" * 70)
        print("注意: KenLM 模型训练失败（lmplz 命令不可用）")
        print("但词频文件已成功生成，可以继续阶段二！")
        print("")
        print("解决方案（可选）：")
        print("  1. 安装 kenlm 二进制工具:")
        print("     apt-get install kenlm  # Ubuntu/Debian")
        print("     或从源码编译: https://github.com/kpu/kenlm")
        print("  2. 继续使用词频文件进行阶段二（推荐）")
        print("")
        print(f"词频统计: {vocab_file}")
        return None, vocab_file


# ==================== 命令行入口 ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='阶段一：KenLM N-gram 训练 + 分词词频统计（SSD 低内存模式）'
    )
    parser.add_argument('--data-dir', default=DATA_DIR,
                       help='Excel 数据目录')
    parser.add_argument('--output-model', default=str(RADIOLOGY_NGRAM),
                       help='输出 KenLM 模型路径')
    parser.add_argument('--output-vocab', default=str(RADIOLOGY_VOCAB),
                       help='输出词频文件路径')
    parser.add_argument('--prepared-corpus',
                       help='直接使用准备好的语料文件训练 KenLM，跳过 Excel 扫描')
    parser.add_argument('--prepared-vocab',
                       help='与 prepared-corpus 对应的词频文件路径')
    parser.add_argument('--ngram-order', type=int, default=NGRAM_ORDER,
                       help=f'N-gram 阶数（默认{NGRAM_ORDER}）')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                       help=f'分块大小（默认{CHUNK_SIZE}）')
    parser.add_argument('--memory-mode', choices=['ssd', 'ram'], default='ssd',
                       help='内存模式：ssd（低内存）或 ram（全内存）')
    
    args = parser.parse_args()
    
    model_file, vocab_file = train_kenlm(
        data_dir=args.data_dir,
        output_model=args.output_model,
        output_vocab=args.output_vocab,
        prepared_corpus=args.prepared_corpus,
        prepared_vocab=args.prepared_vocab,
        ngram_order=args.ngram_order,
        chunk_size=args.chunk_size
    )
    
    if model_file and vocab_file:
        print("\n训练成功！")
        sys.exit(0)
    elif vocab_file:
        print("\n词频生成成功，但 KenLM 训练失败")
        sys.exit(1)
    else:
        print("\n训练失败")
        sys.exit(1)
