import json
import os
import glob
from pathlib import Path
import warnings

import pandas as pd
import ast
import json as _json
from tqdm import tqdm
import re
from collections import Counter
import random
MAX_SAMPLES_PER_PART = 800
warnings.filterwarnings("ignore")
# project repo root (two levels up from this file)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'output'
def merge_excel_files(corpus_dir: str = None) -> pd.DataFrame:
    """Read and merge Excel files named like all_data_match*.xlsx under corpus_dir.

    If `corpus_dir` is None, resolve it relative to the project root based on this
    script's location (two levels up -> project root, then `radiology_data`). This
    makes the function behave the same regardless of current working directory.

    Optimizations:
    - Only read needed columns to reduce memory and parsing time.
    - Force '影像号' to string and strip whitespace so lookups match JSON values.
    - Drop duplicate '影像号' keeping the first occurrence.
    """
    # resolve default corpus_dir relative to this script to avoid cwd-dependency
    if corpus_dir is None:
        corpus_dir = '~/work/python/Radiology_Entities/radiology_data/radiology_data'
    pattern = os.path.join(corpus_dir, "all_data_match*.xlsx")
    excel_files = glob.glob(pattern)

    if not excel_files:
        raise ValueError(f"未找到任何符合模式的Excel文件: {pattern}")

    # usecols = ['影像号', '部位', '标准化部位', '部位详情']
    dataframes = []
    for file in excel_files:
        try:
            df = pd.read_excel(file,  dtype={'影像号': str})
            dataframes.append(df)
            print(f"Read file {file} with {len(df)} records")
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    if not dataframes:
        raise ValueError("未找到任何有效的Excel文件（或读取失败）")

    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Finished merging {len(excel_files)} files; total {len(combined_df)} ")
    # 返回并保留影像号列（可选构建索引）
    return combined_df