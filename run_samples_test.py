#!/usr/bin/env python3
"""
医学影像报告质控测试脚本
批量运行 samples.xlsx 中的样本，输出质控结果到Excel

使用方法:
    python run_samples_test.py                    # 运行所有样本
    python run_samples_test.py --verbose          # 显示详细输出
    python run_samples_test.py --tag 部位缺失      # 只运行特定标签
    python run_samples_test.py --output result.xlsx # 指定输出文件名
    python run_samples_test.py --no-llm           # 禁用LLM验证

环境变量:
    USE_LLM_VALIDATION=true/false  # 在.env文件中设置，或使用 --no-llm 覆盖
"""

import os

# 尝试加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装，使用系统环境变量

import pandas as pd
import argparse
import sys
from datetime import datetime
from NLP_analyze import Report, Report_Quality


def safe_str(value):
    """将值转换为字符串，空值转为空字符串"""
    if value is None:
        return ''
    if isinstance(value, list):
        if not value:
            return ''
        return '；'.join(str(x) for x in value)
    return str(value)


def run_single(row):
    """运行单个样本的质控检测
    
    Returns:
        (result_dict, error_msg) - 成功时error_msg为空
    """
    try:
        # 处理输入数据
        applyTable = row['applyTable'] if pd.notna(row.get('applyTable')) else ''
        ReportStr = row['ReportStr'] if pd.notna(row.get('ReportStr')) else ''
        ConclusionStr = row['ConclusionStr'] if pd.notna(row.get('ConclusionStr')) else ''
        StudyPart = row['StudyPart'] if pd.notna(row.get('StudyPart')) else ''
        Sex = row['Sex'] if pd.notna(row.get('Sex')) else ''
        modality = row['modality'] if pd.notna(row.get('modality')) else ''
        
        # 创建Report对象并运行质控
        report = Report(
            ConclusionStr=ConclusionStr,
            ReportStr=ReportStr,
            modality=modality,
            StudyPart=StudyPart,
            Sex=Sex,
            applyTable=applyTable
        )
        
        result = Report_Quality(report)
        return result, None
        
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description='医学影像报告质控批量测试')
    parser.add_argument('--input', default='samples.xlsx', help='输入Excel文件路径 (默认: samples.xlsx)')
    parser.add_argument('--output', default=None, help='输出Excel文件路径 (默认: samples_test_YYYYMMDD_HHMMSS.xlsx)')
    parser.add_argument('--tag', default=None, help='只测试指定标签的样本')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细输出',default=True)
    parser.add_argument('--limit', type=int, default=None, help='限制测试数量')
    parser.add_argument('--no-llm', action='store_true', help='禁用LLM验证')
    
    args = parser.parse_args()
    
    # 控制LLM验证开关
    if args.no_llm:
        os.environ['USE_LLM_VALIDATION'] = 'false'
        print("[注意] LLM验证已禁用")
    
    # 设置输出文件名
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'samples_test_{timestamp}.xlsx'
    
    # 读取样本
    print(f"读取测试样本: {args.input}")
    try:
        df = pd.read_excel(args.input)
    except FileNotFoundError:
        print(f"错误: 文件 '{args.input}' 不存在")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 无法读取文件 - {e}")
        sys.exit(1)
    
    # 过滤指定标签
    if args.tag:
        df = df[df['tag'] == args.tag]
        print(f"只测试标签为 '{args.tag}' 的样本")
    
    # 限制数量
    if args.limit:
        df = df.head(args.limit)
    
    total = len(df)
    print(f"总共 {total} 条测试样本\n")
    
    if total == 0:
        print("没有需要测试的样本")
        sys.exit(0)
    
    # 运行测试
    results = []
    error_count = 0
    
    for idx, row in df.iterrows():
        tag = row.get('tag', '')
        print(f"[{idx+1}/{total}] {tag} ...", end=' ')
        
        result, error = run_single(row)
        
        if error:
            print(f"错误: {error[:50]}")
            error_count += 1
            result_row = {'error': error}
        else:
            print("完成")
            # 将所有质控结果转换为字符串
            if result.get('Critical_value'):
                QC_Critical_value=[x['category'] for x in result['Critical_value']]
                QC_Critical_value=';'.join(QC_Critical_value)
            else:
                QC_Critical_value=""
            result_row = {
                'QC_partmissing': safe_str(result.get('partmissing')),
                'QC_partinverse': safe_str(result.get('partinverse')),
                'QC_special_missing': safe_str(result.get('special_missing')),
                'QC_conclusion_missing': safe_str(result.get('conclusion_missing')),
                'QC_orient_error': safe_str(result.get('orient_error')),
                'QC_contradiction': safe_str(result.get('contradiction')),
                'QC_sex_error': safe_str(result.get('sex_error')),
                'QC_measure_unit_error': safe_str(result.get('measure_unit_error')),
                'QC_none_standard_term': safe_str(result.get('none_standard_term')),
                'QC_RADS': safe_str(result.get('RADS')),
                'QC_Critical_value': QC_Critical_value,
                'QC_apply_orient': safe_str(result.get('apply_orient')),
                'error': ''
            }
            
            if args.verbose:
                for key, value in result_row.items():
                    if key != 'error' and value:
                        print(f"  {key}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")
        
        results.append(result_row)
    
    # 构建输出DataFrame
    result_df = pd.DataFrame(results)
    
    # 合并原始数据（原始列放在前面）
    for col in result_df.columns:
        df[f'output_{col}'] = result_df[col].values
    
    # 保存结果
    try:
        df.to_excel(args.output, index=False, na_rep='')
        print(f"\n结果已保存: {args.output}")
        print(f"总计: {total}条, 成功: {total - error_count}条, 错误: {error_count}条")
    except Exception as e:
        print(f"\n保存结果失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
