#!/usr/bin/env python3
"""
医学影像报告质控测试脚本
用于批量测试 samples.xlsx 中的样本，并生成对比报告

使用方法:
    python run_samples_test.py                    # 运行所有测试
    python run_samples_test.py --verbose          # 显示详细输出
    python run_samples_test.py --tag 部位缺失      # 只测试特定标签
    python run_samples_test.py --output result.xlsx # 指定输出文件名
"""

import pandas as pd
import argparse
import sys
import re
from datetime import datetime
from NLP_analyze import Report, Report_Quality


def normalize_text(text):
    """标准化文本以便比较"""
    if text is None:
        return ""
    text = str(text)
    # 移除多余空格、换行符
    text = re.sub(r'\s+', '', text)
    # 统一标点
    text = text.replace('，', ',').replace('。', '.').replace('；', ';')
    return text.lower()


def determine_field(expected):
    """根据期望输出确定对应的QC字段"""
    if '漏写部位' in expected:
        return 'QC_partmissing'
    elif '检查项目方位' in expected:
        return 'QC_partinverse'
    elif '检查方式' in expected:
        return 'QC_special_missing'
    elif '测量值' in expected:
        return 'QC_measure_unit_error'
    elif '术语' in expected:
        return 'QC_none_standard_term'
    elif '危急值' in expected:
        return 'QC_Critical_value'
    elif '性别' in expected:
        return 'QC_sex_error'
    elif '语言矛盾' in expected:
        return 'QC_contradiction'
    elif 'RADS' in expected:
        return 'QC_RADS'
    elif '结论' in expected and '方位' in expected:
        return 'QC_orient_error'
    elif '描述与结论' in expected or '结论' in expected:
        return 'QC_conclusion_missing'
    elif '申请单' in expected:
        return 'QC_apply_orient'
    else:
        return None


def is_match(expected, actual, field):
    """判断期望和实际是否匹配"""
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    
    # 如果完全相同
    if expected_norm == actual_norm:
        return True
    
    # 如果期望包含在实际中（部分匹配）
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return True
    
    # 危急值特殊处理：检查是否包含关键信息
    if field == 'QC_Critical_value' and 'category' in actual:
        return True
    
    return False


def run_test(row, verbose=False):
    """运行单个测试"""
    try:
        # 处理 NaN 值
        applyTable = row['applyTable'] if pd.notna(row['applyTable']) else ''
        ReportStr = row['ReportStr'] if pd.notna(row['ReportStr']) else ''
        ConclusionStr = row['ConclusionStr'] if pd.notna(row['ConclusionStr']) else ''
        StudyPart = row['StudyPart'] if pd.notna(row['StudyPart']) else ''
        Sex = row['Sex'] if pd.notna(row['Sex']) else ''
        modality = row['modality'] if pd.notna(row['modality']) else ''
        
        # 创建 Report 对象
        report = Report(
            ConclusionStr=ConclusionStr,
            ReportStr=ReportStr,
            modality=modality,
            StudyPart=StudyPart,
            Sex=Sex,
            applyTable=applyTable
        )
        
        # 运行质控
        result = Report_Quality(report)
        
        # 确定期望输出对应的字段
        expected = str(row['输出'])
        field = determine_field(expected)
        
        if field is None:
            return {
                'success': False,
                'error': f'无法识别期望输出类型: {expected[:50]}',
                'result': result
            }
        
        actual = str(result.get(field, ''))
        match = is_match(expected, actual, field)
        
        if verbose:
            print(f"  字段: {field}")
            print(f"  期望: {expected[:80]}")
            print(f"  实际: {actual[:80]}")
            print(f"  结果: {'✓ 通过' if match else '✗ 失败'}")
        
        return {
            'success': True,
            'match': match,
            'field': field,
            'expected': expected,
            'actual': actual,
            'result': result
        }
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"  错误: {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'result': {}
        }


def main():
    parser = argparse.ArgumentParser(description='医学影像报告质控批量测试')
    parser.add_argument('--input', default='samples.xlsx', help='输入Excel文件路径 (默认: samples.xlsx)')
    parser.add_argument('--output', default=None, help='输出Excel文件路径 (默认: samples_test_YYYYMMDD_HHMMSS.xlsx)')
    parser.add_argument('--tag', default=None, help='只测试指定标签的样本 (如: 部位缺失)')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细输出')
    parser.add_argument('--limit', type=int, default=None, help='限制测试数量')
    
    args = parser.parse_args()
    
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
    stats = {
        'total': total,
        'passed': 0,
        'failed': 0,
        'error': 0,
        'by_tag': {}
    }
    
    for idx, row in df.iterrows():
        tag = row['tag']
        print(f"[{idx+1}/{total}] {tag} ...", end=' ')
        
        if args.verbose:
            print()
        
        test_result = run_test(row, verbose=args.verbose)
        
        if not test_result['success']:
            stats['error'] += 1
            status = '错误'
            match = False
        elif test_result['match']:
            stats['passed'] += 1
            status = '通过'
            match = True
        else:
            stats['failed'] += 1
            status = '失败'
            match = False
        
        if not args.verbose:
            print(status)
        else:
            print()
        
        # 统计按标签
        if tag not in stats['by_tag']:
            stats['by_tag'][tag] = {'total': 0, 'passed': 0, 'failed': 0, 'error': 0}
        stats['by_tag'][tag]['total'] += 1
        if not test_result['success']:
            stats['by_tag'][tag]['error'] += 1
        elif test_result['match']:
            stats['by_tag'][tag]['passed'] += 1
        else:
            stats['by_tag'][tag]['failed'] += 1
        
        # 收集结果
        result_row = {
            'index': idx + 1,
            'tag': tag,
            'match': '✓' if match else '✗',
            'status': status,
            'expected': test_result.get('expected', ''),
            'actual': test_result.get('actual', ''),
            'field': test_result.get('field', ''),
            'error': test_result.get('error', '')
        }
        
        # 添加所有QC结果字段
        if 'result' in test_result:
            for key, value in test_result['result'].items():
                result_row[f'QC_{key}'] = value
        
        results.append(result_row)
    
    # 生成报告
    print("\n" + "=" * 80)
    print("测试报告")
    print("=" * 80)
    
    print(f"\n总体统计:")
    print(f"  总数:   {stats['total']}")
    print(f"  通过:   {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"  失败:   {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"  错误:   {stats['error']} ({stats['error']/stats['total']*100:.1f}%)")
    
    print(f"\n按类型统计:")
    for tag, tag_stats in sorted(stats['by_tag'].items()):
        pass_rate = tag_stats['passed'] / tag_stats['total'] * 100 if tag_stats['total'] > 0 else 0
        print(f"  {tag:15s}: {tag_stats['passed']}/{tag_stats['total']} ({pass_rate:5.1f}%)", end='')
        if tag_stats['error'] > 0:
            print(f" [错误: {tag_stats['error']}]")
        else:
            print()
    
    # 保存结果
    result_df = pd.DataFrame(results)
    
    # 合并原始数据
    for col in df.columns:
        if col not in result_df.columns:
            result_df[col] = df[col].values
    
    # 调整列顺序
    first_cols = ['index', 'tag', 'match', 'status', 'expected', 'actual', 'field', 'error']
    other_cols = [c for c in result_df.columns if c not in first_cols]
    result_df = result_df[first_cols + other_cols]
    
    result_df.to_excel(args.output, index=False)
    print(f"\n结果已保存: {args.output}")
    
    # 显示失败的案例
    if stats['failed'] > 0:
        print("\n" + "=" * 80)
        print("失败案例分析")
        print("=" * 80)
        failed_cases = [r for r in results if r['status'] == '失败']
        for case in failed_cases[:10]:  # 只显示前10个
            print(f"\n[{case['index']}] {case['tag']}")
            print(f"  字段: {case['field']}")
            exp = case['expected'][:100] + '...' if len(case['expected']) > 100 else case['expected']
            act = case['actual'][:100] + '...' if len(case['actual']) > 100 else case['actual']
            print(f"  期望: {exp}")
            print(f"  实际: {act}")
        if len(failed_cases) > 10:
            print(f"\n... 还有 {len(failed_cases) - 10} 个失败案例")
    
    # 返回码
    if stats['failed'] > 0 or stats['error'] > 0:
        sys.exit(1)
    else:
        print("\n✓ 所有测试通过！")
        sys.exit(0)


if __name__ == '__main__':
    main()
