#!/usr/bin/env python3
"""
Trigram vs Bigram 对比测试

展示为什么trigram对医学文本更可靠：
1. 更多上下文信息
2. 能区分相似的双字组合
3. 更少的假阳性
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from grammer.fast_recover import CharacterAnomalyDetector


def test_trigram_advantage():
    """测试trigram的优势"""
    
    # 模拟训练数据
    train_texts = [
        "双肺纹理增粗",
        "双肺纹理增多",
        "肺纹理清晰",
        "肺纹理紊乱",
        "纹理正常",
        "胸部CT扫描",
        "胸部CT平扫",
        "CT增强扫描",
    ]
    
    print("=" * 70)
    print("Trigram vs Bigram 对比测试")
    print("=" * 70)
    
    # 训练两个模型
    print("\n训练Bigram模型...")
    bigram_detector = CharacterAnomalyDetector(use_trigram=False)
    bigram_detector.train(train_texts)
    
    print("训练Trigram模型...")
    trigram_detector = CharacterAnomalyDetector(use_trigram=True)
    trigram_detector.train(train_texts)
    
    # 测试用例
    test_cases = [
        ("双肺纹理增粗", "正常表述"),
        ("双肺文里增粗", "错别字：纹理->文里"),
        ("肺文里增多", "错别字：纹理->文里"),
        ("肺纹里增粗", "错别字：理->里"),
        ("胸CT扫描", "正常缩写"),
    ]
    
    print("\n" + "=" * 70)
    print("检测结果对比")
    print("=" * 70)
    
    for text, desc in test_cases:
        print(f"\n测试: '{text}' ({desc})")
        
        # Bigram检测
        bigram_frags = bigram_detector.detect(text)
        bigram_suspicious = len(bigram_frags)
        
        # Trigram检测
        trigram_frags = trigram_detector.detect(text)
        trigram_count = len([f for f in trigram_frags if f.strategy == 'trigram_rarity'])
        bigram_in_trigram = len([f for f in trigram_frags if f.strategy == 'bigram_rarity'])
        
        print(f"  Bigram模型: {bigram_suspicious}个可疑")
        print(f"  Trigram模型: {trigram_count}个trigram可疑 + {bigram_in_trigram}个bigram补充")
        
        # 显示具体的trigram检测
        if trigram_count > 0:
            print(f"    Trigram检测:")
            for f in trigram_frags:
                if f.strategy == 'trigram_rarity':
                    print(f"      - '{f.text}': {f.reason}")


def demonstrate_context_advantage():
    """展示上下文优势"""
    
    print("\n" + "=" * 70)
    print("上下文信息优势示例")
    print("=" * 70)
    
    examples = [
        {
            'text': "肺文里增多",
            'explanation': [
                ("肺文", "罕见bigram，可疑"),
                ("文里", "罕见bigram，可疑"),
                ("肺文里", "罕见trigram，确认错误"),
            ]
        },
        {
            'text': "肺纹理增粗", 
            'explanation': [
                ("肺纹", "常见bigram"),
                ("纹理", "常见bigram"),
                ("肺纹理", "常见trigram，确认正确"),
            ]
        },
    ]
    
    for example in examples:
        print(f"\n文本: '{example['text']}'")
        print("分析:")
        for ngram, desc in example['explanation']:
            ngram_type = "trigram" if len(ngram) == 3 else "bigram"
            print(f"  - '{ngram}' ({ngram_type}): {desc}")


def show_trigram_coverage():
    """展示trigram覆盖的医学术语"""
    
    print("\n" + "=" * 70)
    print("Trigram覆盖的常见医学术语")
    print("=" * 70)
    
    common_terms = [
        "肺纹理", "支气管", "低密度", "高密度",
        "钙化影", "结节影", "条索影", "磨玻璃",
        "增强扫描", "平扫", "占位性", "胸腔积液",
        "心包积液", "主动脉", "冠状动脉", "肺动脉",
    ]
    
    print("\n这些术语通常是固定3字搭配，trigram能有效识别：")
    for i, term in enumerate(common_terms, 1):
        marker = "✓" if len(term) == 3 else " "
        print(f"  {marker} {term}")


if __name__ == '__main__':
    test_trigram_advantage()
    demonstrate_context_advantage()
    show_trigram_coverage()
    
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
Trigram对医学文本更可靠，因为：
1. 医学术语多为固定3字搭配（如"肺纹理增粗"）
2. 能区分相似的双字组合（如"肺文里" vs "肺纹理"）
3. 上下文信息更丰富，假阳性更低
4. 对位置不敏感（错别字在中间也能捕获）

建议：生产环境启用trigram（默认已启用）
    """)
