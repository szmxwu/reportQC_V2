"""
医学报告错别字检测引擎 v3.0

三策略检测：
    1. 拼音混淆检测（分词后匹配，避免子串误报）
    2. 高危通用词检测（纯中文）
    3. 词序错误检测（高频可靠搭配）

项目结构：
    train/      - 训练相关脚本
    inference/  - 推理/检测脚本
    utils/      - 工具函数和配置
    models/     - 预训练模型文件
    output/     - 检测结果输出

快速开始：
    >>> from inference.medical_typo_detector import MedicalTypoDetector
    >>> detector = MedicalTypoDetector()
    >>> detector.load()
    >>> errors = detector.detect("双肺文里增粗")
    >>> print(errors)
    [{'error': '文里', 'suggestion': '纹理', 'type': 'typo'}]
"""

__version__ = '3.0.0'

from inference.medical_typo_detector import MedicalTypoDetector
from inference.word_order_detector import WordOrderDetector

__all__ = ['MedicalTypoDetector', 'WordOrderDetector']
