"""
医学错别字检测器 - 统一入口

封装三阶段流程：
1. KenLM N-gram 训练
2. 双策略黑名单挖掘
3. AC 自动机引擎构建

提供简洁的 API 接口
"""
from pathlib import Path
from typing import List, Dict, Optional, Union

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    RADIOLOGY_NGRAM, RADIOLOGY_VOCAB,
    MEDICAL_CONFUSION, HIGH_RISK_GENERAL,
    WORD_ORDER_TEMPLATES, AC_AUTOMATON,
    ensure_dirs
)
from train.phase3_build_engine_v2 import FastMedicalCorrectorV2
from inference.word_order_detector import WordOrderDetector


class MedicalTypoDetector:
    """
    医学错别字检测器
    
    统一入口类，封装完整的检测流程
    """
    
    def __init__(
        self,
        confusion_path: str = None,
        high_risk_path: str = None,
        lm_path: str = None,
        engine_path: str = None,
        word_order_path: str = None,
        enable_word_order: bool = True
    ):
        """
        初始化检测器
        
        Args:
            confusion_path: 拼音混淆对文件路径
            high_risk_path: 高危通用词文件路径
            lm_path: KenLM 模型路径（可选）
            engine_path: 预构建引擎路径（可选）
            word_order_path: 词序模板文件路径
            enable_word_order: 是否启用词序错误检测
        """
        self.confusion_path = confusion_path or str(MEDICAL_CONFUSION)
        self.high_risk_path = high_risk_path or str(HIGH_RISK_GENERAL)
        self.lm_path = lm_path or str(RADIOLOGY_NGRAM)
        self.engine_path = engine_path or str(AC_AUTOMATON)
        self.word_order_path = word_order_path or str(WORD_ORDER_TEMPLATES)
        self.enable_word_order = enable_word_order
        
        # 内部引擎
        self._corrector = None
        self._word_order_detector = None
        self._lm_loaded = False
    
    def load(self):
        """加载检测引擎"""
        ensure_dirs()
        
        self._corrector = FastMedicalCorrectorV2()
        
        # 优先加载预构建引擎
        engine_file = Path(self.engine_path)
        if engine_file.exists():
            print(f"加载预构建引擎: {self.engine_path}")
            self._corrector.load(self.engine_path)
        else:
            # 从黑名单文件构建
            print("从黑名单文件构建引擎...")
            self._corrector.load_blacklists(
                self.confusion_path,
                self.high_risk_path
            )
            # 保存供下次使用
            self._corrector.save(self.engine_path)
        
        # 加载词序检测器
        if self.enable_word_order and Path(self.word_order_path).exists():
            self._word_order_detector = WordOrderDetector(self.word_order_path)
        
        # 尝试加载 KenLM（可选，用于纠错功能）
        try:
            from pycorrector import Corrector
            if Path(self.lm_path).exists():
                self._lm = Corrector(language_model_path=self.lm_path)
                self._lm_loaded = True
                print(f"KenLM 模型已加载: {self.lm_path}")
            else:
                print(f"KenLM 模型不存在（可选）: {self.lm_path}")
                print("提示: 检测功能仍可正常工作，纠错功能将使用基于规则的修正")
                self._lm_loaded = False
        except Exception as e:
            print(f"KenLM 加载失败（可选）: {e}")
            print("提示: 检测功能仍可正常工作")
            self._lm_loaded = False
    
    def scan(self, text: str) -> List[Dict]:
        """
        快速扫描文本中的错误
        
        O(N) 复杂度，使用 AC 自动机 + 词序检测
        
        Args:
            text: 输入文本
            
        Returns:
            错误列表，每项包含：
            - error: 错误词
            - suggestion: 建议修正（高危词为 null）
            - type: 错误类型 ('typo', 'general_high_risk', 'word_order')
            - position: (start, end) 位置
            - score: 风险分数（仅高危词）或置信度（词序错误）
        """
        if self._corrector is None:
            self.load()
        
        # 1. AC自动机检测（拼音混淆 + 高危词）
        matches = self._corrector.scan(text)
        errors = [
            {
                'error': m.error,
                'suggestion': m.suggestion,
                'type': m.error_type,
                'position': m.position,
                'score': m.score if m.error_type == 'general_high_risk' else None
            }
            for m in matches
        ]
        
        # 2. 词序错误检测
        if self._word_order_detector:
            word_order_errors = self._word_order_detector.detect(text)
            for e in word_order_errors:
                errors.append({
                    'error': e['error'],
                    'suggestion': e['suggestion'],
                    'type': 'word_order',
                    'position': e['position'],
                    'score': e['confidence'],  # 用置信度作为score
                    'forward_freq': e.get('forward_freq'),
                    'backward_freq': e.get('backward_freq')
                })
        
        # 按位置排序
        errors.sort(key=lambda x: x['position'][0])
        
        return errors
    
    def correct(self, text: str) -> Dict:
        """
        纠错：返回修正后的文本
        
        Args:
            text: 输入文本
            
        Returns:
            {
                'source': 原文,
                'target': 修正后文本,
                'errors': [
                    {
                        'error': 错误词,
                        'suggestion': 建议,
                        'type': 类型,
                        'position': (start, end)
                    }
                ]
            }
        """
        if self._corrector is None:
            self.load()
        
        return self._corrector.correct(text)
    
    def detect(self, text: str) -> List[Dict]:
        """
        检测（scan 的别名）
        
        Args:
            text: 输入文本
            
        Returns:
            错误列表
        """
        return self.scan(text)
    
    def batch_detect(self, texts: List[str]) -> List[List[Dict]]:
        """
        批量检测
        
        Args:
            texts: 文本列表
            
        Returns:
            每个文本的错误列表
        """
        if self._corrector is None:
            self.load()
        
        return [self.scan(text) for text in texts]
    
    def get_stats(self) -> Dict:
        """获取检测器统计信息"""
        if self._corrector is None:
            return {'loaded': False}
        
        stats = self._corrector.get_stats()
        stats['lm_loaded'] = self._lm_loaded
        stats['word_order_enabled'] = self._word_order_detector is not None
        if self._word_order_detector:
            stats['word_order_patterns'] = len(self._word_order_detector.word_patterns)
        return stats


# ==================== 便捷函数 ====================

def detect(text: str, **kwargs) -> List[Dict]:
    """
    便捷检测函数
    
    Args:
        text: 输入文本
        **kwargs: 传递给 MedicalTypoDetector 的参数
        
    Returns:
        错误列表
    """
    detector = MedicalTypoDetector(**kwargs)
    return detector.scan(text)


def correct(text: str, **kwargs) -> Dict:
    """
    便捷纠错函数
    
    Args:
        text: 输入文本
        **kwargs: 传递给 MedicalTypoDetector 的参数
        
    Returns:
        纠错结果
    """
    detector = MedicalTypoDetector(**kwargs)
    return detector.correct(text)


# ==================== 完整流程执行 ====================

def run_full_pipeline(
    data_dir: str = None,
    skip_phase1: bool = False,
    skip_phase2: bool = False,
    skip_phase3: bool = False
) -> MedicalTypoDetector:
    """
    执行完整的三阶段流程
    
    Args:
        data_dir: 数据目录
        skip_phase1: 跳过阶段一（如果已有 KenLM 模型）
        skip_phase2: 跳过阶段二（如果已有黑名单）
        skip_phase3: 跳过阶段三（如果已有引擎）
        
    Returns:
        配置好的 MedicalTypoDetector 实例
    """
    from phase1_train_kenlm import train_kenlm
    from phase2_mine_blacklist import mine_blacklist
    from phase3_build_engine import build_engine
    
    print("=" * 70)
    print("医学错别字检测 - 完整流程")
    print("=" * 70)
    
    # 阶段一：KenLM 训练
    if not skip_phase1:
        model_file, vocab_file = train_kenlm(data_dir=data_dir)
        if not vocab_file:
            raise RuntimeError("阶段一失败：无法生成词频文件")
    else:
        print("跳过阶段一（使用现有模型）")
    
    # 阶段二：黑名单挖掘
    if not skip_phase2:
        confusion_file, high_risk_file = mine_blacklist()
        if not confusion_file or not high_risk_file:
            raise RuntimeError("阶段二失败：无法生成黑名单")
    else:
        print("跳过阶段二（使用现有黑名单）")
    
    # 阶段三：引擎构建
    if not skip_phase3:
        build_engine()
    else:
        print("跳过阶段三（使用现有引擎）")
    
    # 创建检测器
    detector = MedicalTypoDetector()
    detector.load()
    
    print("\n" + "=" * 70)
    print("完整流程完成！检测器已就绪")
    print("=" * 70)
    
    return detector


# ==================== 命令行入口 ====================

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='医学错别字检测器'
    )
    parser.add_argument('text', nargs='?', help='要检测的文本')
    parser.add_argument('--scan', action='store_true', help='快速扫描模式')
    parser.add_argument('--correct', action='store_true', help='纠错模式')
    parser.add_argument('--full-pipeline', action='store_true', 
                       help='执行完整三阶段流程')
    parser.add_argument('--data-dir', help='数据目录')
    
    args = parser.parse_args()
    
    if args.full_pipeline:
        # 执行完整流程
        detector = run_full_pipeline(data_dir=args.data_dir)
        
        if args.text:
            result = detector.correct(args.text)
            print(f"\n原文: {result['source']}")
            print(f"修正: {result['target']}")
            print(f"\n发现 {len(result['errors'])} 个错误:")
            for err in result['errors']:
                if err['suggestion']:
                    print(f"  - {err['error']} -> {err['suggestion']} ({err['type']})")
                else:
                    print(f"  - {err['error']} (高危通用词)")
    
    elif args.text:
        # 快速检测
        detector = MedicalTypoDetector()
        detector.load()
        
        if args.correct:
            result = detector.correct(args.text)
            print(f"原文: {result['source']}")
            print(f"修正: {result['target']}")
        else:
            errors = detector.scan(args.text)
            print(f"文本: {args.text}")
            print(f"发现 {len(errors)} 个错误:")
            for err in errors:
                if err['suggestion']:
                    print(f"  - {err['error']} -> {err['suggestion']} @ {err['position']}")
                else:
                    print(f"  - {err['error']} (风险分: {err['score']:.1f}) @ {err['position']}")
    
    else:
        parser.print_help()
    
    sys.exit(0)
