#!/usr/bin/env python3
"""
分层语法检测器 v3.0 - 支持多线程LLM调用

架构：
    第一层: 快速召回（多策略并行，高召回率）
    └── 输出: 可疑片段列表
    
    第二层: LLM精校（多线程批量验证）
    └── 输出: 确认的语法错误

性能优化：
- 多线程LLM调用（默认8线程）
- 批量验证（每批5个片段）
- 异步支持（async/await）

使用：
    detector = LayeredGrammarDetector(
        model_dir='grammer/models',
        llm_service=llm_service,
        llm_workers=8  # LLM并发数
    )
    
    # 同步调用
    errors = detector.detect(text, use_llm=True)
    
    # 异步调用（推荐）
    errors = await detector.detect_async(text)
"""

import os
import sys
import re
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# 添加项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 导入快速检测层
try:
    from grammer.fast_recover import FastRecoverDetector, SuspiciousFragment
except ImportError:
    from fast_recover import FastRecoverDetector, SuspiciousFragment


@dataclass
class GrammarError:
    """确认的语法错误"""
    text: str
    position: Tuple[int, int]
    suggestion: str
    error_type: str
    confidence: float
    reason: str
    llm_verified: bool


class LLMGrammarValidator:
    """
    LLM语法验证层 - 支持多线程批量验证
    
    优化策略：
    1. 批量验证：一次验证多个片段
    2. 多线程并发：并行调用LLM
    3. 本地缓存：避免重复验证相同片段
    """
    
    # 批量验证Prompt
    BATCH_PROMPT = """你是一位医学报告编辑专家。请判断以下医学影像报告中的可疑片段是否确实存在语法错误或错别字。

【完整报告】
{full_text}

【可疑片段列表】
{suspicious_list}

【判断标准】
1. 是否存在错别字？（如"肺文里"应为"肺纹理"）
2. 是否存在语法不通顺？
3. 是否使用了非标准医学表述？
4. 注意：有些可能是医学专业术语，不要误判

请对每个片段进行判断，以JSON数组格式输出：
[
    {{
        "index": 0,
        "is_error": true/false,
        "error_type": "错别字/语法错误/标点错误/正常",
        "suggestion": "修正建议",
        "confidence": 0.95,
        "reason": "判断理由"
    }},
    ...
]
"""
    
    def __init__(self, llm_client=None, num_workers: int = 8, batch_size: int = 5):
        """
        Args:
            llm_client: LLM客户端（如OpenAI client）
            num_workers: LLM并发线程数
            batch_size: 每批验证的片段数
        """
        self.llm = llm_client
        self.num_workers = num_workers
        self.batch_size = batch_size
        self._cache = {}  # 本地缓存
        self._cache_lock = threading.Lock()
    
    def _get_cache_key(self, text: str, fragment_text: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(f"{text}:{fragment_text}".encode()).hexdigest()[:16]
    
    def _check_cache(self, text: str, fragment: SuspiciousFragment) -> Optional[GrammarError]:
        """检查缓存"""
        key = self._get_cache_key(text, fragment.text)
        with self._cache_lock:
            return self._cache.get(key)
    
    def _save_cache(self, text: str, fragment: SuspiciousFragment, error: GrammarError):
        """保存到缓存"""
        key = self._get_cache_key(text, fragment.text)
        with self._cache_lock:
            self._cache[key] = error
    
    def validate(self, text: str, fragments: List[SuspiciousFragment]) -> List[GrammarError]:
        """
        批量验证可疑片段 - 多线程版本
        
        Args:
            text: 完整文本
            fragments: 可疑片段列表
        
        Returns:
            确认的语法错误列表
        """
        if not fragments:
            return []
        
        # 检查缓存
        uncached_fragments = []
        cached_errors = []
        
        for fragment in fragments:
            cached = self._check_cache(text, fragment)
            if cached:
                cached_errors.append(cached)
            else:
                uncached_fragments.append(fragment)
        
        if not uncached_fragments:
            return cached_errors
        
        # 分批处理
        batches = []
        for i in range(0, len(uncached_fragments), self.batch_size):
            batch = uncached_fragments[i:i + self.batch_size]
            batches.append((text, batch, i))
        
        print(f"LLM验证: {len(fragments)}个片段 -> {len(batches)}批（并发{self.num_workers}）")
        
        # 多线程并行验证
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(self._validate_batch, *batch): batch 
                for batch in batches
            }
            
            # 收集结果
            for future in as_completed(future_to_batch):
                batch_errors = future.result()
                errors.extend(batch_errors)
                
                # 保存到缓存
                batch = future_to_batch[future]
                for err in batch_errors:
                    # 找到对应的fragment并缓存
                    for frag in batch[1]:
                        if frag.text == err.text:
                            self._save_cache(text, frag, err)
                            break
        
        # 合并缓存结果
        errors.extend(cached_errors)
        
        return errors
    
    async def validate_async(self, text: str, 
                           fragments: List[SuspiciousFragment]) -> List[GrammarError]:
        """异步批量验证"""
        if not fragments:
            return []
        
        # 分批
        batches = []
        for i in range(0, len(fragments), self.batch_size):
            batch = fragments[i:i + self.batch_size]
            batches.append((text, batch, i))
        
        # 并发执行
        tasks = [self._validate_batch_async(*batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        # 合并结果
        errors = []
        for batch_errors in results:
            errors.extend(batch_errors)
        
        return errors
    
    def _validate_batch(self, text: str, batch: List[SuspiciousFragment], 
                       start_index: int) -> List[GrammarError]:
        """验证一批片段（同步）"""
        # 构造可疑片段列表
        suspicious_list = []
        for i, frag in enumerate(batch):
            suspicious_list.append(
                f"[{i}] \"{frag.text}\" (位置: {frag.position}, "
                f"检测策略: {frag.strategy}, 可疑理由: {frag.reason})"
            )
        
        prompt = self.BATCH_PROMPT.format(
            full_text=text[:500],  # 限制长度
            suspicious_list='\n'.join(suspicious_list)
        )
        
        # 如果没有LLM客户端，使用启发式规则
        if self.llm is None:
            return self._heuristic_validate(text, batch, start_index)
        
        # 实际LLM调用
        try:
            # TODO: 替换为实际LLM调用
            # response = self.llm.chat.completions.create(...)
            # return self._parse_llm_response(response, batch)
            return self._heuristic_validate(text, batch, start_index)
        except Exception as e:
            print(f"LLM验证失败: {e}")
            return self._heuristic_validate(text, batch, start_index)
    
    async def _validate_batch_async(self, text: str, batch: List[SuspiciousFragment],
                                    start_index: int) -> List[GrammarError]:
        """验证一批片段（异步）"""
        # 目前使用同步方法的包装
        # 如果有支持async的LLM客户端，可以直接调用
        return self._validate_batch(text, batch, start_index)
    
    def _heuristic_validate(self, text: str, batch: List[SuspiciousFragment],
                           start_index: int) -> List[GrammarError]:
        """启发式验证（无LLM时的回退）"""
        errors = []
        
        # 常见错别字映射（核心错误）
        common_typos = {
            '文里': ('纹理', '错别字'),
            '低密渡': ('低密度', '音近错误'),
            '末见': ('未见', '形近错误'),
            '曾强': ('增强', '形近错误'),
            '影象': ('影像', '习惯错误'),
            '追体': ('椎体', '形近错误'),
            '密谋度': ('密度', '音近错误'),
            '种块': ('肿块', '形近错误'),
            '曾粗': ('增粗', '形近错误'),
            '曾多': ('增多', '形近错误'),
        }
        
        for i, frag in enumerate(batch):
            # 检查是否匹配已知错别字
            matched = False
            for wrong, (correct, err_type) in common_typos.items():
                if wrong in frag.text:
                    errors.append(GrammarError(
                        text=frag.text,
                        position=frag.position,
                        suggestion=frag.text.replace(wrong, correct),
                        error_type=err_type,
                        confidence=0.9,
                        reason=f"匹配常见错别字库: {wrong}->{correct}",
                        llm_verified=False
                    ))
                    matched = True
                    break
            
            # 模式异常通常是真的错误
            if not matched and frag.strategy == 'pattern':
                errors.append(GrammarError(
                    text=frag.text,
                    position=frag.position,
                    suggestion="需要人工检查",
                    error_type='pattern_anomaly',
                    confidence=frag.score,
                    reason=frag.reason,
                    llm_verified=False
                ))
            
            # 高置信度的统计异常也可能是错误
            if not matched and frag.score > 0.9 and '罕见' in frag.reason:
                errors.append(GrammarError(
                    text=frag.text,
                    position=frag.position,
                    suggestion="建议核查",
                    error_type='suspicious',
                    confidence=frag.score * 0.7,
                    reason=frag.reason,
                    llm_verified=False
                ))
        
        return errors
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        with self._cache_lock:
            return {
                'cache_size': len(self._cache),
                'max_size': 10000,  # 可以设置上限
            }


class LayeredGrammarDetector:
    """
    分层语法检测器 - Trigram增强版
    
    整合快速召回层和LLM验证层，支持多线程
    """
    
    def __init__(self, model_dir: str = None, llm_client = None, 
                 llm_workers: int = 8, use_trigram: bool = True):
        """
        Args:
            model_dir: 预训练模型目录
            llm_client: LLM客户端
            llm_workers: LLM并发线程数
            use_trigram: 是否使用trigram（推荐启用）
        """
        self.fast_detector = FastRecoverDetector(model_dir, use_trigram=use_trigram)
        self.llm_validator = LLMGrammarValidator(
            llm_client, 
            num_workers=llm_workers
        )
        
        # 召回层阈值（低阈值=高召回）
        self.recall_threshold = 0.5
        # LLM验证阈值（高阈值=高精度）
        self.precision_threshold = 0.7
        self.use_trigram = use_trigram
    
    def train(self, texts: List[str], output_dir: str):
        """训练快速召回层"""
        self.fast_detector.train(texts, output_dir)
    
    def fast_detect(self, text: str) -> List[SuspiciousFragment]:
        """仅使用快速召回层"""
        return self.fast_detector.detect(text, min_score=self.recall_threshold)
    
    def detect(self, text: str, use_llm: bool = True) -> List[GrammarError]:
        """
        完整检测流程（同步）
        
        Args:
            text: 待检测文本
            use_llm: 是否使用LLM验证
        
        Returns:
            确认的语法错误列表
        """
        # 第一步：快速召回
        suspicious = self.fast_detect(text)
        
        if not suspicious:
            return []
        
        # 如果没有LLM，直接返回启发式结果
        if not use_llm:
            return self.llm_validator._heuristic_validate(text, suspicious, 0)
        
        # 第二步：LLM验证（多线程）
        errors = self.llm_validator.validate(text, suspicious)
        
        # 过滤低置信度
        errors = [e for e in errors if e.confidence >= self.precision_threshold]
        
        return errors
    
    async def detect_async(self, text: str) -> List[GrammarError]:
        """完整检测流程（异步）"""
        suspicious = self.fast_detect(text)
        if not suspicious:
            return []
        
        errors = await self.llm_validator.validate_async(text, suspicious)
        errors = [e for e in errors if e.confidence >= self.precision_threshold]
        
        return errors
    
    def detect_batch(self, texts: List[str], use_llm: bool = True) -> List[List[GrammarError]]:
        """
        批量检测
        
        注意：这会对每个文本分别调用LLM，如需跨文本优化请使用其他方法
        """
        results = []
        for text in texts:
            errors = self.detect(text, use_llm=use_llm)
            results.append(errors)
        return results
    
    async def detect_batch_async(self, texts: List[str]) -> List[List[GrammarError]]:
        """批量检测（异步）"""
        tasks = [self.detect_async(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict:
        """获取检测器统计信息"""
        return {
            'llm_cache': self.llm_validator.get_cache_stats(),
            'recall_threshold': self.recall_threshold,
            'precision_threshold': self.precision_threshold,
        }


def quick_detect(text: str, model_dir: str = None) -> Dict:
    """快速检测入口"""
    detector = LayeredGrammarDetector(model_dir)
    
    # 快速召回
    suspicious = detector.fast_detect(text)
    
    result = {
        "suspicious_count": len(suspicious),
        "suspicious_fragments": [
            {
                "text": f.text,
                "position": f.position,
                "score": f.score,
                "strategy": f.strategy,
                "reason": f.reason
            }
            for f in suspicious[:5]
        ],
        "confirmed_errors": []
    }
    
    # 启发式确认
    errors = detector.detect(text, use_llm=False)
    result["confirmed_errors"] = [
        {
            "text": e.text,
            "suggestion": e.suggestion,
            "type": e.error_type,
            "confidence": e.confidence
        }
        for e in errors
    ]
    
    return result


# 测试
if __name__ == '__main__':
    test_cases = [
        "双肺纹理增粗，未见明显异常密度影。",
        "双肺文里增粗，见低密渡影。",
        "胸部CT扫描未见异常。",
        "肺纹理理理紊乱，追体边缘骨质增生。",
    ]
    
    print("=" * 70)
    print("分层语法检测器 v3.0 测试 - 多线程LLM")
    print("=" * 70)
    
    # 使用多线程LLM验证器
    detector = LayeredGrammarDetector(llm_workers=4)
    
    for text in test_cases:
        print(f"\n原文: {text}")
        
        # 快速召回
        suspicious = detector.fast_detect(text)
        print(f"  快速召回: {len(suspicious)}个可疑片段")
        for f in suspicious[:2]:
            print(f"    - [{f.strategy}] '{f.text}' (score={f.score:.2f})")
        
        # 完整检测（启发式，但使用多线程框架）
        errors = detector.detect(text, use_llm=False)
        if errors:
            print(f"  确认错误:")
            for e in errors:
                print(f"    ⚠ {e.text} -> {e.suggestion} ({e.error_type})")
        else:
            print(f"  ✓ 无确认错误")
    
    # 显示缓存统计
    print("\n" + "=" * 70)
    print("检测器统计:")
    print(detector.get_stats())
