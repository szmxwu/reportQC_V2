#!/usr/bin/env python3
"""
分层语法检测器 v3.0 - 支持多线程LLM调用和实体感知的短句验证

架构：
    第一层: 快速召回（多策略并行，高召回率）
    └── 输出: 可疑片段列表
    
    第二层: LLM精校（基于实体短句的批量验证）
    └── 输出: 确认的语法错误

关键优化：
- 使用text_extrac_process拆分基于实体的短句
- 只提供可疑片段所在的短句给LLM，节省token
- 多线程LLM调用（默认8线程）
- 本地缓存避免重复验证

使用：
    detector = LayeredGrammarDetector(
        model_dir='grammer/models',
        llm_workers=8
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

# 导入实体提取（用于短句拆分）
try:
    from Extract_Entities import text_extrac_process
    EXTRACT_ENTITIES_AVAILABLE = True
except ImportError:
    EXTRACT_ENTITIES_AVAILABLE = False
    print("警告: Extract_Entities 不可用，将使用简单句子拆分")

# 加载 .env 配置
def load_env_config():
    """加载 .env 文件中的 LLM 配置"""
    config = {
        'llm_base_url': 'http://192.0.0.193:9997/v1',
        'llm_model': 'qwen3',
        'llm_api_key': '',
        'llm_timeout': 30,
        'llm_max_tokens': 2048,
        'llm_batch_size': 5,
        'llm_confidence_threshold': 0.7,
        'use_llm_validation': True,
    }
    
    env_path = REPO_ROOT / '.env'
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 移除行内注释（以#开头的内容）
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    
                    if key == 'LLM_BASE_URL':
                        config['llm_base_url'] = value
                    elif key == 'LLM_MODEL':
                        config['llm_model'] = value
                    elif key == 'LLM_API_KEY':
                        config['llm_api_key'] = value
                    elif key == 'LLM_TIMEOUT':
                        try:
                            config['llm_timeout'] = int(value)
                        except:
                            pass
                    elif key == 'LLM_MAX_TOKENS':
                        try:
                            config['llm_max_tokens'] = int(value)
                        except:
                            pass
                    elif key == 'LLM_BATCH_SIZE':
                        try:
                            config['llm_batch_size'] = int(value)
                        except:
                            pass
                    elif key == 'LLM_CONFIDENCE_THRESHOLD':
                        try:
                            config['llm_confidence_threshold'] = float(value)
                        except:
                            pass
                    elif key == 'USE_LLM_VALIDATION':
                        config['use_llm_validation'] = value.lower() in ('true', '1', 'yes', 'on')
    
    return config


# 全局配置
ENV_CONFIG = load_env_config()


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


class SentenceSplitter:
    """
    句子拆分器 - 基于实体或简单规则
    
    优先使用 Extract_Entities 的 text_extrac_process，
    如果不可用则回退到简单拆分
    """
    
    def __init__(self):
        self.use_entity_split = EXTRACT_ENTITIES_AVAILABLE
    
    def split(self, text: str) -> List[Dict]:
        """
        将文本拆分为短句列表
        
        Returns:
            短句列表，每个元素包含:
            - text: 短句文本
            - start: 在原文中的起始位置
            - end: 在原文中的结束位置
        """
        if self.use_entity_split:
            return self._split_by_entities(text)
        else:
            return self._split_simple(text)
    
    def _split_by_entities(self, text: str) -> List[Dict]:
        """使用实体提取拆分短句"""
        try:
            # 调用 text_extrac_process
            anchors = text_extrac_process(
                report_text=text,
                version='报告',
                modality='CT',
                train_mode=False
            )
            
            # 提取短句（去重）
            sentences = []
            seen_texts = set()
            
            for anchor in anchors:
                short_text = anchor.get('short_sentence', '')
                if not short_text or short_text in seen_texts:
                    continue
                
                seen_texts.add(short_text)
                
                # 计算在原文中的位置
                start = text.find(short_text)
                if start == -1:
                    continue
                
                sentences.append({
                    'text': short_text,
                    'start': start,
                    'end': start + len(short_text),
                    'keyword': anchor.get('keyword', ''),
                    'position': anchor.get('position', ''),
                })
            
            # 按位置排序
            sentences.sort(key=lambda x: x['start'])
            
            # 如果没有提取到短句，回退到简单拆分
            if not sentences:
                return self._split_simple(text)
            
            return sentences
            
        except Exception as e:
            print(f"实体拆分失败: {e}")
            return self._split_simple(text)
    
    def _split_simple(self, text: str) -> List[Dict]:
        """简单句子拆分（按标点符号）"""
        # 按标点符号分割
        pattern = r'([。！？；.!?;\n]+)'
        parts = re.split(pattern, text)
        
        sentences = []
        current_pos = 0
        
        for i in range(0, len(parts), 2):
            sentence = parts[i] if i < len(parts) else ''
            # 包含分隔符
            if i + 1 < len(parts):
                sentence += parts[i + 1]
            
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 计算位置
            start = text.find(sentence, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(sentence)
            current_pos = end
            
            sentences.append({
                'text': sentence,
                'start': start,
                'end': end,
                'keyword': '',
                'position': '',
            })
        
        return sentences
    
    def find_sentence_for_fragment(self, fragment: SuspiciousFragment, 
                                   sentences: List[Dict]) -> Optional[Dict]:
        """
        找到包含可疑片段的短句
        
        Args:
            fragment: 可疑片段
            sentences: 短句列表
        
        Returns:
            包含该片段的短句，如果没有则返回None
        """
        frag_start, frag_end = fragment.position
        
        for sent in sentences:
            # 检查片段是否在短句范围内（允许一些边界误差）
            if (sent['start'] <= frag_start < sent['end']) or \
               (sent['start'] < frag_end <= sent['end']) or \
               (frag_start <= sent['start'] < frag_end):
                return sent
        
        return None


class LLMGrammarValidator:
    """
    LLM语法验证层 - 单片段验证模式
    
    设计原则：
    1. 一个prompt执行一个任务，防止输出错乱
    2. 每个可疑片段单独验证
    3. 只输出YES/NO，不输出fix
    4. 多线程并发提高性能
    """
    
    # 单片段验证Prompt - 极简，只判断YES/NO
    SINGLE_PROMPT = """
## 目标：你是一名放射科质控专家，以下是n-gram算法找到的罕见短语片段，请判断短语片段附近是否有语法错误。

句子：'{sentence}'
罕见短语：【{fragment}】

## 规则:
### 先在句子中找到罕见短语的位置，然后联系上下文判断：
- 这是错别字/语法错误，会引发误解 → YES
- 这是合理的罕见短语 → NO
- 这是被截断的正常短语 → NO
- 再次阅读整个句子验证你的结论
## 输出要求:
- 只输出YES或NO，不要解释。"""
    
    def __init__(self, llm_client=None, num_workers: int = 8, batch_size: int = None):
        """
        Args:
            llm_client: LLM客户端（如OpenAI client）
            num_workers: LLM并发线程数
            batch_size: 每批验证的片段数（默认从.env读取）
        """
        self.llm = llm_client
        self.num_workers = num_workers
        self.batch_size = batch_size or ENV_CONFIG['llm_batch_size']
        self.confidence_threshold = ENV_CONFIG['llm_confidence_threshold']
        self.timeout = ENV_CONFIG['llm_timeout']
        self.max_tokens = ENV_CONFIG['llm_max_tokens']
        
        self._cache = {}  # 本地缓存
        self._cache_lock = threading.Lock()
        
        # 初始化句子拆分器
        self.sentence_splitter = SentenceSplitter()
        
        # 如果没有提供llm_client，尝试创建一个
        if self.llm is None:
            self.llm = self._create_llm_client()
    
    def _create_llm_client(self):
        """创建LLM客户端"""
        try:
            import openai
            client = openai.OpenAI(
                base_url=ENV_CONFIG['llm_base_url'],
                api_key=ENV_CONFIG['llm_api_key'] or 'dummy',
                timeout=self.timeout,
            )
            print(f"✓ LLM客户端创建成功: {ENV_CONFIG['llm_base_url']}")
            return client
        except ImportError:
            print("✗ openai未安装，LLM验证将使用启发式规则")
            return None
        except Exception as e:
            print(f"✗ LLM客户端创建失败: {e}")
            return None
    
    def _get_cache_key(self, text: str, fragment_text: str) -> str:
        """生成缓存键"""
        import hashlib
        return hashlib.md5(f"{text}:{fragment_text}".encode()).hexdigest()[:16]
    
    def _check_cache(self, text: str, fragment: SuspiciousFragment) -> Optional[GrammarError]:
        """检查缓存"""
        key = self._get_cache_key(text, fragment.text)
        with self._cache_lock:
            return self._cache.get(key)
    
    def _save_cache(self, text: str, fragment: SuspiciousFragment, error: Optional[GrammarError]):
        """保存到缓存"""
        key = self._get_cache_key(text, fragment.text)
        with self._cache_lock:
            self._cache[key] = error
            # 限制缓存大小
            if len(self._cache) > 10000:
                # 简单LRU：移除最早的10%
                keys_to_remove = list(self._cache.keys())[:1000]
                for k in keys_to_remove:
                    del self._cache[k]
    
    def validate(self, text: str, fragments: List[SuspiciousFragment]) -> List[GrammarError]:
        """
        验证可疑片段 - 单片段验证模式，多线程并行
        
        Args:
            text: 完整文本
            fragments: 可疑片段列表
        
        Returns:
            确认的语法错误列表
        """
        if not fragments:
            return []
        
        # 将文本拆分为短句
        sentences = self.sentence_splitter.split(text)
        
        # 准备验证任务（每个片段一个任务）
        tasks = []
        cached_errors = []
        
        for fragment in fragments:
            # 检查缓存
            cached = self._check_cache(text, fragment)
            if cached:
                cached_errors.append(cached)
                continue
            
            # 找到包含该片段的短句
            sentence = self.sentence_splitter.find_sentence_for_fragment(fragment, sentences)
            
            if sentence:
                tasks.append((fragment, sentence['text']))
            else:
                # 使用片段上下文作为句子
                tasks.append((fragment, fragment.context))
        
        if not tasks:
            return cached_errors
        
        # 多线程并行验证（每个片段独立验证）
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务（每个片段一个独立任务）
            future_to_task = {}
            for fragment, sentence_text in tasks:
                future = executor.submit(
                    self._validate_single,
                    fragment,
                    sentence_text
                )
                future_to_task[future] = (fragment, sentence_text)
            
            # 收集结果
            for future in as_completed(future_to_task):
                fragment, sentence_text = future_to_task[future]
                try:
                    error = future.result()
                    if error:
                        errors.append(error)
                        self._save_cache(text, fragment, error)
                    else:
                        self._save_cache(text, fragment, None)
                except Exception as e:
                    print(f"验证片段失败 '{fragment.text}': {e}")
                    # 回退到启发式规则
                    err = self._heuristic_validate_single(fragment)
                    if err:
                        errors.append(err)
                        self._save_cache(text, fragment, err)
        
        # 合并缓存结果
        errors.extend(cached_errors)
        
        return errors
    
    def _validate_single(self, fragment: SuspiciousFragment, sentence_text: str) -> Optional[GrammarError]:
        """
        验证单个片段 - 一个prompt一个任务
        
        Args:
            fragment: 可疑片段
            sentence_text: 包含该片段的句子
        
        Returns:
            如果是错误返回GrammarError，否则返回None
        """
        if not self.llm:
            return self._heuristic_validate_single(fragment)
        
        # 构造prompt
        prompt = self.SINGLE_PROMPT.format(
            sentence=sentence_text,
            fragment=fragment.text
        )
        
        # 调用LLM
        result = self._call_llm_simple(prompt)
        
        # 解析结果
        if result and result.upper() == 'YES':
            return GrammarError(
                text=fragment.text,
                position=fragment.position,
                suggestion="",  # 简化版不输出fix
                error_type='grammar_error',
                confidence=0.85,
                reason='LLM判定为错误',
                llm_verified=True
            )
        
        return None
    
    def _call_llm_simple(self, prompt: str, max_retries: int = 3) -> str:
        """
        调用LLM - 简化版，只返回YES/NO
        
        Args:
            prompt: prompt文本
            max_retries: 最大重试次数
        
        Returns:
            "YES" 或 "NO"
        """
        import time
        import openai
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = 2 ** attempt
                    time.sleep(delay)
                
                response = self.llm.chat.completions.create(
                    model=ENV_CONFIG['llm_model'],
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10,  # 只需要YES/NO
                    timeout=self.timeout
                )
                
                result = response.choices[0].message.content.strip().upper()
                
                # 提取YES/NO
                if 'YES' in result:
                    return 'YES'
                elif 'NO' in result:
                    return 'NO'
                else:
                    # 无法识别，默认NO（宁可放过）
                    return 'NO'
                    
            except openai.APITimeoutError:
                if attempt == max_retries - 1:
                    return 'NO'  # 超时默认NO
                continue
            except Exception as e:
                if attempt == max_retries - 1:
                    return 'NO'  # 错误默认NO
                continue
        
        return 'NO'
    
    def _heuristic_validate_single(self, fragment: SuspiciousFragment) -> Optional[GrammarError]:
        """启发式验证单个片段"""
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
        
        # 检查是否匹配已知错别字
        for wrong, (correct, err_type) in common_typos.items():
            if wrong in fragment.text:
                return GrammarError(
                    text=fragment.text,
                    position=fragment.position,
                    suggestion=fragment.text.replace(wrong, correct),
                    error_type=err_type,
                    confidence=0.85,
                    reason=f"匹配常见错别字库: {wrong}->{correct}",
                    llm_verified=False
                )
        
        # 模式异常通常是真的错误
        if fragment.strategy == 'pattern':
            return GrammarError(
                text=fragment.text,
                position=fragment.position,
                suggestion="需要人工检查",
                error_type='pattern_anomaly',
                confidence=fragment.score,
                reason=fragment.reason,
                llm_verified=False
            )
        
        return None
    
    async def validate_async(self, text: str, 
                           fragments: List[SuspiciousFragment]) -> List[GrammarError]:
        """异步批量验证"""
        # 目前使用同步方法的包装
        # 如果有支持async的LLM客户端，可以直接调用
        return self.validate(text, fragments)
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        with self._cache_lock:
            return {
                'cache_size': len(self._cache),
                'max_size': 10000,
            }


class LayeredGrammarDetector:
    """
    分层语法检测器 - Trigram增强版
    
    整合快速召回层和LLM验证层
    """
    
    def __init__(self, model_dir: str = None, llm_client = None, 
                 llm_workers: int = 8, use_trigram: bool = True):
        """
        Args:
            model_dir: 预训练模型目录
            llm_client: LLM客户端（可选，默认从.env创建）
            llm_workers: LLM并发线程数
            use_trigram: 是否使用trigram
        """
        self.fast_detector = FastRecoverDetector(model_dir, use_trigram=use_trigram)
        self.llm_validator = LLMGrammarValidator(
            llm_client, 
            num_workers=llm_workers
        )
        
        # 召回层阈值（提高阈值减少误报）
        self.recall_threshold = 0.75  # 原为0.5，提高后减少正常短语的误报
        # LLM验证阈值（高阈值=高精度）
        self.precision_threshold = ENV_CONFIG['llm_confidence_threshold']
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
        if not use_llm or not ENV_CONFIG['use_llm_validation']:
            return [e for e in [self.llm_validator._heuristic_validate_single(f) for f in suspicious] if e]
        
        # 第二步：LLM验证（多线程 + 实体感知短句）
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
        """批量检测"""
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
            'use_llm_validation': ENV_CONFIG['use_llm_validation'],
            'llm_model': ENV_CONFIG['llm_model'],
            'llm_base_url': ENV_CONFIG['llm_base_url'],
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
    print("分层语法检测器 v3.1 测试 - 实体感知短句LLM验证")
    print("=" * 70)
    
    # 使用多线程LLM验证器
    detector = LayeredGrammarDetector(llm_workers=4)
    
    print(f"\n配置信息:")
    stats = detector.get_stats()
    print(f"  LLM模型: {stats['llm_model']}")
    print(f"  LLM地址: {stats['llm_base_url']}")
    print(f"  LLM验证: {'启用' if stats['use_llm_validation'] else '禁用'}")
    print(f"  置信度阈值: {stats['precision_threshold']}")
    
    for text in test_cases:
        print(f"\n原文: {text}")
        
        # 快速召回
        suspicious = detector.fast_detect(text)
        print(f"  快速召回: {len(suspicious)}个可疑片段")
        for f in suspicious[:2]:
            print(f"    - [{f.strategy}] '{f.text}' (score={f.score:.2f})")
        
        # 完整检测（启发式）
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
