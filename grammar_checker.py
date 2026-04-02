"""
医学报告语法检测器 - 分层架构
Stage 1: 快速召回（规则+统计）
Stage 2: LLM精校
"""
import os
import re
import json
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GrammarError:
    """语法错误数据类"""
    position: int
    text: str
    context: str
    error_type: str
    suggestion: str = ""
    confidence: float = 0.0


class MedicalTermWhitelist:
    """医学术语白名单 - 避免误报"""
    
    def __init__(self):
        self.terms: Set[str] = set()
        self.prefixes: Set[str] = set()
        self.suffixes: Set[str] = set()
        self._load_default_terms()
    
    def _load_default_terms(self):
        """加载默认医学术语"""
        # 解剖部位
        anatomy = [
            "肺", "肝", "肾", "心", "脾", "胰", "胆", "胃", "肠",
            "脑", "脊髓", "骨骼", "肌肉", "血管", "淋巴结",
            "上叶", "下叶", "左叶", "右叶", "实质", "皮质", "髓质"
        ]
        
        # 医学术语后缀
        suffixes = [
            "炎", "症", "瘤", "癌", "结", "灶", "影", "斑", "点",
            "扩张", "狭窄", "积液", "肿大", "萎缩", "钙化"
        ]
        
        # 检查相关
        exams = [
            "平扫", "增强", "CT", "MRI", "MR", "DR", "DX", "MG",
            "造影", "扫描", "成像"
        ]
        
        # 程度描述
        degrees = [
            "轻度", "中度", "重度", "明显", "显著", "少许", "少量",
            "多发", "单发", "弥漫", "局限"
        ]
        
        self.terms.update(anatomy + exams + degrees)
        self.suffixes.update(suffixes)
        
        # 构建前缀（用于模糊匹配）
        for term in anatomy:
            if len(term) >= 2:
                self.prefixes.add(term[:2])
    
    def is_medical_term(self, text: str) -> bool:
        """判断是否为医学术语"""
        # 精确匹配
        if text in self.terms:
            return True
        
        # 后缀匹配
        for suffix in self.suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                return True
        
        # 包含医学术语
        for term in self.terms:
            if term in text and len(text) < len(term) + 3:
                return True
        
        return False
    
    def load_from_corpus(self, corpus_path: str, min_freq: int = 100):
        """从语料库加载高频术语"""
        # TODO: 实现从corpus提取术语
        pass


class TypoRuleEngine:
    """错别字规则引擎"""
    
    def __init__(self):
        self.rules = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """加载默认错别字规则"""
        # 医学常见错别字 (pattern, suggestion, description)
        self.rules = [
            # 形近字
            (r'肺文里', '肺纹理', '形近字错误'),
            (r'低密渡', '低密度', '形近字错误'),
            (r'高密渡', '高密度', '形近字错误'),
            (r'曾强', '增强', '形近字错误'),
            (r'末见', '未见', '形近字错误'),
            
            # 音近字
            (r'支气管[^，,；;。]*?严', '支气管炎症', '音近字'),
            
            # 医学术语错字
            (r'钙化造', '钙化灶', '术语错字'),
            (r'占位性炳变', '占位性病变', '术语错字'),
            
            # 标点错误
            (r'[^，,；;。\s]\n[^，,；;。\s]', None, '换行缺少标点'),
            
            # 重复字
            (r'([，,；;。])\1+', r'\1', '重复标点'),
        ]
    
    def check(self, text: str) -> List[Dict]:
        """检查文本"""
        errors = []
        for pattern, suggestion, desc in self.rules:
            for match in re.finditer(pattern, text):
                errors.append({
                    'position': match.start(),
                    'text': match.group(),
                    'suggestion': suggestion or '检查标点',
                    'type': desc,
                    'context': text[max(0, match.start()-10):match.end()+10]
                })
        return errors


class NgramAnomalyDetector:
    """N-gram 统计异常检测器"""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams: Dict[str, int] = defaultdict(int)
        self.total = 0
        self.threshold = 1e-6  # 概率阈值
        self.min_count = 5  # 最小出现次数
    
    def train(self, texts: List[str]):
        """训练N-gram模型"""
        print(f"Training {self.n}-gram model...")
        for text in texts:
            # 清洗文本
            text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
            # 构建N-gram
            for i in range(len(text) - self.n + 1):
                ngram = text[i:i+self.n]
                self.ngrams[ngram] += 1
                self.total += 1
        
        # 过滤低频N-gram
        self.ngrams = {k: v for k, v in self.ngrams.items() if v >= self.min_count}
        print(f"N-gram vocabulary size: {len(self.ngrams)}")
    
    def check(self, text: str, window: int = 3) -> List[Dict]:
        """检测异常N-gram"""
        # 如果模型未训练，返回空
        if self.total == 0 or len(self.ngrams) == 0:
            return []
        
        anomalies = []
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i+self.n]
            count = self.ngrams.get(ngram, 0)
            prob = count / self.total if self.total > 0 else 0
            
            # 极低概率或不存在
            if prob < self.threshold and count == 0:
                context_start = max(0, i - window)
                context_end = min(len(text), i + self.n + window)
                anomalies.append({
                    'position': i,
                    'ngram': ngram,
                    'probability': prob,
                    'context': text[context_start:context_end]
                })
        
        return anomalies
    
    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump({
                'ngrams': dict(self.ngrams),
                'total': self.total,
                'n': self.n
            }, f)
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.ngrams = defaultdict(int, data['ngrams'])
            self.total = data['total']
            self.n = data['n']


class GrammarChecker:
    """分层语法检测器主类"""
    
    def __init__(self, 
                 ngram_model_path: Optional[str] = None,
                 use_llm: bool = True):
        self.whitelist = MedicalTermWhitelist()
        self.typo_engine = TypoRuleEngine()
        self.ngram_detector = NgramAnomalyDetector(n=3)
        self.use_llm = use_llm and os.getenv('USE_LLM_VALIDATION', 'true').lower() == 'true'
        
        # 加载N-gram模型
        if ngram_model_path and os.path.exists(ngram_model_path):
            self.ngram_detector.load(ngram_model_path)
            print(f"Loaded N-gram model from {ngram_model_path}")
        
        # LLM验证器
        if self.use_llm:
            from llm_service import get_llm_validator
            self.llm = get_llm_validator()
            self.llm_available = self.llm.available()
        else:
            self.llm_available = False
    
    def train_ngram(self, corpus_path: str, save_path: str):
        """从语料库训练N-gram模型"""
        texts = []
        
        # 支持多种格式
        if corpus_path.endswith('.txt'):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
        elif corpus_path.endswith('.jsonl'):
            import json
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data.get('ReportStr', '') + data.get('ConclusionStr', ''))
        elif os.path.isdir(corpus_path):
            # 目录，遍历所有txt文件
            import glob
            for filepath in glob.glob(os.path.join(corpus_path, '*.txt')):
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.extend(f.readlines())
        
        print(f"Training on {len(texts)} documents...")
        self.ngram_detector.train(texts)
        self.ngram_detector.save(save_path)
        print(f"Model saved to {save_path}")
    
    def stage1_recall(self, text: str) -> List[Dict]:
        """Stage 1: 快速召回"""
        candidates = []
        
        # 1. 规则检测
        typo_errors = self.typo_engine.check(text)
        candidates.extend(typo_errors)
        
        # 2. N-gram异常检测
        anomalies = self.ngram_detector.check(text)
        for ano in anomalies:
            # 跳过医学术语
            if not self.whitelist.is_medical_term(ano['ngram']):
                candidates.append({
                    'position': ano['position'],
                    'text': ano['ngram'],
                    'context': ano['context'],
                    'type': '统计异常',
                    'suggestion': ''
                })
        
        # 去重
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = (c['position'], c['text'])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        
        return unique_candidates
    
    def stage2_verify(self, text: str, candidates: List[Dict]) -> List[GrammarError]:
        """Stage 2: LLM精校"""
        if not self.llm_available or not candidates:
            return []
        
        verified = []
        
        for cand in candidates:
            # 跳过明确的医学术语
            if self.whitelist.is_medical_term(cand['text']):
                continue
            
            # LLM验证
            try:
                prompt = f"""判断以下文本是否存在语法错误或错别字：

【上下文】
{cand['context']}

【可疑片段】
{cand['text']}

请以JSON输出：
{{"is_error": true/false, "confidence": 0.9, "suggestion": "修正建议", "reason": "判断理由"}}"""
                
                result = self.llm._call_llm(prompt)
                parsed = self.llm._parse_json_response(result)
                
                is_error = parsed.get('is_error', False)
                confidence = parsed.get('confidence', 0.5)
                suggestion = parsed.get('suggestion', '')
                
                # 高置信度错误
                if is_error and confidence >= 0.7:
                    verified.append(GrammarError(
                        position=cand['position'],
                        text=cand['text'],
                        context=cand['context'],
                        error_type=cand['type'],
                        suggestion=suggestion,
                        confidence=confidence
                    ))
                    
            except Exception as e:
                print(f"LLM verification error: {e}")
                continue
        
        return verified
    
    def check(self, text: str) -> List[GrammarError]:
        """
        完整检测流程
        
        Returns:
            List[GrammarError]: 确认的语法错误列表
        """
        # Stage 1: 快速召回
        candidates = self.stage1_recall(text)
        
        if not candidates:
            return []
        
        # Stage 2: LLM精校（异步场景下可以批量处理）
        if self.use_llm:
            return self.stage2_verify(text, candidates)
        else:
            # 无LLM模式：返回规则检测结果（可能包含误报）
            return [
                GrammarError(
                    position=c['position'],
                    text=c['text'],
                    context=c['context'],
                    error_type=c['type'],
                    suggestion=c['suggestion'],
                    confidence=0.5  # 规则检测默认置信度
                )
                for c in candidates[:5]  # 限制数量避免过多误报
            ]


def test_grammar_checker():
    """测试语法检测器"""
    checker = GrammarChecker(use_llm=False)
    
    test_cases = [
        "双肺肺文里增多、紊乱",  # 错别字：文里->纹理
        "肝脏形态正常，肝边缘欠光整",  # 正常描述
        "低密渡灶，边界清",  # 错别字：渡->度
    ]
    
    print("测试语法检测器:\n")
    for text in test_cases:
        print(f"文本: {text}")
        errors = checker.check(text)
        if errors:
            for e in errors:
                print(f"  → 错误: '{e.text}' ({e.error_type}) 建议: {e.suggestion}")
        else:
            print("  → 无错误")
        print()


if __name__ == '__main__':
    test_grammar_checker()
