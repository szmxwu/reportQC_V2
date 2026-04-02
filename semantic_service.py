"""
语义匹配服务 - 封装 BGE Embedding 和 Rerank API
基于 xinferrence OpenAI 兼容接口
"""
import os
import numpy as np
import requests
from functools import lru_cache
from typing import List, Optional, Dict
from dotenv import load_dotenv

# 加载环境变量
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


# 模块级别的lru_cache（类方法上的lru_cache在实例间不共享）
_embedding_cache = {}
_cache_hits = 0
_cache_misses = 0
_cache_size = int(os.getenv('EMBEDDING_CACHE_SIZE', '100000'))


def _get_embedding_cached(text: str, base_url: str, model: str, timeout: int, api_key: str) -> Optional[np.ndarray]:
    """
    内部函数：获取Embedding（供lru_cache装饰）
    注意：所有可变/不可序列化参数必须转为字符串
    """
    global _embedding_cache, _cache_hits, _cache_misses
    
    if not text or not text.strip():
        return None
    
    cache_key = text.strip()
    
    # 检查缓存
    if cache_key in _embedding_cache:
        _cache_hits += 1
        return _embedding_cache[cache_key]
    
    _cache_misses += 1
    
    try:
        session = requests.Session()
        session.headers.update({'Content-Type': 'application/json'})
        if api_key:
            session.headers['Authorization'] = f'Bearer {api_key}'
        
        resp = session.post(
            f"{base_url}/embeddings",
            json={"model": model, "input": text},
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        
        embedding = np.array(data['data'][0]['embedding'])
        
        # 存入缓存（LRU淘汰）
        if len(_embedding_cache) >= _cache_size:
            # 简单LRU：删除最早插入的
            oldest_key = next(iter(_embedding_cache))
            del _embedding_cache[oldest_key]
        _embedding_cache[cache_key] = embedding
        
        return embedding
        
    except Exception as e:
        print(f"Embedding API error: {e}")
        return None


class SemanticMatcher:
    """基于 xinferrence 的语义匹配服务"""
    
    def __init__(self):
        self.base_url = os.getenv('XINFERENCE_BASE_URL', 'http://192.0.0.188:9997/v1')
        self.api_key = os.getenv('XINFERENCE_API_KEY', '')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'bge-large-zh-v1.5')
        self.rerank_model = os.getenv('RERANK_MODEL', 'bge-reranker-v2-m3')
        self.timeout = int(os.getenv('API_TIMEOUT', '10'))
        
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json'
        })
        if self.api_key:
            self._session.headers['Authorization'] = f'Bearer {self.api_key}'
    
    def available(self) -> bool:
        """检查服务是否可用"""
        try:
            resp = self._session.get(
                f"{self.base_url}/models", 
                timeout=5
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"Semantic service unavailable: {e}")
            return False
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的Embedding向量（带缓存）
        """
        return _get_embedding_cached(
            text, 
            self.base_url, 
            self.embedding_model, 
            self.timeout,
            self.api_key
        )
    
    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """
        使用Rerank模型对候选进行精排
        
        Args:
            query: 查询文本（报告描述）
            passages: 候选文本列表（结论列表）
        
        Returns:
            每个passage的 relevance_score (0-1)
        """
        if not passages:
            return []
        
        if not query or not query.strip():
            return [0.0] * len(passages)
        
        try:
            resp = self._session.post(
                f"{self.base_url}/rerank",
                json={
                    "model": self.rerank_model,
                    "query": query.strip(),
                    "documents": [p.strip() for p in passages if p.strip()],
                    "top_n": len(passages)
                },
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            # 构建结果数组（保持原始顺序）
            scores = [0.0] * len(passages)
            for item in data.get('results', []):
                idx = item.get('index', -1)
                if 0 <= idx < len(scores):
                    scores[idx] = item.get('relevance_score', 0.0)
            
            return scores
            
        except Exception as e:
            print(f"Rerank API error: {e}")
            return [0.0] * len(passages)
    
    def cosine_sim(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': hit_rate,
            'size': len(self._embedding_cache)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# 全局单例
_matcher = None

def get_matcher() -> SemanticMatcher:
    """获取全局语义匹配器实例"""
    global _matcher
    if _matcher is None:
        _matcher = SemanticMatcher()
    return _matcher


def test_service():
    """测试服务功能"""
    print("Testing semantic service...")
    matcher = get_matcher()
    
    # 测试可用性
    print(f"\n1. Service available: {matcher.available()}")
    
    # 测试Embedding
    print("\n2. Testing embedding...")
    text1 = "右肺上叶可见多发钙化影"
    text2 = "双肺支气管扩张伴感染"
    
    emb1 = matcher.get_embedding(text1)
    emb2 = matcher.get_embedding(text2)
    
    if emb1 is not None and emb2 is not None:
        sim = matcher.cosine_sim(emb1, emb2)
        print(f"   '{text1}' <-> '{text2}'")
        print(f"   Cosine similarity: {sim:.4f}")
    else:
        print("   Failed to get embeddings")
    
    # 测试Rerank
    print("\n3. Testing rerank...")
    query = "右肺上叶可见多发钙化影"
    passages = [
        "双肺支气管扩张伴感染",
        "右肺上叶多发钙化灶，考虑陈旧性病变",
        "心脏增大，心包积液"
    ]
    
    scores = matcher.rerank(query, passages)
    print(f"   Query: '{query}'")
    for i, (passage, score) in enumerate(zip(passages, scores)):
        print(f"   [{i+1}] {score:.4f}: {passage}")
    
    # 缓存统计
    print("\n4. Cache stats:", matcher.get_cache_stats())


if __name__ == '__main__':
    test_service()
