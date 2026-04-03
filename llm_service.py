"""
LLM 后置精筛服务
用于对规则引擎筛选出的候选问题进行LLM验证，降低假阳性
"""
import os
import json
import requests
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from json_repair import repair_json
# 加载环境变量
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)


class LLMValidator:
    """LLM验证器 - 用于精筛规则引擎的候选结果"""
    
    def __init__(self):
        self.base_url = os.getenv('LLM_BASE_URL', 'http://192.0.0.193:9997/v1')
        self.model = os.getenv('LLM_MODEL', 'qwen3')
        self.api_key = os.getenv('LLM_API_KEY', '')
        self.timeout = int(os.getenv('LLM_TIMEOUT', '30'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '2048'))
        self.batch_size = int(os.getenv('LLM_BATCH_SIZE', '5'))
        self.confidence_threshold = float(os.getenv('LLM_CONFIDENCE_THRESHOLD', '0.7'))
        self.use_llm = os.getenv('USE_LLM_VALIDATION', 'true').lower() == 'true'
        
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json'
        })
        if self.api_key:
            self._session.headers['Authorization'] = f'Bearer {self.api_key}'
        
        # 缓存
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size = int(os.getenv('LLM_CACHE_SIZE', '500'))
    
    def available(self) -> bool:
        """检查LLM服务是否可用"""
        try:
            resp = self._session.get(
                f"{self.base_url}/models",
                timeout=5
            )
            return resp.status_code == 200
        except Exception as e:
            print(f"LLM service unavailable: {e}")
            return False
    
    def validate_conclusion_missing(
        self, 
        description: str, 
        conclusion: str, 
        suspected_missing: str
    ) -> Tuple[bool, float, str]:
        """
        验证结论缺失是否为真阳性
        
        Returns:
            (is_true_positive, confidence, reason)
        """
        cache_key = f"cm:{hash(description + conclusion + suspected_missing)}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        
        prompt = f"""你是一位医学影像报告质控专家。请判断以下描述是否真的在结论中遗漏。

【报告描述】
{description}

【结论】
{conclusion}

【疑似遗漏的描述】
{suspected_missing}

请判断：
1. 该描述是否确实在结论中未体现？
2. 是否为同一部位/病变的不同表述？
3. 置信度（0-1）

请以JSON格式输出：
{{"is_missing": true/false, "confidence": 0.9, "reason": "简要说明原因"}}"""

        try:
            result = self._call_llm(prompt)
            parsed = self._parse_json_response(result)
            
            is_missing = parsed.get('is_missing', True)
            confidence = parsed.get('confidence', 0.5)
            reason = parsed.get('reason', '')
            
            result_tuple = (is_missing, confidence, reason)
            
            # 缓存结果
            if len(self._cache) < self._cache_size:
                self._cache[cache_key] = result_tuple
            
            return result_tuple
            
        except Exception as e:
            print(f"LLM validation error: {e}")
            # 超时或错误时返回待审核标记
            return (True, 0.0, f"LLM验证失败: {str(e)}")
    
    def validate_orient_error(
        self,
        description: str,
        conclusion: str
    ) -> Tuple[bool, float, str]:
        """验证方位错误是否为真阳性"""
        cache_key = f"oe:{hash(description + conclusion)}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        
        prompt = f"""你是一位医学影像报告质控专家。请判断以下方位描述是否矛盾。

【描述部分】
{description}

【结论部分】
{conclusion}

请判断：
1. 左右方位是否确实矛盾？
2. 是否为表述习惯差异（如"左侧"vs"左叶"）？
3. 置信度（0-1）

请以JSON格式输出：
{{"is_error": true/false, "confidence": 0.9, "reason": "简要说明原因"}}"""

        try:
            result = self._call_llm(prompt)
            parsed = self._parse_json_response(result)
            
            is_error = parsed.get('is_error', True)
            confidence = parsed.get('confidence', 0.5)
            reason = parsed.get('reason', '')
            
            result_tuple = (is_error, confidence, reason)
            
            if len(self._cache) < self._cache_size:
                self._cache[cache_key] = result_tuple
            
            return result_tuple
            
        except Exception as e:
            print(f"LLM validation error: {e}")
            return (True, 0.0, f"LLM验证失败: {str(e)}")
    
    def validate_contradiction(
        self,
        statement1: str,
        statement2: str
    ) -> Tuple[bool, float, str]:
        """验证矛盾是否为真阳性"""
        cache_key = f"ct:{hash(statement1 + statement2)}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        
        prompt = f"""你是一位医学影像报告质控专家。请判断以下两句话是否语义矛盾。

【描述1】
{statement1}

【描述2】
{statement2}

请判断：
1. 这两句话是否描述同一部位/病变？
2. 描述是否相互矛盾（一个正常一个异常）？
3. 置信度（0-1）

请以JSON格式输出：
{{"is_contradiction": true/false, "confidence": 0.9, "reason": "简要说明原因"}}"""

        try:
            result = self._call_llm(prompt)
            parsed = self._parse_json_response(result)
            
            is_contradiction = parsed.get('is_contradiction', True)
            confidence = parsed.get('confidence', 0.5)
            reason = parsed.get('reason', '')
            
            result_tuple = (is_contradiction, confidence, reason)
            
            if len(self._cache) < self._cache_size:
                self._cache[cache_key] = result_tuple
            
            return result_tuple
            
        except Exception as e:
            print(f"LLM validation error: {e}")
            return (True, 0.0, f"LLM验证失败: {str(e)}")
    
    def batch_validate(
        self,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        批量验证候选问题
        
        candidates: [
            {"type": "conclusion_missing", "description": "...", "conclusion": "...", "suspected": "..."},
            {"type": "orient_error", "description": "...", "conclusion": "..."},
            ...
        ]
        
        Returns: 验证后的候选列表（过滤掉假阳性）
        """
        if not self.use_llm or not self.available():
            return candidates
        
        validated = []
        
        for candidate in candidates:
            try:
                if candidate['type'] == 'conclusion_missing':
                    is_valid, conf, reason = self.validate_conclusion_missing(
                        candidate['description'],
                        candidate['conclusion'],
                        candidate['suspected']
                    )
                elif candidate['type'] == 'orient_error':
                    is_valid, conf, reason = self.validate_orient_error(
                        candidate['description'],
                        candidate['conclusion']
                    )
                elif candidate['type'] == 'contradiction':
                    is_valid, conf, reason = self.validate_contradiction(
                        candidate['statement1'],
                        candidate['statement2']
                    )
                else:
                    validated.append(candidate)
                    continue
                
                # 置信度判断
                if conf == 0.0 and "LLM验证失败" in reason:
                    # 超时或错误，标记为待审核
                    candidate['needs_review'] = True
                    candidate['review_reason'] = reason
                    validated.append(candidate)
                elif is_valid and conf >= self.confidence_threshold:
                    # 高置信度真阳性
                    candidate['confidence'] = conf
                    candidate['llm_reason'] = reason
                    validated.append(candidate)
                elif is_valid and conf >= 0.5:
                    # 中置信度，保留但标记
                    candidate['confidence'] = conf
                    candidate['llm_reason'] = reason
                    candidate['weak_positive'] = True
                    validated.append(candidate)
                # else: 低置信度，丢弃（假阳性）
                
            except Exception as e:
                print(f"Validation error for candidate: {e}")
                candidate['needs_review'] = True
                candidate['review_reason'] = f"验证异常: {str(e)}"
                validated.append(candidate)
        
        return validated
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM API"""
        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一个医学影像报告质控专家，专门识别报告中的错误和矛盾。请严格按照要求的JSON格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": self.max_tokens
            },
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']
    
    def _parse_json_response(self, response: str) -> Dict:
        """解析LLM的JSON响应"""
        try:
            # 尝试直接解析
            response=repair_json(response)
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])
            except:
                pass
            # 返回默认值
            return {"is_valid": True, "confidence": 0.5, "reason": "解析失败"}
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        total = self._cache_hits + self._cache_misses
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / total if total > 0 else 0,
            'size': len(self._cache)
        }


# 全局单例
_validator = None

def get_llm_validator() -> LLMValidator:
    """获取全局LLM验证器实例"""
    global _validator
    if _validator is None:
        _validator = LLMValidator()
    return _validator


def test_llm_validator():
    """测试LLM验证器"""
    print("Testing LLM Validator...")
    validator = get_llm_validator()
    
    print(f"\n1. Service available: {validator.available()}")
    
    # 测试结论缺失验证
    print("\n2. Testing conclusion missing validation...")
    desc = "右肺上叶可见多发钙化影。双肺支气管扩张伴感染。"
    conc = "双肺支气管扩张伴感染。"
    suspected = "右肺上叶可见多发钙化影"
    
    is_missing, conf, reason = validator.validate_conclusion_missing(desc, conc, suspected)
    print(f"   Is missing: {is_missing}, Confidence: {conf:.2f}")
    print(f"   Reason: {reason}")
    
    # 测试方位错误验证
    print("\n3. Testing orient error validation...")
    desc = "左肺上叶前段另见钙化结节"
    conc = "右肺上叶前段钙化结节"
    
    is_error, conf, reason = validator.validate_orient_error(desc, conc)
    print(f"   Is error: {is_error}, Confidence: {conf:.2f}")
    print(f"   Reason: {reason}")
    
    print(f"\n4. Cache stats: {validator.get_cache_stats()}")


if __name__ == '__main__':
    test_llm_validator()
