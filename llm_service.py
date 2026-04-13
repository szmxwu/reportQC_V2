"""
LLM 后置精筛服务
用于对规则引擎筛选出的候选问题进行LLM验证，降低假阳性
"""
import os
import json
import requests
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
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
1. 该描述是否涉及重要诊断，但确实在结论中未体现？
2. 是否为同一部位/病变的不同表述？
3. 置信度（0-1）

举例：
1. 【报告描述】左额叶见高密度影。【结论】基底节多发梗死。【疑似遗漏的描述】左额叶见高密度影
    输出：{{"is_missing": true, "confidence": 1.0, "reason": "左额叶重要征象遗漏"}}
2. 【报告描述】左额叶见高密度影，伴中线结构右移。【结论】左额叶出血。【疑似遗漏的描述】伴中线结构右移
    输出：{{"is_missing": false, "confidence": 0.8, "reason": "中线右移是间接征象，不是主要诊断"}}
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
    
    def validate_sex_error(
        self,
        patient_sex: str,
        suspected_keywords: List[str],
        report_content: str
    ) -> Tuple[bool, float, str]:
        """
        验证性别错误是否为真阳性
        
        Args:
            patient_sex: 患者性别 ("男" 或 "女")
            suspected_keywords: 疑似性别冲突的关键词列表
            report_content: 完整的报告内容（用于上下文判断）
            
        Returns:
            (is_error, confidence, reason)
        """
        cache_key = f"se:{hash(patient_sex + ''.join(suspected_keywords) + report_content[:100])}"
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1
        
        keywords_str = "、".join(suspected_keywords)
        
        prompt = f"""你是一位医学影像报告质控专家。请判断以下报告是否存在性别错误。

【患者性别】
{patient_sex}

【报告中出现的疑似性别冲突关键词】
{keywords_str}

【报告内容】
{report_content}

【判断规则】
1. 男性患者报告中出现女性专属解剖部位/疾病（如子宫、卵巢、阴道、输卵管、妊娠等）
2. 女性患者报告中出现男性专属解剖部位/疾病（如前列腺、精囊、睾丸、阴茎等）
3. 某些情况需要特殊考虑：
   - 先天性异常（如两性畸形）
   - 术后改变（如子宫切除术后男性化特征）
   - 引用既往史（如"既往子宫切除术后"）
   - 误用词（如"前列腺"用于描述女性盆腔结构）
   - 引用他人报告或对比描述

【输出要求】
请以JSON格式输出，包含以下字段：
- is_error: true/false，是否确实存在性别错误
- confidence: 0-1之间的置信度
- reason: 简要说明判断原因（如果是假阳性，请说明原因）

【示例】
1. 患者性别：男，关键词：子宫、卵巢
   输出：{{"is_error": true, "confidence": 0.95, "reason": "男性报告中不应出现子宫、卵巢等女性专属器官"}}

2. 患者性别：女，关键词：前列腺
   输出：{{"is_error": true, "confidence": 0.95, "reason": "女性报告中不应出现前列腺"}}

3. 患者性别：男，关键词：子宫（报告描述："盆腔术后改变，子宫已切除"）
   输出：{{"is_error": false, "confidence": 0.85, "reason": "引用既往手术史，非当前解剖结构描述"}}

4. 患者性别：女，关键词：前列腺（报告描述："前列腺形态正常"但患者为女性）
   输出：{{"is_error": false, "confidence": 0.8, "reason": "可能是笔误或模板错误，但女性确实没有前列腺，需结合临床"}}"""

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

    def validate_grammar_error(
        self,
        grammar_type: str,
        sentence: str,
        error_phrase: str,
        suggestion: Optional[str] = None,
        full_text: str = '',
        rule_score: Optional[float] = None,
        rule_source: str = '',
    ) -> Tuple[bool, float, str]:
        """验证 grammer 子系统命中的候选是否为真阳性。"""
        cache_key = (
            f"ge:{grammar_type}:{hash(sentence + error_phrase + (suggestion or '') + full_text[:120])}"
        )
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1

        rule_score_str = "" if rule_score is None else str(rule_score)
        suggestion_str = suggestion or ""
        prompt = f"""你是一位医学影像报告质控专家。请判断下面的语法错误候选是否应该被保留。

【错误类型】
{grammar_type}

【命中短句】
{sentence}

【疑似错误短语】
{error_phrase}

【建议修正】
{suggestion_str}

【规则来源】
{rule_source}

【规则分数】
{rule_score_str}

【原始全文】
{full_text}

【判定要求】
1. 若错误类型是 typo：判断该短语是否真的是医学影像报告中的错别字或拼音混淆，且建议修正是否正确。
2. 若错误类型是 general_high_risk：判断该短语是否属于通用语料高频、但放射报告里不应出现或极不自然的高危词；如果它在当前句子中是正常医学表达、机构简称、设备名、病史引用或固定搭配，则不要保留。
3. 若错误类型是 word_order：判断该短语在当前医学影像报告语境下是否存在明显词序错误；如果只是另一种常见表达、模板省略或上下文不足，则不要保留。
4. 重点降低误报，但不要无原则放过明显错误。

请以 JSON 输出：
{{"is_valid": true/false, "confidence": 0.0-1.0, "reason": "简要原因"}}"""

        try:
            result = self._call_llm(prompt)
            parsed = self._parse_json_response(result)

            is_valid = bool(parsed.get('is_valid', True))
            confidence = float(parsed.get('confidence', 0.5))
            reason = str(parsed.get('reason', ''))

            result_tuple = (is_valid, confidence, reason)
            if len(self._cache) < self._cache_size:
                self._cache[cache_key] = result_tuple
            return result_tuple

        except Exception as e:
            print(f"LLM grammar validation error: {e}")
            return (True, 0.0, f"LLM验证失败: {str(e)}")
    
    def _validate_single(self, candidate: Dict) -> Optional[Dict]:
        """
        验证单个候选
        
        Returns: 验证后的候选字典，如果验证失败返回None（将被丢弃）
        """
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
            elif candidate['type'] == 'sex_error':
                is_valid, conf, reason = self.validate_sex_error(
                    candidate['patient_sex'],
                    candidate['suspected_keywords'],
                    candidate['report_content']
                )
            else:
                # 未知类型直接返回
                return candidate
            
            # 置信度判断
            if conf == 0.0 and "LLM验证失败" in reason:
                # 超时或错误，标记为待审核
                candidate['needs_review'] = True
                candidate['review_reason'] = reason
                return candidate
            elif is_valid and conf >= self.confidence_threshold:
                # 高置信度真阳性
                candidate['confidence'] = conf
                candidate['llm_reason'] = reason
                return candidate
            elif is_valid and conf >= 0.5:
                # 中置信度，保留但标记
                candidate['confidence'] = conf
                candidate['llm_reason'] = reason
                candidate['weak_positive'] = True
                return candidate
            else:
                # 低置信度，丢弃（假阳性）
                return None
                
        except Exception as e:
            print(f"Validation error for candidate: {e}")
            candidate['needs_review'] = True
            candidate['review_reason'] = f"验证异常: {str(e)}"
            return candidate
    
    def batch_validate(
        self,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        批量验证候选问题 - 使用线程池并发处理
        
        candidates: [
            {"type": "conclusion_missing", "description": "...", "conclusion": "...", "suspected": "..."},
            {"type": "orient_error", "description": "...", "conclusion": "..."},
            {"type": "contradiction", "statement1": "...", "statement2": "..."},
            ...
        ]
        
        Returns: 验证后的候选列表（过滤掉假阳性）
        """
        if not self.use_llm or not self.available():
            return candidates
        
        if not candidates:
            return []
        
        validated = []
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            # 提交所有任务
            future_to_candidate = {
                executor.submit(self._validate_single, candidate): candidate 
                for candidate in candidates
            }
            
            # 收集结果
            for future in as_completed(future_to_candidate):
                result = future.result()
                if result is not None:
                    validated.append(result)
        
        return validated

    def batch_validate_grammar_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """批量验证 grammer 子系统候选。"""
        if not self.use_llm or not self.available():
            return candidates

        if not candidates:
            return []

        validated = []

        def _validate_candidate(candidate: Dict) -> Optional[Dict]:
            try:
                is_valid, conf, reason = self.validate_grammar_error(
                    grammar_type=candidate.get('grammar_type', ''),
                    sentence=candidate.get('sentence', ''),
                    error_phrase=candidate.get('error_phrase', ''),
                    suggestion=candidate.get('suggestion'),
                    full_text=candidate.get('full_text', ''),
                    rule_score=candidate.get('rule_score'),
                    rule_source=candidate.get('rule_source', ''),
                )

                if conf == 0.0 and "LLM验证失败" in reason:
                    candidate['needs_review'] = True
                    candidate['review_reason'] = reason
                    return candidate
                if is_valid and conf >= self.confidence_threshold:
                    candidate['confidence'] = conf
                    candidate['llm_reason'] = reason
                    return candidate
                if is_valid and conf >= 0.5:
                    candidate['confidence'] = conf
                    candidate['llm_reason'] = reason
                    candidate['weak_positive'] = True
                    return candidate
                return None
            except Exception as e:
                candidate['needs_review'] = True
                candidate['review_reason'] = f"验证异常: {str(e)}"
                return candidate

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            future_to_candidate = {
                executor.submit(_validate_candidate, candidate): candidate
                for candidate in candidates
            }
            for future in as_completed(future_to_candidate):
                result = future.result()
                if result is not None:
                    validated.append(result)

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
