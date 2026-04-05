# -*- coding: utf-8 -*-
"""
报告质控系统配置模块

集中管理所有配置读取，避免重复初始化和循环依赖
"""

import os
import re
import configparser

# 全局配置实例（单例模式）
_conf = None
_conf_user = None

def _load_config():
    """加载配置文件（仅执行一次）"""
    global _conf, _conf_user
    if _conf is None:
        _conf = configparser.ConfigParser()
        _conf.read('config/config.ini', encoding='UTF-8')
        
        _conf_user = configparser.ConfigParser()
        _conf_user.read('config/user_config.ini', encoding='UTF-8')
    return _conf, _conf_user

def get_config():
    """获取系统配置"""
    _load_config()
    return _conf

def get_user_config():
    """获取用户配置"""
    _load_config()
    return _conf_user

# 系统配置项（延迟加载）
class SystemConfig:
    """系统配置（来自config.ini）"""
    
    @staticmethod
    def key_part():
        return get_config().get("contradiction", "key_part").split("|")
    
    @staticmethod
    def ignore_sentence():
        return get_config().get("clean", "Ignore_sentence")
    
    @staticmethod
    def aspects():
        """获取所有aspect配置"""
        conf = get_config()
        aspects = []
        n = 1
        while True:
            try:
                aspect = conf.get("contradiction", "aspect" + str(n))
                aspects.append(aspect)
                n += 1
            except:
                break
        return aspects
    
    @staticmethod
    def exclud():
        return get_config().get("contradiction", "exclud")
    
    @staticmethod
    def sub_part():
        return get_config().get("contradiction", "sub_part")
    
    @staticmethod
    def stopwords():
        return get_config().get("clean", "stopwords")
    
    @staticmethod
    def semantics_stopwords():
        return get_config().get("semantics", "stopwords")
    
    @staticmethod
    def miss_ignore_pattern():
        pattern = get_config().get("report_conclusion", "miss_ignore")
        return re.compile(pattern, flags=re.I)
    
    @staticmethod
    def orient_ignore():
        return get_config().get("report_conclusion", "orient_ignore")
    
    @staticmethod
    def upper_position():
        return get_config().get("clean", "upper_position")
    
    @staticmethod
    def critical_ignore_words():
        return get_config().get("Critical", "IgnoreWords")
    
    @staticmethod
    def ignore_part():
        return get_config().get("missing", "ignore_part")
    
    @staticmethod
    def enhance():
        return get_config().get("missing", "enhance")
    
    @staticmethod
    def dwi():
        return get_config().get("missing", "dwi")
    
    @staticmethod
    def swi():
        return get_config().get("missing", "swi")
    
    @staticmethod
    def perfusion():
        return get_config().get("missing", "perfusion")
    
    @staticmethod
    def mrs():
        return get_config().get("missing", "MRS")

# 用户配置项（延迟加载）
class UserConfig:
    """用户配置（来自user_config.ini）"""
    
    @staticmethod
    def default_non_standard():
        return get_user_config().get("report_score", "defult_non_standard")
    
    @staticmethod
    def mr_non_standard():
        return get_user_config().get("report_score", "MR_non_standard")
    
    @staticmethod
    def ct_non_standard():
        return get_user_config().get("report_score", "CT_non_standard")
    
    @staticmethod
    def cm_max():
        return float(get_user_config().get("measure", "cm_max"))
    
    @staticmethod
    def mm_max():
        return float(get_user_config().get("measure", "mm_max"))
    
    @staticmethod
    def m_max():
        return float(get_user_config().get("measure", "m_max"))
    
    @staticmethod
    def positive_level():
        return int(get_user_config().get("positive", "level"))
    
    @staticmethod
    def male_keywords():
        return get_user_config().get("sex", "MaleKeyWords")
    
    @staticmethod
    def female_keywords():
        return get_user_config().get("sex", "FemaleKeyWords")
    
    @staticmethod
    def a_level():
        return get_user_config().get("report_score", "A_level").split(',')
    
    @staticmethod
    def b_level():
        return get_user_config().get("report_score", "B_level").split(',')
    
    @staticmethod
    def c_level():
        return get_user_config().get("report_score", "C_level").split(',')
    
    @staticmethod
    def part_correct_score():
        return int(get_user_config().get("report_score", "part_correct_score"))
    
    @staticmethod
    def conclusion_sum_score():
        return int(get_user_config().get("report_score", "conclusion_sum_score"))
    
    @staticmethod
    def report_sum_score():
        return int(get_user_config().get("report_score", "report_sum_score"))
    
    @staticmethod
    def language_score():
        return int(get_user_config().get("report_score", "language_score"))
    
    @staticmethod
    def standard_term_score():
        return int(get_user_config().get("report_score", "standard_term_score"))
    
    @staticmethod
    def audit_score():
        return int(get_user_config().get("report_score", "audit_score"))
    
    @staticmethod
    def subtraction():
        return get_user_config().get("Part_standard", "subtraction").split("|")
    
    @staticmethod
    def position_orientation():
        return get_user_config().get("Part_standard", "Position_orientation")
    
    @staticmethod
    def exam_orientation():
        return get_user_config().get("Part_standard", "Exam_orientation")
    
    @staticmethod
    def exam_enhance():
        return get_user_config().get("Part_standard", "Exam_enhance")
    
    @staticmethod
    def dr_complexity():
        return float(get_user_config().get("Complexity", "DRcomplexity"))
    
    @staticmethod
    def mg_complexity():
        return float(get_user_config().get("Complexity", "MGcomplexity"))
    
    @staticmethod
    def ct_complexity():
        return float(get_user_config().get("Complexity", "CTcomplexity"))
    
    @staticmethod
    def mr_complexity():
        return float(get_user_config().get("Complexity", "MRcomplexity"))
    
    @staticmethod
    def check_modality():
        return get_user_config().get("Check", "Modality")
    
    @staticmethod
    def missing_exclud():
        return get_user_config().get("missing", "exclud")

# 术后相关配置
class PostoperativeConfig:
    """术后相关配置 - 用于降低术后相关描述的匹配阈值"""
    
    @staticmethod
    def patterns():
        """术后相关模式列表"""
        # 默认模式，可通过配置文件扩展
        return ['术后', '术后改变', '术后复查', '术区', '术后状态', '术后改变', '术后所见']
    
    @staticmethod
    def threshold():
        """术后相关匹配的降低阈值"""
        # 默认0.5，可通过环境变量调整
        return float(os.getenv('POSTOPERATIVE_THRESHOLD', '0.5'))
    
    @staticmethod
    def is_postoperative_related(text: str) -> bool:
        """检查文本是否与术后相关"""
        if not text:
            return False
        patterns = PostoperativeConfig.patterns()
        return any(p in text for p in patterns)

class LLMConfig:
    """LLM验证配置"""
    
    @staticmethod
    def USE_LLM_VALIDATION():
        """延迟读取环境变量，确保.env文件已加载"""
        return os.getenv('USE_LLM_VALIDATION', 'true').lower() == 'true'
