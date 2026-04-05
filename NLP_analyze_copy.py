# -*- coding: utf-8 -*-'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import pickle
import time
from datetime import datetime
import jieba  
from Extract_Entities import text_extrac_process, report_extrac_process
from semantic_service import get_matcher
from llm_service import get_llm_validator
import warnings
import configparser
from gensim.models import Word2Vec
from scipy import spatial
import copy
from multiprocessing import Pool,cpu_count
from typing import Optional, List, Dict
from pydantic import BaseModel,Field

warnings.filterwarnings("ignore")
jieba.load_userdict("config/user_dic_expand.txt")
class Report(BaseModel):
    """报告医生的数据结构.<br>
    ConclusionStr：报告结论<br>
    ReportStr:报告描述<br>
    modality:设备类型<br>
    StudyPart:检查条目名称<br>
    Sex:性别<br>
    applyTable:申请单，由既往史、临床症状、主诉、现病史拼合<br>
    """

    ConclusionStr: str= Field(title="报告结论",example= """
1.左乳术后改变，双肺及胸膜多发结节，部分较前饱满、稍大；肝内多发低密度影较前增多、范围增大；右侧心膈角稍大淋巴结。结合病史均考虑转移瘤可能性大。
2.右肺少许炎症，右侧胸膜增厚，较前进展。右侧少量胸腔积液较前稍多。双肺少许慢性炎症同前。主动脉及冠脉硬化，心包积液大致同前。
3.胆囊结石，胆囊炎可能，胆囊底部局限性粘膜增厚。
4.附见：胸12及腰5椎体呈楔形改变；左侧锁骨胸骨端低密度结节。胸骨密度不均。双肾多发囊肿；双肾小结石或钙化灶；双侧肾上腺结节状增粗。
        """)   
    ReportStr: str= Field(title="报告描述",example="""
“左乳腺癌术后多发转移”复查， 与2023/12/8日片比较(号码:0002093420):
   左乳缺如。右肺见少许斑片状模糊影与胸膜牵拉，右侧胸膜增厚，较前略进展；两肺多发结节，部分较前饱满、增大，结节较小增强扫描无法评估。右肺中叶及左肺上叶下舌段见斑片影及条索影，双下肺少许条索影。气管、支气管管腔完整，管壁光滑无增厚，管腔未见狭窄或阻塞。双肺门、纵隔见数枚淋巴结，同前；心影稍大，心包积液同前；主动脉、冠脉多发钙化。右侧腋窝多小淋巴结大致同前。右侧心膈角稍大淋巴结。右侧胸腔少量积液较前增多。
    附见：胸骨见少许斑点状高密度影，所见大致同前；左侧锁骨胸骨端见结节状低密度影，周边可见硬化边，大致同前。胸12、腰5椎体呈楔形改变，大致同前。
    肝内见多发稍低密度影，边界不清，肝左叶为著，，较前病灶增多、范围增大，增强扫描呈动脉期明显不均匀强化，其内多发无强化坏死区。肝内、外胆管未见明显扩张。胆囊形态、大小未见明确异常，腔内见类圆形高密度影，径约25×14mm；胆囊底部见一枚结节样突起约10mm，大致同前。胆囊壁增厚强化；胆囊窝脂肪间隙清晰。胰腺形态、大小未见明确异常，内见钙化灶，胰管未见明显扩张；胰腺周围脂肪间隙清晰；脾形态、大小、密度未见明确异常；胆、胰、脾增强扫描未见明显异常强化。腹膜后见多发稍大淋巴结。未见腹水征。
    附见：双侧肾上腺结节状增粗，以左侧内肢为著，增强扫描呈明显均匀强化。两肾见多发囊状稍低密度影，增强扫描未见明显强化。双肾见点样致密影。
        """)   
    modality:str= Field(title="设备类型",example='CT') #
    StudyPart: str= Field(title="检查条目名称",example='CT胸部/肺平扫+增强,CT上腹部/肝胆/脾/胰平扫+增强')  #
    Sex: str= Field(title="性别",example='女')  #
    applyTable: str= Field(default="",title="申请单",example="""
                    病史:主诉:左膝外伤后肿痛不适近3个月，要求复查。
                    诊断:膝关节损伤""")  #

class AuditReport(BaseModel):
    """审核医生的数据结构.
    beforeConclusionStr:报告医生的结论<br>
    beforeReportStr:报告医生的描述<br>
    afterConclusionStr：审核医生的结论<br>
    afterReportStr:审核医生的描述<br>
    modality:设备类型<br>
    StudyPart:检查条目名称<br>
    Sex:性别<br>
    report_doctor：报告医生姓名或工号<br>
    audit_doctor：审核医生姓名或工号<br>
    applyTable:申请单，由既往史、临床症状、主诉、现病史拼合<br>
    """
    beforeConclusionStr: str= Field(title="报告医生的结论",example= """
1.两肺上叶少许慢性陈旧灶，同前相仿。随访复查。
2.两肺下叶小肺气囊，大致同前。
        """)  
    afterConclusionStr: str= Field(title="审核医生的结论",example="""
1.两肺上叶少许陈旧灶，同前相仿。随访复查。
2.两肺下叶小肺气囊，大致同前。
3.附见：不均匀脂肪肝；肝右叶低密度灶；胆囊结石。左侧肾上腺增粗。
        """)
    beforeReportStr: str= Field(title="报告医生的描述",example=  """
与2022/9/12日片比较(号码:0001680102):
    两肺上叶见少许小片索状影伴小点状钙化，同前相仿。两肺下叶数枚无肺纹理透亮区，大者径约6mm，大致同前。气管、支气管管腔完整，管壁光滑无增厚，管腔未见狭窄或阻塞。肺门、纵隔未见淋巴结肿大，心脏、大血管未见异常。两侧胸腔未见积液。
    """)
    afterReportStr: str= Field(title="审核医生的描述",example="""
与2022/9/12日片比较(号码:0001680102):
    两肺上叶见少许小片索状影伴小点状钙化，同前相仿。两肺下叶数枚无肺纹理透亮区，大者径约6mm，大致同前。气管、支气管管腔完整，管壁光滑无增厚，管腔未见狭窄或阻塞。肺门、纵隔未见淋巴结肿大，心脏、大血管未见异常。两侧胸腔未见积液。
   附见：肝脏密度不均匀减低。肝右叶低信号灶，胆囊内高密度结节。
    """)
    modality: str= Field(title="设备类型",example='CT')
    StudyPart: str= Field(title="检查条目名称",example='X线计算机体层（CT）扫加收,胸部/肺平扫(去除文胸，项链等）')
    Sex:str= Field(title="性别",example='女')
    report_doctor: Optional[str]= Field(title="报告医生姓名或工号",example='报告医生A')
    audit_doctor: Optional[str]= Field(title="审核医生姓名或工号",example='审核医生B')
    applyTable: str= Field(default="",title="申请单",example="""
                    病史:主诉:左膝外伤后肿痛不适近3个月，要求复查。
                    诊断:膝关节损伤""")  #
class PositionModel(BaseModel):
    """
    历史检查条目或模板输入数据结构<br>
    index：序号<br>
    position：历史条目名称或模板名称（需包含解剖部位信息和检查方式方法信息）
    """
    index: object= Field(title="序号",example='65287172131302241')
    position: str= Field(title="历史条目名称或模板名称",example='颅脑平扫,CT上臂/肱骨平扫,胸部/肺平扫(去除文胸，项链等）,CT全腹部平扫,上臂/肱骨(三维)（右）')

class AbstractModel(BaseModel):
    """
    摘要输入数据结构<br>
    modality：设备类型<br>
    result_str：报告结论<br>
    date：检查时间，格式为2024-01-01 12:00:00
    """
    modality: str= Field(title="设备类型",example='病理')
    result_str: str=Field(title="报告结论",example="（左下肺结节）微浸润性肺腺癌，吻合钉切缘未见癌。") #报告结论文本
    date: str= Field(title="检查时间",example='2024-01-01 12:00:00')

class HistoryInfo(BaseModel):
    """病史对照，发现漏诊。<br>
    report:报告医生的数据结构.<br>
        *ConclusionStr:报告结论<br>
        *ReportStr:报告描述<br>
        *modality:设备类型<br>
        *StudyPart:检查条目名称<br>
        *Sex:性别<br>
    abstract:摘要输入数据结构<br>
        *modality：设备类型<br>
        *result_str：报告结论<br>
        *date：检查时间，格式为2024-01-01 12:00:00
    """
    report:Report= Field(title="完整报告",example="""
                         {"ConclusionStr": "1、 胆囊造瘘术后复查：胆总管末段结石；胆囊结石伴胆囊炎；胆囊引流管留置。2、轻度脂肪肝；肝小囊肿可能。3、双肾囊肿；左肾结石。4、子宫上方右侧髂血管旁一囊状影，拟淋巴管囊肿可能，必要时进一步检查。附见右侧胸腔积液。",
                         "ReportStr": "胆囊造瘘术后复查：胆管、胆囊：胆总管末段可见结石影，径约11mm。胆囊增大，壁增厚，腔内可见多发不规则高密度影，以及引流管留置。肝内、外胆管未见明显扩张。肝脏：形态、大小及各叶比例未见明确异常。肝密度轻度下降，肝S8可见一小圆形低密度影，径约7mm，余肝实质未见明显异常密度影。 胰腺：形态、大小、密度未见明确异常，胰管未见明显扩张；胰腺周围脂肪间隙清晰。 脾脏：形态、大小、密度未见明确异常。 胃、十二指肠及所见肠曲：未见明确异常。腹膜、肠系膜，腹膜后淋巴结：未见明确异常。双肾可见类圆形低密度影，径约10mm。左肾内可见结石影，径约7mm。双肾上腺形态、大小、密度未见明确异常，双侧肾盂、肾盏未见明显扩张积水。膀胱:充盈尚可，形态、大小未见明显异常，腔内未见明确异常密度影，膀胱壁未见明显增厚。子宫上方右侧髂血管旁可见一囊状密度影，径约20mm。子宫及双侧附件区:未见明确异常。肠曲：分布、形态及密度未见异常。附见右肺支气管轻度扩张，右肺下叶可见斑片模糊影，右侧胸腔少量积液。",
                         "StudyPart": "CT全腹部平扫",
                         "Sex": "女",
                         "modality": "CT"}
                         """)
    abstract:List[AbstractModel]=Field(title="相关病史",example="""
                                 [{"modality": "申请单", "result_str": "病史:肝圆韧带肿瘤术后复查。体格：肝圆韧带肿瘤术后复查。主诉：协诊体格检查：肝圆韧带肿瘤术后复查诊断:腹腔肿瘤术后复查","date": "2023/10/9 11:15:40"},
                                 {"modality": "CT", "result_str": "1.双侧斜裂胸膜稍增厚，余胸部CT平扫未见明确异常征象。2.附见：肝胃间隙软组织结节影，建议进一步检查；双肾细小结石。","date": "2023/7/11 12:22:13"},
                                 {"modality": "MR", "result_str": "肝胃间隙结节状异常信号，性质待定，请结合临床相关检查考虑。","date": "2023/7/11 8:44:16"},
                                 {"modality": "PS", "result_str": "（脐部脂肪瘤）符合脂肪瘤。","date": "2023/7/14 15:12:29"},
                                 {"modality": "PS", "result_str": "（肝缘韧带肿物）梭型细胞肿瘤，待石蜡及免疫组化继续评价。","date": "2023/7/14 9:57:44"},
                                 {"modality": "ES", "result_str": "胃窦壁外低回声灶", "date": "2023/7/13 11:37:51"}]""") 
    
# 加载配置文件规则
conf = configparser.ConfigParser()
conf.read('config/config.ini', encoding='UTF-8')
conf_user = configparser.ConfigParser()
conf_user.read('config/user_config.ini', encoding='UTF-8')
#系统配置
key_part = conf.get("contradiction", "key_part").split("|")
Ignore_sentence = conf.get("clean", "Ignore_sentence")
aspects=[]
n=1
while True: 
    try:
        aspect=conf.get("contradiction", "aspect"+str(n))
        aspects.append(aspect)
        n+=1
    except:
        break
    
exclud = conf.get("contradiction", "exclud")
sub_part = conf.get("contradiction", "sub_part")
enhance = conf.get("missing", "enhance")
dwi = conf.get("missing", "dwi")
swi = conf.get("missing", "swi")
perfusion = conf.get("missing", "perfusion")
MRS = conf.get("missing", "MRS")
stopwords = conf.get("clean", "stopwords")
semantics_stopwords=conf.get("semantics", "stopwords")
miss_ignore_pattern=re.compile(conf.get("report_conclusion", "miss_ignore"),flags=re.I)
upper_position=conf.get("clean", "upper_position")
CriticalIgnoreWords = conf.get("Critical", "IgnoreWords")
ignore_part=conf.get("missing", "ignore_part")
orient_ignore=conf.get("report_conclusion", "orient_ignore")

#自定义配置
defult_non_standard = conf_user.get("report_score", "defult_non_standard")
MR_non_standard = conf_user.get("report_score", "MR_non_standard")
CT_non_standard= conf_user.get("report_score", "CT_non_standard")
cm_max = float(conf_user.get("measure", "cm_max"))
mm_max = float(conf_user.get("measure", "mm_max"))
m_max = float(conf_user.get("measure", "m_max"))
positive_level = int(conf_user.get("positive", "level"))
MaleKeyWords = conf_user.get("sex", "MaleKeyWords")
FemaleKeyWords = conf_user.get("sex", "FemaleKeyWords")
A_level = conf_user.get("report_score", "A_level").split(',')
B_level = conf_user.get("report_score", "B_level").split(',')
C_level = conf_user.get("report_score", "C_level").split(',')
part_correct_score_sum= int(conf_user.get("report_score", "part_correct_score"))
conclusion_sum_score=int(conf_user.get("report_score", "conclusion_sum_score"))
report_sum_score=int(conf_user.get("report_score", "report_sum_score"))
language_score_sum=int(conf_user.get("report_score", "language_score"))
standard_term_score_sum=int(conf_user.get("report_score", "standard_term_score"))
audit_score_sum=int(conf_user.get("report_score", "audit_score"))
subtraction = conf_user.get("Part_standard", "subtraction").split("|")
Position_orientation = conf_user.get("Part_standard", "Position_orientation")
Exam_orientation = conf_user.get("Part_standard", "Exam_orientation")
Exam_enhance = conf_user.get("Part_standard", "Exam_enhance")
DRcomplexity=float(conf_user.get("Complexity", "DRcomplexity"))
MGcomplexity=float(conf_user.get("Complexity", "MGcomplexity"))
CTcomplexity=float(conf_user.get("Complexity", "CTcomplexity"))
MRcomplexity=float(conf_user.get("Complexity", "MRcomplexity"))
defautComlexity=float(conf_user.get("Complexity", "defautComlexity"))
# threshold=float(conf_user.get("Grammer", "perplexity"))
missing_exclud = conf_user.get("missing", "exclud")
check_modality=conf_user.get("Check", "Modality")
punctuations=conf_user.get("punctuation_norm", "punctuations").split("|")
applytable_exclud=conf_user.get("applytable", "exclud")
partlist_len={}
partlist_len['CT']=int(conf_user.get("Part_standard", "CT_partlist_len"))
partlist_len['MR']=int(conf_user.get("Part_standard", "MR_partlist_len"))
partlist_len['DR']=int(conf_user.get("Part_standard", "DX_partlist_len"))
partlist_len['MG']=int(conf_user.get("Part_standard", "MG_partlist_len"))
partlist_len_add=conf_user.get("Part_standard", "partlist_len_add").split("|")

# ============ BGE Embedding & Rerank 配置 ============
import os
USE_BGE_RERANK = os.getenv('USE_BGE_RERANK', 'true').lower() == 'true'
RERANK_THRESHOLD = float(os.getenv('RERANK_THRESHOLD', '0.7'))
FALLBACK_TO_WORD2VEC = os.getenv('FALLBACK_TO_WORD2VEC', 'true').lower() == 'true'

# 新重构的报告结论检查模块（可选切换，默认启用）
USE_NEW_CONCLUSION_CHECKER = os.getenv('USE_NEW_CONCLUSION_CHECKER', 'true').lower() == 'true'
if USE_NEW_CONCLUSION_CHECKER:
    try:
        from report_analyze.report_conclusion_checker import check_report_conclusion as check_report_conclusion_new
    except ImportError as e:
        print(f"Warning: 无法导入新的结论检查模块: {e}")
        USE_NEW_CONCLUSION_CHECKER = False

# 新重构的矛盾检查模块（可选切换）
USE_NEW_CONTRADICTION_CHECKER = os.getenv('USE_NEW_CONTRADICTION_CHECKER', 'false').lower() == 'true'
if USE_NEW_CONTRADICTION_CHECKER:
    try:
        from report_analyze.contradiction_checker import check_contradiction as check_contradiction_new
    except ImportError as e:
        print(f"Warning: 无法导入新的矛盾检查模块: {e}")
        USE_NEW_CONTRADICTION_CHECKER = False

# 初始化语义匹配器
_matcher = None
def get_semantic_matcher():
    global _matcher
    if _matcher is None:
        _matcher = get_matcher()
    return _matcher

# ============ partlist 和 axis 辅助函数（适配 Extract_Entities.py 的 List[tuple] 结构） ============

def should_ignore_entity(entity):
    """
    判断实体是否应该被忽略（基于 original_short_sentence）
    """
    sentence = entity.get('original_short_sentence', '')
    if re.search(rf"{Ignore_sentence}", sentence, re.I):
        return True
    return False


def add_ignore_field(entities):
    """
    为实体列表添加 ignore 字段
    """
    for entity in entities:
        entity['ignore'] = should_ignore_entity(entity)
    return entities


def get_all_partlist_elements(item):
    """
    获取实体所有可能的 partlist 元素（展平所有 tuple）
    用于处理歧义情况下的多 partlist
    """
    partlists = item.get('partlist', [])
    if not partlists:
        return set()
    # 展平所有 tuple 中的所有元素
    elements = set()
    for pl in partlists:
        if isinstance(pl, (list, tuple)):
            elements.update(pl)
        else:
            elements.add(pl)
    return elements


def get_all_axis_ranges(item):
    """
    获取实体所有可能的 axis 范围
    返回 [(start1, end1), (start2, end2), ...]
    """
    axes = item.get('axis', [])
    if not axes:
        return []
    return axes


def position_in_any_partlist(position, item):
    """
    检查 position 是否在 item 的任一 partlist 中
    """
    partlists = item.get('partlist', [])
    if not partlists:
        return False
    for pl in partlists:
        if isinstance(pl, (list, tuple)):
            if position in pl:
                return True
        elif position == pl:
            return True
    return False


def any_partlist_is_subset(item1, item2):
    """
    检查 item1 的任一 partlist 是否是 item2 的任一 partlist 的子集
    任一匹配即返回 True
    """
    pl1_list = item1.get('partlist', [])
    pl2_list = item2.get('partlist', [])
    if not pl1_list or not pl2_list:
        return False
    
    for pl1 in pl1_list:
        set1 = set(pl1) if isinstance(pl1, (list, tuple)) else {pl1}
        for pl2 in pl2_list:
            set2 = set(pl2) if isinstance(pl2, (list, tuple)) else {pl2}
            if set1.issubset(set2):
                return True
    return False


def any_partlist_intersects(item1, item2):
    """
    检查 item1 的任一 partlist 是否与 item2 的任一 partlist 有交集
    """
    pl1_list = item1.get('partlist', [])
    pl2_list = item2.get('partlist', [])
    if not pl1_list or not pl2_list:
        return False
    
    for pl1 in pl1_list:
        set1 = set(pl1) if isinstance(pl1, (list, tuple)) else {pl1}
        for pl2 in pl2_list:
            set2 = set(pl2) if isinstance(pl2, (list, tuple)) else {pl2}
            if set1 & set2:  # 有交集
                return True
    return False


def get_first_partlist(item):
    """
    获取第一个 partlist（用于需要单一 partlist 的场景）
    """
    partlists = item.get('partlist', [])
    if partlists:
        return partlists[0]
    return ()


def format_partlist_for_join(item):
    """
    将 partlist 格式化为可 join 的字符串列表
    如果有多个 partlist，取第一个
    """
    partlists = item.get('partlist', [])
    if not partlists:
        return []
    pl = partlists[0]
    return list(pl) if isinstance(pl, (list, tuple)) else [pl]


# ============ 原有函数 ============

def Get_special_exam(exam):
    """从excel部位字典中读取keyword."""
    partdic = {}
    for index, row in exam.iterrows():
        Partlist = []
        parts = [[], [], []]
        if row["三级项目"] is not np.nan:
            parts[2] = row['三级项目'].split("|")
            Partlist.append(parts[2][0])

        if row["二级项目"] is not np.nan:
            parts[1] = row['二级项目'].split("|")
            Partlist.append(parts[1][0])

        if row["一级项目"] is not np.nan:
            parts[0] = row['一级项目'].split("|")
            Partlist.append(parts[0][0])

        for i in range(3):
            newlist = Partlist[::-1]
            newlist = newlist[:i+1]
            if parts[i] != []:
                partdic[tuple(newlist)] = parts[i]
    return partdic


# 特殊检查项目字典读取
ct_special = Get_special_exam(pd.read_excel("config/exam_special.xlsx", sheet_name=0))
dr_special = Get_special_exam(pd.read_excel("config/exam_special.xlsx", sheet_name=1))
mr_special = Get_special_exam(pd.read_excel("config/exam_special.xlsx", sheet_name=2))
# 危急值规则表
ruletable_dict = pd.read_excel('config/criticalvalue.xlsx').to_dict('records')  # 危急值规则表

# 预加载分词词库和向量模型
jieba.initialize()
# wVmodel = Word2Vec.load('model/report_word2vec_processed_balance_med_jieba.m')  # 基于1亿放射文本语料+医学专业书籍的预训练模型，200个向量
wVmodel = Word2Vec.load('model/finetuned_word2vec.m')# 监督微调模型，200个向量
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def avg_feature_vector(sentence, model):  # 文本转向量
    words = jieba.lcut(sentence)
    feature_vec = np.zeros((model.wv.vector_size, ), dtype='float32')
    n_words = 0
    for word in words:
        try:
            feature_vec = np.add(feature_vec, model.wv[word])
            n_words += 1
        except:
            pass
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def sentence_semantics(s1, s2,model=wVmodel):  # 句子的相似度比较
    s1=re.sub(rf"{semantics_stopwords}",'',s1)
    s2=re.sub(rf"{semantics_stopwords}",'',s2)
    if s1=="" or s2=='':
        return 0.0
    s1_afv = avg_feature_vector(s1, model)
    s2_afv = avg_feature_vector(s2, model)
    if np.sum(s1_afv)==0 or np.sum(s2_afv)==0:
        sim=0
    else:
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
    if sim>0.5:
        s1_o=re.findall("[左右]",s1)
        s2_o=re.findall("[左右]",s2)
        if s1_o!=s2_o:
            sim-=0.05
    return sim

def struc_sim(s1_anylaze,s2_anylaze,model=False,complete=False):
    if s1_anylaze==[] or s2_anylaze==[]:
        return sentence_semantics(s1_anylaze['original_short_sentence'],s2_anylaze['original_short_sentence'])
    s1_position=" ".join(format_partlist_for_join(s1_anylaze)) 
    s2_position=" ".join(format_partlist_for_join(s2_anylaze))
    position_sim=sentence_semantics(s1_position,s2_position)
    if s1_anylaze['orientation']!=s2_anylaze['orientation']:
        position_sim-=0.1
    s1_test=[x for x in s1_anylaze['illness'].split(",") if x!='']
    s2_test=[x for x in s2_anylaze['illness'].split(",") if x!='']
    if s1_test!=[] and s2_test!=[]:
        if complete:
            illness_sim=sentence_semantics(s1_anylaze['illness'],s2_anylaze['illness'])
        else:
            illness_sim=sentence_semantics(s1_test[0],s2_test[0])
    else:
        illness_sim=0
    if model==False:
        if s1_anylaze['positive']!=s2_anylaze['positive'] :
            illness_sim-=0.2
        if s1_anylaze['measure']!=s2_anylaze['measure'] or s1_anylaze['percent']!=s2_anylaze['percent'] or s1_anylaze['volume']!=s2_anylaze['volume']:
            illness_sim-=0.1
    if model:#用于模板比较，更强调部位
        sim=position_sim*0.7+illness_sim*0.3
    else:
        sim=position_sim*0.4+illness_sim*0.6
    return sim




def detect_abnormal_medical_terms(sentence: str) -> List[str]:
    """
    检测医学句子中不符合解剖结构的术语
    
    参数:
        sentence: 需要检查的医学文本句子
        
    返回:
        包含所有不符合解剖常识的词语列表
    """
    # 医学术语规则库（部位: (最大允许值, 正则模式列表)）
    # print("sentence=",sentence)
    term_rules: Dict[str, tuple] = {
        "指骨": (5, [r"第([一二三四五六七八九十\d]+)指骨", r"指骨([1-5])"]),
        "趾骨": (5, [r"第([一二三四五六七八九十\d]+)趾骨", r"趾([1-5])"]),
        "掌骨": (5, [r"第([一二三四五六七八九十\d]+)掌骨", r"掌骨([1-5])"]),
        "跖骨": (5, [r"第([一二三四五六七八九十\d]+)跖骨", r"跖骨([1-5])"]),
        "颈椎": (7, [r"第([一二三四五六七八九十\d]+)颈椎", r"C(\d+)", r"颈椎([1-7])"]),
        "胸椎": (12, [r"第([一二三四五六七八九十\d]+)胸椎",  r"胸(\d+)椎体"]),
        "腰椎": (5, [r"第([一二三四五六七八九十\d]+)腰椎", r"L(\d+)", r"腰(\d+)椎体"]),
        "骶椎": (5, [r"第([一二三四五六七八九十\d]+)骶椎",  r"骶(\d+)椎体"]),
        "尾椎": (4, [r"第([一二三四五六七八九十\d]+)尾椎", r"尾([1-4])"]),
        "肋骨": (12, [r"第([一二三四五六七八九十\d]+)肋", 
                    r"肋(\d+)",
                    r"第([一二三四五六七八九十\d]+)(腋|后|前|软|背)肋",
                    r"([一二三四五六七八九十\d]+)(腋|后|前|软|背)肋",
                    r"([一二三四五六七八九十\d]+)十([一二三四五六七八九])(腋|后|前|软|背)肋",
               ]),
        "脑神经": (12, [r"第([一二三四五六七八九十\d]+)对脑神经"]),
        "颅神经": (12, [r"第([一二三四五六七八九十\d]+)对颅神经"]),
        "腕骨": (8, [r"第([一二三四五六七八九十\d]+)腕骨"]),
        "跗骨": (7, [r"第([一二三四五六七八九十\d]+)跗骨"])
    }

    # 数字转换字典
    cn_num = {"零":0, "一":1, "二":2, "三":3, "四":4, "五":5, "六":6, "七":7, "八":8, "九":9, "十":10}
    roman_num = {"Ⅰ":1, "Ⅱ":2, "Ⅲ":3, "Ⅳ":4, "Ⅴ":5, "Ⅵ":6, "Ⅶ":7, "Ⅷ":8, "Ⅸ":9, "Ⅹ":10}

    def convert_number(num_str: str) -> int:
        """处理多种数字表达形式"""
        if num_str.isdigit():
            return int(num_str)
        if num_str in cn_num:
            return cn_num[num_str]
        if num_str in roman_num:
            return roman_num[num_str]
        if num_str.endswith("十"):  # 处理汉字"十一"等
            return 10 + cn_num.get(num_str[1], 0)
        return -1

    abnormal_terms = []
    
    # 遍历所有解剖部位规则
    for part, (max_val, patterns) in term_rules.items():
        for pattern in patterns:
            # 匹配所有可能格式
            matches = re.finditer(pattern, sentence, flags=re.IGNORECASE)
            for match in matches:
                num_str = match.group(1).upper()  # 统一处理大小写
                num = convert_number(num_str)
                
                # 数值有效性验证
                if num == -1 or num > max_val:
                    term = match.group(0)
                    abnormal_terms.append(term)
    
    return list(set(abnormal_terms))  # 去重后返回


def CheckSex(all_analyze, Sex: str):  # 检查性别错误
    if len(all_analyze) == 0:
        return ""
    global MaleKeyWords, FemaleKeyWords
    body_part = []
    [body_part.extend(get_all_partlist_elements(d)) for d in all_analyze]
    body_part.extend([d['original_short_sentence'] for d in all_analyze])
    body_part = set(body_part)
    if Sex == "男" or Sex == "M":
        mat = re.findall(FemaleKeyWords," ".join(body_part))
        if mat:
            return ("男性报告中出现：%s" % "；".join(mat))
    if Sex == "女" or Sex == "F":
        mat = re.findall(MaleKeyWords," ".join(body_part))
        if mat:
            return ("女性报告中出现：%s" % "；".join(mat))
    return ""

def CheckMeasure(Str):  # 检查测量值错误
    if Str.strip() == "":
        return ""
    global mm_max, cm_max, m_max
    Str = Str.replace("*", "×")
    Str=re.sub("ml","毫升",Str,flags=re.I)
    relink =  r'(?<![a-zA-Z])(\d+(\.\d+)?(?:mm|cm|m|\*|×))(?![a-zA-Z])|(\d+(\.\d+)?(?:毫米|米))' 
    mes = re.findall(relink, Str, flags=re.I)
    mes=[x for group in mes for x in group if x!='']
    # print(mes)
    result = []
    allValues=[]
    unit = ""
    for x in reversed(mes):
        if unit != "" and "×" in x:
            x = x.replace("×", unit)
        # print('x=',x,"unit=",unit)

        i = x.lower().find("cm")
        if i < 0:
            i = x.find("厘米")
        if i >= 0:
            try:
                value = (float(x[:i]))
                unit = "cm"
                allValues.append({
                    "value":value*10,
                    "text":x
                })
                if value > cm_max and re.search("下肢长度|下肢全长|下肢长约|下肢力线",Str)==None:
                    p=Str.find(x)
                    if Str[p+len(x)]!="2" and Str[p+len(x)]!="3" and Str[p+len(x)]!="0":
                        result.append(x)
            except:
                pass
        else:
            i = x.lower().find("mm")
            if i < 0:
                i = x.find("毫米")
            if i >= 0:
                try:
                    value = (float(x[:i]))
                    unit = "mm"
                    allValues.append({
                    "value":value,
                    "text":x
                    })
                    if value > mm_max and re.search("下肢长度|下肢全长|下肢长约|下肢力线",Str)==None :
                        #排除平方
                        p=Str.find(x)
                        if Str[p+len(x)]!="2" and Str[p+len(x)]!="3" and Str[p+len(x)]!="0":
                            result.append(x)
                except:
                    pass
            else:
                i = x.lower().find("m")
                if i < 0:
                    i = x.find("米")
                if i >= 0:
                    try:
                        value = (float(x[:i]))
                        unit = "m"
                        allValues.append({
                        "value":value*100,
                        "text":x
                        })
                        if value > m_max:
                            result.append(x)
                    except:
                        pass
    if len(result) > 0:
        return ("测量值%s过大，请检查" % result)
    else:
        return ""
#检查部位缺失
def part_missing(studypart_analyze, ReportStr_analyze, Conclusion_analyze, modality):
    """检查部位缺失."""
    if re.search(check_modality,modality) is None:
        return [],[]
    missing = set()
    inverse=set()
    all_analyze = []
    all_analyze.extend(ReportStr_analyze)
    all_analyze.extend(Conclusion_analyze)
    starts = set(x['start'] for x in studypart_analyze)
    for start in starts:
        studyparts = [
            x for x in studypart_analyze if x['start'] == start]
        find = False
        #此处不可双向检测，会产生大量的假阳性。因为检查部位studyparts一般都是父节点（例如：腹部），报告部位all_analyze一般都是子节点（腹部-肝脏-肝左叶）
        #我们的目标是检测每一个检查部位，在报告部位中是否有对应的子节点，如果存在则不算遗漏部位
        #假设存在非常特殊的检查部位，建议在replace.xlsx中将其标准化，不必修改这里的逻辑
        for studypart in studyparts:
            temp = [x for x in all_analyze if (position_in_any_partlist(studypart['position'], x) or
                                               position_in_any_partlist(x['position'], studypart)) and
                    not((x['orientation'] =='左' and studypart['orientation']=='右') or
                        (x['orientation'] =='右' and studypart['orientation']=='左'))]
            if len(temp) > 0:
                find = True
                break

        if find == False:
            missing.add(studypart['orientation']+studypart['position'])
    for item in all_analyze:
        same_parts=[x for x in studypart_analyze if position_in_any_partlist(x['position'], item) or
                position_in_any_partlist(item['position'], x)]
        if not same_parts:
            continue
        if all(
            ((item['orientation'] =='左' and x['orientation']=='右') or
            (item['orientation'] =='右' and x['orientation']=='左')) for x in same_parts
            ):
            inverse.add(item['original_short_sentence'])
    for i in inverse:
        if i in missing:
            missing.remove(i)
    if len(ReportStr_analyze)>0:   #急诊报告（无描述）不检测部位缺失         
        return list(missing),list(set(inverse))
    else:
        return [],list(inverse)



def applytable_error(apply_analyze,studypart_analyze, ReportStr_analyze, Conclusion_analyze, modality):
    """检测申请单方位错误

    Args:
        apply_analyze (_type_): _description_
        studypart_analyze (_type_): _description_
        ReportStr_analyze (_type_): _description_
        Conclusion_analyze (_type_): _description_
        modality (_type_): _description_

    Returns:
        _type_: _description_
    """
    if not apply_analyze:
        return ""
    apply_parts=[]
    for s in studypart_analyze:       
        if re.search(upper_position,s['position']) is not None:
            part = [d for d in apply_analyze if any_partlist_is_subset(d, s) or 
                                                any_partlist_is_subset(s, d) ]
        else:
            part = [d for d in apply_analyze if position_in_any_partlist(d['position'], s) or 
                                                position_in_any_partlist(s['position'], d) ]
        if part:
            apply_parts.extend(part)
    if apply_parts:
        all_analyze=[]
        all_analyze.extend(ReportStr_analyze)
        all_analyze.extend(Conclusion_analyze)
        # all_analyze=[d for d in all_analyze if d['positive']]
        if all_analyze==[]:
            return ""
        # 使用新模块或旧函数
        if USE_NEW_CONCLUSION_CHECKER:
            _, inverse, _ = check_report_conclusion_new(apply_parts, all_analyze, modality)
        else:
            _, inverse = Check_report_conclusion(apply_parts, all_analyze, modality)
        if inverse:
            result= f"以下可能存在方位不符:{'；'.join(inverse)}"
            return result.replace("[描述]", "[报告]").replace("[结论]", "[申请单]")
    return ""

        
# 检测特殊检查错误

def check_special_missing(StudyPart, primaryStr, ReportStr_analyze, Conclusion_analyze, modality):
    """检查特殊项目缺失."""
    if re.search(check_modality,modality) is None:
        return []
    missing = []
    global enhance, dwi, swi
    ReportStrList = []
    ReportStrList.extend(ReportStr_analyze)
    ReportStrList.extend(Conclusion_analyze)
    # 注意：新模块已在抽取时完成文本预处理，此处不再调用 Str_replace
    if ReportStrList == []:
        return []
    primaryList = set([x['original_short_sentence'] for x in ReportStrList])
    for ReportStr in primaryList:
        if (re.search(rf"{enhance}", ReportStr) and re.search("建议|纹理增强|必要|请|^完善", ReportStr) == None and
                re.search(rf"{Exam_enhance}", StudyPart) == None and (modality == "CT" or modality == "MR")):
            # print(re.search(enhance, ReportStr), ReportStr)
            missing.append("本检查无增强，但出现“%s”描述" % ReportStr)
            break
    partlist = [" ".join(format_partlist_for_join(x)) for x in ReportStrList]
    partlist = " ".join(partlist)
    primaryList=re.split("[。；，？,?\n\t\r]",primaryStr)
    enhance_mark=any([re.search(rf"{enhance+'|'+perfusion}", primary)!=None for primary in primaryList])
    if re.search(missing_exclud,partlist)==None:
        if (re.search(rf"{Exam_enhance}", StudyPart) and enhance_mark==False and
                len(re.findall("动脉", partlist)) < 4  and
                len(re.findall("静脉", partlist)) < 2 
                and modality == "CT" ):
            missing.append("增强未描述")
        if (re.search(rf"{Exam_enhance}", StudyPart) and "平扫" not in StudyPart 
            and enhance_mark==False and  len(re.findall("动脉", partlist)) < 4 
            and modality == "MR" ):
            missing.append("增强未描述")
    if re.search(rf"{dwi}", StudyPart) and re.search(dwi, primaryStr) == None and modality == "MR":
        missing.append("弥散成像未描述")
    if re.search(rf"{swi}", StudyPart) and re.search(swi, primaryStr) == None and modality == "MR":
        missing.append("磁敏感成像未描述")
    if re.search(rf"{perfusion}", StudyPart,re.I) and re.search(perfusion, primaryStr,re.I) == None and (modality == "MR" or 
                                                                                                   modality == "CT"):
        missing.append("灌注成像未描述")
    if re.search(rf"{MRS}", StudyPart,re.I) and re.search(MRS, primaryStr,re.I) == None and modality == "MR":
        missing.append("波谱成像未描述")        
    return missing


def check_match_with_rerank(report_entity, conclusion_candidates, threshold=None):
    """
    使用BGE Rerank模型检查描述和结论是否匹配
    
    Returns:
        (best_match, max_score) - 最佳匹配的结论实体和得分
        如果没有找到匹配，返回 (None, 0)
    """
    if threshold is None:
        threshold = RERANK_THRESHOLD
    
    if not conclusion_candidates:
        return None, 0
    
    try:
        matcher = get_semantic_matcher()
        if not matcher.available():
            return None, -1  # 标记为API不可用
        
        # 准备查询和候选
        query = report_entity['original_short_sentence']
        passages = [c['original_short_sentence'] for c in conclusion_candidates]
        
        # 调用Rerank API
        scores = matcher.rerank(query, passages)
        
        if not scores:
            return None, 0
        
        # 找到最高分
        max_score = max(scores)
        best_idx = scores.index(max_score)
        
        if max_score >= threshold:
            return conclusion_candidates[best_idx], max_score
        else:
            return None, max_score
            
    except Exception as e:
        print(f"Rerank error: {e}")
        return None, -1  # 标记为错误


#描述结论不符合
def Check_report_conclusion(Conclusion_analyze, ReportStr_analyze,modality):
    """检查报告描述与结论的方位、部位匹配."""
    if re.search(check_modality,modality) is None:
        return [],[]
    conclusion_missing = []
    orient_error = []
    if len(ReportStr_analyze) == 0:
        return [], []
    if len(Conclusion_analyze) == 0:
        missing_find=[d['original_short_sentence'] for d in ReportStr_analyze if d['positive'] > 1 and miss_ignore_pattern.search(d['original_short_sentence'])==None]
        return list(set(missing_find)), []
    sentenc_starts = set([d['start'] for d in ReportStr_analyze])
    similarity_list = []
    for start in sentenc_starts:
        sentences = [d for d in ReportStr_analyze if d['start'] == start and d['illness']!='']
        if sentences==[]:
            continue
        missing_find = True
        #建立描述与结论的相似度矩阵
        for s in sentences:
            if re.search(upper_position,s['position']) is not None:
                position_find_list = [d for d in Conclusion_analyze if any_partlist_is_subset(d, s) or 
                                                    any_partlist_is_subset(s, d) ]
            else:
                position_find_list = [d for d in Conclusion_analyze if position_in_any_partlist(d['position'], s) or 
                                                    position_in_any_partlist(s['position'], d) ]
           
            entity_missing = True  # 单个实体是否缺失
            
            if len(position_find_list) > 0:
                # 检查方位匹配
                orient_match_list = [x for x in position_find_list if not((x['orientation']=='左' and s['orientation']=='右') or 
                                                     (x['orientation']=='右' and s['orientation']=='左'))]
                
                # === BGE + Rerank 语义匹配 (新增) ===
                # 注意：Rerank用于判断描述和结论是否语义匹配，不应该过滤掉方位不匹配的候选
                # 因为方位错误检测需要保留方位不匹配的项到similarity_list
                use_rerank = USE_BGE_RERANK and len(position_find_list) > 0
                rerank_found_match = False
                
                if use_rerank:
                    best_match, rerank_score = check_match_with_rerank(s, position_find_list)
                    if rerank_score == -1 and FALLBACK_TO_WORD2VEC:
                        # API失败，回退到Word2Vec
                        use_rerank = False
                    elif best_match is None:
                        # Rerank认为没有足够相似的匹配，可能是真缺失
                        # 但要保留所有候选用于后续方位错误检测
                        pass
                    else:
                        # Rerank找到匹配，说明描述在结论中有对应（无论方位是否匹配）
                        # 方位匹配 → 不是缺失
                        # 方位不匹配 → 可能是方位错误，也不是缺失
                        rerank_found_match = True
                        entity_missing = False
                        missing_find = False
                
                # 如果没有使用Rerank或Rerank失败/回退，使用原有逻辑
                if not use_rerank:
                    if orient_match_list:
                        entity_missing = False
                        missing_find = False
                
                # === 构建similarity_list用于方位错误检测 ===
                # 注意：无论Rerank结果如何，都需要将所有候选加入similarity_list
                # 因为方位错误检测需要比较方位不匹配的情况
                if s['positive'] > 1:
                    position_find_list_filtered = [d for d in position_find_list if d['ignore']==False]
                else:
                    position_find_list_filtered = position_find_list
                    
                for position_find in position_find_list_filtered:
                    if not position_in_any_partlist(position_find['position'], s):
                        continue
                    if re.search(orient_ignore,s['original_short_sentence']) is not None or re.search(orient_ignore,position_find['original_short_sentence']) is not None:
                        continue
                    semantics=struc_sim(position_find,s)
                    if semantics>0 and s['positive'] > 1:
                        similarity_list.append({'report':s['original_short_sentence'].split(",")[0],
                                                'report_orientation':s['orientation'],
                                                'positive':s['positive'],
                                                'report_position':s['position'],
                                                'conclusion_position':position_find['position'],
                                                'conclusion':position_find['original_short_sentence'].split(",")[0],
                                                'conclusion_orientation':position_find['orientation'],
                                                'semantics': semantics
                                                })
            
            # 检查当前实体是否缺失结论（立即处理，不依赖循环变量）
            if entity_missing and s['positive'] > 1 and s.get('ignore', False) == False:
                conclusion_missing.append(s['original_short_sentence'])
    
    non_orientation_match=[x for x in similarity_list if (x['report_orientation']=='左' and x['conclusion_orientation']=='右') or  
                                               (x['report_orientation']=='右' and x['conclusion_orientation']=='左')]
    orientation_match=[x for x in similarity_list if x not in non_orientation_match]


    for orientation_item in non_orientation_match:
        report_df=[x['semantics'] for x in orientation_match if x['report']==orientation_item['report']]
        conclusion_df=[x['semantics'] for x in orientation_match if x['conclusion']==orientation_item['conclusion']]
        report_probability=1 if report_df==[] else orientation_item['semantics']-max(report_df)
        conclusion_probability=1 if conclusion_df==[] else orientation_item['semantics']-max(conclusion_df)
        if (((conclusion_probability>0.006 and orientation_item['semantics']>=0.5) or 
            (conclusion_probability==1  and (orientation_item['semantics']>=0.4 or 
                                             any([orientation_item['report'] in x for x in conclusion_missing]))) )
            and  report_probability>0):
            orientation_item['report_probability']=report_probability
            orientation_item['conclusion_probability']=conclusion_probability
            orientation_item['max_conclusion_probability']=0 if conclusion_df==[] else max(conclusion_df)
            orientation_item['match_probability']=report_probability+conclusion_probability
    orientation_result={}
    for item in non_orientation_match:
        if 'match_probability' not in item:
            continue
        name=item['report']
        if name not in orientation_result or item['match_probability']>orientation_result[name]['match_probability']:
            orientation_result[name]=item
    non_orientation_match=list(orientation_result.values())
    orientation_result={}
    for item in non_orientation_match:
        if 'match_probability' not in item:
            continue
        name=item['conclusion']
        if name not in orientation_result or item['match_probability']>orientation_result[name]['match_probability']:
            orientation_result[name]=item
    orient_error=["[描述]"+x['report']+"；[结论]"+x['conclusion']
                  for x in orientation_result.values() if  re.search("左|右",x['report']) or re.search("左|右",x['conclusion'])]
    for  orient in orientation_result.values():
        conclusion_missing=[x for x in conclusion_missing if orient['report'] not in x]
    conclusion_missing = [x for x in set(conclusion_missing) if miss_ignore_pattern.search(x)==None]
    return conclusion_missing, orient_error


def check_contradiction(ReportStr_analyze, Conclusion_analyze,modality):
    """检查报告矛盾语句."""
    if re.search(check_modality,modality) is None:
        return []
    contradiction = []
    all_analyze = []
    all_analyze.extend([x for x in ReportStr_analyze if x['ignore'] == False])
    all_analyze.extend([x for x in Conclusion_analyze if x['ignore'] == False])

    if len(all_analyze) == 0:
        return contradiction
    for part in key_part:
        key_analyze = [d for d in all_analyze if (position_in_any_partlist(part, d)) and (
            d['orientation'] == "左" or d['orientation'] == "双" or d['orientation'] == "")]
        contradiction.extend(get_contradiction(key_analyze))
        key_analyze = [d for d in all_analyze if (position_in_any_partlist(part, d)) and (
            d['orientation'] == "右" or d['orientation'] == "双" or d['orientation'] == "")]
        contradiction.extend(get_contradiction(key_analyze))
        key_analyze = [d for d in all_analyze if (position_in_any_partlist(part, d)) and (
            d['orientation'] == "双" or d['orientation'] == "")]
        contradiction.extend(get_contradiction(key_analyze))
    return sorted(set(contradiction),key=contradiction.index)

def get_contradiction(key_analyze):
    if len(key_analyze) == 0:
        return []
    positive_key = []
    positive_key = [d for d in key_analyze if d['positive'] > 1 and
                         re.search(rf"{exclud}", d['original_short_sentence']) == None]


    negative_key = [d for d in key_analyze if d['positive'] <= 1 and 
                    d['measure']==0 and 
                    len(re.sub(stopwords,"",d['illness']))>1 and re.search(rf"{exclud}", d['original_short_sentence']) == None]
    if len(negative_key) == 0 or len(positive_key) == 0:
        return []

    found=False
    for negSentence in negative_key:
        for aspect in aspects:
            if re.search(rf"{aspect}", negSentence['original_short_sentence']):
                positive_str=get_positive_key(negSentence,aspect,positive_key)
                found=True
                if positive_str!="":
                    return [negSentence['original_short_sentence'], positive_str]
        if found==False:
            positive_str=get_positive_key(negSentence,"",positive_key)
            if positive_str!="":
                # 子部位匹配逻辑（纯规则引擎）
                pos_sub_part=re.findall(sub_part,positive_str)
                neg_sub_part=re.findall(sub_part,negSentence['original_short_sentence'])
                if pos_sub_part==[] or neg_sub_part==[] or (set(pos_sub_part) & set(neg_sub_part)): #子部位存在交集
                    return [negSentence['original_short_sentence'], positive_str]
                else:
                    return []
    return []
def get_positive_key(negSentence,aspect,positive_key):
    positive_key=[d for d in positive_key if position_in_any_partlist(negSentence['position'], d)]
    if aspect!="":
        positive_key = [d for d in positive_key if re.search(rf"{aspect}", d['original_short_sentence'])]

    if positive_key:
        positive_key=[{**dic, 'semantics': sentence_semantics(negSentence['original_short_sentence'],dic['original_short_sentence'])} for dic in positive_key]  
        semantics_min=min([x['semantics'] for x in positive_key])
        if semantics_min<=0.78:
            return [x['original_short_sentence'] for x in positive_key if x['semantics']==semantics_min][0]
    return ""


def check_contradiction_bge(sentence1: str, sentence2: str) -> bool:
    """
    使用BGE模型判断两句话是否语义矛盾
    
    Returns:
        True if 矛盾, False otherwise
    """
    if not USE_BGE_RERANK:
        return False
    
    try:
        matcher = get_semantic_matcher()
        if not matcher.available():
            return False
        
        # Step 1: 语义相似度检查
        emb1 = matcher.get_embedding(sentence1)
        emb2 = matcher.get_embedding(sentence2)
        
        if emb1 is None or emb2 is None:
            return False
        
        sim = matcher.cosine_sim(emb1, emb2)
        
        # 相似度太低，描述的是不同病变 → 不矛盾
        if sim < 0.3:
            return False
        
        # 高度相似，可能是重复描述 → 不矛盾
        if sim > 0.85:
            return False
        
        # Step 2: Rerank判断语义关系
        query = f"描述1：{sentence1}\n描述2：{sentence2}"
        passages = [
            "这两句话描述矛盾，一个是正常/阴性，一个是异常/阳性",
            "这两句话描述不矛盾，可能是同一事物的不同角度描述"
        ]
        
        scores = matcher.rerank(query, passages)
        
        if not scores or len(scores) < 2:
            return False
        
        # 如果"矛盾"得分显著高于"不矛盾"
        contradiction_score = scores[0]
        non_contradiction_score = scores[1]
        
        # 矛盾得分 > 0.35 且 高于不矛盾得分（放宽阈值）
        if contradiction_score > 0.35 and contradiction_score > non_contradiction_score + 0.05:
            return True
        
        return False
        
    except Exception as e:
        print(f"Contradiction BGE error: {e}")
        return False


# %%危急值
def CheckCritical(Conclusion_analyze, ReportStr_analyze, modality):
    """危急值检测."""
    all_analyze = ReportStr_analyze + Conclusion_analyze
    # print(Conclusion_analyze,ReportStr_analyze,modality)
    CriticalList = []
    for row in ruletable_dict:
        if row['设备'] is not np.nan:
            if not re.search(rf"{row['设备']}", modality, re.I):
                continue
        if row["结论"] is np.nan:
            continue
        if len(Conclusion_analyze) == 0:
            break
        # Conclusion_list = set([d['original_short_sentence'] for d in Conclusion_analyze])
        for c in Conclusion_analyze:
            sentence=c['original_short_sentence']
            if sentence.strip() == '':
                continue
            # 这里用一个词典来存储忽略词
            if re.search(rf"{CriticalIgnoreWords}", sentence, re.I):
                continue
            result = re.search(rf"{row['结论']}", sentence, re.I)
            if not result:
                continue
            if row['部位'] is not np.nan:
                part_result=position_in_any_partlist(row['部位'], c)
                if not part_result:
                    continue

            # print({"index":index,"category":row['类别'],"description":result.group()})
            if row['描述值'] > 0:
                # print(row['部位'],row['描述值'])
                if row['描述值'] < 1:
                    critical_des = [d for d in all_analyze if position_in_any_partlist(row['部位'], d) and
                                    d['percent'] >= row['描述值']]
                else:
                    critical_des = [d for d in all_analyze if position_in_any_partlist(row['部位'], d) and
                                    d['measure'] >= row['描述值']]
                if len(critical_des) > 0:
                    if critical_des[0]['original_short_sentence'] != sentence:
                        sentence = sentence + "：" + critical_des[0]['original_short_sentence']
                    if not any(x['index'] == row['序号'] for x in CriticalList):
                        CriticalList.append({"index": row['序号'],
                                             "category": row['类别'],
                                             "description": sentence})
            elif row['描述'] is not np.nan:
                if row['描述'].strip() != "":
                    critical_des = [d for d in ReportStr_analyze if re.search(
                        rf"{row['描述']}", d['original_short_sentence'])]
                    if len(critical_des) > 0:
                        if not any(x['index'] == row['序号'] for x in CriticalList):
                            CriticalList.append({"index": row['序号'],
                                                 "category": row['类别'],
                                                 "description": sentence + "：" + critical_des[0]['original_short_sentence']})

            else:
                if not any(x['index'] == row['序号'] for x in CriticalList):
                    CriticalList.append(
                        {"index": row['序号'], "category": row['类别'], "description": sentence})
    return CriticalList

#%%RADS分类检测
def CheckRADS(StudyPartStr:str,ConclusionStr:str,modality:str):
    if (modality=="MG" or modality=="DR" or modality=="DX" or modality=="MR" ) and "乳腺" in StudyPartStr:
        if re.search("BI.*RADS",ConclusionStr,flags=re.I |re.DOTALL)==None:
            return "缺少BI-RADS分类"
    if modality=="MR"  and "前列腺" in StudyPartStr:
        if re.search("PI.*RADS",ConclusionStr,flags=re.I |re.DOTALL)==None:
            return "缺少PI-RADS分类"
    return ""


# %% LLM验证函数 - 并发批量验证所有类型错误

def _batch_validate_with_llm(
    conclusion_missing_list: list,
    orient_error_list: list,
    contradiction_list: list,
    description: str,
    conclusion: str
) -> tuple:
    """
    统一批量LLM验证函数 - 并发验证所有类型的错误
    
    将结论缺失、方位错误、矛盾检测的结果一起提交给LLM进行批量验证，
    减少多次调用的开销，提高处理速度。
    
    Args:
        conclusion_missing_list: 结论缺失列表
        orient_error_list: 方位错误列表
        contradiction_list: 矛盾列表（成对出现）
        description: 报告描述原文
        conclusion: 报告结论原文
        
    Returns:
        tuple: (filtered_conclusion_missing, filtered_orient_error, filtered_contradiction)
    """
    # 检查是否需要验证
    has_conclusion_missing = bool(conclusion_missing_list)
    has_orient_error = bool(orient_error_list)
    has_contradiction = bool(contradiction_list and len(contradiction_list) >= 2)
    
    if not (has_conclusion_missing or has_orient_error or has_contradiction):
        return conclusion_missing_list, orient_error_list, contradiction_list
    
    if os.getenv('USE_LLM_VALIDATION', 'true').lower() != 'true':
        return conclusion_missing_list, orient_error_list, contradiction_list
    
    try:
        validator = get_llm_validator()
        if not validator.available():
            return conclusion_missing_list, orient_error_list, contradiction_list
        
        # 构建统一的验证候选列表
        candidates = []
        
        # 1. 结论缺失候选
        for item in conclusion_missing_list:
            candidates.append({
                'type': 'conclusion_missing',
                'description': description,
                'conclusion': conclusion,
                'suspected': item,
                '_original': item
            })
        
        # 2. 方位错误候选
        for item in orient_error_list:
            match = re.search(r'\[描述\](.+?)；\[结论\](.+)', item)
            if match:
                candidates.append({
                    'type': 'orient_error',
                    'description': match.group(1),
                    'conclusion': match.group(2),
                    '_original': item
                })
            else:
                # 格式不匹配，直接保留
                candidates.append({
                    'type': 'orient_error',
                    'description': item,
                    'conclusion': '',
                    '_original': item,
                    '_keep': True  # 标记为直接保留
                })
        
        # 3. 矛盾候选（成对出现）
        for i in range(0, len(contradiction_list) - 1, 2):
            stmt1 = contradiction_list[i]
            stmt2 = contradiction_list[i + 1] if i + 1 < len(contradiction_list) else ''
            if isinstance(stmt1, str) and isinstance(stmt2, str):
                candidates.append({
                    'type': 'contradiction',
                    'statement1': stmt1,
                    'statement2': stmt2,
                    '_pair_idx': i // 2
                })
        
        if not candidates:
            return conclusion_missing_list, orient_error_list, contradiction_list
        
        # 批量LLM验证（一次调用验证所有）
        validated = validator.batch_validate(candidates)
        
        # 分别收集结果
        confirmed_conclusion_missing = []
        conclusion_needs_review = []
        
        confirmed_orient_error = []
        orient_needs_review = []
        
        confirmed_contradiction = []
        contradiction_needs_review = []
        
        for v in validated:
            vtype = v.get('type', '')
            
            if vtype == 'conclusion_missing':
                original = v.get('_original', '')
                if v.get('needs_review'):
                    conclusion_needs_review.append(original)
                elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                    confirmed_conclusion_missing.append(original)
                elif v.get('weak_positive'):
                    confirmed_conclusion_missing.append(f"[弱阳性]{original}")
                # 低置信度丢弃
                
            elif vtype == 'orient_error':
                if v.get('_keep'):
                    # 格式不匹配的直接保留
                    confirmed_orient_error.append(v.get('_original', ''))
                    continue
                    
                original = v.get('_original', '')
                if v.get('needs_review'):
                    orient_needs_review.append(original)
                elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                    confirmed_orient_error.append(original)
                elif v.get('weak_positive'):
                    confirmed_orient_error.append(f"[弱阳性]{original}")
                # 低置信度丢弃
                
            elif vtype == 'contradiction':
                stmt1 = v.get('statement1', '')
                stmt2 = v.get('statement2', '')
                if v.get('needs_review'):
                    contradiction_needs_review.append((stmt1, stmt2))
                elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                    confirmed_contradiction.extend([stmt1, stmt2])
                elif v.get('weak_positive'):
                    confirmed_contradiction.extend([f"[弱阳性]{stmt1}", f"[弱阳性]{stmt2}"])
                # 低置信度丢弃
        
        # 添加待审核标记
        if conclusion_needs_review:
            confirmed_conclusion_missing.append(f"[待审核]{len(conclusion_needs_review)}项")
        if orient_needs_review:
            confirmed_orient_error.append(f"[待审核]{len(orient_needs_review)}项")
        if contradiction_needs_review:
            confirmed_contradiction.append(f"[待审核]{len(contradiction_needs_review)}项")
        
        return (
            confirmed_conclusion_missing if confirmed_conclusion_missing else [],
            confirmed_orient_error if confirmed_orient_error else [],
            confirmed_contradiction if confirmed_contradiction else []
        )
        
    except Exception as e:
        print(f"批量LLM验证失败: {e}")
        # 失败时保留原结果
        return conclusion_missing_list, orient_error_list, contradiction_list


# 保留旧函数以向后兼容（可选，新代码应使用_batch_validate_with_llm）
def _validate_conclusion_missing_with_llm(conclusion_missing_list: list, description: str, conclusion: str) -> list:
    """使用LLM验证结论缺失问题，过滤假阳性
    
    仅当规则引擎返回 conclusion_missing 不为空时才会被调用。
    LLM的作用是阻拦规则引擎的假阳性，保留高置信度的真阳性。
    
    Args:
        conclusion_missing_list: 规则引擎检测到的结论缺失列表
        description: 报告描述原文
        conclusion: 报告结论原文
        
    Returns:
        经过LLM验证过滤后的结论缺失列表
    """
    if not conclusion_missing_list:
        return conclusion_missing_list
    
    if os.getenv('USE_LLM_VALIDATION', 'true').lower() != 'true':
        return conclusion_missing_list
    
    try:
        validator = get_llm_validator()
        if not validator.available():
            return conclusion_missing_list
        
        # 构建验证候选
        candidates = []
        for item in conclusion_missing_list:
            candidates.append({
                'type': 'conclusion_missing',
                'description': description,
                'conclusion': conclusion,
                'suspected': item
            })
        
        # LLM批量验证
        validated = validator.batch_validate(candidates)
        
        # 收集验证结果：只保留高置信度真阳性
        confirmed_missing = []
        needs_review = []
        
        for v in validated:
            if v.get('needs_review'):
                # LLM验证失败或超时，标记为待审核
                needs_review.append(v)
            elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                # 高置信度真阳性
                confirmed_missing.append(v['suspected'])
            elif v.get('weak_positive'):
                # 中置信度，标记弱阳性但仍保留
                confirmed_missing.append(f"[弱阳性]{v['suspected']}")
            # 低置信度视为假阳性，丢弃
        
        # 添加待审核标记
        if needs_review:
            confirmed_missing.append(f"[待审核]{len(needs_review)}项")
        
        return confirmed_missing if confirmed_missing else []
        
    except Exception as e:
        print(f"结论缺失LLM验证失败: {e}")
        return conclusion_missing_list  # 失败时保留原结果


def _validate_orient_error_with_llm(orient_error_list: list) -> list:
    """使用LLM验证方位错误问题，过滤假阳性
    
    仅当规则引擎返回 orient_error 不为空时才会被调用。
    LLM的作用是阻拦规则引擎的假阳性，保留高置信度的真阳性。
    
    Args:
        orient_error_list: 规则引擎检测到的方位错误列表
        格式: ["[描述]xxx；[结论]xxx", ...]
        
    Returns:
        经过LLM验证过滤后的方位错误列表
    """
    if not orient_error_list:
        return orient_error_list
    
    if os.getenv('USE_LLM_VALIDATION', 'true').lower() != 'true':
        return orient_error_list
    
    try:
        validator = get_llm_validator()
        if not validator.available():
            return orient_error_list
        
        # 构建验证候选
        candidates = []
        original_items = {}  # 保存原始项用于后续映射
        
        for idx, item in enumerate(orient_error_list):
            # 解析方位错误格式：[描述]...；[结论]...
            match = re.search(r'\[描述\](.+?)；\[结论\](.+)', item)
            if match:
                candidates.append({
                    'type': 'orient_error',
                    'description': match.group(1),
                    'conclusion': match.group(2),
                    'original': item,
                    '_idx': idx
                })
                original_items[idx] = item
            else:
                # 格式不匹配，直接保留
                original_items[idx] = item
        
        if not candidates:
            return orient_error_list
        
        # LLM批量验证
        validated = validator.batch_validate(candidates)
        
        # 收集验证结果
        confirmed_errors = []
        needs_review = []
        
        for v in validated:
            original_item = v.get('original', '')
            
            if v.get('needs_review'):
                needs_review.append(original_item)
            elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                # 高置信度真阳性
                confirmed_errors.append(original_item)
            elif v.get('weak_positive'):
                # 中置信度，标记弱阳性
                confirmed_errors.append(f"[弱阳性]{original_item}")
            # 低置信度视为假阳性，丢弃
        
        # 添加待审核标记
        if needs_review:
            confirmed_errors.append(f"[待审核]{len(needs_review)}项")
        
        return confirmed_errors if confirmed_errors else []
        
    except Exception as e:
        print(f"方位错误LLM验证失败: {e}")
        return orient_error_list  # 失败时保留原结果


def _validate_contradiction_with_llm(contradiction_list: list) -> list:
    """使用LLM验证语言矛盾问题，过滤假阳性
    
    仅当规则引擎返回 contradiction 不为空时才会被调用。
    LLM的作用是阻拦规则引擎的假阳性，保留高置信度的真阳性。
    
    Args:
        contradiction_list: 规则引擎检测到的矛盾列表
        格式: [stmt1, stmt2, stmt3, stmt4, ...] 成对出现
        
    Returns:
        经过LLM验证过滤后的矛盾列表
    """
    if not contradiction_list or len(contradiction_list) < 2:
        return contradiction_list
    
    if os.getenv('USE_LLM_VALIDATION', 'true').lower() != 'true':
        return contradiction_list
    
    try:
        validator = get_llm_validator()
        if not validator.available():
            return contradiction_list
        
        # 构建验证候选（矛盾成对出现）
        candidates = []
        for i in range(0, len(contradiction_list) - 1, 2):
            stmt1 = contradiction_list[i]
            stmt2 = contradiction_list[i + 1] if i + 1 < len(contradiction_list) else ''
            if isinstance(stmt1, str) and isinstance(stmt2, str):
                candidates.append({
                    'type': 'contradiction',
                    'statement1': stmt1,
                    'statement2': stmt2,
                    '_pair_idx': i // 2
                })
        
        if not candidates:
            return contradiction_list
        
        # LLM批量验证
        validated = validator.batch_validate(candidates)
        
        # 收集验证结果
        confirmed_contradictions = []
        needs_review = []
        
        for v in validated:
            stmt1 = v.get('statement1', '')
            stmt2 = v.get('statement2', '')
            
            if v.get('needs_review'):
                needs_review.append((stmt1, stmt2))
            elif not v.get('weak_positive') and v.get('confidence', 0.5) >= 0.7:
                # 高置信度真阳性
                confirmed_contradictions.extend([stmt1, stmt2])
            elif v.get('weak_positive'):
                # 中置信度，标记弱阳性
                confirmed_contradictions.extend([f"[弱阳性]{stmt1}", f"[弱阳性]{stmt2}"])
            # 低置信度视为假阳性，丢弃
        
        # 添加待审核标记
        if needs_review:
            confirmed_contradictions.append(f"[待审核]{len(needs_review)}项")
        
        return confirmed_contradictions if confirmed_contradictions else []
        
    except Exception as e:
        print(f"矛盾LLM验证失败: {e}")
        return contradiction_list  # 失败时保留原结果


# %%报告医生质控函数
def Report_Quality(ReportTxt: Report, debug=False):
    """对初诊医生的报告进行质控，返回阳性率、报告错误等信息."""
    import time
    timers = {}
    total_start = time.time()
    
    # === Stage 1: 实体抽取 ===
    t0 = time.time()
    studypart_analyze = text_extrac_process(
        report_text=ReportTxt.StudyPart, version="标题", modality=ReportTxt.modality) if ReportTxt.StudyPart else []
    Conclusion_analyze = text_extrac_process(report_text=ReportTxt.ConclusionStr, version="报告", 
                                             modality=ReportTxt.modality, add_info=studypart_analyze) if ReportTxt.ConclusionStr else []
    ReportStr_analyze = text_extrac_process(report_text=ReportTxt.ReportStr, version="报告", 
                                            modality=ReportTxt.modality, add_info=studypart_analyze) if ReportTxt.ReportStr else []
    ReportTxt.applyTable=ReportTxt.applyTable.replace("  ", "\n")
    apply_analyze=text_extrac_process(report_text=ReportTxt.applyTable, version="报告", 
                                      modality=ReportTxt.modality, add_info=studypart_analyze) if ReportTxt.applyTable else []
    timers['实体抽取'] = time.time() - t0
    
    # 为所有实体添加 ignore 字段
    t0 = time.time()
    Conclusion_analyze = add_ignore_field(Conclusion_analyze)
    ReportStr_analyze = add_ignore_field(ReportStr_analyze)
    timers['字段处理'] = time.time() - t0
    
    missing = []
    inverse = []
    special_missing = []
    sex_error = []
    conclusion_missing = []
    orient_error = []
    contradiction = []
    none_standard_term = []
    apply_orient=''
    RADS=""

    # 检查漏写部位
    t0 = time.time()
    if studypart_analyze and re.search(ignore_part,ReportTxt.StudyPart) is None:
        missing,inverse = part_missing(
            studypart_analyze, ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
    timers['部位缺失检查'] = time.time() - t0
    
    # 检查漏写特殊检查
    t0 = time.time()
    special_missing = check_special_missing(
        ReportTxt.StudyPart, ReportTxt.ReportStr+ReportTxt.ConclusionStr,
        ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
    timers['特殊检查'] = time.time() - t0

    # 检查描述与结论相符
    t0 = time.time()
    conclusion_perf = {}
    if "骨龄" not in ReportTxt.StudyPart:
        if USE_NEW_CONCLUSION_CHECKER:
            conclusion_missing, orient_error, conclusion_perf = check_report_conclusion_new(
                                                Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)
        else:
            conclusion_missing, orient_error = Check_report_conclusion(
                                                Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)
    timers['描述结论检查'] = time.time() - t0
    
    # 提取Rerank性能统计
    if conclusion_perf:
        timers['Rerank调用'] = conclusion_perf.get('rerank_time', 0)
        timers['Rerank次数'] = conclusion_perf.get('rerank_calls', 0)
    
    # 检查矛盾
    t0 = time.time()
    if USE_NEW_CONTRADICTION_CHECKER:
        contradiction = check_contradiction_new(ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
    else:
        contradiction = check_contradiction(ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
    timers['矛盾检查'] = time.time() - t0
    
    # === Stage 2-4: 统一批量LLM验证 ===
    t0 = time.time()
    llm_validation_triggered = False
    if conclusion_missing or orient_error or contradiction:
        llm_validation_triggered = True
        conclusion_missing, orient_error, contradiction = _batch_validate_with_llm(
            conclusion_missing,
            orient_error,
            contradiction,
            ReportTxt.ReportStr,
            ReportTxt.ConclusionStr
        )
    timers['LLM验证'] = time.time() - t0
    
    # 检查性别错误
    all_analyze = ReportStr_analyze + Conclusion_analyze

    sex_error = CheckSex(all_analyze, ReportTxt.Sex)

    # 判断测量单位错误
    measure_unit_error = CheckMeasure(ReportTxt.ReportStr+"\n"+ReportTxt.ConclusionStr)

    #判断术语不规范
    global MR_non_standard, defult_non_standard,CT_non_standard
    term_list = []
    if len(ReportStr_analyze) > 0:
        term_list.extend([a['original_short_sentence'] for a in ReportStr_analyze])
    if len(Conclusion_analyze) > 0:
        term_list.extend([a['original_short_sentence'] for a in Conclusion_analyze])
    term_list = set(term_list)
    for term in term_list:
        if ReportTxt.modality == "MR":
            mat = re.findall(f"{MR_non_standard}|{defult_non_standard}", term,re.I)
        else:
            mat = re.findall(CT_non_standard +"|"+defult_non_standard, term,re.I)
        mat.extend(detect_abnormal_medical_terms(term))
        if mat != []:
            none_standard_term.append(
                term.replace("\n","") + " 中术语“" + ",".join(mat) + "”不标准")
    
    # if none_standard_term==[]:
    #     none_standard_term="未检测到常见术语错误"
    # else:
    #     none_standard_term="；".join(none_standard_term)
    
    #判断申请单方位错误
    apply_orient=applytable_error(apply_analyze,studypart_analyze,ReportStr_analyze,Conclusion_analyze,ReportTxt.modality)
    
    # 判断危急值
    Critical_value = CheckCritical(
        Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)
    # print('------Critical_value---\n',Critical_value)

    #检查RADS分类
    t0 = time.time()
    RADS=CheckRADS(ReportTxt.StudyPart, ReportTxt.ConclusionStr,ReportTxt.modality)
    timers['RADS检查'] = time.time() - t0
    
    total_time = time.time() - total_start
    timers['总计'] = total_time
    
    if debug:
        print("\n" + "="*60)
        print("详细耗时统计")
        print("="*60)
        
        # 提取Rerank调用次数（整数，不是时间）
        rerank_calls = timers.pop('Rerank次数', 0)
        
        # 按耗时排序
        sorted_timers = sorted(timers.items(), key=lambda x: x[1], reverse=True)
        for name, t in sorted_timers:
            pct = (t / total_time * 100) if total_time > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"{name:20s}: {t:6.3f}s ({pct:5.1f}%) {bar}")
        
        # 显示Rerank调用次数
        if rerank_calls:
            print(f"{'Rerank调用次数':20s}: {int(rerank_calls):>6d}次")
        
        print("="*60)

    return {
        "partmissing": missing,#报告部位缺失
        "partinverse":inverse, #检查项目方位错误 
        "special_missing": special_missing,#检查方法错误或缺失
        "conclusion_missing": conclusion_missing,#结论与描述不对应
        "orient_error": orient_error,#结论与描述方位（左右）不符合
        "contradiction": contradiction,#语言矛盾
        "sex_error": sex_error,#性别错误
        "measure_unit_error": measure_unit_error,#测量单位错误
        "none_standard_term":none_standard_term,#术语不规范
        'RADS':RADS,#分类
        "Critical_value": Critical_value,#危急值
        "apply_orient":apply_orient,
    }




if __name__ == "__main__":
    # 报告医生数据结构
    a = Report(
        ReportStr = """
   左肾盂肾盏明显扩张积水，左肾皮质明显变薄，左侧输尿管支架留置。右肾实质内未见明显异常密度影，增强后未见异常强化灶；右输尿管下段约右骶髂关节下缘局限性狭窄，管壁增厚，增强扫描未见异常强化灶，以上右泌尿系轻度扩张。膀胱充盈，未见确切异常密度影，未见明确膀胱壁异常，未见异常强化灶。
    所示肝、胆、胰、脾位置及形态正常，实质内密度尚均。腹膜后未见肿大淋巴结。
   CTA：双侧可见肾动脉由主动脉分出，经由肾门进入肾实质内，左侧肾动脉较对侧稍细，余双侧肾动脉未见明显异常。双侧未见副肾动脉。


        """,
   ConclusionStr = """
1.左侧输尿管支架留置，左肾明显扩张积水；左肾皮质萎缩.
2.右侧输尿管下段局限性狭窄，管壁增厚，未见明确占位性病变，以上右侧泌尿系轻度扩张积水，请结合临床及相关检查。
3.左侧肾动脉较对侧稍细。

   """,
        StudyPart = '肾脏、输尿管，膀胱增强+双肾(肾上腺)+肾动脉',
        Sex = '男',
        modality = "CT",
        applyTable=""
 
    )
    print("Report_Quality=", Report_Quality(a, debug=False))


