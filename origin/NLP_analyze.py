# -*- coding: utf-8 -*-'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import pickle
import time
from datetime import datetime
import jieba  
from keyword_extraction import *
import warnings
import configparser
from gensim.models import Word2Vec
from scipy import spatial
import copy
from multiprocessing import Pool,cpu_count
from typing import Optional, List, Dict
from pydantic import BaseModel,Field
# from llm import Structure_sentence_LLM
# from jinja2 import Template
# import pkuseg
# seg = pkuseg.pkuseg(model_name='medicine')
warnings.filterwarnings("ignore")
jieba.load_userdict("user_dic_expand.txt")
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
conf.read('config/system_config.ini', encoding='UTF-8')
conf_user = configparser.ConfigParser()
conf_user.read('config/user_config.ini', encoding='UTF-8')
#系统配置
key_part = conf.get("contradiction", "key_part").split("|")
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

def shorten_partlist(StudyPart,modality):
    "剪枝，缩短partlist到指定长度"
    global partlist_len,partlist_len_add
    for d in StudyPart:
        if d['root'] in partlist_len_add:
            d['position']=d['partlist'][partlist_len[modality]] if len(d['partlist'])>partlist_len[modality] else d['partlist'][-1]
        else:
            d['position']=d['partlist'][partlist_len[modality]-1] if len(d['partlist'])>partlist_len[modality]-1 else d['partlist'][-1]
    return StudyPart

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
        return sentence_semantics(s1_anylaze['primary'],s2_anylaze['primary'])
    s1_position=" ".join(s1_anylaze['partlist']) 
    s2_position=" ".join(s2_anylaze['partlist'])
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


import re
from typing import List, Dict

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

# %%历史检查,高级查询同部位判断

def check_history_match(StudyPartStr: str, HistoryPart_list: List[PositionModel]):
    """
    历史检查部位相似度比较，返回与当前检查相似的历史检查列表序号
    输入格式
      主检索部位或当前检查部位：StudyPartStr="肝胆胰脾"
      其他检查名称列表：HistoryPart_list=[{"index":0,"position":"上腹部"},{"index":1,"position":"颈椎"},{"index":2,"position":"胸部/纵膈"}]
    返回格式：匹配检查的index列表。例如匹配结果为上腹部、胸部/纵隔，则返回[0,2]"""
    matches = []
    Special_Processor = KeywordProcessor()
    Special_Processor.add_keywords_from_dict(dr_special)
    StudyPart = get_orientation_position(StudyPartStr, title=True, match=True)
    if len(StudyPart) == 0:
        return matches
    StudyPart = drop_dict_duplicates(StudyPart, ["partlist", "orientation"])
    for his in HistoryPart_list:
        Study_special = Special_Processor.extract_keywords(StudyPartStr)
        History_special = Special_Processor.extract_keywords(his.position)
        if Study_special != [] or History_special != []:
            special_match = False
            for x in Study_special:
                for y in History_special:
                    if not (set(x).isdisjoint(set(y))):
                        special_match = True
                        break
            if special_match == False:
                continue

        his_position = get_orientation_position(
            his.position, title=True, match=True)
        if len(his_position) == 0:
            continue
        for row in StudyPart:
            temp = [x for x in his_position if ((x['position'] in row['partlist']) or 
                (row['position'] in x['partlist'])) 
                and ((x['orientation'] == row['orientation']) or
                     ('双' in x['orientation']) or ('双' in row['orientation']) or
                     (x['orientation'] == '') or (row['orientation'] == ''))]
            if len(temp) > 0:
                matches.append(his.index)
                break
    return matches

# %%内部测试用,比较两个部位名称是否相似

def check_match(StudyPartStr: str, HistoryPartStr: str):
    # 输入格式
    # 当前检查部位：StudyPartStr="肝胆胰脾"
    # 其他检查名称列表：HistoryPart='上腹部、胸部/纵隔'
    # 返回格式：True/False。例如匹配结果为True
    Special_Processor = KeywordProcessor()
    Special_Processor.add_keywords_from_dict(dr_special)
    Study_special = Special_Processor.extract_keywords(StudyPartStr)
    History_special = Special_Processor.extract_keywords(HistoryPartStr)
    if Study_special != [] or History_special != []:
        special_match = False
        for x in Study_special:
            for y in History_special:
                if not (set(x).isdisjoint(set(y))):
                    special_match = True
                    break
        if special_match == False:
            return False

    StudyPart = get_orientation_position(StudyPartStr, title=True, match=True)
    if len(StudyPart) == 0:
        return False
    StudyPart = drop_dict_duplicates(StudyPart, ["partlist", "orientation"])
    his_position = get_orientation_position(
        HistoryPartStr, title=True, match=True)

    if len(his_position) == 0:
        return False
    for row in StudyPart:
        temp = [x for x in his_position if ((x['position'] in row['partlist']) or 
            (row['position'] in x['partlist'])) and ((x['orientation'] == row['orientation']) or
                 ('双' in x['orientation']) or ('双' in row['orientation']) or
                 (x['orientation'] == '') or (row['orientation'] == ''))]
        if len(temp) > 0:
            return True
    return False


def mode_filter(Sex,Age,modeList):
    """#根据条件筛选模板

    Args:
        Sex (_type_): _description_
        Age (_type_): _description_
        modeList (_type_): _description_
    """
    temp=[]
    if Sex.strip().upper() in ['男','M','MALE']:
        temp=[x for x in modeList if "女" not  in x['primary']]
    elif Sex.strip().upper() in ['女','F','FEMALE']:
        temp=[x for x in modeList if "男" not  in x['primary']]
    if temp:
        modeList=temp
    
    Age=Age.replace(" ","")
    mat = re.search(r'\d{1,3}(?=(Y|岁|Year))',Age,flags=re.I)
    temp=[]
    if mat:
        Age=int(mat.group())
        if Age<=6:
            temp=[x for x in modeList if re.search("儿童|5岁",x['primary']) is not None]
        elif Age>=60:
            temp=[x for x in modeList if "老年" in x['primary']]
    else:
        mat = re.search(r'\d{1,3}(?=(M|月|month))',Age,flags=re.I)
        if mat:
            temp=[x for x in modeList if "婴儿" in x['primary']]
        else:
            mat = re.search(r'\d{1,3}(?=(Y|天|Day))',Age,flags=re.I)
            if mat:
                temp=[x for x in modeList if "新生儿" in x['primary'] or "早产儿" in x['primary']]
    if temp:
        modeList=temp
    return modeList
# %%报告模板展开判断，展开最相似的模板

def check_mode_match(StudyPartStr: str,Sex:str,Age:str, ModePart_list: List[PositionModel]):
    """报告知识库模板相似度判断，返回与当前检查部位最相似的知识库模板列表 
    输入格式:
        当前检查部位：StudyPartStr="肝胆胰脾"
        患者性别:Sex=男 or 女
        患者年龄:Age=15D,3m,45Y
        模板名称列表：ModePart_list=[{"index":0,"position":"上腹部"},{"index":1,"position":"颈椎"},{"index":2,"position":"胸部/纵膈"}]
    返回格式：匹配度match>=0的list，展开这个list包含的模板序号index"""
    # start=time.time()
    result_list = []
    StudyPart = get_orientation_position(StudyPartStr, title=True,match=True)
    if len(StudyPart) == 0 or len(ModePart_list)==0:
        return result_list
    StudyPart = drop_dict_duplicates(StudyPart, ["partlist"])
    mode_position = []
    #把所有参数一起输入函数，以加快执行时间
    mode_long_str=""
    for mode in ModePart_list:
        modeStr=re.split(stop_pattern[:-1]+"\\.]",mode.position)
        for s in modeStr:
            if s:
                mode_long_str+= s+f"~{mode.index}\n"

    temp = get_orientation_position(mode_long_str, title=True,match=True)
    if len(temp) > 0:
        for t in temp:
            t['index'] = re.search(r"~(\d+)$",t['primary'])
            if not t['index']:
                continue
            else:
                try:
                    t['index']=t['index'].group(1)
                except:
                    continue
            t['sentence'] = [x.position  for x in ModePart_list if  str(x.index)==t['index']][0]
        mode_position.extend([x for x in temp if x['index']])
    #初步筛选部位符合的模板
    model_matched=[]
    for row in StudyPart:
        mode = {}
        temp = [x for x in mode_position if ((x['position'] in row['partlist']) or 
            (row['position'] in x['partlist'])) and ((x['orientation'] == row['orientation']) or
                 ('双' in x['orientation']) or ('双' in row['orientation']) or
                 (x['orientation'] == '') or (row['orientation'] == ''))]
        if len(temp) > 0:
            temp=[{**dic, 'match': struc_sim(row,dic,True)} for dic in temp]
            model_matched.extend(temp)
    #对筛选出来的模板进行重排序打分,这里考虑部分匹配和一对多匹配的情况，部分匹配需要减分，一对多算均分
    model_index=set([x['index'] for x in model_matched])
    reMarked_model=[]
    for i in model_index:
        model_part=[m for m in model_matched if m['index']==i]
        parts=set([x['position'] for x in model_part])
        match=0
        for part in parts:
            match+=np.mean([x['match'] for x in model_part if x['position']==part])
        match=match/len([x for x in mode_position if x['index']==i]) 
        for m in model_part:
            m['match']=match
        reMarked_model.extend(model_part)
    #对于每个部位，按方位、年龄、性别再次筛选后，挑选出评分最高的模板
    for row in StudyPart:
        model_part=[x for x in reMarked_model if ((x['position'] in row['partlist']) or 
            (row['position'] in x['partlist'])) and ((x['orientation'] == row['orientation']) or
                 ('双' in x['orientation']) or ('双' in row['orientation']) or
                 (x['orientation'] == '') or (row['orientation'] == ''))]
        model_part = sorted(model_part, key=lambda x:x['match'], reverse=True)
        model_part=mode_filter(Sex,Age,model_part)
        if model_part:
            result_list.append({'index':model_part[0]['index'],
                                'position':model_part[0]['sentence'],
                                'match':model_part[0]['match']
                })
    list_obj_dict = {i.get("index"): i for i in result_list}
    result_list = list(list_obj_dict.values())
    if any([type(x.index)==str for x in ModePart_list ]):
        return result_list
    else:
        return [{**d, 'index': int(d['index'])} for d in result_list]
# %%病史摘要函数
def History_Summary(StudyPartStr: str, abstract: List[AbstractModel],debug=False):
    """病史摘要.从患者历史检查、病历中抽取中与当前检查部位相关的，且为阳性的记录，
    并将其结构化后返回"""
    result = []
    studypart_analyze = get_orientation_position(
        StudyPartStr, title=True) if StudyPartStr else []
    if studypart_analyze == []:
        return []
    for history in abstract:
        summary = get_orientation_position(history.result_str, add_info=[
            s['axis'] for s in studypart_analyze]) if history.result_str else []
        if summary == []:
            continue
        samePartSummary = []
        for studypart in studypart_analyze:
            temp = [x for x in summary if (x['position'] in studypart['partlist'] or
                                           studypart['position'] in x['partlist']) and
                    not((x['orientation']=='左' and studypart['orientation']=='右') or 
                        (x['orientation']=='右' and studypart['orientation']=='左')) and 
                    x['positive'] == True]
            samePartSummary.extend(temp)
        modality = history.modality
        if modality == "US":
            modality = "超声"
        if modality == "ES":
            modality = "内镜"
        if modality == "PS":
            modality = "病理"
        if modality == "CR":
            modality = "DR"    
        samePartSummary = [{**dic, 'modality': modality}
                           for dic in samePartSummary]
        samePartSummary = [{**dic, 'result_str': history.result_str}
                           for dic in samePartSummary]
        #处理非标准格式的日期字符串
        try:
            history.date = re.sub("[\u4e00-\u9fa5]", "", history.date)
            history.date = history.date.replace("  ", " ")
            date = datetime.strptime(
                history.date.replace("-", '/'), "%Y/%m/%d %H:%M:%S")
            samePartSummary = [{**dic, 'date': date} for dic in samePartSummary]
            result.extend(samePartSummary)
        except:
            continue
    result = sorted(result, key=lambda x: x['position'])
    recent_result = []
    position_list = set([x['position'] for x in result])
    modality_list = set([x['modality'] for x in result])
    for modality in modality_list:
        for position in position_list:
            position_dic = [x for x in result if x['position'] == position and
                            x['modality'] == modality]
            if position_dic == []:
                continue
            position_dic = sorted(position_dic, key=lambda x: x['date'])
            recent_date = position_dic[-1]['date']
            recent_result.extend(
                [x for x in position_dic if x['date'] == recent_date])
    recent_result = [dict(t)
                     for t in {tuple(d.items()) for d in recent_result}]
    recent_result = sorted(
        recent_result, key=lambda x: x['date'], reverse=True)
    for dic in recent_result:
        if debug==False:
            del dic['partlist']
            del dic['axis']
            del dic['words']
            del dic['index']
            del dic['illness']
            del dic['orientation']
            del dic['start']
            del dic['positive']
            del dic['measure']
            del dic['percent']
            del dic['root']
            del dic['ambiguity']
            del dic['ignore']
            del dic['word_start']
            del dic['deny']
            del dic['volume']
        dic['date'] = dic['date'].strftime("%Y-%m-%d")
    recent_result = [dict(t)
                     for t in {tuple(d.items()) for d in recent_result}]
    recent_result = sorted(recent_result, key=lambda x: x['date'], reverse=True)
    return recent_result

# %%病史摘要函数forVB

def History_Summary_VB(abstract: List[AbstractModel]):
    """病史摘要 for VB，输入为post.从患者历史检查中抽取中与StudyPartStr部位相关的，且属性positve==True的记录，
    并将其结构化后返回"""
    result = []
    root_list = []
    StudyPartStr = abstract[0].result_str
    abstract = abstract[1:]
    studypart_analyze = get_orientation_position(
        StudyPartStr, title=True) if StudyPartStr else []
    if studypart_analyze == []:
        return []
    for history in abstract:
        summary = get_orientation_position(history.result_str, add_info=[
            s['axis'] for s in studypart_analyze]) if history.result_str else []
        if summary == []:
            continue
        samePartSummary = []
        for studypart in studypart_analyze:
            temp = [x for x in summary if (x['position'] in studypart['partlist'] or
                                           studypart['position'] in x['partlist']) and
                    x['positive'] == True]
            samePartSummary.extend(temp)

        modality = history.modality
        if modality == "US":
            modality = "超声"
        if modality == "ES":
            modality = "内镜"
        if modality == "PS":
            modality = "病理"
        samePartSummary = [{**dic, 'modality': modality}
                           for dic in samePartSummary]
        samePartSummary = [{**dic, 'result_str': history.result_str}
                           for dic in samePartSummary]
        history.date = re.sub("星期|一|二|三|四|五|六|日|天|上|下|午", "", history.date)
        history.date = history.date.replace("  ", " ")
        date = datetime.strptime(
            history.date.replace("-", '/'), "%Y/%m/%d %H:%M:%S")
        samePartSummary = [{**dic, 'date': date} for dic in samePartSummary]
        result.extend(samePartSummary)
    result = sorted(result, key=lambda x: x['position'])
    recent_result = []
    position_list = set([x['position'] for x in result])
    modality_list = set([x['modality'] for x in result])
    for modality in modality_list:
        for position in position_list:
            position_dic = [x for x in result if x['position'] == position and
                            x['modality'] == modality]
            if position_dic == []:
                continue
            position_dic = sorted(position_dic, key=lambda x: x['date'])
            recent_date = position_dic[-1]['date']
            recent_result.extend(
                [x for x in position_dic if x['date'] == recent_date])
    for dic in recent_result:
        del dic['partlist']
        del dic['axis']
        del dic['words']
        del dic['index']
        del dic['illness']
        del dic['orientation']
        del dic['start']
        del dic['positive']
        del dic['measure']
        del dic['percent']
        del dic['root']
        del dic['ambiguity']
        del dic['ignore']
        del dic['word_start']
        del dic['deny']
        dic['date'] = dic['date'].strftime("%Y-%m-%d")
    recent_result = [dict(t)
                     for t in {tuple(d.items()) for d in recent_result}]
    recent_result = sorted(
        recent_result, key=lambda x: x['date'], reverse=True)
    return recent_result

# 计算部位数量
def merge_intervals(a_list_of_intervals):
    # 求两个区间的并集
    a_list_of_intervals = [list(x) for x in a_list_of_intervals]
    sorted_intervals = sorted(a_list_of_intervals, key=lambda x: x[0])
    temp = []
    for interval in sorted_intervals:
        if temp and temp[-1][1] >= interval[0]:
            temp[-1][1] = max(temp[-1][1], interval[1])
        else:
            temp.append(interval)
    return temp

def CheckSex(all_analyze, Sex: str):  # 检查性别错误
    if len(all_analyze) == 0:
        return "未发现性别错误"
    global MaleKeyWords, FemaleKeyWords
    body_part = []
    [body_part.extend(x) for x in [d['partlist'] for d in all_analyze]]
    body_part.extend([d['primary'] for d in all_analyze])
    body_part = set(body_part)
    if Sex == "男" or Sex == "M":
        mat = re.findall(FemaleKeyWords," ".join(body_part))
        if mat:
            return ("男性报告中出现：%s" % "；".join(mat))
    if Sex == "女" or Sex == "F":
        mat = re.findall(MaleKeyWords," ".join(body_part))
        if mat:
            return ("女性报告中出现：%s" % "；".join(mat))
    return "未发现性别错误"

def CheckMeasure(Str):  # 检查测量值错误
    if Str.strip() == "":
        return "未发现测量单位明显错误"
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
        return "未发现测量单位明显错误"
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
    starts = set(x['word_start'] for x in studypart_analyze)
    for start in starts:
        studyparts = [
            x for x in studypart_analyze if x['word_start'] == start]
        find = False
        #此处不可双向检测，会产生大量的假阳性。因为检查部位studyparts一般都是父节点（例如：腹部），报告部位all_analyze一般都是子节点（腹部-肝脏-肝左叶）
        #我们的目标是检测每一个检查部位，在报告部位中是否有对应的子节点，如果存在则不算遗漏部位
        #假设存在非常特殊的检查部位，建议在replace.xlsx中将其标准化，不必修改这里的逻辑
        for studypart in studyparts:
            temp = [x for x in all_analyze if (studypart['position'] in x['partlist'] or
                                               x['position'] in studypart['partlist']) and
                    not((x['orientation'] =='左' and studypart['orientation']=='右') or
                        (x['orientation'] =='右' and studypart['orientation']=='左'))]
            if len(temp) > 0:
                find = True
                break

        if find == False:
            missing.add(studypart['orientation']+studypart['position'])
    for item in all_analyze:
        same_parts=[x for x in studypart_analyze if x['position'] in item['partlist'] or
                item['position'] in x['partlist']]
        if not same_parts:
            continue
        if all(
            ((item['orientation'] =='左' and x['orientation']=='右') or
            (item['orientation'] =='右' and x['orientation']=='左')) for x in same_parts
            ):
            inverse.add(item['primary'])
    for i in inverse:
        if i in missing:
            missing.remove(i)
    if len(ReportStr_analyze)>0:   #急诊报告（无描述）不检测部位缺失         
        return list(missing),list(set(inverse))
    else:
        return [],list(inverse)

def applytable_check(StudyPart:str,applyTable:str,modality:str):
    """检测申请单不规范

    Returns:
        _type_: _description_
    """
    missing = set()
    add_info=[]
    for row in ApplyReplaceTable:
        if row['原始值'] is np.nan:
            continue
        if row['替换值'] is np.nan:
            applyTable = applyTable.replace(row["原始值"], "")
        else:
            applyTable = applyTable.lower().replace(row["原始值"].lower(), row["替换值"])
    
    if StudyPart != "" and re.search(applytable_exclud,StudyPart) is None :
        studypart_analyze = get_orientation_position(StudyPart, title=True)
        studypart_analyze=shorten_partlist(studypart_analyze,modality)
    else:
        return None
    if len(studypart_analyze) > 0:
        add_info = [a['axis'] for a in studypart_analyze]
    else:
        return None
    apply_analyze=get_orientation_position(applyTable, add_info=add_info) if applyTable else []

    starts = set(x['word_start'] for x in studypart_analyze)
    for start in starts:
        studyparts = [
            x for x in studypart_analyze if x['word_start'] == start]
        find = False
        for studypart in studyparts:
            temp = [x for x in apply_analyze if (studypart['position'] in x['partlist'] or
                                               x['position'] in studypart['partlist']) and
                    not((x['orientation'] =='左' and studypart['orientation']=='右') or
                        (x['orientation'] =='右' and studypart['orientation']=='左'))]
            if len(temp) > 0:
                find = True
                break

        if find == False:
            missing.add(studypart['orientation']+studypart['position'])
    if len(missing)>0:
        return f"申请单中缺乏对“{','.join(missing)}”的检查原因详细描述"
    else:
        return None




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
            part = [d for d in apply_analyze if set(d['partlist']).issubset(set(s['partlist'])) or 
                                                set(s['partlist']).issubset(set(d['partlist'])) ]
        else:
            part = [d for d in apply_analyze if d['position'] in s['partlist'] or 
                                                s['position'] in d['partlist'] ]
        if part:
            apply_parts.extend(part)
    if apply_parts:
        all_analyze=[]
        all_analyze.extend(ReportStr_analyze)
        all_analyze.extend(Conclusion_analyze)
        # all_analyze=[d for d in all_analyze if d['positive']]
        if all_analyze==[]:
            return ""
        _,inverse=Check_report_conclusion(apply_parts, all_analyze,  modality)
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
    StudyPart=Str_replace(StudyPart, title=True)
    if ReportStrList == []:
        return []
    primaryList = set([x['primary'] for x in ReportStrList])
    for ReportStr in primaryList:
        if (re.search(rf"{enhance}", ReportStr) and re.search("建议|纹理增强|必要|请|^完善", ReportStr) == None and
                re.search(rf"{Exam_enhance}", StudyPart) == None and (modality == "CT" or modality == "MR")):
            # print(re.search(enhance, ReportStr), ReportStr)
            missing.append("本检查无增强，但出现“%s”描述" % ReportStr)
            break
    partlist = [" ".join(y) for y in (x['partlist'] for x in ReportStrList)]
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

def get_positive_dict(Conclusion_analyze, StandardPart):
    """获取阳性部位字典."""
    global positive_level
    MainPartdict = {}
    OtherPartdict = {}
    whole = False
    ResultItems = copy.deepcopy(Conclusion_analyze)
    if len(ResultItems) == 0:
        return {"主要部位": MainPartdict, "其他部位": OtherPartdict, "whole": whole}
    for item in ResultItems:
        if len(item['partlist']) > positive_level:
            item['childpart'] = item['partlist'][positive_level]
        else:
            item['childpart'] = item['partlist'][-1]
    ResultItems = drop_dict_duplicates(
        ResultItems, ["childpart", "root", 'positive'])
    if ("胸部" in StandardPart) and ("心脏" not in StandardPart):
        for item in ResultItems:
            if item["root"] == "心脏":
                item["root"] = "胸部"
    rootparts = set([d['root'] for d in ResultItems]) | StandardPart
    for part in rootparts:
        pos_Items = [d for d in ResultItems if d['root'] == part and d['positive']]
        if len(pos_Items) > 0:
            whole = True
        if part in StandardPart:
            if len(pos_Items) > 0:
                MainPartdict[part] = list(
                    set([d['childpart'] for d in pos_Items]))
            else:
                MainPartdict[part] = ["阴性"]
        else:
            if len(pos_Items) > 0:
                OtherPartdict[part] = list(
                    set([d['childpart'] for d in pos_Items]))
            else:
                OtherPartdict[part] = ["阴性"]
    return {"主要部位": MainPartdict, "其他部位": OtherPartdict, "whole": whole}
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
        missing_find=[d['primary'] for d in ReportStr_analyze if d['positive'] and miss_ignore_pattern.search(d['primary'])==None]
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
                position_find_list = [d for d in Conclusion_analyze if set(d['partlist']).issubset(set(s['partlist'])) or 
                                                    set(s['partlist']).issubset(set(d['partlist'])) ]
            else:
                position_find_list = [d for d in Conclusion_analyze if d['position'] in s['partlist'] or 
                                                    s['position'] in d['partlist'] ]
           
            if len(position_find_list) > 0:
                if [x for x in position_find_list if not((x['orientation']=='左' and s['orientation']=='右') or 
                                                     (x['orientation']=='右' and s['orientation']=='左'))]:
                    missing_find = False
                    
                if s['positive']:
                    position_find_list=[d for d in position_find_list if d['ignore']==False]
                for position_find in position_find_list:
                    if position_find['position'] not in s['partlist']:
                        continue
                    if re.search(orient_ignore,s['primary']) is not None or re.search(orient_ignore,position_find['primary']) is not None:
                        continue
                    # semantics=sentence_semantics( position_find['primary'].split(",")[0],s['primary'].split(",")[0])
                    semantics=struc_sim(position_find,s)
                    if semantics>0 and s['positive']:
                        similarity_list.append({'report':s['primary'].split(",")[0],
                                                'report_orientation':s['orientation'],
                                                'positive':s['positive'],
                                                'report_position':s['position'],
                                                'conclusion_position':position_find['position'],
                                                'conclusion':position_find['primary'].split(",")[0],
                                                'conclusion_orientation':position_find['orientation'],
                                                'semantics': semantics
                                                })
   
        if missing_find and s['positive'] and s['ignore'] == False:
            conclusion_missing.append(s['primary'])
    
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
# 语言矛盾:

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
        key_analyze = [d for d in all_analyze if (part in d['partlist']) and (
            d['orientation'] == "左" or d['orientation'] == "双" or d['orientation'] == "")]
        contradiction.extend(get_contradiction(key_analyze))
        key_analyze = [d for d in all_analyze if (part in d['partlist']) and (
            d['orientation'] == "右" or d['orientation'] == "双" or d['orientation'] == "")]
        contradiction.extend(get_contradiction(key_analyze))
        key_analyze = [d for d in all_analyze if (part in d['partlist']) and (
            d['orientation'] == "双" or d['orientation'] == "")]
        contradiction.extend(get_contradiction(key_analyze))
    return sorted(set(contradiction),key=contradiction.index)

def get_contradiction(key_analyze):
    if len(key_analyze) == 0:
        return []
    positive_key = []
    positive_key = [d for d in key_analyze if d['positive'] == True and
                         re.search(rf"{exclud}", d['primary']) == None]


    negative_key = [d for d in key_analyze if d['positive'] == False and 
                    d['measure']==0 and 
                    len(re.sub(stopwords,"",d['illness']))>1 and re.search(rf"{exclud}", d['primary']) == None]
    if len(negative_key) == 0 or len(positive_key) == 0:
        return []

    found=False
    for negSentence in negative_key:
        for aspect in aspects:
            if re.search(rf"{aspect}", negSentence['primary']):
                positive_str=get_positive_key(negSentence,aspect,positive_key)
                found=True
                if positive_str!="":
                    return [negSentence['primary'], positive_str]
        if found==False:
            positive_str=get_positive_key(negSentence,"",positive_key)
            if positive_str!="":
                pos_sub_part=re.findall(sub_part,positive_str)
                neg_sub_part=re.findall(sub_part,negSentence['primary'])
                if pos_sub_part==[] or neg_sub_part==[] or (set(pos_sub_part) & set(neg_sub_part)): #子部位存在交集
                    return [negSentence['primary'], positive_str]
                else:
                    return []
    return []
def get_positive_key(negSentence,aspect,positive_key):
    positive_key=[d for d in positive_key if negSentence['position'] in d['partlist']]
    if aspect!="":
        positive_key = [d for d in positive_key if re.search(rf"{aspect}", d['primary'])]

    if positive_key:
        positive_key=[{**dic, 'semantics': sentence_semantics(negSentence['primary'],dic['primary'])} for dic in positive_key]  
        semantics_min=min([x['semantics'] for x in positive_key])
        if semantics_min<=0.78:
            return [x['primary'] for x in positive_key if x['semantics']==semantics_min][0]
    return ""

# %%报告复杂度
def reportComplexity(Conclusion_analyze, ReportStr_analyze, modality):
    """计算报告复杂度
        计算公式：复杂度=(病灶数量+1)*权重
        如果是其它设备类型，则使用defautComlexity作为权重
    Args:
        Conclusion_analyze (_type_): 结论
        ReportStr_analyze (_type_): 描述
        modality (_type_): 设备类型

    Returns:
        复杂度数值
    """
    if  "CT" in modality:
        baseScore = CTcomplexity
    elif  "MR" in modality:
        baseScore = MRcomplexity
    elif  "MG" in modality:
        baseScore = MGcomplexity
    elif modality=="DR" or modality=="DX":
        baseScore=DRcomplexity
    else:
        if defautComlexity==0:
            baseScore = DRcomplexity
        else:
            baseScore=defautComlexity
    if len(Conclusion_analyze)>len(ReportStr_analyze):
        analyze=Conclusion_analyze
    else:
        analyze=ReportStr_analyze
    if len(analyze) == 0 :
        return 0
    if [x for x in analyze if x['positive'] == True] == []:
        return baseScore
    illness_count = len(
        [x for x in analyze if x['positive'] == True])
    # print("illness_count=", illness_count)
    return baseScore*(illness_count+1)
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
        # Conclusion_list = set([d['primary'] for d in Conclusion_analyze])
        for c in Conclusion_analyze:
            sentence=c['primary']
            if sentence.strip() == '':
                continue
            # 这里用一个词典来存储忽略词
            if re.search(rf"{CriticalIgnoreWords}", sentence, re.I):
                continue
            result = re.search(rf"{row['结论']}", sentence, re.I)
            if not result:
                continue
            if row['部位'] is not np.nan:
                part_result=[x for x in c['partlist'] if x==row['部位']]
                if not part_result:
                    continue

            # print({"index":index,"category":row['类别'],"description":result.group()})
            if row['描述值'] > 0:
                # print(row['部位'],row['描述值'])
                if row['描述值'] < 1:
                    critical_des = [d for d in all_analyze if row['部位'] in d['partlist'] and
                                    d['percent'] >= row['描述值']]
                else:
                    critical_des = [d for d in all_analyze if row['部位'] in d['partlist'] and
                                    d['measure'] >= row['描述值']]
                if len(critical_des) > 0:
                    if critical_des[0]['primary'] != sentence:
                        sentence = sentence + "：" + critical_des[0]['primary']
                    if not any(x['index'] == row['序号'] for x in CriticalList):
                        CriticalList.append({"index": row['序号'],
                                             "category": row['类别'],
                                             "description": sentence})
            elif row['描述'] is not np.nan:
                if row['描述'].strip() != "":
                    critical_des = [d for d in ReportStr_analyze if re.search(
                        rf"{row['描述']}", d['primary'])]
                    if len(critical_des) > 0:
                        if not any(x['index'] == row['序号'] for x in CriticalList):
                            CriticalList.append({"index": row['序号'],
                                                 "category": row['类别'],
                                                 "description": sentence + "：" + critical_des[0]['primary']})

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
# %%获取标准化部位及部位数量
def get_standardPart(studypart:str, modality:str):
    """解析检查部位获得标准化的部位列表，包括部位分类、检查方式、患者方位、检查方位、特殊检查."""
    studypart_analyze = get_orientation_position(studypart, title=True) if studypart else []
    if len(studypart_analyze) == 0:
        return {
            'StandardPart':'',
            'parts_sum': 0,
            'Part_list':[]
            }
    starts = set([d['word_start'] for d in studypart_analyze])
    "在多个备选部位中挑选坐标范围最长的部位"
    clearUp = []
    for start in starts:
        temp = [x for x in studypart_analyze if x['word_start'] == start]
        if len(temp) == 1:
            clearUp.extend(temp)
        else:
            axis_length = max([x['axis'][1]-x['axis'][0] for x in temp])
            temp = [x for x in temp if x['axis']
                    [1]-x['axis'][0] == axis_length]
            clearUp.extend(temp)

    axis_list = []
    # if modality=="CT":
    for sub in subtraction:
        sub_index = [i for i, d in enumerate(
            clearUp) if sub in d['primary']]
        for i in sub_index:
            if clearUp[i]['axis'][1]-clearUp[i]['axis'][0]>1:
                clearUp[i]['axis'] = (
                    clearUp[i]['axis'][0], clearUp[i]['axis'][1]-1)
    axis_list += [d['axis'] for d in clearUp]
    axis_list = merge_intervals(axis_list)
    parts_sum = sum(np.ceil([x[1]-x[0] for x in axis_list]))

    studyparts = set([d['root'] for d in clearUp])
    result = []
    for root in studyparts:
        part_dict = {}
        part_analyze = [d for d in clearUp if d['root'] == root]
        primary = ''.join([d['primary'] for d in part_analyze])
        primary = primary.replace("两", '双')
        part_dict['Position_orientation'] = ''
        part_dict['Exam_orientation'] = ''
        part_dict['Exam_method'] = '平扫'
        part_dict['Exam_special'] = ''
        part_dict['root'] = root
        part_dict['position'] = list(set(d['position'] for d in part_analyze))
        part_dict['Position_orientation'] = re.findall(Position_orientation, primary)
        if (("左" in part_dict['Position_orientation'] and "右" in part_dict['Position_orientation']) or
                "双" in part_dict['Position_orientation']):
            part_dict['Position_orientation'] = '双'
            if part_analyze[0]['axis'][0]>299:
                parts_sum += 1
        part_dict['Position_orientation'] = list(
            set(part_dict['Position_orientation']))
        part_dict['Exam_orientation'] = re.findall(Exam_orientation, primary)
        part_dict['Exam_orientation'] = list(
            set(part_dict['Exam_orientation']))
        if "平扫" in primary and "增强" in primary:
            part_dict['Exam_method'] = '平扫+增强'
        elif re.search(rf"{Exam_enhance}", primary):
            part_dict['Exam_method'] = "增强"
        if modality == "DR" or modality == "CR" or modality == "MG" or modality == "DX":
            dict_special = dr_special
            if len(part_dict['Exam_orientation']) > 1:
                parts_sum += len(part_dict['Exam_orientation'])-1
        elif modality == "MR":
            dict_special = mr_special
        else:
            dict_special = ct_special
            if len(part_dict['Exam_orientation']) > 1:
                parts_sum += len(part_dict['Exam_orientation'])-1    
        Special_Processor = KeywordProcessor()
        Special_Processor.add_keywords_from_dict(dict_special)
        part_dict['Exam_special'] = list(
            set(Special_Processor.extract_keywords(primary)))
        result.append(part_dict)

    return {
        'StandardPart':",".join(studyparts),
        'parts_sum': parts_sum,
        'Part_list':result
        }

# %%解析报告
def report_analysis(ReportTxt: Report):
    """将报告的描述与结论字段的自然语言转换为结构化字段，以便存入数据库中.
    输出格式：part_level0-5：六级部位列表；position：标准化部位名称；word：原始部位名称；
    illness：疾病描述；measure：轴向测量值；percent：百分比值；primary：原始报告语句
    """
    studypart_analyze = get_orientation_position(ReportTxt.StudyPart, title=True) if ReportTxt.StudyPart else []
    result=[]
    Conclusion_analyze = get_orientation_position(ReportTxt.ConclusionStr, add_info=[
        s['axis'] for s in studypart_analyze]) if ReportTxt.ConclusionStr else []
    Report_analyze = get_orientation_position(ReportTxt.ReportStr, add_info=[
        s['axis'] for s in studypart_analyze]) if ReportTxt.ReportStr else []
    if len(Report_analyze)>0:
        Report_analyze=[{**dic, 'Field': 'report'} for dic in Report_analyze]
    if len(Conclusion_analyze)>0:
        Conclusion_analyze=[{**dic, 'Field': 'Conclusion'} for dic in Conclusion_analyze]
    result.extend(Report_analyze)
    result.extend(Conclusion_analyze)
    for dic in result:
        dic['part_level0']=dic['partlist'][0] if dic['partlist'] else None
        dic['part_level1']=dic['partlist'][1] if len(dic['partlist'])>1 else None
        dic['part_level2']=dic['partlist'][2] if len(dic['partlist'])>2 else None
        dic['part_level3']=dic['partlist'][3] if len(dic['partlist'])>3 else None
        dic['part_level4']=dic['partlist'][4] if len(dic['partlist'])>4 else None
        dic['part_level5']=dic['partlist'][5] if len(dic['partlist'])>5 else None
        del dic['partlist']
        del dic['axis']
        del dic['index']
        del dic['start']
        del dic['ambiguity']
        del dic['ignore']
        del dic['word_start']
        del dic['deny']    
    return result
#%%查询语句转sql
def text_to_SQL(searchStr: str):
    """把输入的自然语言查询语句转换为SQL查询语句"""
    if "AND" in searchStr.upper() and "OR" in searchStr.upper():
        return {"status":400,
                "SQL":"不支持同时使用AND和OR"}
    sql_condition=""
    searchStr=re.sub(r'[^a-zA-Z]and[^a-zA-Z]',",",searchStr,flags=re.I)
    if re.search(r'[^a-zA-Z]or[^a-zA-Z]',searchStr,re.I):
        search_list=re.split(r'[^a-zA-Z]or[^a-zA-Z]',searchStr,flags=re.I)
        for itemStr in search_list:
            sql_sentence=get_search_sql(itemStr)
            if sql_sentence=="":
                continue
            if sql_condition!="":
                sql_condition+=" UNION ("+sql_sentence+")"
            else:
                sql_condition="("+sql_sentence+")"
    else:
        sql_condition=get_search_sql(searchStr)
    if "select" not in sql_condition:
        return {"status":400,
                "SQL":sql_condition}
    else:
        return {"status":200,
                "SQL": sql_condition}
def get_search_sql(itemStr):
    global punctuation
    text_analyze = get_orientation_position(itemStr) if itemStr else []
    text_analyze=drop_dict_duplicates(text_analyze,['position','illness'])
    if len(text_analyze)==0:
        return "无法解析出有效的部位信息"
    text_analyze_positive=[x for x in text_analyze if x['positive']]
    text_analyze_negative=[x for x in text_analyze if not x['positive']]
    if len(text_analyze_positive)==0:
        return "至少需要一个阳性条件"
    sql_Sentence=get_sql_subSentence(text_analyze_positive,True)
    for negative in text_analyze_negative:
        negative['positive']=True
        sql_Sentence=sql_Sentence+" WHERE subquery0.影像号 NOT IN (" +get_sql_subSentence([negative],False)+")"
    return sql_Sentence
    
def get_sql_subSentence(text_analyze,positive):
    condition=""
    sql_query="select subquery0.影像号"
    query_n=0
    for sub_item in text_analyze:
        sub_item['positive']=True
        if condition!="":
            condition+=" INNER JOIN (select 影像号,primary_sentence from StructuredReport where "
        else:
            condition+="(select 影像号,primary_sentence from StructuredReport where "
        condition+=get_sql_condition(sub_item)
        if query_n==0:
            condition+=") AS subquery0"
        else:
            condition+=") AS subquery%s on subquery%s.影像号=subquery%s.影像号" %(query_n,query_n-1,query_n)
        if positive:
            sql_query+=",subquery%s.primary_sentence as 条件%s" %(query_n,query_n+1)
        query_n+=1
    if positive:
        sql_query += ",subquery%s.描述,subquery%s.结论 from %s" %(query_n,query_n,condition)
        sql_query+=""" INNER JOIN (select 影像号,描述,结论 from Report) AS subquery%s 
                on subquery%s.影像号=subquery%s.影像号 """ %(query_n,query_n-1,query_n)
    else:
        sql_query +=" from "+condition
    return sql_query.replace("\n","").replace("  ","")
def get_sql_condition(sub_item:dict):
    condition=""
    condition+="""(part_level0='%s' OR part_level1='%s' OR part_level2='%s' OR 
                        part_level3='%s' OR part_level4='%s' OR part_level5='%s') 
                """ %(sub_item['position'],sub_item['position'],sub_item['position'],
                      sub_item['position'],sub_item['position'],sub_item['position'])
    if sub_item['orientation']=='左' or sub_item['orientation']=='右':
        condition+="AND orientation='%s' " %sub_item['orientation']
    condition+="AND positive=%s " %int(sub_item['positive'])
    sub_item['primary']=sub_item['primary']+"\n"
    if re.search("大于等于|>=|≥|≧",sub_item['primary']):
        if re.search("[^a-zA-Z]cm[^a-zA-Z]|[^a-zA-Z]mm[^a-zA-Z]|厘米|毫米",sub_item['primary'],re.I):
            condition+="AND measure>=%s " %sub_item['measure']
        elif re.search("%|％|百分之",sub_item['primary'],re.I):
            condition+="AND value_percent>=%s " %sub_item['percent']
    elif re.search("小于等于|<=|≤|≦",sub_item['primary']):
        if re.search("[^a-zA-Z]cm[^a-zA-Z]|[^a-zA-Z]mm[^a-zA-Z]|厘米|毫米",sub_item['primary'],re.I):
            condition+="AND measure<=%s " %sub_item['measure']
        elif re.search("%|％|百分之",sub_item['primary'],re.I):
            condition+="AND value_percent<=%s " %sub_item['percent']
    elif re.search("小于|<|﹤|＜",sub_item['primary']):
        if re.search("[^a-zA-Z]cm[^a-zA-Z]|[^a-zA-Z]mm[^a-zA-Z]|厘米|毫米",sub_item['primary'],re.I):
            condition+="AND measure<%s " %sub_item['measure']
        elif re.search("%|％|百分之",sub_item['primary'],re.I):
            condition+="AND value_percent<%s " %sub_item['percent']
    elif re.search("大于|>|﹥|＞",sub_item['primary']):
        if re.search("[^a-zA-Z]cm[^a-zA-Z]|[^a-zA-Z]mm[^a-zA-Z]|厘米|毫米",sub_item['primary'],re.I):
            condition+="AND measure>%s " %sub_item['measure']
        elif re.search("%|％|百分之",sub_item['primary'],re.I):
            condition+="AND value_percent>%s " %sub_item['percent']
    elif re.search("等于|=|＝",sub_item['primary']):
        if re.search("[^a-zA-Z]cm[^a-zA-Z]|[^a-zA-Z]mm[^a-zA-Z]|厘米|毫米",sub_item['primary'],re.I):
            condition+="AND measure=%s " %sub_item['measure']
        elif re.search("%|％|百分之",sub_item['primary'],re.I):
            condition+="AND value_percent=%s " %sub_item['percent']
    keyword=re.sub(r'[\d\.]{1,}(?:mm|cm|m|厘米|毫米|米|×)|[\d\.]{1,}(?:%|％)|大于|小于|等于|>=|≥|≧|<=|≤|≦|<|﹤|＜|>|﹥|＞|百分之',
                   "",sub_item['illness'],flags=re.IGNORECASE)
    keyword=re.sub(punctuation,"",keyword) 
    if keyword not in "正常|未见异常|阳性|阴性|疾病|病变":
        condition+="AND illness like '%%%s%%' " %keyword.replace(" ","")
    return condition

# %%标准化检查部位及费用
def get_standar_Fee(studypart: str, modality: str):
    standardPart_result = get_standardPart(studypart, modality)
    parts_sum = standardPart_result['parts_sum']
    standardPartlist = pd.DataFrame(standardPart_result['Part_list'])
    part_fee = int(parts_sum)*300
    Exam_special = [subelement for sublist in standardPart_result['Part_list']
                    for element in sublist['Exam_special'] for subelement in element]
    if "三维" in ",".join(Exam_special):
        special_Fee = 100
    else:
        special_Fee = 0
    if modality == 'CT':
        spiral = 100
    else:
        spiral = 0
    sum_Fee = part_fee+special_Fee+100
    fee_dic = {"部位数量": parts_sum,
               "部位收费": part_fee,
               "特殊收费": special_Fee,
               "螺旋费": spiral,
               "总费用": sum_Fee}
    return standardPartlist, fee_dic

def punctuation_norm(report):
    pun = [";", ",", "，", ".", "。"]
    pun_index = []
    punctuation = 0
    for i, ch in enumerate(report):
        if ch in pun:
            pun_index.append(i)
    pun = np.array(pun_index, dtype=int)
    diff = np.diff(pun)
    if 1 in set(diff):
        punctuation = 1
    #这里需要明确告知是什么错误，并且把错误放入配置文件中，以便修改
    return "标点符号有错误" if punctuation else "未发现标点符号错误"
# %%报告医生质控函数
def Report_Quality(ReportTxt: Report, debug=False):
    """对初诊医生的报告进行质控，返回阳性率、报告错误等信息."""
    start_time = time.time()
    studypart_analyze = get_orientation_position(
        ReportTxt.StudyPart, title=True) if ReportTxt.StudyPart else []
    Conclusion_analyze = get_orientation_position(ReportTxt.ConclusionStr, add_info=[
        s['axis'] for s in studypart_analyze]) if ReportTxt.ConclusionStr else []
    ReportStr_analyze = get_orientation_position(ReportTxt.ReportStr, add_info=[
        s['axis'] for s in studypart_analyze]) if ReportTxt.ReportStr else []
    ReportTxt.applyTable=ReportTxt.applyTable.replace("  ", "\n")
    apply_analyze=get_orientation_position(ReportTxt.applyTable, add_info=[
        s['axis'] for s in studypart_analyze]) if ReportTxt.applyTable else []
    StandardPart = set()
    GetPositivelist = {}
    missing = []
    inverse = []
    special_missing = []
    sex_error = []
    conclusion_missing = []
    orient_error = []
    contradiction = []
    standardPart_result = []
    none_standard_term = []
    apply_orient=''
    RADS=""
    # 获取标准化部位和部位数量
    if len(studypart_analyze)>0:
        StandardPart = set(x['root'] for x in studypart_analyze)

    # 获取阳性部位字典

    GetPositivelist = get_positive_dict(Conclusion_analyze, StandardPart)

    # 补充缺失部位
    if not StandardPart and GetPositivelist:
        if GetPositivelist['其他部位']:
            StandardPart = GetPositivelist['其他部位']
            GetPositivelist['主要部位'] = GetPositivelist['其他部位']
            GetPositivelist['其他部位'] = {}

    # 检查漏写部位
    if studypart_analyze and re.search(ignore_part,ReportTxt.StudyPart) is None:
        missing,inverse = part_missing(
            studypart_analyze, ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
    missing = "未检查到漏写部位" if not missing else "可能漏写部位: " + "；".join(missing)
    inverse = "未检查到检查项目方位错误" if not inverse else "检查项目方位可能错误: " + "；".join(inverse)
    # 检查漏写特殊检查
    special_missing = check_special_missing(
        ReportTxt.StudyPart, ReportTxt.ReportStr+ReportTxt.ConclusionStr,
        ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
    special_missing = "未检查到漏写特殊检查" if not special_missing else "请注意检查方式: " + \
        "；".join(special_missing)

    # 检查描述与结论相符
    if "骨龄" not in ReportTxt.StudyPart:
        conclusion_missing, orient_error = Check_report_conclusion(
                                            Conclusion_analyze, ReportStr_analyze,ReportTxt.modality)
    if len(conclusion_missing)>1:
        conclusion_missing ="结论可能遗漏: " + "；".join(f"{i+1}.{item}" for i,item in enumerate(conclusion_missing))
    else:
        conclusion_missing = "未检查到描述与结论不符" if not conclusion_missing else "结论可能遗漏: " + "".join(conclusion_missing)
    orient_error = "未检查到描述与结论方位不符" if not orient_error else "以下描述与结论可能方位不符: " + \
        "；".join(orient_error)

    # 检查矛盾
    contradiction = check_contradiction(ReportStr_analyze, Conclusion_analyze,ReportTxt.modality)
    contradiction = "未检测到语言矛盾" if not contradiction else "以下语言可能矛盾： " + \
        "；".join(contradiction)

    # 检查性别错误
    all_analyze = ReportStr_analyze + Conclusion_analyze

    sex_error = CheckSex(all_analyze, ReportTxt.Sex)

    # 判断测量单位错误
    measure_unit_error = CheckMeasure(ReportTxt.ReportStr+"\n"+ReportTxt.ConclusionStr)

    #判断术语不规范
    global MR_non_standard, defult_non_standard,CT_non_standard
    term_list = []
    if len(ReportStr_analyze) > 0:
        term_list.extend([a['primary'] for a in ReportStr_analyze])
    if len(Conclusion_analyze) > 0:
        term_list.extend([a['primary'] for a in Conclusion_analyze])
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
    
    if none_standard_term==[]:
        none_standard_term="未检测到常见术语错误"
    else:
        none_standard_term="；".join(none_standard_term)
    # 标点符号
    all_report=ReportTxt.ReportStr+'\n'+ReportTxt.ConclusionStr    
    punctuation = punctuation_norm(all_report)
    
    #判断申请单方位错误
    apply_orient=applytable_error(apply_analyze,studypart_analyze,ReportStr_analyze,Conclusion_analyze,ReportTxt.modality)
    
    # 判断危急值
    Critical_value = CheckCritical(
        Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)
    # print('------Critical_value---\n',Critical_value)
    # 计算报告复杂度
    complexity = reportComplexity(
        Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)
    #检查RADS分类
    RADS=CheckRADS(ReportTxt.StudyPart, ReportTxt.ConclusionStr,ReportTxt.modality)
    if debug:
        end_time = time.time()
        print("耗时:%.2f秒" % (end_time-start_time))

    return {
        "GetPositivelist": GetPositivelist,#阳性率，whole为人次阳性率
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
        "complexity": complexity,#复杂度
        "punctuation": punctuation,  # 标点
        "apply_orient":apply_orient,
    }

# %%诊断文本比较函数
def get_diagnose_difference(analyze, before_analyze, sum_score):
    """比较报告文本与审核文本的差异并评价差异大小."""
    sum_diagnosis = max(len([report for report in analyze if report['positive'] == True]),
                        len([report for report in before_analyze if report['positive'] == True]))
    if sum_diagnosis == 0:
        sum_diagnosis = max(len(analyze), len(before_analyze))
    before_analyze = [
        report for report in before_analyze if report['positive'] == True]
    analyze = [report for report in analyze if report['positive'] == True]
    before_audit = set(report['primary'] for report in before_analyze)
    audit = set(report['primary'] for report in analyze)
    # print("audit=",audit,"before_audit:",before_audit)
    dic1 = {}
    dic2 = {}
    add_list = []
    modify_report = {}
    add_report = {}
    del_report = {}
    del_list = []
    orientation_list = []
    modify_orientation = {}
    if before_audit == {} and audit == {}:
        return ({"add_report": {"item": [], "deduct": 0},
                "modify_report": {"item": [], "deduct": 0},
                 "del_report": {"item": [], "deduct": 0},
                 "modify_orientation": {"item": [], "deduct": 0},
                 "sum_score": 20})
    if before_audit == audit:
        return ({"add_report": {"item": [], "deduct": 0},
                "modify_report": {"item": [], "deduct": 0},
                 "del_report": {"item": [], "deduct": 0},
                 "modify_orientation": {"item": [], "deduct": 0},
                 "sum_score": 20})

    addsentence = audit-before_audit
    delsentence = before_audit-audit
    opposite_str = ""
    if addsentence != set():
        if delsentence == set():
            add_list = list(addsentence)
        for s1 in addsentence:
            for s2 in delsentence:
                sim = sentence_semantics(s1, s2)
                diff=set(s1) ^ set(s2)
                if sim>0.9 and "左" in diff and "右" in diff:
                    # print("s1=",s1)
                    # print("s2=",s2)
                    orientation_list.append(s1)
                    opposite_str = s2
                else:
                    s1_m,s1_p,s1_volume,s1_positive=find_measure(s1)
                    s2_m,s2_p,s2_volume,s2_positive=find_measure(s2)
                    if s1_m!=s2_m or s1_p!=s2_p or s1_volume!=s2_volume or s1_positive!=s2_positive:
                        sim=0
                    if sim < 0:
                        sim = 0
                    if s1 in dic1:
                        if sim > dic1[s1]:
                            dic1[s1] = sim
                    else:
                        dic1[s1] = sim
                    if s2 in dic2:
                        if sim > dic2[s2]:
                            dic2[s2] = sim
                    else:
                        dic2[s2] = sim
        # print("dic1=",dic1)
        # print("dic2=",dic2)
        if opposite_str != '' and (opposite_str in dic2):
            del dic2[opposite_str]
        # print("opposite_str=",opposite_str)
        for s1 in dic1:
            if dic1[s1] < 0.5:
                add_list.append(s1)
            else:
                modify_report[s1] = dic1[s1]
        for s2 in dic2:
            if dic2[s2] < 0.5:
                del_list.append(s2)
    else:
        del_list = list(delsentence)
    add_report = {"item": add_list, "deduct": np.ceil(
        len(add_list)/sum_diagnosis*sum_score)}
    sum_score -= add_report['deduct']
    del_report = {"item": del_list, "deduct": np.ceil(
        len(del_list)/sum_diagnosis*sum_score)}
    sum_score -= del_report['deduct']
    modify_orientation = {"item": orientation_list, "deduct": np.ceil(
        len(orientation_list)/sum_diagnosis*sum_score)}
    sum_score -= modify_orientation['deduct']
    for key in modify_report:
        modify_report[key] = np.ceil(
            (1-modify_report[key])/sum_diagnosis*sum_score)
        sum_score -= modify_report[key]
    if sum_score < 0:
        sum_score = 0
    return {"add_report": add_report,
            "modify_report": {'item': [key for key in modify_report.keys()],
                              'deduct': sum([value for value in modify_report.values()])},
            "del_report": del_report,
            "modify_orientation": modify_orientation,
            "sum_score": sum_score}

def study_include(studypart_analyze: list[dict],sentence_analyze:dict)->bool:
    """判断是否是附见检查

    Args:
        studypart_analyze (list[dict]): _description_
        sentence_analyze (dict): _description_

    Returns:
        bool: _description_
    """
    temp = [x for x in studypart_analyze if ((x['position'] in sentence_analyze['partlist']) or 
            (sentence_analyze['position'] in x['partlist'])) and ((x['orientation'] == sentence_analyze['orientation']) or
                 ('双' in x['orientation']) or ('双' in sentence_analyze['orientation']) or
                 (x['orientation'] == '') or (sentence_analyze['orientation'] == ''))]
    if len(temp)>0:
        return True
    else:
        temp=sentence_analyze['partlist']
        study=[x['partlist'] for x in studypart_analyze]
        study=[num for sublist in study for num in sublist]
        if "胸部" in study and "心脏" in temp:
            return True
        else:
            return False
def compare_paragraphs_revised(before_analyze: list[dict], analyze: list[dict],studypart_analyze: list[dict], sum_score) -> dict:
    added_sentences = []
    attached_added_sentences=[]
    added_deduct=[]
    modified_pairs = []
    measure_pairs=[]
    modify_deduct=[]  # 用于存储修改扣分
    deleted_sentences = []
    attached_deleted_sentences = []
    deleted_deduct=[]
    orientation_list=[] #方位错误
    len_p1 = len(before_analyze)
    len_p2 = len(analyze)
    sum_diagnosis = max(len(set([report['primary'] for report in analyze if report['positive'] == True])),
                        len(set([report['primary'] for report in before_analyze if report['positive'] == True])))
    if sum_diagnosis == 0:
        sum_diagnosis = max(len(analyze), len(before_analyze))

    before_audit = set(report['primary'] for report in before_analyze)
    audit = set(report['primary'] for report in analyze)
    if before_audit==audit or before_audit=={} or audit=={}:
        return ({"add_report": {"item": [],"attached_item":[], "deduct": 0},
            "modify_report": {"item": [],"measure": [], "deduct": 0},
            "del_report": {"item": [],"attached_item":[], "deduct": 0},
            "modify_orientation": {"item": [], "deduct": 0},
            "sum_score": sum_score})
    p1_matched_flags = [False] * len_p1
    p2_matched_flags = [False] * len_p2

    for i in range(len_p1):
        if p1_matched_flags[i]:
            continue
        for j in range(len_p2):
            if p2_matched_flags[j]:
                continue
            if before_analyze[i]['primary']==analyze[j]['primary']:
                p1_matched_flags[i]=True
                p2_matched_flags[j]=True
                break
            
    candidates = []
    for i in range(len_p1):
        if p1_matched_flags[i]:
            continue
        for j in range(len_p2):
            if p2_matched_flags[j]:
                continue
            similarity = struc_sim(before_analyze[i], analyze[j],complete=True) # 调用传入的相似度函数
            if similarity > 0.41 and similarity<1:
                candidates.append((similarity, i, j)) # (similarity, p1_index, p2_index)

    # 按相似度降序排序
    candidates.sort(key=lambda x: x[0], reverse=True)

    for sim, p1_idx, p2_idx in candidates:
        # print(before_analyze[p1_idx]['primary'], analyze[p2_idx]['primary'],sim)
        if not p1_matched_flags[p1_idx] and not p2_matched_flags[p2_idx]:
            s1_dict = before_analyze[p1_idx]
            s2_dict = analyze[p2_idx]

            
            modified_pairs.append(f'"{s1_dict["primary"]}" 修改为 "{s2_dict["primary"]}"')
            deduct=0
            if s1_dict['measure']!=s2_dict['measure']>0 :
                measure_pairs.append(f'测量值"{s1_dict["measure"]}" 修改为 "{s2_dict["measure"]}"')
                deduct=sum_score/sum_diagnosis
            if  s1_dict['percent']!=s2_dict['percent']>0 :
                measure_pairs.append(f'百分比"{s1_dict["measure"]}" 修改为 "{s2_dict["measure"]}"')
                deduct=sum_score/sum_diagnosis
            if  s1_dict['volume']!=s2_dict['volume']>0 :
                measure_pairs.append(f'体积"{s1_dict["measure"]}" 修改为 "{s2_dict["measure"]}"')
                deduct=sum_score/sum_diagnosis
            if  s1_dict['positive']!=s2_dict['positive']:
                deduct=sum_score/sum_diagnosis
            diff=set(s1_dict['primary']) ^ set(s2_dict['primary'])
            if  sum(k in diff for k in ("左", "右")) >= 2 and sim>0.8:
                orientation_list.append(s1_dict['primary'])
            if  s1_dict['positive']!=s2_dict['positive']:
                deduct=sum_score/sum_diagnosis
            else:
                deduct=np.ceil(sum_score/sum_diagnosis*(1-sim))
                
            modify_deduct.append(deduct)
            # 标记已匹配（无论是相同还是修改）
            p1_matched_flags[p1_idx] = True
            p2_matched_flags[p2_idx] = True
    orientation_deduct=sum_score if orientation_list else 0

    for j in range(len_p2):
        if not p2_matched_flags[j] and analyze[j]['primary'] not in added_sentences and analyze[j]['primary'] not in ''.join(modified_pairs):
            if study_include(studypart_analyze,analyze[j]):
                added_sentences.append(analyze[j]['primary'])
                added_deduct.append(sum_score/sum_diagnosis)
            else:
                attached_added_sentences.append(analyze[j]['primary'])
                added_deduct.append(sum_score/sum_diagnosis*0.5)

    for i in range(len_p1):
        if not p1_matched_flags[i]  and before_analyze[i]['primary'] not in deleted_sentences and before_analyze[i]['primary'] not in ''.join(modified_pairs):
            if study_include(studypart_analyze,before_analyze[i]):
                deleted_sentences.append(before_analyze[i]['primary'])
                deleted_deduct.append(sum_score/sum_diagnosis)
            else:
                attached_deleted_sentences.append(before_analyze[i]['primary'])
                deleted_deduct.append(sum_score/sum_diagnosis*0.5)
    sum_deduct=sum(added_deduct)+sum(modify_deduct)+sum(deleted_deduct)+orientation_deduct
    return {"add_report": {"item": list(set(added_sentences)), 
                           "attached_item": list(set(attached_added_sentences)),
                           "deduct": sum(added_deduct) if sum(added_deduct)<=sum_score else sum_score},
            "modify_report": {"item": list(set(modified_pairs)),
                              "measure": list(set(measure_pairs)), 
                              "deduct": sum(modify_deduct)  if sum(added_deduct)<=sum_score else sum_score},
            "del_report": {"item":list(set(deleted_sentences)),
                           "attached_item":list(set(attached_deleted_sentences)), 
                           "deduct": sum(deleted_deduct)  if sum(added_deduct)<=sum_score else sum_score},
            "modify_orientation": {"item": list(set(orientation_list)), 
                                   "deduct": orientation_deduct},
            "sum_score": (sum_score-sum_deduct) if (sum_score-sum_deduct)>0 else 0
            }

def explain_text(result):
    text=""
    if result['add_report']['item']: #遗漏病变
        text+="### 新增:\n" +'\n'.join([f"- {sentence}" for sentence in result['add_report']['item']]) +"\n"
    # if 'attached_item' in result['add_report']:
    #     if result['add_report']['attached_item']: #遗漏附见病变
    #         text+="新增(附见):" +";".join(result['add_report']['attached_item'])+"\n"
    if result['modify_report']['item']: #修改病变
        text+="### 纠正:\n" +'\n'.join([f"- {sentence}" for sentence in result['modify_report']['item']]) +"\n"
    # if result['modify']['measure']: #修改病变
    #     text+="数值:" +'\n'.join([f"- {sentence}" for sentence in result['modify']['measure']])
    if result['del_report']['item']: #删除病变
        text+="### 删除:\n" +'\n'.join([f"- {sentence}" for sentence in result['del_report']['item']]) 
    # if 'attached_item' in result['del_report']:
    #     if result['del_report']['attached_item']: #遗漏附见病变
    #         text+="删除(附见):" +";".join(result['del_report']['attached_item'])+"\n"
    # if result['orientation']['item']: #方位错误
    #     text+="方位错误:" +";".join(result['orientation']['item'])+"\n"    
    return text
# %%审核医生质控函数
def Audit_Quality(ReportTxt: AuditReport, debug=False):
    """对审核医生的报告进行质控，返回阳性率，报告错误，以及报告评级（比较对初诊报告的修改程度）."""
    studypart_analyze = []
    Conclusion_analyze = []
    ReportStr_analyze = []
    StandardPart = set()
    #parts_sum = 0
    GetPositivelist = {}
    missing = []
    inverse = []
    special_missing = []
    sex_error = []
    conclusion_missing = []
    orient_error = []
    contradiction = []
    standardPart_result = []
    add_info = []
    apply_orient=''
    RADS=None
    # 获取标准化部位
    start_time = time.time()
    if ReportTxt.StudyPart != "":
        studypart_analyze = get_orientation_position(
            ReportTxt.StudyPart, title=True)
    if len(studypart_analyze) > 0:
        add_info = [a['axis'] for a in studypart_analyze]
        StandardPart = set(x['root'] for x in studypart_analyze)
    apply_analyze=get_orientation_position(ReportTxt.applyTable, add_info=add_info) if ReportTxt.applyTable else []
    
    # 获取阳性部位字典

    Conclusion_analyze = get_orientation_position(ReportTxt.afterConclusionStr, add_info=add_info)
    GetPositivelist = get_positive_dict(Conclusion_analyze, StandardPart)
    # 补充缺失部位
    if not StandardPart and GetPositivelist.get('其他部位'):
        StandardPart = list(GetPositivelist['其他部位'])
        GetPositivelist['主要部位'] = GetPositivelist['其他部位']
        GetPositivelist['其他部位'] = {}
    # 检查漏写部位
    if ReportTxt.afterReportStr.strip() != '':
        ReportStr_analyze = get_orientation_position(
            ReportTxt.afterReportStr, add_info=add_info)
    if len(studypart_analyze) > 0 and re.search(ignore_part,ReportTxt.StudyPart) is None:
        after_missing,after_inverse = part_missing(
            studypart_analyze, ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
        missing = "未检查到漏写部位" if not after_missing else "可能漏写部位: " + "；".join(after_missing)
        inverse = "未检查到检查项目方位错误" if not after_inverse else "检查项目方位可能错误: " + "；".join(after_inverse)
    # 检查漏写特殊检查
    special_missing = check_special_missing(
        ReportTxt.StudyPart, ReportTxt.afterReportStr+ReportTxt.afterConclusionStr,
        ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
    special_missing = "未检查到漏写特殊检查" if not special_missing else "请注意检查方式: " + \
        "；".join(special_missing)
    # 检查描述与结论相符
    if "骨龄" not in ReportTxt.StudyPart:
        conclusion_missing, orient_error = Check_report_conclusion(
                    Conclusion_analyze, ReportStr_analyze,ReportTxt.modality)
    if len(conclusion_missing)>1:
        conclusion_missing ="结论可能遗漏: " + "；".join(f"{i+1}.{item}" for i,item in enumerate(conclusion_missing))
    else:
        conclusion_missing = "未检查到描述与结论不符" if not conclusion_missing else "结论可能遗漏: " + "".join(conclusion_missing)
    orient_error = "未检查到描述与结论方位不符" if not orient_error else "以下描述与结论可能方位不符: " + \
        "；".join(orient_error)
    # 检查矛盾
    contradiction = check_contradiction(ReportStr_analyze, Conclusion_analyze,ReportTxt.modality)
    contradiction = "未检测到语言矛盾" if not contradiction else "以下语言可能矛盾： " + \
        "；".join(contradiction)
    # 检查性别错误
    all_analyze = ReportStr_analyze + Conclusion_analyze

    sex_error = CheckSex(all_analyze, ReportTxt.Sex)
    #标点符号
    all_report=ReportTxt.afterReportStr+"\n"+ReportTxt.afterConclusionStr
    punctuation = punctuation_norm(all_report)
    
    # 判断测量单位错误
    measure_unit_error = CheckMeasure(ReportTxt.afterReportStr+"\n"+ReportTxt.afterConclusionStr)

    #判断术语不规范
    global MR_non_standard, defult_non_standard
    none_standard_term = []
    term_list = []
    if len(ReportStr_analyze) > 0:
        term_list.extend([a['primary'] for a in ReportStr_analyze])
    if len(Conclusion_analyze) > 0:
        term_list.extend([a['primary'] for a in Conclusion_analyze])
    term_list = set(term_list)
    for term in term_list:
        if ReportTxt.modality == "MR":
            mat = re.findall(f"{MR_non_standard}|{defult_non_standard}", term,re.I)
        else:
            mat = re.findall(CT_non_standard +"|"+defult_non_standard, term,re.I)
        mat.extend(detect_abnormal_medical_terms(term))
        if mat != []:
            none_standard_term.append(
                term + " 中术语“" + ",".join(mat) + "”不标准")
    if none_standard_term==[]:
        none_standard_term="未检测到常见术语错误"
    else:
        none_standard_term="；".join(none_standard_term)

    #判断申请单方位错误
    apply_orient=applytable_error(apply_analyze,studypart_analyze,ReportStr_analyze,Conclusion_analyze,ReportTxt.modality)
    
    #检查RADS分类
    RADS=CheckRADS(ReportTxt.StudyPart, ReportTxt.afterConclusionStr,ReportTxt.modality)

    # 判断危急值
    Critical_value = CheckCritical(
        Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)

    # 计算报告复杂度
    complexity = reportComplexity(
        Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)

    #报告评级
    global part_correct_score_sum,conclusion_sum_score,report_sum_score,language_score_sum,standard_term_score_sum,audit_score_sum
    # 一、报告部位规范性
    part_correct_score = part_correct_score_sum
    part_correct_explain = ""
    before_missing = []
    beforeReportStr_analyze = []
    beforeConclusion_analyze = []
    before_inverse=""
    if ReportTxt.beforeReportStr.strip() != '':
        beforeReportStr_analyze = get_orientation_position(
            ReportTxt.beforeReportStr, add_info=add_info)
    if ReportTxt.beforeConclusionStr.strip() != '':
        beforeConclusion_analyze = get_orientation_position(
            ReportTxt.beforeConclusionStr, add_info=add_info)

    if len(studypart_analyze) > 0:
        before_missing,before_inverse = part_missing(
            studypart_analyze, beforeReportStr_analyze, beforeConclusion_analyze, ReportTxt.modality)
    if len(StandardPart) == 0 or len(beforeReportStr_analyze) == 0:
        part_correct_score = 0
    if len(before_missing) > 0:
        audit_missing = set(before_missing)-set(after_missing)
        if len(audit_missing) > 0:
            part_correct_explain = "报告漏写部位:"+"；".join(audit_missing)
            part_correct_score = 0
    if len(before_inverse) > 0:
        audit_inverse = set(before_inverse)-set(after_inverse)
        if len(audit_inverse) > 0:
            part_correct_explain = "检查项目方位错误:"+"；".join(audit_inverse) if part_correct_explain=="" else part_correct_explain+"\n检查项目方位错误:"+"；".join(audit_inverse)
            part_correct_score = 0
    # 二、结论修改
    #conclusion_sum_score = 20
    conclusion_score = {"add_report": {"item": [], "deduct": 0},
                        "modify_report": {"item": [], "deduct": 0},
                        "del_report": {"item": [], "deduct": 0},
                        "modify_orientation": {"item": [], "deduct": 0},
                        "sum_score": conclusion_sum_score}

    if len(Conclusion_analyze) > 0 and len(beforeConclusion_analyze) > 0:
        # conclusion_score = get_diagnose_difference(
        #     Conclusion_analyze, beforeConclusion_analyze, conclusion_sum_score)
        conclusion_score =compare_paragraphs_revised(beforeConclusion_analyze,Conclusion_analyze,studypart_analyze, conclusion_sum_score)
    # 三、描述修改
    # report_sum_score = 20
    report_score = {"add_report": {"item": [], "deduct": 0},
                    "modify_report": {"item": [], "deduct": 0},
                    "del_report": {"item": [], "deduct": 0},
                    "modify_orientation": {"item": [], "deduct": 0},
                    "sum_score": report_sum_score}

    if len(ReportStr_analyze) > 0 and len(beforeReportStr_analyze) > 0:
        # report_score = get_diagnose_difference(
        #     ReportStr_analyze, beforeReportStr_analyze, report_sum_score)
        report_score =compare_paragraphs_revised(beforeReportStr_analyze,ReportStr_analyze,studypart_analyze, report_sum_score)
    # 四、一般性语言错误
    language_score = language_score_sum
    language_explain = ""
    measure_error = CheckMeasure(ReportTxt.beforeReportStr+"\n"+ReportTxt.beforeConclusionStr)
    if "值" in measure_error:
        afer_measure = [m['measure']
                        for m in ReportStr_analyze if m['measure'] > 0]
        before_measure = [m['measure']
                          for m in beforeReportStr_analyze if m['measure'] > 0]
        measure_difference = [
            str(int(x))+'mm' for x in before_measure if x not in afer_measure]
        if measure_difference != []:
            language_score=0
            language_explain = "测量值错误:" + " ".join(measure_difference)
    modify_orientation = []
    modify_orientation.extend(conclusion_score['modify_orientation']["item"])
    modify_orientation.extend(report_score['modify_orientation']["item"])
    if modify_orientation != []:
        language_score =0
        if language_explain == "":
            language_explain = "方位错误:" + "；".join(modify_orientation)
        else:
            language_explain = language_explain + \
                "\n方位错误:" + "；".join(modify_orientation)
    before_all_analyze = beforeReportStr_analyze + beforeConclusion_analyze
    before_sex_error = CheckSex(before_all_analyze, ReportTxt.Sex)
    if "未" not in before_sex_error:
        if language_explain == "":
            language_explain = before_sex_error
        else:
            language_explain = language_explain+"\n" + before_sex_error
        language_score=0
    if language_score < 0:
        language_score = 0

    # 五、用语不规范
    standard_term_score = standard_term_score_sum
    standard_term_explain = []
    term_list = []
    if len(beforeReportStr_analyze) > 0:
        term_list.extend([a['primary'] for a in beforeReportStr_analyze])
    if len(beforeConclusion_analyze) > 0:
        term_list.extend([a['primary'] for a in beforeConclusion_analyze])
    term_list = set(term_list)
    for term in term_list:
        if ReportTxt.modality == "MR":
            mat = re.findall(MR_non_standard, term)
        else:
            mat = re.findall(defult_non_standard, term)
        mat.extend(detect_abnormal_medical_terms(term))
        if mat != []:
            standard_term_explain.append(
                term.replace("\n","") + " 中术语“" + ",".join(mat) + "”不标准")
            standard_term_score -= standard_term_score_sum/5
    if standard_term_score < 0:
        standard_term_score = 0

    # 六、二级审核
 
    if (ReportTxt.report_doctor == "" or ReportTxt.audit_doctor == "" or ReportTxt.audit_doctor == ReportTxt.report_doctor or
       len(beforeConclusion_analyze) == 0 or len(beforeReportStr_analyze) == 0):
        audit_score = 0
    else:
        audit_score = audit_score_sum

    # 报告分级
    sum_score = part_correct_score + \
        conclusion_score['sum_score']+report_score['sum_score'] + \
        language_score+standard_term_score+audit_score
    if sum_score<0:
        sum_score=0
    if sum_score >= int(A_level[0]):
        report_level = "A"
    elif sum_score < int(B_level[1]) and sum_score >= int(B_level[0]):
        report_level = "B"
    else:
        report_level = "C"
    if debug:
        end_time = time.time()
        print("总耗时:%.2f秒" % (end_time-start_time))

    return {
           "GetPositivelist": GetPositivelist,
            "partmissing": missing,
            "partinverse":inverse, #检查项目方位错误 
            "special_missing": special_missing,
            "conclusion_missing": conclusion_missing,
            "orient_error": orient_error,
            "contradiction": contradiction,
            "sex_error": sex_error,
            "measure_unit_error": measure_unit_error,
            "none_standard_term":none_standard_term,
            "punctuation": punctuation,  # 标点
            "apply_orient":apply_orient, #申请单方位
            'RADS':RADS,#分类
            "Critical_value": Critical_value,
            "complexity": complexity,
            "part_correct_score": part_correct_score, "part_correct_explain": part_correct_explain,
            "conclusion_score": conclusion_score,
            "report_score": report_score,
            "llm_explain":explain_text(conclusion_score),
            "language_score": language_score, "language_explain": language_explain,
            "standard_term_explain": standard_term_explain, "standard_term_score": standard_term_score,
            "audit_score": audit_score,
            "sum_score": sum_score, "report_level": report_level}

#%%漏诊分析
def Missed_diagnosis(HistoryInfoText:HistoryInfo):
    abstractText=HistoryInfoText.abstract
    StudyPartStr=HistoryInfoText.report.StudyPart
    ConclusionStr=HistoryInfoText.report.ConclusionStr
    modality=HistoryInfoText.report.modality
    recent_result=History_Summary(StudyPartStr, abstractText,debug=True)
    recent_result=[x for x in recent_result if x['modality']!=modality and x['modality']!='住院']
    if recent_result==[]:
        return None
    missed=[]
    studypart_analyze = get_orientation_position(
        StudyPartStr, title=True) if StudyPartStr else []
    Conclusion_analyze = get_orientation_position(ConclusionStr, add_info=[
        s['axis'] for s in studypart_analyze]) if ConclusionStr else []
    Conclusion_analyze=[x for x in Conclusion_analyze if x['positive']==True]
    primary=set([x['primary'] for x in recent_result])
    for start in primary:
        sentence = [d for d in recent_result if d['primary'] == start]
        find=False
        for other in sentence:
            temp=[x for x in Conclusion_analyze if (x['position'] in other['partlist'] 
                or other['position'] in x['partlist']) and 
                not((x['orientation']=='左' and other['orientation']=='右') or 
                    (other['orientation']=='左' and x['orientation']=='右'))]
            if len(temp)>0:
                find=True
        if find==False:
            missed.append(other['date']+other['modality']+":"+other['primary'])
    if missed!=[]:
        return ";".join(missed)
    else:
        return None


    


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
    print("Report_Quality=", Report_Quality(a, debug=True))