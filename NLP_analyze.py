# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
import jieba  
from Extract_Entities import text_extrac_process
from semantic_service import get_matcher
import warnings
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

# 新模块化结构
from report_analyze import (
    check_report_conclusion,
    check_contradiction,
    batch_validate_with_llm,
    SystemConfig,
    UserConfig,
)

# 导入 miss_ignore_pattern 用于结论缺失检测过滤
miss_ignore_pattern = SystemConfig.miss_ignore_pattern()

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
    
# 加载配置（通过新配置模块）
# 仅保留实际使用的配置变量
Ignore_sentence = SystemConfig.ignore_sentence()
enhance = SystemConfig.enhance()
dwi = SystemConfig.dwi()
swi = SystemConfig.swi()
perfusion = SystemConfig.perfusion()
MRS = SystemConfig.mrs()
upper_position = SystemConfig.upper_position()
CriticalIgnoreWords = SystemConfig.critical_ignore_words()
ignore_part = SystemConfig.ignore_part()

defult_non_standard = UserConfig.default_non_standard()
MR_non_standard = UserConfig.mr_non_standard()
CT_non_standard = UserConfig.ct_non_standard()
cm_max = UserConfig.cm_max()
mm_max = UserConfig.mm_max()
m_max = UserConfig.m_max()
MaleKeyWords = UserConfig.male_keywords()
FemaleKeyWords = UserConfig.female_keywords()
Exam_enhance = UserConfig.exam_enhance()
check_modality = UserConfig.check_modality()
missing_exclud = UserConfig.missing_exclud()

# 初始化语义匹配器（供report_analyze模块使用）
_matcher = None
def get_semantic_matcher():
    global _matcher
    if _matcher is None:
        _matcher = get_matcher()
    return _matcher

# ============ partlist 辅助函数（适配 Extract_Entities.py 的 List[tuple] 结构） ============

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


# 危急值规则表
ruletable_dict = pd.read_excel('config/criticalvalue.xlsx').to_dict('records')

# 初始化jieba
jieba.initialize()


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
        # 检查报告结论
        _, inverse, _ = check_report_conclusion(apply_parts, all_analyze, modality)
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




# %%危急值
def CheckCritical(
    Conclusion_analyze: List[Dict], 
    ReportStr_analyze: List[Dict], 
    modality: str
) -> List[Dict]:
    """
    危急值检测
    
    根据危急值规则表，检查报告描述和结论中是否存在危急值。
    支持两种检测模式：
    1. 数值型：根据 percent/measure 阈值判断
    2. 描述型：根据正则表达式匹配描述文本
    
    Args:
        Conclusion_analyze: 解析后的结论实体列表
        ReportStr_analyze: 解析后的描述实体列表  
        modality: 设备类型 (CT/MR/DR等)
    
    Returns:
        List[Dict]: 危急值列表，每项包含:
            - index: 规则序号
            - category: 危急值类别
            - description: 描述文本（可能包含关联的描述实体）
    """
    # 快速返回：空结论
    if not Conclusion_analyze:
        return []
    
    # 合并所有实体用于后续查找
    all_analyze = ReportStr_analyze + Conclusion_analyze
    
    # 使用集合记录已处理的规则序号，避免重复检查
    processed_indices = set()
    critical_list = []
    
    # 预编译忽略词正则（提高性能）
    ignore_pattern = re.compile(rf"{CriticalIgnoreWords}", re.I)
    
    for rule in ruletable_dict:
        rule_idx = rule.get('序号')
        
        # 跳过已处理的规则
        if rule_idx in processed_indices:
            continue
        
        # 设备类型过滤
        rule_devices = rule.get('设备')
        if pd.notna(rule_devices):
            if not re.search(rf"{rule_devices}", modality, re.I):
                continue
        
        # 结论规则检查
        rule_conclusion = rule.get('结论')
        if pd.isna(rule_conclusion):
            continue
        
        conclusion_pattern = re.compile(rf"{rule_conclusion}", re.I)
        
        # 遍历结论实体进行匹配
        for entity in Conclusion_analyze:
            sentence = entity.get('original_short_sentence', '').strip()
            if not sentence:
                continue
            
            # 忽略词过滤
            if ignore_pattern.search(sentence):
                continue
            
            # 结论正则匹配
            if not conclusion_pattern.search(sentence):
                continue
            
            # 部位匹配检查
            rule_part = rule.get('部位')
            if pd.notna(rule_part):
                if not position_in_any_partlist(rule_part, entity):
                    continue
            
            # === 匹配成功，构建危急值记录 ===
            critical_item = _build_critical_item(
                rule=rule,
                sentence=sentence,
                all_analyze=all_analyze,
                report_analyze=ReportStr_analyze
            )
            
            if critical_item:
                critical_list.append(critical_item)
                processed_indices.add(rule_idx)
                break  # 该规则已匹配，跳出实体循环
    
    return critical_list


def _build_critical_item(
    rule: Dict,
    sentence: str,
    all_analyze: List[Dict],
    report_analyze: List[Dict]
) -> Optional[Dict]:
    """
    构建危急值记录项
    
    根据规则类型（数值型/描述型/纯结论型）构建对应的危急值记录
    
    Args:
        rule: 危急值规则字典
        sentence: 匹配的结论句子
        all_analyze: 所有实体列表（用于查找关联描述）
        report_analyze: 描述实体列表
    
    Returns:
        Dict或None: 危急值记录项
    """
    rule_idx = rule.get('序号')
    rule_category = rule.get('类别', '')
    rule_part = rule.get('部位')
    desc_threshold = rule.get('描述值', 0)
    desc_pattern = rule.get('描述')
    
    # 情况1: 数值型阈值检查（percent >= 1 或 measure >= 阈值）
    if desc_threshold > 0:
        # 查找满足数值条件的描述实体
        matching_descriptions = _find_critical_descriptions(
            rule_part=rule_part,
            threshold=desc_threshold,
            all_analyze=all_analyze
        )
        
        if matching_descriptions:
            # 合并结论和描述
            desc_sentence = matching_descriptions[0].get('original_short_sentence', '')
            if desc_sentence and desc_sentence != sentence:
                sentence = f"{sentence}：{desc_sentence}"
            
            return {
                "index": rule_idx,
                "category": rule_category,
                "description": sentence
            }
        return None
    
    # 情况2: 描述型规则（需要匹配描述文本）
    if pd.notna(desc_pattern) and str(desc_pattern).strip():
        desc_regex = re.compile(rf"{desc_pattern}")
        matching_descriptions = [
            d for d in report_analyze 
            if desc_regex.search(d.get('original_short_sentence', ''))
        ]
        
        if matching_descriptions:
            desc_sentence = matching_descriptions[0].get('original_short_sentence', '')
            return {
                "index": rule_idx,
                "category": rule_category,
                "description": f"{sentence}：{desc_sentence}"
            }
        return None
    
    # 情况3: 纯结论型（无需额外检查）
    return {
        "index": rule_idx,
        "category": rule_category,
        "description": sentence
    }


def _find_critical_descriptions(
    rule_part: str,
    threshold: float,
    all_analyze: List[Dict]
) -> List[Dict]:
    """
    查找满足数值阈值的描述实体
    
    Args:
        rule_part: 规则指定的部位
        threshold: 阈值（<1 表示百分比，>=1 表示绝对数值）
        all_analyze: 所有实体列表
    
    Returns:
        List[Dict]: 满足条件的实体列表
    """
    is_percentage = threshold < 1
    results = []
    
    for entity in all_analyze:
        # 部位匹配
        if pd.notna(rule_part):
            if not position_in_any_partlist(rule_part, entity):
                continue
        
        # 数值检查
        if is_percentage:
            if entity.get('percent', 0) >= threshold:
                results.append(entity)
        else:
            if entity.get('measure', 0) >= threshold:
                results.append(entity)
    
    return results

#%%RADS分类检测
def CheckRADS(StudyPartStr:str,ConclusionStr:str,modality:str):
    if (modality=="MG" or modality=="DR" or modality=="DX" or modality=="MR" ) and "乳腺" in StudyPartStr:
        if re.search("BI.*RADS",ConclusionStr,flags=re.I |re.DOTALL)==None:
            return "缺少BI-RADS分类"
    if modality=="MR"  and "前列腺" in StudyPartStr:
        if re.search("PI.*RADS",ConclusionStr,flags=re.I |re.DOTALL)==None:
            return "缺少PI-RADS分类"
    return ""

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
        conclusion_missing, orient_error, conclusion_perf = check_report_conclusion(
                                            Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)
    timers['描述结论检查'] = time.time() - t0
    
    # 提取描述结论检查内部详细性能统计
    if conclusion_perf:
        timers['Rerank调用'] = conclusion_perf.get('rerank_time', 0)
        timers['Rerank次数'] = conclusion_perf.get('rerank_calls', 0)
        timers['语义相似度计算'] = conclusion_perf.get('semantic_sim_time', 0)
        timers['语义计算次数'] = conclusion_perf.get('semantic_sim_calls', 0)
        
        # 内部详细步骤
        detailed = conclusion_perf.get('detailed', {})
        for name, t in detailed.items():
            timers[f'[结论检查]{name}'] = t
    
    # 检查矛盾
    t0 = time.time()
    contradiction = check_contradiction(ReportStr_analyze, Conclusion_analyze, ReportTxt.modality)
    timers['矛盾检查'] = time.time() - t0
    
    # === Stage 2-4: 统一批量LLM验证 ===
    t0 = time.time()
    llm_validation_triggered = False
    if conclusion_missing or orient_error or contradiction:
        llm_validation_triggered = True
        conclusion_missing, orient_error, contradiction = batch_validate_with_llm(
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
    
   
    #判断申请单方位错误
    apply_orient=applytable_error(apply_analyze,studypart_analyze,ReportStr_analyze,Conclusion_analyze,ReportTxt.modality)
    
    # 判断危急值
    Critical_value = CheckCritical(
        Conclusion_analyze, ReportStr_analyze, ReportTxt.modality)

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
        
        # 提取整数计数项（不是时间）
        rerank_calls = timers.pop('Rerank次数', 0)
        semantic_calls = timers.pop('语义计算次数', 0)
        
        # 过滤掉值为0的计时器
        filtered_timers = {k: v for k, v in timers.items() if v > 0.001}
        
        # 按耗时排序
        sorted_timers = sorted(filtered_timers.items(), key=lambda x: x[1], reverse=True)
        for name, t in sorted_timers:
            pct = (t / total_time * 100) if total_time > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"{name:20s}: {t:6.3f}s ({pct:5.1f}%) {bar}")
        
        # 显示调用次数
        print()
        print("调用次数统计:")
        if rerank_calls:
            print(f"  Rerank调用: {int(rerank_calls)}次")
        if semantic_calls:
            print(f"  语义相似度计算: {int(semantic_calls)}次")
        
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
鼻腔左侧钩突(位于中鼻甲前上方的骨性突起)形态异常其尖端部分可见一局限性软组织密度影，大小约1cmx0.8cm，边缘较清晰，与周围组织分界尚可，未见明显骨质破坏。
        """,
   ConclusionStr = """
鼻腔右侧钩突肥大并伴有软组织肿块，考虑慢性炎症或肿瘤性病变可能性大建议进一步IRI检查及活检以明确诊断。

   """,
        StudyPart = '副鼻窦',
        Sex = '男',
        modality = "CT",
        applyTable=""
 
    )
    print("Report_Quality=", Report_Quality(a, debug=False))


