from NLP_analyze import *
from tqdm import tqdm
from multiprocessing import Pool
from time import time
import os
import json
def process_report_with_process_pool(report, num_processes=16):
    pool = Pool(num_processes)
   
    # 将报告库分割成num_processes个部分
    report_chunks = [report[i::num_processes] for i in range(num_processes)]
    
    # 使用map_async来并行化
    # report_list = pool.map_async(process_report, report_chunks).get(999999)  # 999999是一个足够大的超时时间
    report_list = list(tqdm(pool.imap_unordered(process_report, report_chunks), total=len(report_chunks), desc="报告查错进度"))
    # 合并多个对象
    total_df = pd.concat(report_list,axis=0)
    pool.close()
    pool.join()
    return total_df



def process_report(report_chunks):
    # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #       '准备处理报告子任务%s条' % len(report_chunks.index))
    df = []
    start_time = time()
    
    for index, row in tqdm(report_chunks.iterrows(),total=len(report_chunks),desc="子进程进度"):
        a = Report(
            ReportStr=row['描述'],
            ConclusionStr=row['结论'],
            StudyPart = row['部位'],
            Sex =  row['性别'],
            modality = row['类型'],
            applyTable=row['申请单']
        )
        Quality=Report_Quality(a)
        Quality['ReportStr']=row['描述']
        Quality['ConclusionStr']=row['结论']
        Quality['StudyPart']=row['部位']
        Quality['Sex']=row['性别']
        Quality['modality']=row['类型']
        Quality['applyTable']=row['申请单']
        Quality['影像号']=row['影像号']
        Quality['阳性']=Quality['GetPositivelist']['whole']
        df.append(Quality)
        # if index % 1000 == 0:
        #     print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #           "%s//%s" % (index, report_chunks.index[-1]))
        #     if index > 0:
        #         end_time = time()
        #         print("  period=%.2f秒,速度=%.2f秒/条" %
        #               ((end_time-start_time), (end_time-start_time)/1000))
        #         start_time = time()
        #     else:
        #         start_time = time()
    return pd.DataFrame(df)

def struct_report(report_chunks):
    # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #       '准备处理报告子任务%s条' % len(report_chunks.index))
    df = []
    start_time = time()
    for index, row in report_chunks.iterrows():
        a = Report(
            ReportStr=row['描述'],
            ConclusionStr=row['结论'],
            StudyPart = row['部位'],
            Sex =  row['性别'],
            modality = row['类型'],
        )
        result=report_analysis(a)
        if len(result) > 0:
            result = [{**dic, '影像号': row['影像号']} for dic in result]
            df.extend(result)
        if index % 1000 == 0:
            # print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #       "%s//%s" % (index, len(report_chunks)))
            if index > 0:
                end_time = time()
                # print("  period=%.2f秒,速度=%.2f秒/条" %
                #       ((end_time-start_time), (end_time-start_time)/1000))
                start_time = time()
            else:
                start_time = time()
    df = pd.DataFrame(df)
    return df
def test_file(filename="龙岗耳鼻咽喉医院--报告实时查错数据分析内容.xlsx"):
    report = pd.read_excel(filename,converters={'影像号': str})
    report = report.drop_duplicates()
    report = report.reset_index()
    report=report.astype('str')
    sum_df=len(report)
    start=time()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'已读取文件%s条' % len(report.index))
    df=process_report_with_process_pool(report,num_processes=12)
    # df=process_report(report)

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"分析报告数量: ",len(df))

    df1=df[['partmissing','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df1=df1[df1['partmissing'].str.find("未检")<0]
    print("部位缺失:%.2f%%" %(len(df1)/sum_df*100),len(df1))
    df2=df[['special_missing','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df2=df2[df2['special_missing'].str.find("未检")<0]
    print("特殊检查:%.2f%%" %(len(df2)/sum_df*100),len(df2))
    df3=df[['conclusion_missing','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df3=df3[df3['conclusion_missing'].str.find("未检")<0]
    print("结论描述不对应:%.2f%%" %(len(df3)/sum_df*100),len(df3))
    df4=df[['orient_error','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df4=df4[df4['orient_error'].str.find("未检")<0]
    print("方位错误:%.2f%%" %(len(df4)/sum_df*100),len(df4))
    df5=df[['contradiction','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df5=df5[df5['contradiction'].str.find("未检")<0]
    print("语言矛盾:%.2f%%" %(len(df5)/sum_df*100),len(df5))
    df6=df[['none_standard_term','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df6=df6[df6['none_standard_term'].str.find("未检")<0]
    print("术语不标准:%.2f%%" %(len(df6)/sum_df*100),len(df6))
    df7=df[['RADS','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df7=df7[df7['RADS'].str.find("缺少")>=0]
    print("RADS缺失:%.2f%%" %(len(df7)/sum_df*100),len(df7))
    df8=df[['measure_unit_error','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df8=df8[df8['measure_unit_error'].str.find("过大")>=0]
    print("测量错误:%.2f%%" %(len(df8)/sum_df*100),len(df8))
    df9=df[['sex_error','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df9=df9[df9['sex_error'].str.find("报告")>=0]
    print("性别错误:%.2f%%" %(len(df9)/sum_df*100),len(df9))
    df10=df[['punctuation','ReportStr','ConclusionStr','StudyPart','Sex','modality','影像号']]
    df10=df10[df10['punctuation'].str.find("未发现")<0]
    print("标点错误:%.2f%%" %(len(df10)/sum_df*100),len(df10))
    df11=df[['apply_orient','ReportStr','ConclusionStr','StudyPart','Sex','modality','applyTable','影像号']]
    df11=df11[df11['apply_orient'].str.find("申请单")>=0]
    print("申请单错误:%.2f%%" %(len(df11)/sum_df*100),len(df11))
    df12=df[['partinverse','ReportStr','ConclusionStr','StudyPart','Sex','modality','applyTable','影像号']]
    df12=df12[df12['partinverse'].str.find("可能错误")>=0]
    print("检查项目方位错误:%.2f%%" %(len(df12)/sum_df*100),len(df12))
    
    
    sum_time=time()-start
    print("分析结束,耗时%.2f分钟，平均%.1f毫秒/条" %(sum_time/60,sum_time*1000/len(df)))
    
    # file_path = '报告错误分析.xlsx'
    file_path=filename.replace(".xlsx","(错误分析).xlsx")
    final_df=pd.DataFrame([])
    if os.path.exists(file_path):
        all_sheets = pd.read_excel(file_path, sheet_name=None,converters={'影像号': str})
        sheet_names = list(all_sheets.keys())[1:-1]  # 取前11个工作表
        dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9,df10,df11,df12]
        all_changes = []
        for idx in range(len(sheet_names)):
            sheet_name = sheet_names[idx]
            old_df = all_sheets[sheet_name]
            new_df = dfs[idx]
            # 标准化列顺序（按新DataFrame的列顺序）
            old_df = old_df[new_df.columns]
            # 使用indicator合并查找差异
            merged = old_df.merge(new_df, on=['影像号'], 
                                how='outer', indicator=True)
            # 处理删除的行（存在于旧表但不在新DF）
            deleted = merged[merged['_merge'] == 'left_only'].copy()
            if not deleted.empty:
                deleted.insert(0, '表名', sheet_name)
                deleted.insert(1, '修改', '删除')
                all_changes.append(deleted.drop('_merge', axis=1))
            # 处理新增的行（存在于新DF但不在旧表）
            added = merged[merged['_merge'] == 'right_only'].copy()
            if not added.empty:
                added.insert(0, '表名', sheet_name)
                added.insert(1, '修改', '增加')
                all_changes.append(added.drop('_merge', axis=1))
        # 合并最终结果
        final_df = pd.concat(all_changes, ignore_index=True) if all_changes else pd.DataFrame()

    with pd.ExcelWriter(file_path) as writer:
        df.to_excel(writer,sheet_name="全部",index=False)
        df1.to_excel(writer,sheet_name="报告部位缺失",index=False)
        df2.to_excel(writer,sheet_name="检查方法错误",index=False)
        df3.to_excel(writer,sheet_name="结论与描述不对应",index=False)
        df4.to_excel(writer,sheet_name="方位不符合",index=False)
        df5.to_excel(writer,sheet_name="语言矛盾",index=False)
        df6.to_excel(writer,sheet_name="术语不规范",index=False)
        df7.to_excel(writer,sheet_name="RADS缺失",index=False)
        df8.to_excel(writer,sheet_name="测量错误",index=False)
        df9.to_excel(writer,sheet_name="性别错误",index=False)
        df10.to_excel(writer,sheet_name="标点错误",index=False)
        df11.to_excel(writer,sheet_name="申请单方位错误",index=False)
        df12.to_excel(writer,sheet_name="检查项目方位错误",index=False)
        if not final_df.empty:
            final_df.to_excel(writer,sheet_name="修改",index=False)
        
    with open('report_error.log', 'a', encoding='utf-8') as f:  
        f.write(filename+"执行结束时间 "+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+
                "分析报告数量%s条,耗时%.2f分钟，平均%.1f毫秒/条\n" %(len(df),sum_time/60,sum_time*1000/len(df)))
        f.write("  部位缺失:%s\t" %(len(df1)))
        f.write("  特殊检查:%s\t" %(len(df2)))
        f.write("  结论描述不对应:%s\t" %(len(df3)))
        f.write("  方位错误:%s\t" %(len(df4)))
        f.write("  语言矛盾:%s\t" %(len(df5)))
        f.write("  术语不标准:%s\t" %(len(df6)))
        f.write("  RADS缺失:%s\t" %(len(df7)))
        f.write("  测量错误:%s\t" %(len(df8)))
        f.write("  性别错误:%s\n" %(len(df9)))
        f.write("  标点错误:%s\t" %(len(df10)))
        f.write("  申请单方位错误:%s\n" %(len(df11)))
        f.write("  检查项目方位错误:%s\n" %(len(df12)))
def test_examples(filename="samples验证.xlsx"):
    report = pd.read_excel(filename)

    report['输出']=np.nan
    report['Positive']=np.nan
    # start_time = time()
    a = Report
    for index, row in report.iterrows():
        a.ReportStr=row['ReportStr']
        a.ConclusionStr=row['ConclusionStr']
        a.StudyPart = row['StudyPart']
        a.Sex =  row['Sex']
        a.modality = row['modality']
        Quality=Report_Quality(a)
        report['Positive'].loc[index]=Quality['GetPositivelist']['whole']
        err_text=""
        for value in Quality.values():
            if type(value)!=str or len(value)==0:
                continue
            if value[0]!="未":
                err_text+=value+";"
        err_text=err_text[:-1]
        if Quality['Critical_value']!=[]:
            err_text+="危急值:"+Quality['Critical_value'][0]["category"]
        report['输出'].loc[index]=err_text
    report.to_excel(filename,index=False)

def test_struc_report(filename="报告样本.xlsx"):  
    report = pd.read_excel(filename)
    report = report.drop_duplicates()
    report = report.reset_index()
    report=report.astype('str')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          '已读取文件%s条' % len(report.index))
    total_df=[]
    start_time = time()
    num_processes=16
    pool = Pool(num_processes)
   
    # 将报告库分割成num_processes个部分
    report_chunks = [report[i::num_processes] for i in range(num_processes)]
    
    # 使用map_async来并行化
    # report_list = pool.map_async(struct_report, report_chunks).get(999999)  # 999999是一个足够大的超时时间
    report_list = list(tqdm(pool.imap_unordered(struct_report, report_chunks), total=len(report_chunks), desc="结构化进度"))
    # 合并多个对象
    total_df = pd.concat(report_list,axis=0)
    pool.close()
    pool.join()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"已经完成",len(total_df),"耗时%.2fs,平均时间%.2fs" %((time()-start_time),(time()-start_time)/len(total_df)) )
    # try:
    #     total_df[:int(len(total_df)/2)].to_excel("结构化数据库1.xlsx", index=False)
    #     total_df[int(len(total_df)/2):].to_excel("结构化数据库2.xlsx", index=False)
    # except:
    #     print("保存失败")
    return total_df

def save_sample(a:Report)->str:
    df=pd.read_excel("samples验证.xlsx")
    df.loc[len(df.index)] = {'ReportStr':a.ReportStr , 
                             'ConclusionStr':a.ConclusionStr, 
                             'StudyPart':a.StudyPart, 
                             'Sex':a.Sex, 
                             'modality': a.modality}
    df.to_excel("samples验证.xlsx")
    return "sample saved"
# %%测试数据
# if __name__ == "__main__":

#     r = AuditReport(  
#         beforeReportStr = """
#     平扫左回旋支见钙化。
#     冠状动脉分布呈右势型；右冠起源于右冠窦，左冠起源于左冠窦。
#     左主干：长度较短，分叉为左前降、左旋支，管腔通畅，未见明确狭窄或斑块。
#     左前降：管腔通畅，未见斑块及明显狭窄。诸对角支管腔通畅，未见明确狭窄或斑块；
#     左旋支：中远段管壁可见钙化斑块，管腔狭窄程度约10-20%。余段管腔通畅，未见明确狭窄或斑块；诸钝缘支管腔通畅，未见明确狭窄或斑块；
#     右冠状动脉：管腔通畅，未见明确狭窄或斑块；其分支后降支管腔通畅，未见明确狭窄或斑块；
#     心内结构未见明确异常；心包未见积液。


#            """,

#         beforeConclusionStr = """
# 冠状动脉轻度硬化，左回旋支中远段管壁钙化斑块伴管腔轻度狭窄。
#                     """,
#         afterReportStr = """
#     平扫左回旋支见钙化。
#     冠状动脉分布呈右势型；右冠起源于右冠窦，左冠起源于左冠窦。
#     左主干：长度较短，分叉为左前降、左旋支，管腔通畅，未见明确狭窄或斑块。
#     左前降：管腔通畅，未见斑块及明显狭窄。诸对角支管腔通畅，未见明确狭窄或斑块；
#     左旋支：中远段管壁可见钙化斑块，管腔狭窄程度约10-20%。余段管腔通畅，未见明确狭窄或斑块；诸钝缘支管腔通畅，未见明确狭窄或斑块；
#     右冠状动脉：管腔通畅，未见明确狭窄或斑块；其分支后降支管腔通畅，未见明确狭窄或斑块；
#     心内结构未见明确异常；心包未见积液。

  
#                """,
#         afterConclusionStr = """
# 左回旋支中远段管壁钙化斑块伴管腔轻度狭窄。

#                     """,

#         StudyPart = 'CT全腹部平扫',
#         Sex = '女',
#         modality = "CT",
#         report_doctor = "a",
#         audit_doctor = "b")
#     print("Audit_Quality=", Audit_Quality(r, debug=True))
#     data=Audit_Quality(r, debug=True)
#     with open("modify.json","w",encoding='utf-8') as f:
#         f.write(json.dumps(data,ensure_ascii=False,indent=4))
# %%测试数据2
# if __name__ == "__main__":
#     # 审核医生数据结构
#     r = AuditReport(  
#         beforeReportStr = """
#     腰椎椎体缘轻度骨质增生、硬化，前纵韧带钙化，椎小关节面增白、硬化。椎管未见明显狭窄。附见：腰4/5椎间盘膨出，腰5/骶1椎间盘向后突出。肝脏密度减低。

#            """,

#         beforeConclusionStr = """
# 1.腰椎退行性变。
# 2.附见：腰4/5椎间盘膨出，腰5/骶1椎间盘向后突出。肝脏密度减低。
#                     """,
#         afterReportStr = """
#     腰椎椎体缘轻度骨质增生、硬化，前纵韧带钙化，椎小关节面增白、硬化。椎管未见明显狭窄。附见：腰4/5椎间盘膨出，腰5/骶1椎间盘向后突出。肝脏密度减低。
  
#                """,
#         afterConclusionStr = """
# 1.腰椎退行性变。
# 2.腰4/5椎间盘膨出，腰5/骶1椎间盘向后突出。附见脂肪肝。

#                     """,

#         StudyPart = 'CT腰椎/腰髓平扫',
#         Sex = '女',
#         modality = "CT",
#         report_doctor = "a",
#         audit_doctor = "b")

#     print("Audit_Quality=", Audit_Quality(r, debug=True))

# %%危急值测试
# if __name__ == "__main__":
#     c = Report  # 危急值数据结构

#     c.ReportStr = """
#     左额叶见4*2cm高密度影。
#     """
#     c.ConclusionStr = """
# 左额叶血肿

#         """
#     c.StudyPart = '颅脑'
#     c.Sex = '女'
#     c.modality = "CT"
#     print("Report_Quality=", Report_Quality(c, debug=True))
# %%摘要测试
# if __name__ == "__main__":
#     StudyPartStr = '颅脑平扫+增强'
#     data = [{'modality': '病史', "result_str": '2.5小时前车祸撞伤头部，无意识障碍，无恶心未呕吐，肿痛我科就诊。无大小便失禁。'},
#             {'modality': '体格', "result_str": '神清，对答流利，双侧瞳孔等大等圆约3mm，对光反射灵敏，头枕部触见皮下血肿，挫伤，无出血，压痛，鼻口腔未见血迹'},
#             {'modality': '诊断', "result_str": '头部的损伤'}]

#     print(History_Summary(StudyPartStr, data))

# %%测试
# if __name__ == "__main__":
#     a = Report  # 报告医生数据结构

#     a.ReportStr = """
# 鼻腔左侧钩突(位于中鼻甲前上方的骨性突起)形态异常，其尖端部分可见一局限性软组织密度影，大小约lcmx0.8cm，边缘较清晰，与周围组织分界尚可，未见明显骨质破坏。
#     """
#     a.ConclusionStr = """
# 鼻腔右侧钩突肥大并伴有软组织肿块，考虑慢性炎症或肿瘤性病变可能性大，建议进一步MRI检查及活检以明确诊断。
#       """
        

#     a.StudyPart = '副鼻窦'
#     a.Sex = '女'
#     a.modality = "CT"
#     print("Report_Quality=", Report_Quality(a, debug=True))

# %%测试2
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
    print("Report_Quality=", Report_Quality(a, debug=True),save_sample(a))
    
    # test_file("桂林医学附属医院-萨米训练数据.xlsx")

    # searchStr="大脑血肿>3cm"
    # print(text_to_SQL(searchStr))
    # test_examples()
    # result=[]
    # excel_fils=["邢台-深圳市龙华人民医院.xlsx","华南医院_龙岗中心医院.xlsx","唐山妇幼.xlsx","龙岗三院.xlsx","龙岗人民.xlsx","放射ai训练数据.xlsx","龙岗耳鼻咽喉医院.xlsx","报告样本.xlsx"]
    # for f in tqdm(excel_fils):
    #     temp=test_struc_report(f)
    #     temp=temp[['primary','word']]
    #     # temp['positive']=temp['positive'].astype(int)
    #     result.append(temp)
    # result=pd.concat(result,axis=0)
    # total_df=result.drop_duplicates()
    # total_df[:int(len(total_df)/2)].to_excel("data1.xlsx", index=False)
    # total_df[int(len(total_df)/2):].to_excel("data2.xlsx", index=False)
        
#%%相似度测试
# if __name__ == "__main__":
#     s1="左侧中耳、乳突术后改变"
#     s2="右侧乳突炎"
#     print("“"+s1+"”VS“"+s2+"”")
#     print("model0=%.3f" %(sentence_semantics(s1,s2,model=wVmodel)))

    
# %%测试3
# if __name__ == "__main__":
#     a = Report  # 报告医生数据结构

#     a.ReportStr = """   
# 颈部MRI扫描  平扫：stir_fse_cor;stir_fse_tra;t2_fse_tra;t1_fse_tra  增强:t1_fse_wfi_tra_water/echo1;  t1_fse_wfi_cor_water/echo1;t1_fse_sag  右侧腭扁桃体明显肿大，信号尚均匀，T1呈低信号，T2呈等稍高信号，边界清，边缘呈分叶状，较大截面约为38.7mm×19.6mm，增强扫描呈轻中度不均匀强化，病灶向前累及右侧部分舌体部，向下达舌根部会厌谷。  双侧颈Ⅱ区、右侧颈Ⅲ区可见多发大小不等簇状淋巴结显示，部分肿大，信号不均，较大的约为17.8mm×11.5mm，增强扫描呈环形强化，中央区见斑片状低强化区。  左侧腭扁桃体未见增大，信号未见明确异常。  余双侧颈区可见多发小淋巴结显示，部分稍大，增强扫描可见中央线状强化。  颈余部结构未见明确异常。
#   """
#     a.ConclusionStr = """
# 右侧腭扁桃体恶性肿瘤伴双侧颈区部分淋巴结转移，扁桃体Ca可能性大，右侧舌体及舌根部-会厌谷受累及，建议组织学定性。  余颈区散在淋巴结显示，请结合临床。  会厌谷淋巴组织增生。  颈椎退行性变。  颅内散在T2高信号，建议进一步检查。
#  """
#     # a.ConclusionStr=""
#     # a.ReportStr=""
#     a.StudyPart = '颈部磁共振(MRI)平扫+增强(0.5T-1.5T)'
#     a.Sex = '男'
#     a.modality = "MR"
#     print("Report_Quality=", Report_Quality(a, debug=True))
#%%费用测试
# if __name__ == "__main__":
#     print(get_standar_Fee("肾脏、输尿管、膀胱","CT"))
# %%匹配测试
# if __name__ == "__main__":
    # StudyPartStr = "髋关节置换术后正位,右髋关节侧位,打印14×17吋激光片"
    # data = [
    #     {'index': 0, 'position': '*髋关节置换评估正位,右股骨中上段正侧位.'},
    #     {'index': 1,
    #         'position': '颅脑平扫,CT上臂/肱骨平扫,胸部/肺平扫(去除文胸，项链等）,CT全腹部平扫,上臂/肱骨(三维)（右）'},
    #     {'index': 2, 'position': '颅脑DWI,颅脑平扫+增强_1.5T,颅脑SWI'},
    #     {'index': 3,
    #         'position': '脑动脉增强'}
    # ]
    # print("检查部位:", StudyPartStr)
    # HistoryPart_list=[]
    # for i in range(len(data)):
    #     HistoryPart_list.append(PositionModel(index=data[i]['index'],position=data[i]['position']))
    
    # print("历史部位匹配：", check_history_match(StudyPartStr, HistoryPart_list))

    # StudyPartStr="腰椎正侧位.,胸部正位"
    # Sex="女"
    # Age="21Y"
    # print("检查部位:",StudyPartStr)
    # data=[{'index':"012345678",'position':'胸部正位+左侧位'},{'index':"1",'position':'腰椎'},
    #                 {'index':"2",'position':'踝关节'},{'index':"3",'position':'右膝关节正位'},{'index':"4",'position':'右踝关节'},
    #                 {'index':"5",'position':'左膝关节正侧位'},{'index':"5",'position':'右胫腓骨正侧位'}]
    # data=[{'index':11,'position':'胸部正位+左侧位'},{'index':1,'position':'胸部正位+左侧'},
    #         {'index':2,'position':'颈椎'},{'index':3,'position':'胸椎'},
    #         {'index':4,'position':'腰椎'},{'index':1,'position':'骶尾椎'},
    #         {'index':65287172131302241,'position':'胸部床旁摄影(5岁以下)'},  
    #         {'index':8,'position':'*胸部床旁摄影'}
    #         ]

    # ModePart_list=[]
    # for i in range(len(data)):
    #     ModePart_list.append(PositionModel(index=data[i]['index'],position=data[i]['position']))
    # start=time()
    # print("模板1匹配：", check_mode_match(StudyPartStr,Sex,Age,ModePart_list))
    # esp=time()-start
    # print("单线程耗时%.2f秒" %esp)
    
    # StudyPartStr="腰椎正侧位.,颈椎正侧位."
    # Sex="女"
    # Age="34Y"
    # print("检查部位:",StudyPartStr)
    # # data=[{'index':0,'position':'胫腓骨'},{'index':1,'position':'膝关节'},
    # #                 {'index':2,'position':'踝关节'},{'index':3,'position':'右膝关节正位'},{'index':4,'position':'右踝关节'},
    # #                 {'index':5,'position':'左膝关节正侧位'},{'index':5,'position':'右胫腓骨正侧位'}]
    # data=[{'index':0,'position':'脊柱全长'},{'index':1,'position':'颈椎正侧位,DR胸部正侧位片'},
    #         {'index':2,'position':'颈椎'},{'index':3,'position':'腰椎'}]
    # ModePart_list=[]
    # for i in range(len(data)):
    #     ModePart_list.append(PositionModel(index=data[i]['index'],position=data[i]['position']))
    # print("模板2匹配：", check_mode_match(StudyPartStr,Sex,Age,ModePart_list))
#%%摘要测试
# if __name__ == "__main__":
#     data = [
#         {"modality": "US", "result_str": "右侧前臂实质性病变,性质待查考虑脂肪瘤可能",
#             "date": "2023/10/9 11:15:40"},
#         {"modality": "CT", "result_str": "1.双侧斜裂胸膜稍增厚，余胸部CT平扫未见明确异常征象。2.附见：肝胃间隙软组织结节影，建议进一步检查；双肾细小结石。",
#             "date": "2023/7/11 12:22:13"},
#         {"modality": "MR", "result_str": "肝胃间隙结节状异常信号，性质待定，请结合临床相关检查考虑。",
#         "date": "2023/7/11 8:44:16"},
#         {"modality": "住院", "result_str": " 1.结缔组织交界性肿瘤 2.动态未定肿瘤 3.皮下脂肪瘤 4.慢性胃炎 5.幽门螺旋杆菌感染 6.白细胞减少",
#             "date": "2023/7/10 00:00:00"},
#         {"modality": "住院",
#             "result_str": " 1.分娩时Ⅰ度会阴裂伤 2.头位顺产 3.单胎活产 4.孕40周 5.妊娠合并轻度贫血 6.葡萄糖6-磷酸脱氢酶\[G6PD\]缺乏性贫血", "date": "2022/2/1 00:00:00"},
#         {"modality": "PS", "result_str": "（肝缘韧带肿物）未检测出KIT/PDGFRA基因突变",
#         "date": "2023/7/21 15:13:13"},
#         {"modality": "PS", "result_str": "（脐部脂肪瘤）符合脂肪瘤。",
#             "date": "2023/7/14 15:12:29"},
#         {"modality": "PS", "result_str": "（肝缘韧带肿物）梭型细胞肿瘤，待石蜡及免疫组化继续评价。",
#         "date": "2023/7/14 9:57:44"},
#         {"modality": "ES", "result_str": "胃窦壁外低回声灶", "date": "2023/7/13 11:37:51"},
#         {"modality": "US", "result_str": "顺产后第3天，子宫体积增大，宫内未见异常声像。双侧附件区未见明显异常声像。盆腔未见游离液性暗区。",
#         "date": "2022/2/3 8:41:19"},
#         {"modality": "US", "result_str": "宫内妊娠，晚孕，单活胎，头位，枕左前LOA。羊水量正常范围。胎盘成熟度Ⅱ 级。胎儿－胎盘循环功能正常。根据胎儿生物学测量，估计胎儿双顶径、头围及腹围相当于孕37-38周+大小，股骨及肱骨长相当于孕34-35周+大小。", "date": "2022/1/26 10:35:55"},
#         {"modality": "US", "result_str": "宫内妊娠，晚孕，单活胎，头位，枕左前LOA。羊水量正常范围。胎盘成熟度Ⅰ 级。胎儿－胎盘循环功能正常。",
#         "date": "2021/11/22 11:23:04"},
#         {"modality": "US", "result_str": "宫内妊娠，中孕，单活胎。羊水量正常范围。胎盘零级。根据胎儿生物学测量，估计孕龄约为21周+5天。",
#         "date": "2021/9/28 15:37:35"},
#         {"modality": "US", "result_str": "宫内妊娠，单胎，胎儿存活。胎儿颈项部透明层厚度(NT)在正常值范围内（正常值为<2.5mm）。根据胎儿生物学测量，估计孕龄约为12周＋2天。", "date": "2021/7/21 14:35:01"}]

# #     # start_time = time()
#     StudyPartStr="右桡骨平扫+增强_1.5T"
#     Summary_list=[]
#     for i in range(len(data)):
#         Summary_list.append(AbstractModel(modality=data[i]['modality'],
#                                           result_str=data[i]['result_str'],
#                                           date=data[i]['date']
#                                           ))
#     print("摘要", History_Summary(StudyPartStr,Summary_list))
# %%重复部位匹配测试
# if __name__ == "__main__":
#     StudyPartStr = '全腹部增强'
#     HistoryPartStr = '全腹部平扫'
#     print('match=', check_match(StudyPartStr, HistoryPartStr))
#     Special_Processor = KeywordProcessor()
#     Special_Processor.add_keywords_from_dict(dr_special)
#     print("Exam_special1=", Special_Processor.extract_keywords("StudyPartStr"))
#     print("Exam_special2=", Special_Processor.extract_keywords("HistoryPartStr"))
#     print("studypart=", get_orientation_position(StudyPartStr, title=True))
#     print("HistoryPart=", get_orientation_position(HistoryPartStr, title=True))
# %%批量验证

# if __name__ == "__main__":  # 批量样例验证
#     report = pd.read_excel("d://2022.1-7月记录.xlsx").to_dict('records')
#     all_df = []
#     print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "数据读取完成，开始分析...")
#     debug_range = range(0, 100)
#     for i, row in enumerate(tqdm(report)):
#         # print(i,row['检查号'])
#         # if i not in debug_range:
#         #     continue
#         rp = AuditReport
#         if (row['报告医生'] is np.nan or row['审核医生'] is np.nan or row['部位'] is np.nan or
#             row['报告描述'] is np.nan or row['报告结论'] is np.nan or row['审核描述'] is np.nan or
#                 row['审核结论'] is np.nan):
#             continue
#         rp.modality = row['类型']
#         rp.Sex = row['性别']
#         rp.StudyPart = row['部位']
#         rp.beforeConclusionStr = row['报告结论']
#         rp.afterConclusionStr = row['审核结论']
#         rp.beforeReportStr = row['报告描述']
#         rp.afterReportStr = row['审核描述']
#         rp.report_doctor = row['报告医生']
#         rp.audit_doctor = row['审核医生']
#         try:
#             result = Audit_Quality(rp)
#         except Exception as e:
#             print("\n", i, e)
#             continue
#         row["标准部位"] = result['StandardPart']
#         row["标准部位列表"] = result['standardPart_result']
#         row["部位数量"] = result['parts_sum']
#         row['阳性率'] = result['GetPositivelist']
#         row['部位缺失'] = result['partmissing'],
#         row['结论缺失'] = result['conclusion_missing'],
#         row['方向错误'] = result['orient_error']
#         row['语言冲突'] = result['contradiction']
#         row['性别错误'] = result['sex_error']
#         row['测量错误'] = result['measure_unit_error']
#         row['危急值'] = result['Critical_value']
#         row['部位评分'] = result['part_correct_score']
#         row['部位评分解释'] = result['part_correct_explain']
#         row['结论评分(新增)'] = result['conclusion_score']['add_report']
#         row['结论评分(修改)'] = result['conclusion_score']['modify_report']
#         row['结论评分(删除)'] = result['conclusion_score']['del_report']
#         row['结论评分(方向)'] = result['conclusion_score']['modify_orientation']
#         row['结论评分(总分)'] = result['conclusion_score']['sum_score']
#         row['描述评分(新增)'] = result['report_score']['add_report']
#         row['描述评分(修改)'] = result['report_score']['modify_report']
#         row['描述评分(删除)'] = result['report_score']['del_report']
#         row['描述评分(方向)'] = result['report_score']['modify_orientation']
#         row['描述评分(总分)'] = result['report_score']['sum_score']
#         row['语言评分'] = result['language_score']
#         row['语言评分解释'] = result['language_explain']
#         row['标准语评分解释'] = result['standard_term_explain']
#         row['标准语评分'] = result['standard_term_score']
#         row['审核评分'] = result['audit_score']
#         row['总分'] = result['sum_score']
#         row['报告等级'] = result['report_level']

#         all_df.append(row)

#     df_sum = len(all_df)
#     print(f"analyzed: {df_sum}")
#     # df.to_excel("实体抽取+阳性测试1.xlsx")
#     pd.DataFrame.from_dict(all_df).to_excel("实时查错+报告评级.xlsx")
# %%批量验证匹配人民医院
# if __name__ == "__main__":
# testCT = pd.read_excel("J:\\人民医院5K一标注.xlsx", sheet_name=0)
# testMR = pd.read_excel("J:\\人民医院5K一标注.xlsx", sheet_name=1)
# testMG = pd.read_excel("J:\\人民医院5K一标注.xlsx", sheet_name=2)
# testDR = pd.read_excel("J:\\人民医院5K一标注.xlsx", sheet_name=3)
# testMR['modality'] = 'MR'
# testDX['modality'] = 'DX'
# testCT['modality'] = 'CT'
# testMG['modality'] = 'MG'
# testdf = pd.concat([testMR, testCT, testDX, testMG], axis=0)
# testdf = testdf.drop(['Has Elements', '差别'], axis=1)
# testdf = pd.read_excel("J:\\人民医院标注.xlsx")
# testdf['predictedValue'] = False
# testdf['diff'] = 0
# for i in range(len(testdf)):
#     testdf['predictedValue'].iloc[i] = check_match(
#         testdf['element1'].iloc[i], testdf['element2'].iloc[i])
#     testdf['diff'].iloc[i] = testdf['predictedValue'].iloc[i] - \
#         testdf['GrandTruth'].iloc[i]
# accuracy = testdf['diff'][testdf['diff'] == 0].count()
# precision = (testdf['diff'][(testdf['GrandTruth'] == 1) & (testdf['predictedValue'] == True)].count() /
#              testdf['diff'][(testdf['predictedValue'] == 1)].count())
# recall = (testdf['diff'][(testdf['GrandTruth'] == 1) & (testdf['predictedValue'] == True)].count() /
#           testdf['diff'][(testdf['GrandTruth'] == 1)].count())
# print("accuracy=", accuracy/len(testdf))
# print("precision=", precision)
# print("recall=", recall)
# testdf.to_excel("j:\\人民医院标注2.xlsx", index=False)
# %%批量验证匹配全市
# if __name__ == "__main__":
#     # testMR = pd.read_excel("J:\\全市10K一标注.xlsx", sheet_name=0)
#     # testDX = pd.read_excel("J:\\全市10K一标注.xlsx", sheet_name=1)
#     # testCT = pd.read_excel("J:\\全市10K一标注.xlsx", sheet_name=2)
#     # testMG = pd.read_excel("J:\\全市10K一标注.xlsx", sheet_name=3)
#     # testMR['modality'] = 'MR'
#     # testDX['modality'] = 'DX'
#     # testCT['modality'] = 'CT'
#     # testMG['modality'] = 'MG'
#     # testdf = pd.concat([testMR, testCT, testDX, testMG], axis=0)
#     # testdf = testdf.drop(['Has Elements', '差别'], axis=1)
# testdf = pd.read_excel("J:\\标注修改4.xlsx")
# testdf['predictedValue'] = False
# testdf['diff'] = 0
# for i in range(len(testdf)):
#     testdf['predictedValue'].iloc[i] = check_match(
#         testdf['element1'].iloc[i], testdf['element2'].iloc[i])
#     testdf['diff'].iloc[i] = testdf['predictedValue'].iloc[i] - \
#         testdf['GrandTruth'].iloc[i]
# accuracy = testdf['diff'][testdf['diff'] == 0].count()
# precision = (testdf['diff'][(testdf['GrandTruth'] == 1) & (testdf['predictedValue'] == True)].count() /
#               testdf['diff'][(testdf['predictedValue'] == 1)].count())
# recall = (testdf['diff'][(testdf['GrandTruth'] == 1) & (testdf['predictedValue'] == True)].count() /
#           testdf['diff'][(testdf['GrandTruth'] == 1)].count())
# print("accuracy=", accuracy/len(testdf))
# print("precision=", precision)
# print("recall=", recall)
# testdf.to_excel("j:\\全市部位测试集.xlsx", index=False)
# %%批量验证匹配省医

# testdf = pd.read_excel("d:\\省医数据(标注).xlsx")
# testdf['匹配'] = False
# testdf['diff'] = 0
# for i in range(len(testdf)):
#     testdf['匹配'].iloc[i] = check_match(
#         testdf['部位1'].iloc[i], testdf['部位2'].iloc[i])
#     testdf['diff'].iloc[i] = testdf['匹配'].iloc[i] - \
#         testdf['人工标注'].iloc[i]
# accuracy = testdf['diff'][testdf['diff'] == 0].count()
# precision = (testdf['diff'][(testdf['人工标注'] == 1) & (testdf['匹配'] == True)].count() /
#               testdf['diff'][(testdf['匹配'] == 1)].count())
# recall = (testdf['diff'][(testdf['人工标注'] == 1) & (testdf['匹配'] == True)].count() /
#           testdf['diff'][(testdf['人工标注'] == 1)].count())
# print("accuracy=", accuracy/len(testdf))
# print("precision=", precision)
# print("recall=", recall)
# testdf.to_excel("D:\\省医部位测试集.xlsx", index=False)
# %%测试
# if __name__ == "__main__":
#     print(check_match("MR颈椎/颈髓平扫_1.5T,MR胸椎/胸髓平扫_1.5T", "3T颈椎间盘MRI平扫+增强"))
# %%自然语言转SQL测试
# if __name__ == "__main__":
#     searchStr="前降支狭窄大于50%"
#     sql=text_to_SQL(searchStr)
#     print(sql)
# if __name__ == "__main__":
#     sentence="双膈面光整"
#     print(get_abnormal_words(sentence))
#%%建立结构化数据库

    


# #%%异常词测试
# # if __name__ == "__main__":
# #     sentence="肝左叶内实质性异常声像"
# #     print(get_abnormal_words(sentence,modality='CT'))
# #%%标准化部位测试
# if __name__ == "__main__":
    # studypart ="MR垂体平扫+增强_1.5T"
    # modality ='MR'
    # print(get_standardPart(studypart,modality))
# #%%报告错误分析
# if __name__ == "__main__":  
    # test_file()

#%%误诊测试
# if __name__ == "__main__":  
#     data = [{"modality": "MR", "result_str": "1.头颅MRI平扫及MRA平扫未见异常征象。2.颈椎退行性变；椎间盘变性；C3/4-C5/6椎间盘轻度向后突出。","date":"2024/5/20 9:31:46"}]

#     Summary_list=[]
#     for i in range(len(data)):
#         Summary_list.append(AbstractModel(modality=data[i]['modality'],
#                                         result_str=data[i]['result_str'],
#                                         date=data[i]['date']
#                                         ))
    
#     a = Report  # 报告医生数据结构
#     a.ReportStr = """
#     与2023/12/26日片比较：
#    左肺下叶可见小片状高密度影，伴支气管轻度扩张；左肺下叶外基底段可见一枚磨玻璃结节，长径约3mm。气管及主要支气管通畅。双侧肺门及纵隔未见增大淋巴结。心脏和大血管结构未见异常。双侧胸腔未见积液。
#    颈椎序列曲度如常，椎间隙未见狭窄，各椎体及附件骨质光整，未见明显破坏，椎管、椎间孔未见变形、狭窄。齿状突居中，寰枢关节前间隙未见明确狭窄。前纵韧带及项韧带钙化。
#    """
#     a.ConclusionStr = """`
# 1.左肺下叶外基底段磨玻璃结节，建议年度随访；左肺下叶少许慢性炎症。
# 2.颈椎骨质未见明显异常；前纵韧带及项韧带钙化。
# 3.附见：脂肪肝。

#         """
#     a.StudyPart = '胸部/肺平扫(去除文胸，项链等）,CT颈椎/颈髓平扫'
#     a.Sex = '女'
#     a.modality = "CT"
#     h=HistoryInfo
#     h.report=a
#     h.abstract=Summary_list
#     print("漏诊:", Missed_diagnosis(h))