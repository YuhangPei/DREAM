results_path='./statistic_results'
mathod_name=['MixTS_margin_0.5LUDE','MixTS_margin_0.25LUDE','MixTS_margin_noLUDE']

import os
import pandas as pd
# pd 输出不省略中间列且不换行
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
type=['sym_015','sym_03','sym_045','sym_06','asym_01','asym_02','asym_03','asym_04','idn_03','idn_04']

datasets = ['ArrowHead','CBF','FaceFour','OSULeaf','Plane','Symbols','Trace','Epilepsy','NATOPS','FingerMovements',]
# create a dataframe to store the results line is a method, column is a type of data
results=pd.DataFrame(columns=type)
# file name is mathod_name+_sym_015.csv sym_03.csv sym_045.csv sym_06.csv asym_01.csv asym_02.csv asym_03.csv asym_04.csv idn_03.csv idn_04.csv
for method in mathod_name:
    # 遍历mathod_name开头的文件
    method_results = {}
    for type_name in type:
        file=method+'_'+type_name+'.csv'
        # 读取文件
        data = pd.read_csv(os.path.join(results_path, file))
        # 读取avg_five_test_f1,std_five_test_f1 保留3位小数后计算均值再保留3位小数 只统计datasets的平均值
        avg_five_test_f1 = data[data['dataset_name'].isin(datasets)]['avg_five_test_f1'].round(3).mean().round(3)
        std_five_test_f1 = data[data['dataset_name'].isin(datasets)]['std_five_test_f1'].round(3).mean().round(3)

        # 保存到method_results
        method_results[type_name] = f'{avg_five_test_f1}±{std_five_test_f1}'
    # 保存到results
    results.loc[method] = method_results.values()
print(results)



