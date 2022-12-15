# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:25:18 2021

@author: u0138175
"""

import pandas as pd
import os
import re

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 

column_selection = 'all'

encoding_dict = {
    "agg": ["static", "agg"],
}
encoding = []
for k, v in encoding_dict.items():
    encoding.append(k)
    
dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "sepsis_cases": ["sepsis_cases_1","sepsis_cases_2", "sepsis_cases_4"],
    "production": ["production"],

}
datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)

#classifiers dictionary
classifier_ref_to_classifiers = {
    "LRmodels": ["LR", "LLM", "GLRM"],
    "MLmodels": [ "XGB","RF"],
    "DLmodels": ['LSTM','CNN'],
}
classifiers = []
for k, v in classifier_ref_to_classifiers.items():
    classifiers.extend(v)

datasets_overview = []
for dataset_name in datasets:
    print('Dataset:', dataset_name)
    for cls_method in classifiers:
        print('Classifier', cls_method)
        if cls_method in classifier_ref_to_classifiers['DLmodels']:
                        results_dir ='./results_dir_DL'
                        cls_encoding = 'embeddings'
        else: 
                        results_dir ='./results_dir_ML'
                        cls_encoding = 'agg'
        method_name = "%s_%s" % (column_selection, cls_encoding)
        outfile = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        result = pd.read_csv(outfile)
        total_list = []
        for i in range(len(result)-6, len(result)):
                        total_list.append(result.iloc[i,0].split(';'))
                        
        #AUC
        auc = round(float(total_list[0][5])*100,2)
                    
        #length
        len_event = total_list[1][1]
        len_case = total_list[1][3]
        len_control = total_list[1][5]             
        
        #parsimony
        pars_event = round(float(total_list[2][1]),2)
        pars_case = round(float(total_list[2][3]),2)
        pars_control = round(float(total_list[2][5]),2)  
        total_pars = pars_event+pars_case+pars_control
                                   
        #percentage parsimonuous model
        perc_pars_event = round(int(pars_event)/int(len_event)*100,2)
        perc_pars_case = round(pars_case/int(len_case)*100,2)
        perc_pars_control= round(pars_control/int(len_control)*100,2)

                  
        #percentages
        perc_event_total = round(pars_event/total_pars*100,2)
        perc_case_total = round(pars_case/total_pars*100,2)
        perc_control_total = round(pars_control/total_pars*100,2)
                    
        #FC
        FC_event = round(float(total_list[3][1]),2)
        FC_case = round(float(total_list[3][3]),2)
        FC_control =round(float(total_list[3][5]),2)
                    
        #monotonicity and LOD
        monotonicity = total_list[4][1]
        try:
            monotonicity = round(float(monotonicity),2)
        except: 
            print('nan')
        LOD = total_list = total_list[5][1]
        LOD = round(float(LOD),2)       
        dataset_list = [dataset_name, cls_encoding, cls_method,auc, len_event, len_case, len_control, pars_event, pars_case, pars_control, perc_pars_event, perc_pars_case, perc_pars_control, perc_event_total, perc_case_total, perc_control_total, FC_event, FC_case, FC_control, monotonicity, LOD]
        datasets_overview.append(dataset_list)

df = pd.DataFrame(datasets_overview, columns =['dataset_name', 'cls_encoding','classifier','AUC','#event columns', '#case columns', '#control columns','pars. event', 'pars. case','pars. control','perc_pars_event','perc_pars_case','perc_pars_control','perc_event_total', 'perc_case_total', 'perc_control_total','FC event', 'FC_case', 'FC control', 'monotonicity','LOD'])

dataset_names = list(set(df.dataset_name))
d = {}
for i in dataset_names:
    data = df[df.dataset_name==i]
    data = data.loc[data.groupby('classifier')['AUC'].idxmax()]
    data['classifier_encoding'] = data['classifier']+'_'+ data['cls_encoding']
    data = data[['dataset_name','classifier_encoding','AUC', 'pars. event','pars. case', 'pars. control', 'FC event', 'FC_case','FC control', 'monotonicity','LOD']]
    dataset_result = "dataset_"+i
    d[dataset_result]= data

df.to_csv('results.csv')
#d.to_csv('dictionary.csv')