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
    "index": ["static", "index"]
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

            """Deep Learning Models (LSTM and CNN)"""
            if cls_method in classifier_ref_to_classifiers['DLmodels']:
                cls_encoding = 'embeddings'
                results_dir ='./results_dir_DL'
                method_name = "%s_%s" % (column_selection, cls_encoding)
                #methods = encoding_dict[cls_encoding]
                outfile = os.path.join(results_dir,
                                       "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))

                result = pd.read_csv(outfile)
                total_list = []
                for i in range(len(result) - 5, len(result)):
                    total_list.append(result.iloc[i, 0].split(';'))

                # AUC
                auc = total_list[0][5]

                # Length
                len_event = total_list[1][1]
                len_case = total_list[1][3]
                len_control = total_list[1][5]

                total_cols = int(len_event) + int(len_case) + int(len_control)

                # parsimony
                pars_event = float(total_list[2][1]) * int(len_event)
                pars_case = float(total_list[2][3]) * int(len_case)
                pars_control = float(total_list[2][5]) * int(len_control)

                total_pars_cols = pars_event + pars_case + pars_control

                # percentages
                perc_pars_event = int(pars_event * 100) / int(len_event)
                perc_pars_case = pars_case * 100 / int(len_case)
                perc_pars_control = pars_control * 100 / int(len_control)

                perc_pars_total = total_pars_cols * 100 / total_cols

                # percentage parsimonious total
                perc_event_total = pars_event * 100 / total_pars_cols
                perc_case_total = pars_case * 100 / total_pars_cols
                perc_control_total = pars_control * 100 / total_pars_cols

                # FC
                FC_event = float(total_list[3][1]) * 100
                FC_case = float(total_list[3][3]) * 100
                FC_control = float(total_list[3][5]) * 100

                # monotonicity
                monotonicity = total_list[4][1]

                shap_cols = '/'
                feature_cols = '/'

                dataset_list = [dataset_name, cls_encoding, cls_method, auc, len_event, len_case, len_control,
                                    total_cols, pars_event, pars_case, pars_control, total_pars_cols, perc_pars_event,
                                    perc_pars_case, perc_pars_control, perc_event_total, perc_case_total,
                                    perc_control_total, perc_pars_total, FC_event, FC_case, FC_control, monotonicity,
                                    shap_cols, feature_cols]
                datasets_overview.append(dataset_list)

            else:
                for cls_encoding in encoding:
                    #print('Encoding', cls_encoding)
                    results_dir ='./results_dir_ML'
                    method_name = "%s_%s"%(column_selection,cls_encoding)
                    methods = encoding_dict[cls_encoding]
                    outfile = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
                    result = pd.read_csv(outfile)
                    total_list = []
                    if cls_method in classifier_ref_to_classifiers['LRmodels']:
                        for i in range(len(result)-5, len(result)):
                            total_list.append(result.iloc[i,0].split(';'))
                        
                        #AUC
                        auc = total_list[0][5]
                        
                        #Length
                        len_event = total_list[1][1]
                        len_case = total_list[1][3]
                        len_control = total_list[1][5]
    
                        total_cols = int(len_event) +int(len_case)+ int(len_control)
                        
                        #parsimony
                        pars_event = float(total_list[2][1])*int(len_event)
                        pars_case = float(total_list[2][3])*int(len_case)
                        pars_control = float(total_list[2][5])*int(len_control)
                       
                        total_pars_cols = pars_event +pars_case +pars_control
                        
                        #percentages
                        perc_pars_event = int(pars_event*100)/int(len_event)
                        perc_pars_case = pars_case*100/int(len_case)
                        perc_pars_control= pars_control*100/int(len_control)
                       
                        perc_pars_total = total_pars_cols*100/total_cols
                        
                        #percentage parsimonious total
                        perc_event_total = pars_event*100/total_pars_cols
                        perc_case_total = pars_case*100/total_pars_cols
                        perc_control_total = pars_control*100/total_pars_cols
                      
                        #FC
                        FC_event = float(total_list[3][1])*100
                        FC_case = float(total_list[3][3])*100
                        FC_control = float(total_list[3][5])*100
                        
                        #monotonicity
                        monotonicity = total_list[4][1]
                       
                        shap_cols = '/'
                        feature_cols = '/'
                        
                        dataset_list = [dataset_name, cls_encoding, cls_method,auc, len_event, len_case, len_control,total_cols, pars_event, pars_case, pars_control, total_pars_cols, perc_pars_event, perc_pars_case, perc_pars_control, perc_event_total, perc_case_total, perc_control_total, perc_pars_total, FC_event, FC_case, FC_control, monotonicity, shap_cols, feature_cols]
                        datasets_overview.append(dataset_list)

                    elif cls_method in classifier_ref_to_classifiers['MLmodels']:
                        for i in range(len(result)-6, len(result)):
                            total_list.append(result.iloc[i,0].split(';'))
                            
                        #AUC
                        auc = total_list[0][5]
                        
                        #length
                        len_event = total_list[1][1]
                        len_case = total_list[1][3]
                        len_control = total_list[1][5]
                        
                        total_cols = int(len_event)+int(len_case)+int(len_control)
                        
                        #parsimony
                        pars_event = float(total_list[2][1])*int(len_event)
                        pars_case = float(total_list[2][3])*int(len_case)
                        pars_control = float(total_list[2][5])*int(len_control)
                        
                        total_pars_cols = pars_event +pars_case +pars_control
                                              
                        #percentage parsimonuous model
                        perc_pars_event = int(pars_event*100)/int(len_event)
                        perc_pars_case = pars_case*100/int(len_case)
                        perc_pars_control= pars_control*100/int(len_control)
                        perc_pars_total = total_pars_cols*100/total_cols
                        
                        #percentages
                        perc_event_total = pars_event*100/total_pars_cols
                        perc_case_total = pars_case*100/total_pars_cols
                        perc_control_total = pars_control*100/total_pars_cols
                        
                        #FC
                        FC_event = float(total_list[3][1])*100
                        FC_case = float(total_list[3][3])*100
                        FC_control = float(total_list[3][5])*100
                        
                        #monotonicity
                        shap_control = re.findall(r'\d+', str(total_list[5][5]))[0]
                        shap_cols = (total_list[5][1], total_list[5][3],shap_control)
                        #feature_and_shap = result.iloc[-1,0].split(';')
                        
                        monotonicity = total_list[4][1]
                    
                        feature_cols = (total_list[5][6],total_list[5][8],total_list[5][10])
                        
                        dataset_list = [dataset_name, cls_encoding, cls_method,auc, len_event, len_case, len_control, total_cols, pars_event, pars_case, pars_control, total_pars_cols, perc_pars_event, perc_pars_case, perc_pars_control, perc_event_total, perc_case_total, perc_control_total, perc_pars_total, FC_event, FC_case, FC_control, monotonicity, shap_cols, feature_cols]
                        datasets_overview.append(dataset_list)
                        

df = pd.DataFrame(datasets_overview, columns =['dataset_name', 'cls_encoding','classifier','AUC','#event columns', '#case columns', '#control columns','total cols', 'pars. event', 'pars. case','pars. control','total pars cols','perc_pars_event','perc_pars_case','perc_pars_control','perc_event_total', 'perc_case_total', 'perc_control_total','perc_pars_total', 'FC event', 'FC_case', 'FC control', 'monotonicity','shap (event, case, control)', 'feature importance (event, case, control)'])

df.to_csv('results.csv')
#print(df)