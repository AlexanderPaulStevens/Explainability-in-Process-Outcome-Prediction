# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:51:42 2022

@author: u0138175
"""

import pandas as pd

df = pd.read_csv('results2.csv', index_col=0)
dataset_names = list(set(df.dataset_name))
d = {}
for i in dataset_names:
    
    data = df[df.dataset_name==i]
    data = data.loc[data.groupby('classifier')['AUC'].idxmax()]
    data['classifier_encoding'] = data['classifier']+'_'+ data['cls_encoding']
    data = data[['dataset_name','classifier_encoding','AUC', 'pars. event','pars. case', 'pars. control', 'FC event', 'FC_case','FC control', 'monotonicity','LOD']]
    data['monotonicity'] = round(data['monotonicity'],2)
    data['LOD'] = round(data['LOD'],2)
    data['pars. case'] = round(data['pars. case'],2)
    data['pars. event'] = round(data['pars. event'],2)
    data['pars. control'] = round(data['pars. control'],2)
    data['FC event'] = round(data['FC event'],2)
    data['FC_case'] = round(data['FC_case'],2)
    data['FC control'] = round(data['FC control'],2)
    data.AUC = data.AUC*100
    data.AUC = round(data.AUC,2)
    dataset_result = "dataset_"+i
    d[dataset_result]= data

    