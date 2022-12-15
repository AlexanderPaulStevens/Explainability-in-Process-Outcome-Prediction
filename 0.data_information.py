# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
import EncoderFactory
from DatasetManager import DatasetManager

#average of list
def Average(lst):
    return sum(lst) / len(lst)

######PARAMETERS
random_state = 22
train_ratio = 0.8

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
    "sepsis_cases": ["sepsis_cases_1","sepsis_cases_2",'sepsis_cases_4'],
    "production": ["production"],
}
datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)

eventlog_specs = []

for dataset_name in datasets:
            print('Dataset:', dataset_name)
            dataset_manager = DatasetManager(dataset_name)
            data = dataset_manager.read_dataset()  
            data['timesincemidnight'] = data['timesincemidnight']/60
            data['timesincemidnight'] = round(data['timesincemidnight'],0)
            data['timesincecasestart'] = data['timesincecasestart']/60
            data['timesincecasestart'] = round(data['timesincecasestart'],0)
            data['timesincelastevent'] = data['timesincelastevent']/60
            data['timesincelastevent'] = round(data['timesincelastevent'],0)
               
            cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}

            # determine min and max (truncated) prefix lengths
            min_prefix_length = 1
            if "traffic_fines" in dataset_name:
                max_prefix_length = 10
            elif "bpic2017" in dataset_name:
                max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
            else:
                max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
                
            # split into training and test
            train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
            
            #prefix generation of train and test data
            dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
            dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
              
            #get the label of the train and test set
            test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
            train_y = dataset_manager.get_label_numeric(dt_train_prefixes)   

            cases = len(set(data[cls_encoder_args['case_id_col']]))
            events = len(data)
            max_prefix_length
            event_cols = len(set(data[cls_encoder_args['dynamic_cat_cols']+cls_encoder_args['dynamic_num_cols']]))-1
            case_cols = len(set(data[cls_encoder_args['static_cat_cols']+cls_encoder_args['static_num_cols']]))
            activity_col = set(data[cls_encoder_args['dynamic_cat_cols']].filter(like='Act'))
            control_cols = len((activity_col))
            
            #number of activities
            activities = len(data[list(activity_col)[0]])
            
            #average number of activities per case
            activities_per_case = []
            case_id_col = set(train[cls_encoder_args['case_id_col']])
            for i in case_id_col:
                case = train[train[cls_encoder_args['case_id_col']]==i]
                case_activities = len(case[list(activity_col)[0]].unique())
                activities_per_case.append(case_activities)

            avg_act_per_case = round(Average(activities_per_case),0)
            dynamic_cols = len(cls_encoder_args['dynamic_cat_cols']+cls_encoder_args['dynamic_num_cols'])
            static_cols = len(cls_encoder_args['static_cat_cols']+cls_encoder_args['static_num_cols'])
            dataset_list = [dataset_name, events, cases, max_prefix_length, event_cols, case_cols, control_cols, activities, avg_act_per_case, static_cols,dynamic_cols]
            eventlog_specs.append(dataset_list)
            
df_specs = pd.DataFrame(eventlog_specs, columns =['dataset_name', 'events','cases','max prefix length','#event columns', '#case columns', '#control columns', 'activities','avg. act per case','static cols','dynamic cols'])
df_specs.to_csv('data_information.csv')