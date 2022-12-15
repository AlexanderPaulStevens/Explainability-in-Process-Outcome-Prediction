# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:48:52 2021

@author: u0138175
"""

#aix360 packages
from aix360.algorithms.rbm import LogisticRuleRegression
from aix360.algorithms.rbm import FeatureBinarizerFromTrees
#import packages
import os
import pickle
import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler
#packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
import EncoderFactory
from DatasetManager import DatasetManager
#import models
from models.LLM import LLM
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#import metrics
from metrics.parsimony import Parsimony
from metrics.functional_complexity import Functional_complexity
from metrics.faithfulness import Faithfulness, Faithfulness_LLM
from sklearn import metrics
import collections
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
#########REMOVE ERROR PRINTS###################
#pd.options.mode.chained_assignment = None

def transform_data(prefix, feature_combiner):
    #transform train dataset and add the column names back to the dataframe
    dt_named = feature_combiner.transform(prefix)
    names= feature_combiner.get_feature_names()
    dt_named = pd.DataFrame(dt_named)
    dt_named.columns = names
    return dt_named

#split and transform data with sequence encoding mechanism
def split_transform_data(data):
    # split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal") 
    #prefix generation of train and test data
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)   
    #get the label of the train and test set
    test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
    train_y = dataset_manager.get_label_numeric(dt_train_prefixes)   
    #### feature combiner and columns
    feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
    feature_combiner.fit(dt_train_prefixes, train_y)
    dt_train_named = transform_data(dt_train_prefixes, feature_combiner)
    dt_test_named = transform_data(dt_test_prefixes, feature_combiner)
    nr_events = list(dataset_manager.get_prefix_lengths(dt_test_prefixes))
      
    return dt_train_named, dt_test_named, train_y, test_y, nr_events
   
def var_importance(train_data, train_y, n_instances, fb=None):  
     #RMS difference
     print(n_instances)
     n = min(n_instances, len(train_data))
     train_data = train_data.head(n) 
     train_y = train_y[0:n]
     effects_saved = []
     result = train_data.copy()
     if cls_method =='GLRM':
         result = fb.transform(result)
     orig_pred = cls.predict(result)  
     orig_out = metrics.mean_squared_error(train_y, orig_pred)
     teller = 0
     for j in train_data.columns:  # iterate over the columns
         result2 = train_data.copy()
         teller +=1
         print(teller)
         new_items = []
         permuted_values  = set(result2[j].values)
         for i in range(0,len(train_data)):
             value = result2[j].loc[i]
             permuted_list = np.setdiff1d(list(permuted_values),[value])
             if len(permuted_list)<1:
                 random_value = value
             else:
                 random_value = random.choice(permuted_list)
             new_items.append(random_value)
         result2[j] = new_items
         if cls_method =='GLRM':
             result2 = fb.transform(result2)
         perturbed_pred = cls.predict(result2)     
         perturbed_out = metrics.mean_squared_error(train_y, perturbed_pred)
         effect = perturbed_out - orig_out
         print('Variable: ',j, 'perturbation effect: ',effect)
         effects_saved.append(effect)
     return effects_saved

######PARAMETERS

params_dir = './params_dir_ML'
results_dir ='./results_dir_ML'
column_selection = 'all'
train_ratio = 0.8
n_splits = 3
random_state = 22
n_iter=1

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

encoding_dict = {
    "agg": ["static", "agg"]
}
filter ='agg'
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

#classifiers dictionary
classifier_ref_to_classifiers = {
     "LRmodels": ['LR','LLM','GLRM'],
     "MLmodels": ["RF","XGB"],
   }
classifiers = []
for k, v in classifier_ref_to_classifiers.items():
    classifiers.extend(v)
    
for dataset_name in datasets:
    for cls_method in classifiers:
        for cls_encoding in encoding:
                       
###############################################################################
#### INFORMATION ABOUT THE EVENT LOG###########################################
###############################################################################
            print('Dataset:', dataset_name)
            print('Classifier', cls_method)
            print('Encoding', cls_encoding)
            dataset_manager = DatasetManager(dataset_name)
            data = dataset_manager.read_dataset() 
            method_name = "%s_%s"%(column_selection,cls_encoding)
            methods = encoding_dict[cls_encoding]
            
            #extract the optimal parameters
            optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
            if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
                print('problem')
            with open(optimal_params_filename, "rb") as fin:
                args = pickle.load(fin)
                print(args)
            
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
            
            #outfile to save the results in 
            outfile = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
            #transform data
            dt_train_named_original, dt_test_named_original, train_y, test_y, nr_events = split_transform_data(data)
            dt_train_named = dt_train_named_original.copy()
            dt_test_named = dt_test_named_original.copy()
            
            #control flow, event and case columns
            controlflow_columns = dt_train_named.filter(like='Activity').columns
            case_columns = dt_train_named.filter(like='static').columns
            event_columns_act = dt_train_named.filter(like=filter).columns
            event_columns = [x for x in event_columns_act if x not in controlflow_columns]
            len_case,len_event, len_control  = len(case_columns), len(event_columns), len(controlflow_columns)
            print('amount of event columns, case columns and controlflow columns:')
            print(len_event, len_case, len_control)

            preds_all = []
            test_y_all = []
            nr_events_all = []
            test_y_all.extend(test_y)
            nr_events_all.extend(nr_events)
            n_instances = len(dt_train_named_original)
            
###############################################################################
###############################################################################
            
            parsimony = Parsimony(cls_method, event_columns, case_columns, controlflow_columns)
               
            if cls_method =='LR':
                scaler = MinMaxScaler(feature_range=[0,1])
                dt_train_named2 = scaler.fit_transform(dt_train_named)
                dt_test_named2  = scaler.transform(dt_test_named)
                dt_train_named = pd.DataFrame(dt_train_named2, columns = dt_train_named.columns)
                dt_test_named = pd.DataFrame(dt_test_named2, columns = dt_test_named.columns)
                cls = LogisticRegression(C=2**args['C'],solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
                cls.fit(dt_train_named, train_y)  # apply scaling on training data
                coefmodel =pd.DataFrame({'coefficients':abs(cls.coef_.T).tolist(),'variable':dt_train_named.columns.tolist()})
                #print('range:', coefmodel['coefficients'].min(), coefmodel['coefficients'].max())
                preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                pred = cls.predict_proba(dt_test_named)[:,preds_pos_label_idx]
                preds_all.extend(pred)
                
                #auc total
                auc_total = roc_auc_score(test_y_all, preds_all)
                
                #parsimony 
                parsimony_event, parsimony_case, parsimony_control = parsimony.attributes(coefmodel)
                #functional complexity
                functional_complexity = Functional_complexity(dt_test_named_original, event_columns, case_columns, controlflow_columns, cls_method, cls)
                FC_case, FC_event, FC_control = functional_complexity.attributes()

                #monotonicity and LOD
                task_model_effects = var_importance(dt_train_named_original,train_y, n_instances)
                faithfulness = Faithfulness(cls_method, dt_train_named, event_columns, case_columns, controlflow_columns, task_model_effects, cls)
                monotonicity_value, LOD = faithfulness.calculate(coefmodel)

            elif cls_method =='LLM':
                test_y_all = []
                model = LLM(dt_train_named, dt_test_named, train_y, test_y, nr_events_all, args)
                test_y_all, preds_all, coefficients_list, cls_list, cls_index, DT, nr_events_all = model.create_model()
                #auc total
                auc_total = roc_auc_score(test_y_all, preds_all)
                
                #parsimony
                parsimony_event, parsimony_case, parsimony_control = parsimony.LLM_attributes(coefficients_list, DT, dt_train_named_original)
                
                #functional complexity
                functional_complexity = Functional_complexity(dt_test_named_original, event_columns, case_columns, controlflow_columns, cls_method,'None', cls_list, cls_index, DT)
                FC_case, FC_event, FC_control = functional_complexity.attributes()
                
                #monotonicity
                faithfulness = Faithfulness_LLM(cls_method, dt_train_named_original, event_columns, case_columns, controlflow_columns, DT, cls_list,cls_index, coefficients_list)
                monotonicity_value, LOD = faithfulness.calculate()
                
            elif cls_method=='GLRM':
                len_testdata = len(dt_test_named)
                len_traindata = len(dt_train_named)
                total_cols = event_columns.copy()
                total_cols.extend(case_columns)
                total_cols.extend(controlflow_columns)

                #FeaturebinarizerFromTrees (https://github.com/Trusted-AI/AIX360/blob/master/aix360/algorithms/rbm/features.py)
                fb = FeatureBinarizerFromTrees(negations=False, returnOrd=False, threshStr=False, numThresh=10)
                #treeNum. The number of trees to fit. A value greater than one encourages a greater variety of features and thresholds.
                #TreeDepth. The depth of the fitted decision trees. The greater the depth, the more features are generated.
                #treeFeatureSelection. The proportion of randomly chosen input features to consider at each split in the decision tree
                #threshRound. Round the threshold values to the given number of decimal places.
                dt_train_named = fb.fit_transform(dt_train_named, train_y)
                dt_test_named = fb.transform(dt_test_named)
                
                #fit model
                cls= LogisticRuleRegression(lambda0=args['lambda0'], lambda1=args['lambda1'], iterMax= 5000)   
                # Train, print, and evaluate model
                cls.fit(dt_train_named, pd.Series(train_y))
                #predictions
                pred = cls.predict_proba(dt_test_named)
                preds_all.extend(pred)    
                #auc total   len(dt_test_named_original)
                auc_total = roc_auc_score(test_y_all, preds_all)

                
                #parsimony
                parsimony_event, parsimony_case, parsimony_control = parsimony.GLRM_attributes(cls.explain())
                #functional complexity
                functional_complexity = Functional_complexity(dt_test_named_original, event_columns, case_columns, controlflow_columns, cls_method,cls, None,None,None,fb)
                FC_case, FC_event, FC_control = functional_complexity.attributes()
                
                #monotonicity and LOD
                model_effects = var_importance(dt_train_named_original,train_y, n_instances,fb)
                faithfulness = Faithfulness(cls_method, dt_train_named_original, event_columns, case_columns, controlflow_columns, model_effects)
                monotonicity_value, LOD = faithfulness.calculate(None, cls.explain())
        
            elif cls_method=='XGB':
                cls = xgb.XGBClassifier(objective='binary:logistic',
                                                n_estimators=500,
                                                learning_rate= args['learning_rate'],
                                                subsample=args['subsample'],
                                                max_depth=int(args['max_depth']),
                                                colsample_bytree=args['colsample_bytree'],
                                                min_child_weight=int(args['min_child_weight']),
                                                n_jobs=-1,
                                                seed=random_state)
                cls.fit(dt_train_named, train_y)
                
                #predictions
                preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                pred = cls.predict_proba(dt_test_named)[:,preds_pos_label_idx]
                preds_all.extend(pred)
                
                #auc total
                auc_total = roc_auc_score(test_y_all, preds_all)
                
                #parsimony
                coefficients_model = pd.DataFrame(list(zip(dt_train_named.columns, cls.feature_importances_)),columns =['variable', 'coefficients'])
                parsimony_event, parsimony_case, parsimony_control = parsimony.attributes(coefficients_model)

                #functional complexity
                functional_complexity = Functional_complexity(dt_test_named, event_columns, case_columns, controlflow_columns, cls_method, cls)
                FC_event, FC_case, FC_control= functional_complexity.attributes()
                
                #monotonicity and LOD
                model_effects = var_importance(dt_train_named,train_y, len(dt_train_named))
                faithfulness = Faithfulness(cls_method, dt_train_named, event_columns, case_columns, controlflow_columns, model_effects, cls)
                monotonicity_value, LOD = faithfulness.calculate()

            elif cls_method == "RF":
                cls = RandomForestClassifier(n_estimators=500,
                                                        n_jobs=-1,
                                                        random_state=random_state)
                cls.fit(dt_train_named, train_y)

                #predictions
                preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                pred = cls.predict_proba(dt_test_named)[:,preds_pos_label_idx]
                preds_all.extend(pred)
                
                #auc total
                auc_total = roc_auc_score(test_y_all, preds_all)
                
                #parsimony
                coefficients_model = pd.DataFrame(list(zip(dt_train_named.columns, cls.feature_importances_)),columns =['variable', 'coefficients'])
                parsimony_event, parsimony_case, parsimony_control = parsimony.attributes(coefficients_model)

                #functional complexity
                functional_complexity = Functional_complexity(dt_test_named_original, event_columns, case_columns, controlflow_columns, cls_method, cls)
                FC_event, FC_case, FC_control= functional_complexity.attributes()
                
                #monotonicity and LOD
                model_effects = var_importance(dt_train_named,train_y, len(dt_train_named))
                faithfulness = Faithfulness(cls_method, dt_train_named_original, event_columns, case_columns, controlflow_columns,model_effects, cls)
                monotonicity_value, LOD = faithfulness.calculate()
                             
            print('AUC of model', auc_total)
            print('monotonicity value:  ', monotonicity_value)
            print('LOD value:   ', LOD)
            print(outfile)
            with open(outfile, 'w') as fout:
                fout.write("%s;%s;%s;%s;%s;%s\n"%("dataset", "method", "cls", "nr_events", "metric", "score"))
                
                dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all})
                for nr_events, group in dt_results.groupby("nr_events"):
                    if len(set(group.actual)) < 2:
                        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events,-1, "auc", np.nan))
                    else:
                        fout.write("%s;%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method, nr_events,-1, "auc", roc_auc_score(group.actual, group.predicted)))
                fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method,-1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))
                fout.write("%s;%s;%s;%s;%s;%s\n"%("event columns:",len_event,"  case columns:", len_case, "  control columns:", len_control))
                
                fout.write("%s;%s;%s;%s;%s;%s\n"%("parsimony event",parsimony_event,"parsimony case", parsimony_case, "parsimony control", parsimony_control))
                fout.write("%s;%s;%s;%s;%s;%s\n"%("FC event",FC_event,"FC case", FC_case, "FC control", FC_control))
                fout.write("%s;%s\n"%("monotonicity", monotonicity_value))
                fout.write("%s;%s\n"%("LOD", LOD))