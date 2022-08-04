# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:48:52 2021

@author: u0138175
"""

#import packages
#aix360 packages
from AIX360.aix360.algorithms.rbm import LogisticRuleRegression
from AIX360.aix360.algorithms.rbm import FeatureBinarizerFromTrees
import os
import pickle
import pandas as pd
import numpy as np
import random
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#import xgboost as xgb
#import shap
from scipy.stats import spearmanr
#packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
import EncoderFactory
from DatasetManager import DatasetManager

#########REMOVE ERROR PRINTS###################
#pd.options.mode.chained_assignment = None

#create data
def create_data(dataset_name):
  dataset_manager = DatasetManager(dataset_name)
  data = dataset_manager.read_dataset()  
  data['timesincemidnight'] = data['timesincemidnight']/60
  data['timesincemidnight'] = round(data['timesincemidnight'],0)
  data['timesincecasestart'] = data['timesincecasestart']/60
  data['timesincecasestart'] = round(data['timesincecasestart'],0)
  data['timesincelastevent'] = data['timesincelastevent']/60
  data['timesincelastevent'] = round(data['timesincelastevent'],0)
  
  return data

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
    
  #transform train dataset and add the column names back to the dataframe
  dt_train_named = feature_combiner.transform(dt_train_prefixes)
  dt_train_named = pd.DataFrame(dt_train_named)
  names= feature_combiner.get_feature_names()
  dt_train_named.columns = names

  #transform test dataset
  dt_test_named = feature_combiner.transform(dt_test_prefixes)
  dt_test_named = pd.DataFrame(dt_test_named)
  names= feature_combiner.get_feature_names()
  dt_test_named.columns = names

  nr_events = list(dataset_manager.get_prefix_lengths(dt_test_prefixes))
  
  return dt_train_named, dt_test_named, train_y, test_y, nr_events

#create logit leaf model from encoded data
def create_LLM_model(dt_train_named, dt_test_named, train_y, test_y, nr_events_all):
  coefficients_list = []
  #first, create,train and fit a DecisionTreeClassifier
  dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= args['max_depth'], min_samples_leaf= args['min_samples_leaf'], random_state = random_state)
  dt.fit(dt_train_named,train_y)
         
  #homogeneuous segments (buckets)
  dt_train_named['cluster'] = dt.apply(dt_train_named)
  dt_test_named['cluster'] = dt.apply(dt_test_named)

  #temporarily concatenate back, in order to have the correct y-values per segment
  #train data 
  train_y_concat = pd.DataFrame(train_y)
  train_y_concat = train_y_concat.rename(columns={train_y_concat.columns[0]:'label'})
  dt_train_named = pd.concat([dt_train_named,train_y_concat], axis=1)
        
  #test data
  test_y_concat  = pd.DataFrame(test_y)
  test_y_concat  = test_y_concat.rename(columns={test_y_concat.columns[0]:'label'})
  dt_test_named = pd.concat([dt_test_named, test_y_concat], axis=1)
        
  #list of cluster numbers
  cluster_number = list(dt_test_named['cluster'] )
        
  #list of leaves that contain test data 
  leaves = list((dt_train_named['cluster'].unique()))
  a = np.array(cluster_number)
  b = np.array(nr_events_all)

  #reorder the event numbers as the different leaves mixes up the sorting
  nr_events_all = []
  for i in leaves:
      nr_events_all.extend(b[a==i].tolist())

  preds_all = []
  test_y_all = []
  cls_list = []
  cls_index = []
                
  for i in leaves:  
                   #only take the data from the leave, seperate the label from the independent features
                   data_train_x = dt_train_named[dt_train_named['cluster']==i].copy()
                   data_train_y = data_train_x['label'].copy()
                   data_test_x  = dt_test_named[dt_test_named['cluster']==i].copy()
                   data_test_y  = data_test_x['label'].copy()
        
                   #drop the columns 'label' and 'cluster'
                   data_train_x = data_train_x.drop('label', axis=1)
                   data_train_x = data_train_x.drop('cluster', axis=1)
                   data_test_x = data_test_x.drop('label', axis=1)
                   data_test_x = data_test_x.drop('cluster', axis=1)
        
    
                    #if there is only one label in the training data, no need to create a leaf model
                   if len(set(data_train_y))<2:
                        pred = [data_train_y.iloc[0]]*len(data_test_y)
                        preds_all.extend(pred)
                        test_y_all.extend(data_test_y)
                        cls_index.append(i)
                        cls_list.append('dt')
                        coefficients_list.append(len(data_train_x.columns)-args['max_depth'])
                   else:  
                        if data_test_x.empty==True:
                            print('empty test')
                        else:
                            test_y_all.extend(data_test_y)
                            scaler = StandardScaler()
                            #save the scaled data to a new dataframe in order to keep the original column named
                            data_train_x2= scaler.fit_transform(data_train_x)
                            data_test_x2= scaler.transform(data_test_x)
                            data_train_x = pd.DataFrame(data_train_x2, columns = data_train_x.columns)
                            data_test_x = pd.DataFrame(data_test_x2, columns = data_test_x.columns)
                            cls = LogisticRegression(C=2**args['C'],solver='saga', penalty='l1', n_jobs=-1, random_state=random_state)    
                            cls.fit(data_train_x, data_train_y)
                            logmodel=pd.DataFrame({'coefficients':(cls.coef_.T).tolist(),'variable':data_train_x.columns.tolist()})
                            pred = cls.predict_proba(data_test_x)
                            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                            pred = pred[:,preds_pos_label_idx]
                            preds_all.extend(pred)
                            cls_index.append(i)
                            cls_list.append(cls)
                            coefficients_list.append(logmodel)
                        
  return test_y_all, preds_all, coefficients_list, cls_list, cls_index, dt, nr_events_all

#PARSIMONY (and auxilary functions)
def flatten(t):
    return [item for sublist in t for item in sublist]

#functions to extract decision rules from LLM model (3)
def find_path(node_numb, path, x):  
        children_left = dt.tree_.children_left
        children_right = dt.tree_.children_right
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if (children_left[node_numb] !=-1):
            left = find_path(children_left[node_numb], path, x)
        if (children_right[node_numb] !=-1):
            right = find_path(children_right[node_numb], path, x)
        if left or right :
            return True
        path.remove(node_numb)
        return False

def list_of_rules():
    # Leave node
    leave_id = dt.apply(dt_train_named_original)
    paths ={}
    for leaf in np.unique(leave_id):
        path_leaf = []
        find_path(0, path_leaf, leaf)
        paths[leaf] = np.unique(np.sort(path_leaf))
    rules1 = []
    for key in paths:
        rules1.append(get_rule(paths[key], dt_train_named_original.columns))  
    return rules1
    
def get_rule(path, column_names):
    children_left = dt.tree_.children_left
    feature = dt.tree_.feature
    mask = ''
    for index, node in enumerate(path):
        #We check if we are not in the leaf
        if index!=len(path)-1:
            # Do we go under or over the threshold ?
            if (children_left[node] == path[index+1]):
                mask += column_names[feature[node]]
            else:
                mask += column_names[feature[node]]
    # We insert the & at the right places
    mask = mask.replace("\t", "&", mask.count("\t") - 1)
    mask = mask.replace("\t", "")
    return mask

def parsimony(model):
  count_event = 0
  count_case = 0
  count_control = 0
  
  #event columns
  model_event = model[model['variable'].isin(event_columns)]
  model_event = model_event['coefficients'].tolist()
  if cls_method =='LR':
    model_event = flatten(model_event)
  count_event += model_event.count(0.0)

  #case columns
  model_case= model[model['variable'].isin(case_columns)]
  model_case = model_case['coefficients'].tolist()
  if cls_method =='LR':
    model_case = flatten(model_case)
  count_case += model_case.count(0.0)
        
  #controlflow columns
  model_control= model[model['variable'].isin(controlflow_columns)]
  model_control = model_control['coefficients'].tolist()
  if cls_method =='LR':
    model_control = flatten(model_control)
  count_control += model_control.count(0.0)

  parsimony_event = (len_event-count_event)/len_event
  parsimony_case = (len_case-count_case)/len_case 
  parsimony_control = (len_control-count_control)/len_control
  print('event attributes and percentage of event attributes:',len_event-count_event, parsimony_event)
  print('case attributes and percentage of case attributes:', len_case-count_case, parsimony_case)
  print('controlflow attributes and percentage of controlflow attributes', len_control-count_control, parsimony_control)


  return parsimony_event, parsimony_case, parsimony_control

def parsimony_LLM(coefficients_list, len_case, len_event, len_control):


  #PARSIMONY
  count_event = 0
  count_case = 0
  count_control = 0
  for i in coefficients_list:

    if isinstance(i, pd.DataFrame):
        #event columns
        logmodel_event = i[i['variable'].isin(event_columns)]
        logmodel_event = logmodel_event['coefficients'].tolist()
        logmodel_event = flatten(logmodel_event)
        count_event += logmodel_event.count(0.0)
        

        #case columns
        logmodel_case= i[i['variable'].isin(case_columns)]
        logmodel_case = logmodel_case['coefficients'].tolist()
        logmodel_case = flatten(logmodel_case)
        count_case += logmodel_case.count(0.0)
        
        #controlflow columns
        logmodel_control= i[i['variable'].isin(controlflow_columns)]
        logmodel_control = logmodel_control['coefficients'].tolist()
        logmodel_control = flatten(logmodel_control)
        count_control += logmodel_control.count(0.0)
        
  count_event = count_event/(len(coefficients_list))
  count_case= count_case/len(coefficients_list)
  count_control= count_control/len(coefficients_list)
  
  count_event = len_event-count_event
  count_case = len_case-count_case
  count_control = len_control - count_control
  #add the rules of the decision tree
  rules = list_of_rules()
  for i in range(0,len(rules)):
        count_control += rules[i].count('Activity')
        count_event = count_event + rules[i].count('agg')- rules[i].count('Activity')
        count_case += rules[i].count('static')      
  parsimony_event = count_event/len_event
  parsimony_case = count_case/len_case 
  parsimony_control = count_control/len_control
  
  print('event attributes and percentage of event attributes:', count_event, parsimony_event)
  print('case attributes and percentage of case attributes:', count_case, parsimony_case)
  print('controlflow attributes and percentage of controlflow attributes', count_control, parsimony_control)

  return parsimony_event, parsimony_case, parsimony_control

def parsimony_GLRM(model_explainer):
  if cls_encoding == 'agg':
    filter ='agg'
  elif cls_encoding =='index':
    filter ='index'

  #Case columns
  count_case = sum(model_explainer['rule'].str.count('static'))
  #control flow columns
  count_control = sum(model_explainer['rule'].str.count('Activity'))
  #Event columns
  count_event = sum(model_explainer['rule'].str.count(filter)) - sum(model_explainer['rule'].str.count('Activity'))
 
  parsimony_event = (count_event)/len_event
  parsimony_case = (count_case)/len_case 
  parsimony_control = (count_control)/len_control
  
  print('event attributes and percentage of event attributes:',count_event, parsimony_event)
  print('case attributes and percentage of case attributes:', count_case, parsimony_case)
  print('controlflow attributes and percentage of controlflow attributes', count_control, parsimony_control)

  return parsimony_event, parsimony_case, parsimony_control

#FUNCTIONAL COMPLEXITY and auxilary functions
def LLM_test_transform(test_data, cls_list, cls_index): 
  #homogeneuous segments (buckets)
  preds_all = []
  test_data['cluster'] = dt.apply(test_data)
  leaves = list((test_data['cluster'].unique()))
  
  for i in leaves:
      data_test_x  = test_data[test_data['cluster']==i].copy()
      #drop the column 'cluster'
      data_test_x = data_test_x.drop('cluster', axis=1)
      
      cls = cls_list[cls_index.index(i)]
      if type(cls)==str:
          cls = dt   
      pred = cls.predict(data_test_x)
      preds_all.extend(pred)
 
  return preds_all

def distance(lista, listb):
    runsum = 0.0
    for a, b in zip(lista, listb):
        # square the distance of each
        #  then add them back into the sum
        runsum += abs(b - a)   

    # square root it
    return runsum 

def functional_complexity(test_data, n_instances, cls_list=None, cls_index=None):
    NF_event=0
    NF_case=0
    NF_control=0

    #event columns
    print('event columns')
    result = test_data.head(n_instances).copy()
    result2 = result.copy()

    for j in event_columns:
        new_items = []
        permuted_values  = set(result2[j].values)
        for i in range(0,n_instances):
            value = result2[j].loc[i]
            permuted_list = np.setdiff1d(list(permuted_values),[value])
            if len(permuted_list)<1:
                random_value = value
            else:
                random_value = random.choice(permuted_list)
            new_items.append(random_value)
        
        result2[j] = new_items
    if cls_method =='LLM':
        pred1 = LLM_test_transform(pd.DataFrame(result), cls_list, cls_index)
        pred2 = LLM_test_transform(pd.DataFrame(result2), cls_list, cls_index) 
    elif cls_method =='GLRM':
        x_1 = fb.transform(pd.DataFrame(result))
        x_2 = fb.transform(pd.DataFrame(result2))
        pred1 = cls.predict(x_1)
        pred2 = cls.predict(x_2)
    else:
        pred1 = cls.predict(result)        
        pred2 = cls.predict(result2)        
    
    NF_event += distance(pred1, pred2)
    print(NF_event, n_instances, len_event)
    NF_event = NF_event/(n_instances)
    print('NF_event: ', NF_event)
    
    #case columns 
    print('case columns')
    result = test_data.head(n_instances).copy()
    result2 = result.copy()
    for j in case_columns:
        new_items = []
        permuted_values  = set(result2[j].values)
        for i in range(0,n_instances):
            value = result2[j].loc[i]
            permuted_list = np.setdiff1d(list(permuted_values),[value])
            if len(permuted_list)<1:
                random_value = value
            else:
                random_value = random.choice(permuted_list)
            new_items.append(random_value)
        
        result2[j] = new_items
    if cls_method =='LLM':
        pred1 = LLM_test_transform(pd.DataFrame(result), cls_list, cls_index)
        pred2 = LLM_test_transform(pd.DataFrame(result2), cls_list, cls_index) 
    elif cls_method =='GLRM':
        x_1 = fb.transform(pd.DataFrame(result))
        x_2 = fb.transform(pd.DataFrame(result2))
        pred1 = cls.predict(x_1)
        pred2 = cls.predict(x_2)
    else:
        pred1 = cls.predict(result)        
        pred2 = cls.predict(result2)        
    
    NF_case += distance(pred1, pred2)
    print(NF_case, n_instances, len_case)
    NF_case = NF_case/(n_instances)
    print('NF_case: ', NF_case)
   
    #controflow columns 
    result = test_data.head(n_instances).copy()
    result2 = result.copy()
    for j in controlflow_columns:
        new_items = []
        permuted_values  = set(result2[j].values)
        for i in range(0,n_instances):
            value = result2[j].loc[i]
            permuted_list = np.setdiff1d(list(permuted_values),[value])
            if len(permuted_list)<1:
                random_value = value
            else:
                random_value = random.choice(permuted_list)
            new_items.append(random_value)
        
        result2[j] = new_items
    if cls_method =='LLM':
        pred1 = LLM_test_transform(pd.DataFrame(result), cls_list, cls_index)
        pred2 = LLM_test_transform(pd.DataFrame(result2), cls_list, cls_index) 
    elif cls_method =='GLRM':
        x_1 = fb.transform(pd.DataFrame(result))
        x_2 = fb.transform(pd.DataFrame(result2))
        pred1 = cls.predict(x_1)
        pred2 = cls.predict(x_2)
    else:
        pred1 = cls.predict(result)        
        pred2 = cls.predict(result2)        
    NF_control += distance(pred1, pred2)
    print(NF_control, n_instances, len_control)
    NF_control = NF_control/(n_instances)
    print('NF_control: ', NF_control)
    
    return NF_case,NF_event,NF_control

#MONOTONICITY
def var_importance(train_data, n_instances):
    
    #RMS difference
    effects_saved = []
    result = train_data.copy()
    orig_out = cls.predict(result)   
    teller = 0
    for j in result.columns:  # iterate over the columns
        result2 = result.copy()
        teller +=1
        print(teller)
        new_items = []
        permuted_values  = set(result2[j].values)
        for i in range(0,n_instances):
            value = result2[j].loc[i]
            permuted_list = np.setdiff1d(list(permuted_values),[value])
            if len(permuted_list)<1:
                random_value = value
            else:
                random_value = random.choice(permuted_list)
            new_items.append(random_value)
        result2[j] = new_items
        perturbed_out = cls.predict(result2)      
        effect = ((orig_out - perturbed_out) ** 2).mean() ** 0.5
        #print('Variable: ',j, 'perturbation effect: ',effect)
        effects_saved.append(effect)
    return effects_saved


def monotonicity(model):
  
  #feature importance of original model
  feature_importance=pd.DataFrame()
  feature_importance['columns']=dt_train_named_original.columns
  print('perturbation importance')
  perturbation_importance = var_importance(dt_train_named_original, len(dt_train_named_original))
  feature_importance['importances'] = perturbation_importance
  print('shap values')
  sample = shap.sample(dt_train_named, 100)
  if cls_method =='XGB':
      #https://github.com/slundberg/shap/issues/1215
      mybooster = mymodel.get_booster()    
      model_bytearray = mybooster.save_raw()[4:]
      def myfun(self=None):
          return model_bytearray
      mybooster.save_raw = myfun
      explainer = shap.TreeExplainer(mybooster)
  elif cls_method =='RF':
      #shap value feature importance
      explainer = shap.TreeExplainer(model, dt_train_named)
      
  shap_values= explainer.shap_values(sample)
  #takes the mean absolute value for each feature to get feature importance scores
  if cls_method =='RF':
      vals = np.abs(shap_values[0]).mean(0)
  elif cls_method=='XGB':
      vals = np.abs(shap_values).mean(0)
  shap_importances=pd.DataFrame()
  shap_importances['columns']=dt_train_named.columns
  shap_importances['importances_shap'] = vals
  
  #resulting frame
  resulting_frame = pd.concat([shap_importances, feature_importance], join='inner', axis=1)
  resulting_frame.sort_values(by='importances',ascending=False,inplace=True)
  
  #Spearman correlation
  coef, p = spearmanr(resulting_frame['importances_shap'], resulting_frame['importances'])
  
  print('Spearmans correlation coefficient: %.3f' % coef)
  # interpret the significance
  alpha = 0.05
  if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
  else:
    print('Samples are correlated (reject H0) p=%.3f' % p)


  #top 10 of shapley values importances and extract which columns are in it
  shap_importances.sort_values(by='importances_shap',ascending=False,inplace=True)
  shap_top_10 = shap_importances[:10]
  #shap_event
  shap_event = len(shap_top_10[shap_top_10['columns'].isin(event_columns)])
  #shap-case
  shap_case = len(shap_top_10[shap_top_10['columns'].isin(case_columns)])
  #shap-control
  shap_control = len(shap_top_10[shap_top_10['columns'].isin(controlflow_columns)])
  
  #similar for original model
  feature_importance.sort_values(by='importances',ascending=False,inplace=True)
  feature_top_10 = feature_importance[:10]
  #feature-event
  feature_event = len(feature_top_10[feature_top_10['columns'].isin(event_columns)])
  #feature-case
  feature_case = len(feature_top_10[feature_top_10['columns'].isin(case_columns)])
  #feature_control
  feature_control = len(feature_top_10[feature_top_10['columns'].isin(controlflow_columns)])

  return coef, shap_event, shap_case, shap_control, feature_event, feature_case, feature_control


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
    #"agg": ["static", "agg"],
     "index": ["static", "index"]
}
encoding = []
for k, v in encoding_dict.items():
    encoding.append(k)
    
dataset_ref_to_datasets = {
    #"bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
   "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(5,6)],
   # "sepsis_cases": ["sepsis_cases_1","sepsis_cases_2",'sepsis_cases_4'],
   # "production": ["production"],
    #"traffic_fines": ["traffic_fines_%s"%formula for formula in range(1,2)],
    #"bpic2012": ["bpic2012_accepted", 'bpic2012_cancelled', "bpic2012_declined"],
    #"bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
    #"hospital_billing": ["hospital_billing_%s"%suffix for suffix in [2,3]]
}
datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)

#classifiers dictionary
classifier_ref_to_classifiers = {
    "LRmodels": [#'LR',
                 #'LLM',
                 'GLRM'],
    # "MLmodels": [ "RF","XGB"],
   }
classifiers = []
for k, v in classifier_ref_to_classifiers.items():
    classifiers.extend(v)
    
for dataset_name in datasets:
    for cls_method in classifiers:
        for cls_encoding in encoding:
            print('Dataset:', dataset_name)
            print('Classifier', cls_method)
            print('Encoding', cls_encoding)
            dataset_manager = DatasetManager(dataset_name)
            data = create_data(dataset_name)
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
            dt_train_named, dt_test_named, train_y, test_y, nr_events = split_transform_data(data)
            #dt_train_named = dt_train_named.drop('static__label_deviant', 1)
            #dt_train_named = dt_train_named.drop('static__label_regular', 1)
            #dt_test_named = dt_test_named.drop('static__label_deviant', 1)
            #dt_test_named = dt_test_named.drop('static__label_regular', 1)
            dt_train_named_original = dt_train_named.copy()
            dt_test_named_original = dt_test_named.copy()
            
            #control flow, event and case columns
            if cls_encoding == 'agg':
                filter ='agg'
            elif cls_encoding =='index':
                filter ='index'
            controlflow_columns = dt_train_named.filter(like='Activity').columns
            case_columns = dt_train_named.filter(like='static').columns
            event_columns_act = dt_train_named.filter(like=filter).columns
            event_columns = [x for x in event_columns_act if x not in controlflow_columns]
            len_case = len(case_columns)
            len_event =len(event_columns)
            len_control = len(controlflow_columns)
            print('amount of event columns, case columns and controlflow columns:')
            print(len_event, len_case, len_control)

            preds_all = []
            test_y_all = []
            nr_events_all = []
            test_y_all.extend(test_y)
            nr_events_all.extend(nr_events)
            
            if cls_method =='LR':
                scaler = StandardScaler()
                dt_train_named2 = scaler.fit_transform(dt_train_named)
                dt_test_named2  = scaler.transform(dt_test_named)
                dt_train_named = pd.DataFrame(dt_train_named2, columns = dt_train_named.columns)
                dt_test_named = pd.DataFrame(dt_test_named2, columns = dt_test_named.columns)
                
                cls = LogisticRegression(C=2**args['C'],solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
                cls.fit(dt_train_named, train_y)  # apply scaling on training data
                coefmodel =pd.DataFrame({'coefficients':(cls.coef_.T).tolist(),'variable':dt_train_named.columns.tolist()})
                preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                pred = cls.predict_proba(dt_test_named)[:,preds_pos_label_idx]
                preds_all.extend(pred)
                
                #auc total
                auc_total = roc_auc_score(test_y_all, preds_all)

                #parsimony 
                #parsimony_event, parsimony_case, parsimony_control = parsimony(coefmodel)
                #functional complexity
                #FC_case, FC_event, FC_control = functional_complexity(dt_test_named_original, len(dt_test_named_original))
                
                #monotonicity_value 
                monotonicity_value = 1
              
            elif cls_method =='LLM':
                test_y_all = []
                
                test_y_all, preds_all, coefficients_list, cls_list, cls_index, dt, nr_events_all = create_LLM_model(dt_train_named, dt_test_named, train_y, test_y, nr_events_all)
                #auc total
                auc_total = roc_auc_score(test_y_all, preds_all)
                
                #parsimony
                parsimony_event, parsimony_case, parsimony_control = parsimony_LLM(coefficients_list, len_case, len_event, len_control)
                
                #functional complexity
                FC_case, FC_event, FC_control = functional_complexity(dt_test_named_original, len(dt_test_named_original), cls_list, cls_index)
                
                #monotonicity
                monotonicity_value = 1


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
                dt_train_named_original = dt_train_named.copy()
                dt_train_named = fb.fit_transform(dt_train_named, train_y)
                dt_test_named = fb.transform(dt_test_named)

                #fit model
                cls= LogisticRuleRegression(lambda0=args['lambda0'], lambda1=args['lambda1'], iterMax= 100)   
                # Train, print, and evaluate model
                cls.fit(dt_train_named, pd.Series(train_y))
                
                #predictions
                pred = cls.predict_proba(dt_test_named)
                preds_all.extend(pred)    
                
                #auc total   len(dt_test_named_original)
                auc_total = roc_auc_score(test_y_all, preds_all)
                print(auc_total)

                #explanations
                explanations = cls.explain()
                
                
                #parsimony
                parsimony_event, parsimony_case, parsimony_control = parsimony_GLRM(cls.explain())

                #functional complexity
                FC_case, FC_event, FC_control = functional_complexity(dt_test_named_original, len(dt_test_named_original))

                #monotonicity value
                monotonicity_value = 1
                
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
                mymodel = cls.fit(dt_train_named, train_y)
                
                #predictions
                preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                pred = cls.predict_proba(dt_test_named)[:,preds_pos_label_idx]
                preds_all.extend(pred)
                
                #auc total
                auc_total = roc_auc_score(test_y_all, preds_all)

                #parsimony
                coefficients_model = pd.DataFrame(list(zip(dt_train_named.columns, cls.feature_importances_)),columns =['variable', 'coefficients'])
                parsimony_event, parsimony_case, parsimony_control = parsimony(coefficients_model)

                #functional complexity
                FC_event, FC_case, FC_control= functional_complexity(dt_test_named_original, len(dt_test_named_original))

                #monotonicity value
                monotonicity_value, shap_event, shap_case, shap_control, feature_event, feature_case, feature_control = monotonicity(cls)


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
                parsimony_event, parsimony_case, parsimony_control = parsimony(coefficients_model)

                #functional complexity
                FC_event, FC_case, FC_control= functional_complexity(dt_test_named_original, len(dt_test_named_original))

                #monotonicity
                monotonicity_value, shap_event, shap_case, shap_control, feature_event, feature_case, feature_control = monotonicity(cls)
            
            #print('functional complexity of','case columns:  ', FC_case,' event columns:  ', FC_event, 'control columns:  ', FC_control)
            print('monotonicity value:  ', monotonicity_value)
            if cls_method =='RF' or cls_method =='XGB':
                print('monotonicity values: ', shap_event, shap_case, shap_control, feature_event, feature_case, feature_control)
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
       
                if cls_method=='RF' or cls_method =='XGB':
                        fout.write("%s;%s;%s;%s;%s;%s%s;%s;%s;%s;%s;%s\n"%("shap event",shap_event,"shap case", shap_case, "shap control", shap_control, "feature_event", feature_event,'feature_case',feature_case, 'feature_control', feature_control))

                        