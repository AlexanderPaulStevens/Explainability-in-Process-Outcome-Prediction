# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:10:24 2021

@author: u0138175
"""

#import packages
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import make_pipeline
import EncoderFactory
from DatasetManager import DatasetManager
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import hyperopt
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from hyperopt.pyll.base import scope
from AIX360.aix360.algorithms.rbm import LogisticRuleRegression
from AIX360.aix360.algorithms.rbm import FeatureBinarizerFromTrees
import xgboost as xgb

#parameters
params_dir = './params_dir_ML_GLRM'
column_selection = 'all'
train_ratio = 0.8
n_splits = 3
random_state = 22
n_iter=1

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))

encoding_dict = {
    "agg": ["static", "agg"],
    "index": ["static", "index"]
}
encoding = []
for k, v in encoding_dict.items():
    encoding.append(k)
dataset_ref_to_datasets = {
    #"bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"],
    "production": ["production"],
    #"traffic_fines": ["traffic_fines_%s"%formula for formula in range(1,2)],
    #"bpic2012": ["bpic2012_accepted","bpic2012_cancelled","bpic2012_declined"],
    #"bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
    #"hospital_billing": ["hospital_billing_%s"%suffix for suffix in [2,3]]
}

datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)

#classifiers dictionary
classifier_ref_to_classifiers = {
    "LRmodels": ["GLRM"],
    #"MLmodels": ["RF","XGB"],
}
classifiers = []
for k, v in classifier_ref_to_classifiers.items():
    classifiers.extend(v)
    
    
#create and evaluate function
def create_and_evaluate_model(args): 
    global trial_nr
    trial_nr += 1
    score = 0
    for cv_iter in range(n_splits):
        dt_test_prefixes = dt_prefixes[cv_iter]
        dt_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits): 
            if cv_train_iter != cv_iter:
                dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)
        
        #remove training rows with negative values (should be a mistake)
        #dt_train_prefixes = dt_train_prefixes[dt_train_prefixes.select_dtypes(include=[np.number]).ge(0).all(1)]


        preds_all = []
        test_y_all = []
        test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
        train_y = dataset_manager.get_label_numeric(dt_train_prefixes)  
        test_y_all.extend(test_y) 
            
        #feature combiner
        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
        feature_combiner.fit(dt_train_prefixes, train_y)
        #transform train dataset
        
        
        dt_train_named= feature_combiner.transform(dt_train_prefixes)
        dt_train_named = pd.DataFrame(dt_train_named)
        names= feature_combiner.get_feature_names()
        dt_train_named.columns = names
         
        
        #transform test dataset
        dt_test_named = feature_combiner.transform(dt_test_prefixes)
        dt_test_named = pd.DataFrame(dt_test_named)
        names= feature_combiner.get_feature_names()
        dt_test_named.columns = names
        
        if cls_method == "LR":
            cls = LogisticRegression(C=2**args['C'],solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
            pipe = make_pipeline(StandardScaler(), cls)
            pipe.fit(dt_train_named, train_y)  # apply scaling on training data
            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
            pred = pipe.predict_proba(dt_test_named)[:,preds_pos_label_idx]
            preds_all.extend(pred)
                                                 
            
        elif cls_method == 'LLM':
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
            buckets = list((dt_test_named['cluster'].unique()))
            test_y_all = []

            for i in buckets: 
                #only take the data from the bucket, seperate the label from the independent features
                data_train_x = dt_train_named[dt_train_named['cluster']==i].copy()
                data_train_y = data_train_x['label'].copy()
        
                data_test_x  = dt_test_named[dt_test_named['cluster']==i].copy()
                data_test_y  = data_test_x['label'].copy()
            
                #drop the columns
                data_train_x = data_train_x.drop('label', axis=1)
                data_train_x = data_train_x.drop('cluster', axis=1)
                data_test_x = data_test_x.drop('label', axis=1)
                data_test_x = data_test_x.drop('cluster', axis=1)
            
                #if there is only one label in the training data, no need to create a leaf model
                if len(set(data_train_y))<2:
                    pred = [data_train_y.iloc[0]]*len(data_test_y)
                    preds_all.extend(pred)
                    test_y_all.extend(data_test_y)
                    
                else:  
                    #print length of test and training data of the leaf node
                    test_y_all.extend(data_test_y)
                    scaler = StandardScaler()
                    data_train_x= scaler.fit_transform(data_train_x)
                    data_test_x= scaler.transform(data_test_x)
                    cls = LogisticRegression(C=2**args['C'],solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
                    cls.fit(data_train_x, data_train_y)
                    preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                    pred = cls.predict_proba(data_test_x)
                    pred = pred[:,preds_pos_label_idx]
                    preds_all.extend(pred)   
    
        elif cls_method =='GLRM':
            fb = FeatureBinarizerFromTrees(negations=False, returnOrd=False, threshStr=False) 
            
            dt_train_named = fb.fit_transform(dt_train_named, train_y)
            dt_test_named = fb.transform(dt_test_named)
            cls = LogisticRuleRegression(lambda0=args['lambda0'], lambda1=args['lambda1'],
                                         iterMax=100)
            # Train, print, and evaluate model
            cls.fit(dt_train_named, pd.Series(train_y))
            pred = cls.predict_proba(dt_test_named)
            preds_all.extend(pred)   

        elif cls_method =='XGB':
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
            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
            pred = cls.predict_proba(dt_test_named)[:,preds_pos_label_idx]
            preds_all.extend(pred)
        elif cls_method == "RF":
            cls = RandomForestClassifier(n_estimators=500,
                                         max_features=args['max_features'],
                                         n_jobs=-1,
                                         random_state=random_state)
            cls.fit(dt_train_named, train_y)
            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
            pred = cls.predict_proba(dt_test_named)[:,preds_pos_label_idx]
            preds_all.extend(pred)
      
        score += roc_auc_score(test_y_all, preds_all)
        for k, v in args.items():
            fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))   
        fout_all.write("%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, 0))   
    fout_all.flush()
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


# print dataset name
for dataset_name in datasets:
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    data['timesincemidnight'] = data['timesincemidnight']/60
    data['timesincemidnight'] = round(data['timesincemidnight'],0)
    data['timesincecasestart'] = data['timesincecasestart']/60
    data['timesincecasestart'] = round(data['timesincecasestart'],0)
    data['timesincelastevent'] = data['timesincelastevent']/60
    data['timesincelastevent'] = round(data['timesincelastevent'],0)
     
    for cls_method in classifiers:
        for cls_encoding in encoding:
            print('Dataset:', dataset_name)
            print('Classifier', cls_method)
            print('Encoding', cls_encoding)
            
            method_name = "%s_%s"%(column_selection, cls_encoding)            
            methods = encoding_dict[cls_encoding]
            cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True,
                        'max_events': None}

            # determine min and max (truncated) prefix lengths
            min_prefix_length = 1
            if "traffic_fines" in dataset_name:
                max_prefix_length = 10
            elif "bpic2017" in dataset_name:
                max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
            else:
                max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
            
            # split into training and test
            train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
    
    
            # prepare chunks for CV
            dt_prefixes = []
            class_ratios = []
            for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
                class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
                # generate data where each prefix is a separate instance
                dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
            del train
        
            # set up search space
            if cls_method == "LLM":
                space = {"max_depth": scope.int(hp.quniform("max_depth",1,2,1)),
                 "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf",100,1000,10)),
                 'C': hp.uniform('C', -15, 15)}
    
            if cls_method == "XGB":
                space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                 'subsample': hp.uniform("subsample", 0.5, 1),
                 'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                 'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                 'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}
      
            if cls_method =='GLRM':
                space = {'lambda0': hp.uniform("lambda0",0.00001, 0.01),
                 'lambda1': hp.uniform("lambda1", 0.00001,0.01),
                 }
    
            if cls_method == "RF":
                space = {'max_features': hp.uniform('max_features', 0, 1)}
    
            if cls_method == "LR":
                space = {'C': hp.uniform('C', -15, 15)}
            
            # optimize parameters
            trial_nr = 0
            trials = Trials()
            fout_all = open(os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
            if "prefix" in method_name:
                fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "score"))   
            else:
                fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value", "score"))   
            best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=4, trials=trials)
            fout_all.close()
        
            # write the best parameters
            best_params = hyperopt.space_eval(space, best)
            outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
            # write to file
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)
