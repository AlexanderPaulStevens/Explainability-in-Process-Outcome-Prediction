# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:17:13 2022

@author: u0138175
"""
import pandas as pd
import numpy as np
import random
import shap
from scipy.stats import spearmanr
#euclidian distance
from scipy.spatial import distance
import math

class Faithfulness():
    def __init__(self, cls_method, dt_train_named_original, event_columns, case_columns, controlflow_columns, model_effects, cls=None):
        self.cls= cls
        self.cls_method = cls_method
        self.dt_train_named_original = dt_train_named_original
        self.event_columns = event_columns
        self.case_columns = case_columns
        self.controlflow_columns = controlflow_columns
        self.model_effects = model_effects
    
    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def calculate(self, xai_importances=None, model_explainer=None):
        
      #feature importance of original model
      feature_importance=pd.DataFrame()
      feature_importance['variable']= self.dt_train_named_original.columns
      print('perturbation importance')
      perturbation_importance = self.model_effects
      feature_importance['importances'] = perturbation_importance
      
      #feature importance of explainability model
      if self.cls_method == 'RF' or self.cls_method =='XGB':
          print('shap values')
          sample = shap.sample(self.dt_train_named_original, 1000)
          explainer = shap.TreeExplainer(self.cls, sample)
          shap_values= explainer.shap_values(sample, check_additivity=False)
          #takes the mean absolute value for each feature to get feature importance scores
          
          if self.cls_method =='XGB':
              vals = np.abs(shap_values).mean(0)
          else:
              vals = np.abs(shap_values[0]).mean(0)
         
          xai_importances=pd.DataFrame()
          xai_importances['variable']=self.dt_train_named_original.columns
          xai_importances['coefficients'] = vals
              
      elif self.cls_method =='GLRM':
          feature_importances = []
          for i in range(0,len(self.dt_train_named_original.columns)):
              feature_importance_value = abs(model_explainer[(model_explainer['rule'].str.contains(self.dt_train_named_original.columns[i]))]['coefficient']).sum()
              feature_importances.append(feature_importance_value)
          xai_importances=pd.DataFrame()
          xai_importances['variable']= self.dt_train_named_original.columns
          xai_importances['coefficients'] = feature_importances
                                     
      #resulting frame
      resulting_frame = pd.concat([xai_importances, feature_importance], join='inner', axis=1)
      resulting_frame.sort_values(by='importances',ascending=False,inplace=True)
          
      #Spearman correlation
      #assert pd.isna(np.array(resulting_frame['coefficients'])).sum() ==0
      #assert pd.isna(np.array(resulting_frame['importances'])).sum() ==0
      coef, p = spearmanr(resulting_frame['coefficients'], resulting_frame['importances'])
      print(coef)
      #assert math.isnan(coef) == False     
      #top 10 of shapley values importances and extract which columns are in it
      xai_importances.sort_values(by='coefficients',ascending=False,inplace=True)
      shap_top_10 = xai_importances[:10]
      #shap_event
      shap_event = len(shap_top_10[shap_top_10['variable'].isin(self.event_columns)])
      #shap-case
      shap_case = len(shap_top_10[shap_top_10['variable'].isin(self.case_columns)])
      #shap-control
      shap_control = len(shap_top_10[shap_top_10['variable'].isin(self.controlflow_columns)])
      
      #similar for original model
      feature_importance.sort_values(by='importances',ascending=False,inplace=True)
      feature_top_10 = feature_importance[:10]
      #feature-event
      feature_event = len(feature_top_10[feature_top_10['variable'].isin(self.event_columns)])
      #feature-case
      feature_case = len(feature_top_10[feature_top_10['variable'].isin(self.case_columns)])
      #feature_control
      feature_control = len(feature_top_10[feature_top_10['variable'].isin(self.controlflow_columns)])
      
      #LOD
      model_importance = [feature_event, feature_case, feature_control]
      explainability_importance = [shap_event, shap_case, shap_control]
      LOD = distance.euclidean(model_importance, explainability_importance)
    
      return coef, LOD

class Faithfulness_LLM():
    def __init__(self, cls_method, dt_train_named_original, event_columns, case_columns, controlflow_columns, dt, cls_list, cls_index, coefficients_list):
        self.cls_method = cls_method
        self.dt_train_named_original = dt_train_named_original
        self.event_columns = event_columns
        self.case_columns = case_columns
        self.controlflow_columns = controlflow_columns
        self.coefficients_list = coefficients_list
        self.dt = dt
        self.cls_list = cls_list
        self.cls_index = cls_index
        
    def flatten(self, t):
        return [item for sublist in t for item in sublist]
    
    def calculate(self):
         #RMS difference
         result = self.dt_train_named_original.copy()
         result['cluster'] = self.dt.apply(result)
         leaves = list((result['cluster'].unique()))  
         importances_total = np.zeros([len(leaves),self.dt_train_named_original.shape[1]])
         teller = -1
         for j in self.dt_train_named_original.columns:  # iterate over the columns
              result2 = self.dt_train_named_original.copy()
              result2['cluster'] = self.dt.apply(result2)
              teller +=1
              new_items = []
              permuted_values  = set(result2[j].values)
              for i in range(0,len(result2)):
                  value = result2[j].loc[i]
                  permuted_list = np.setdiff1d(list(permuted_values),[value])
                  if len(permuted_list)<1:
                      random_value = value
                  else:
                      random_value = random.choice(permuted_list)
                  new_items.append(random_value)
              result2[j] = new_items
              for l in leaves:
                  data_x  = result[result['cluster']==l].copy()
                  data_x2 = result2[result2['cluster']==l].copy()
                  #drop the column 'cluster'
                  data_x = data_x.drop('cluster', axis=1)
                  data_x2 = data_x2.drop('cluster', axis=1)
                  model = self.cls_list[self.cls_index.index(l)]
                  if type(model)==str:
                      model = self.dt   
                  pred = model.predict(data_x)
                  pred2 = model.predict(data_x2)
                  effect_leave = ((pred - pred2) ** 2).mean() ** 0.5   
                  importances_total[self.cls_index.index(l),teller] = effect_leave
              print(teller)
         mono_list = []
         lod_list = []
         for l in leaves:
            var_importances=pd.DataFrame()
            var_importances['variable']=self.dt_train_named_original.columns 
            var_importances['coefficients'] =  importances_total[self.cls_index.index(l),:]
            
            #MONOTONICITY
            if isinstance(self.coefficients_list[self.cls_index.index(l)], pd.DataFrame):
                feature_odds = self.flatten(self.coefficients_list[self.cls_index.index(l)]['coefficients'])
            else: 
                continue
            feature_importance = self.coefficients_list[self.cls_index.index(l)]
            if len(set(feature_odds))<2:
                continue
            else:
                coef, p = spearmanr(feature_odds,var_importances['coefficients'])
                mono_list.append(coef)
                
                #LOD
                #top 10 of shapley values importances and extract which columns are in it
                var_importances.sort_values(by='coefficients',ascending=False,inplace=True)
                var_top_10 = var_importances[:10]
                #shap_event
                var_event = len(var_top_10[var_top_10['variable'].isin(self.event_columns)])
                #shap-case
                var_case = len(var_top_10[var_top_10['variable'].isin(self.case_columns)])
                #shap-control
                var_control = len(var_top_10[var_top_10['variable'].isin(self.controlflow_columns)])
                
                #similar for original model
                feature_importance.sort_values(by='coefficients',ascending=False,inplace=True)
                feature_top_10 = feature_importance[:10]
                #feature-event
                feature_event = len(feature_top_10[feature_top_10['variable'].isin(self.event_columns)])
                #feature-case
                feature_case = len(feature_top_10[feature_top_10['variable'].isin(self.case_columns)])
                #feature_control
                feature_control = len(feature_top_10[feature_top_10['variable'].isin(self.controlflow_columns)])
                
                #LOD
                model_importance = [feature_event, feature_case, feature_control]
                explainability_importance = [var_event, var_case, var_control]
                LOD = distance.euclidean(model_importance, explainability_importance)
                lod_list.append(LOD)
            
         average_mono = sum(mono_list)/len(mono_list)
         average_lod = sum(lod_list)/len(lod_list)
         return average_mono, average_lod