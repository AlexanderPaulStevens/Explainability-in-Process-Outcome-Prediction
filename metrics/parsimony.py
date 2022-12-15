# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 19:56:57 2022

@author: u0138175
"""
import pandas as pd 
import numpy as np

class Parsimony():
    def __init__(self, cls_method, event_columns, case_columns, controlflow_columns):
        self.cls_method = cls_method
        self.event_columns = event_columns
        self.case_columns = case_columns
        self.controlflow_columns = controlflow_columns    
        self.len_event = len(self.event_columns)
        self.len_case = len(self.case_columns)
        self.len_control = len(self.controlflow_columns)
 

    #PARSIMONY (and auxilary functions)
    def flatten(self, t):
        return [item for sublist in t for item in sublist]
    
    #functions to extract decision rules from LLM model (3)
    def find_path(self, node_numb, path, x):  
            children_left = self.dt.tree_.children_left
            children_right = self.dt.tree_.children_right
            path.append(node_numb)
            if node_numb == x:
                return True
            left = False
            right = False
            if (children_left[node_numb] !=-1):
                left = self.find_path(children_left[node_numb], path, x)
            if (children_right[node_numb] !=-1):
                right = self.find_path(children_right[node_numb], path, x)
            if left or right :
                return True
            path.remove(node_numb)
            return False
    
    def list_of_rules(self):
        # Leave node
        leave_id = self.dt.apply(self.dt_train_named_original)
        paths ={}
        for leaf in np.unique(leave_id):
            path_leaf = []
            self.find_path(0, path_leaf, leaf)
            paths[leaf] = np.unique(np.sort(path_leaf))
        rules1 = []
        for key in paths:
            rules1.append(self.get_rule(paths[key], self.dt_train_named_original.columns))  
        return rules1
        
    def get_rule(self, path, column_names):
        children_left = self.dt.tree_.children_left
        feature = self.dt.tree_.feature
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
    
    def attributes(self, model):
        
      count_event = 0
      count_case = 0
      count_control = 0
      
      #event columns
      model_event = model[model['variable'].isin(self.event_columns)]
      model_event = model_event['coefficients'].tolist()
      if self.cls_method =='LR':
        model_event = self.flatten(model_event)
      count_event += model_event.count(0.0)
    
      #case columns
      model_case= model[model['variable'].isin(self.case_columns)]
      model_case = model_case['coefficients'].tolist()
      if self.cls_method =='LR':
        model_case = self.flatten(model_case)
      count_case += model_case.count(0.0)
            
      #controlflow columns
      model_control= model[model['variable'].isin(self.controlflow_columns)]
      model_control = model_control['coefficients'].tolist()
      if self.cls_method =='LR':
        model_control = self.flatten(model_control)
      count_control += model_control.count(0.0)
      
      parsimony_event = self.len_event-count_event
      parsimony_case = self.len_case-count_case
      parsimony_control = self.len_control-count_control
      #parsimony_event = count_event/(count_event+count_case+count_control)*100
      #parsimony_case = count_case/(count_event+count_case+count_control)*100
      #parsimony_control = count_control/(count_event+count_case+count_control)*100
      print('parsimony event attributes:', parsimony_event)
      print('parsimony case attributes:', parsimony_case)
      print('parsimony controlflow attributes:', parsimony_control)
    
      return parsimony_event, parsimony_case, parsimony_control
    
    def LLM_attributes(self, coefficients_list, dt, dt_train_named_original):
      self.dt_train_named_original = dt_train_named_original
      self.dt = dt
      count_event = 0
      count_case = 0
      count_control = 0
      for i in coefficients_list:
    
        if isinstance(i, pd.DataFrame):
            #event columns
            logmodel_event = i[i['variable'].isin(self.event_columns)]
            logmodel_event = logmodel_event['coefficients'].tolist()
            logmodel_event = self.flatten(logmodel_event)
            count_event += logmodel_event.count(0.0)
            
            #case columns
            logmodel_case= i[i['variable'].isin(self.case_columns)]
            logmodel_case = logmodel_case['coefficients'].tolist()
            logmodel_case = self.flatten(logmodel_case)
            count_case += logmodel_case.count(0.0)
            
            #controlflow columns
            logmodel_control= i[i['variable'].isin(self.controlflow_columns)]
            logmodel_control = logmodel_control['coefficients'].tolist()
            logmodel_control = self.flatten(logmodel_control)
            count_control += logmodel_control.count(0.0)
            
      count_event = count_event/(len(coefficients_list))
      count_case= count_case/len(coefficients_list)
      count_control= count_control/len(coefficients_list)      
      
      count_event = self.len_event-count_event
      count_case = self.len_case-count_case
      count_control = self.len_control - count_control
            
      #add the rules of the decision tree
      rules = self.list_of_rules()
      for i in range(0,len(rules)):
            count_control += rules[i].count('Activity')
            count_event = count_event + rules[i].count('agg')- rules[i].count('Activity')
            count_case += rules[i].count('static') 
      
      parsimony_event = count_event
      parsimony_case = count_case
      parsimony_control = count_control
      #parsimony_event = count_event/(count_event+count_case+count_control)*100
      #parsimony_case = count_case/(count_event+count_case+count_control) *100
      #parsimony_control = count_control/(count_event+count_case+count_control)*100
      
      print('event attributes and percentage of event attributes:', parsimony_event)
      print('case attributes and percentage of case attributes:', parsimony_case)
      print('controlflow attributes and percentage of controlflow attributes', parsimony_control)
    
      return parsimony_event, parsimony_case, parsimony_control
    
    def GLRM_attributes(self, model_explainer):
      filter ='agg'
      #Event columns
      count_event = len(model_explainer[(model_explainer['rule'].str.contains(filter))&(~model_explainer['rule'].str.contains("Activity"))&(~model_explainer['rule'].str.contains("static"))])
      #Case columns
      count_case = len(model_explainer[(model_explainer['rule'].str.contains("static"))&(~model_explainer['rule'].str.contains("Activity"))&(~model_explainer['rule'].str.contains(filter))])
      #control flow columns
      count_control = len(model_explainer[(model_explainer['rule'].str.contains("Activity"))&(~model_explainer['rule'].str.contains("static"))])
      
      parsimony_event = count_event
      parsimony_case = count_case
      parsimony_control = count_control
      print('event attributes and percentage of event attributes:',count_event, parsimony_event)
      print('case attributes and percentage of case attributes:', count_case, parsimony_case)
      print('controlflow attributes and percentage of controlflow attributes', count_control, parsimony_control)
      
      return parsimony_event, parsimony_case, parsimony_control
