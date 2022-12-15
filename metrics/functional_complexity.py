# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 20:20:21 2022

@author: u0138175
"""
import pandas as pd
import numpy as np 
import random

class Functional_complexity():
    def __init__(self, test_data, event_columns, case_columns, controlflow_columns, cls_method, cls=None, cls_list=None, cls_index=None, dt=None, fb=None):
        self.test_data = test_data
        self.n_instances = len(self.test_data)
        self.event_columns = event_columns
        self.case_columns = case_columns
        self.controlflow_columns = controlflow_columns
        self.cls = cls
        self.cls_method = cls_method
            
        if cls_method == 'LLM':
            self.dt = dt
            self.cls_list = cls_list
            self.cls_index = cls_index
        elif cls_method == 'GLRM':
            self.fb = fb

    #FUNCTIONAL COMPLEXITY and auxilary functions
    def LLM_test_transform(self, dt_test): 
      #homogeneuous segments (buckets)
      preds_all = []
      dt_test['cluster'] = self.dt.apply(dt_test)
      leaves = list((dt_test['cluster'].unique()))
      
      for i in leaves:
          data_test_x  = dt_test[dt_test['cluster']==i].copy()
          #drop the column 'cluster'
          data_test_x = data_test_x.drop('cluster', axis=1)
          
          model = self.cls_list[self.cls_index.index(i)]
          if type(model)==str:
              model = self.dt   
          pred = model.predict(data_test_x)
          preds_all.extend(pred)
     
      return preds_all
    
    def distance(self, lista, listb):
        runsum = 0.0
        for a, b in zip(lista, listb):
            # square the distance of each
            #  then add them back into the sum
            runsum += abs(b - a)   
    
        # square root it
        return runsum 
    
    def permute_attributes(self, columns, result, result2):
        NF = 0
        for j in columns:
            new_items = []
            permuted_values  = set(result2[j].values)
            for i in range(0,self.n_instances):
                value = result2[j].loc[i]
                permuted_list = np.setdiff1d(list(permuted_values),[value])
                if len(permuted_list)<1:
                    random_value = value
                else:
                    random_value = random.choice(permuted_list)
                new_items.append(random_value)
            
            result2[j] = new_items
        if self.cls_method =='LLM':
            pred1 = self.LLM_test_transform(pd.DataFrame(result))
            pred2 = self.LLM_test_transform(pd.DataFrame(result2)) 
        elif self.cls_method =='GLRM':
            x_1 = self.fb.transform(pd.DataFrame(result))
            x_2 = self.fb.transform(pd.DataFrame(result2))
            pred1 = self.cls.predict(x_1)
            pred2 = self.cls.predict(x_2)
        else:
            pred1 = self.cls.predict(result)        
            pred2 = self.cls.predict(result2)        
        
        NF += self.distance(pred1, pred2)
        print(NF, self.n_instances, len(columns))
        NF = NF/(self.n_instances)*100
        return NF
        
    def attributes(self):
        
        NF_event=0
        NF_case=0
        NF_control=0
        #event columns
        print('event columns')
        result = self.test_data.copy()
        result2 = result.copy()
        NF_event = self.permute_attributes(self.event_columns,result,result2)
        print('NF_event: ', NF_event)
        
        #case columns 
        print('case columns')
        result = self.test_data.copy()
        result3 = result.copy()
        NF_case = self.permute_attributes(self.case_columns,result,result3)
        print('NF_case: ', NF_case)
        
        #controflow columns 
        print('controlflow columns')
        result = self.test_data.copy()
        result4 = result.copy()
        NF_control = self.permute_attributes(self.controlflow_columns,result,result4)
        print('NF_control: ', NF_control)
        
        return NF_case,NF_event,NF_control