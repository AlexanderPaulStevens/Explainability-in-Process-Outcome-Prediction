# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:55:26 2022

@author: u0138175
"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler

random_state = 22

class LLM:
    
    def __init__(self, dt_train_named, dt_test_named, train_y, test_y, nr_events_all, args):
        self.dt_train_named = dt_train_named
        self.dt_test_named = dt_test_named
        self.train_y = train_y
        self.test_y = test_y
        self.nr_events_all = nr_events_all
        self.args = args
           
    #create logit leaf model from encoded data
    def create_model(self):
      #first, create,train and fit a DecisionTreeClassifier
      dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= self.args['max_depth'], min_samples_leaf= self.args['min_samples_leaf'], random_state = random_state)
      dt.fit(self.dt_train_named,self.train_y)
             
      #homogeneuous segments (buckets)
      self.dt_train_named['cluster'] = dt.apply(self.dt_train_named)
      self.dt_test_named['cluster'] = dt.apply(self.dt_test_named)

      #temporarily concatenate back, in order to have the correct y-values per segment
      #train data 
      self.train_y_concat = pd.DataFrame(self.train_y)
      self.train_y_concat = self.train_y_concat.rename(columns={self.train_y_concat.columns[0]:'label'})
      self.dt_train_named = pd.concat([self.dt_train_named,self.train_y_concat], axis=1)
            
      #test data
      self.test_y_concat  = pd.DataFrame(self.test_y)
      self.test_y_concat  = self.test_y_concat.rename(columns={self.test_y_concat.columns[0]:'label'})
      self.dt_test_named = pd.concat([self.dt_test_named, self.test_y_concat], axis=1)
            
      #list of cluster numbers
      cluster_number = list(self.dt_test_named['cluster'] )
            
      #list of leaves that contain test data 
      leaves = list((self.dt_train_named['cluster'].unique()))
      a = np.array(cluster_number)
      b = np.array(self.nr_events_all)

      #reorder the event numbers as the different leaves mixes up the sorting
      event_list = []
      for i in leaves:
          event_list.extend(b[a==i].tolist())

      preds_all = []
      test_y_all = []
      cls_list = []
      cls_index = []
      coefficients_list = []
                    
      for i in leaves:  
                       #only take the data from the leave, seperate the label from the independent features
                       data_train_x = self.dt_train_named[self.dt_train_named['cluster']==i].copy()
                       data_train_y = data_train_x['label'].copy()
                       data_test_x  = self.dt_test_named[self.dt_test_named['cluster']==i].copy()
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
                            coefficients_list.append(len(data_train_x.columns)-self.args['max_depth'])
                       else:  
                            test_y_all.extend(data_test_y)
                            scaler = MinMaxScaler(feature_range=[0,1])
                            #save the scaled data to a new dataframe in order to keep the original column named
                            data_train_x2= scaler.fit_transform(data_train_x)
                            data_test_x2= scaler.transform(data_test_x)
                            data_train_x = pd.DataFrame(data_train_x2, columns = data_train_x.columns)
                            data_test_x = pd.DataFrame(data_test_x2, columns = data_test_x.columns)
                            cls = LogisticRegression(C=2**self.args['C'],solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)    
                            cls.fit(data_train_x, data_train_y)
                            logmodel=pd.DataFrame({'coefficients': [ele for ele in transposed_coef],'variable':data_train_x.columns.tolist()})
                            pred = cls.predict_proba(data_test_x)
                            preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                            pred = pred[:,preds_pos_label_idx]
                            preds_all.extend(pred)
                            cls_index.append(i)
                            cls_list.append(cls)
                            coefficients_list.append(logmodel)
                            
      return test_y_all, preds_all, coefficients_list, cls_list, cls_index, dt, event_list