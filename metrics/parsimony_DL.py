# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 10:09:43 2022

@author: u0138175
"""

class Parsimony():
    def __init__(self, cls_method, event_columns, case_columns, controlflow_columns):
        self.cls_method = cls_method
        self.event_columns = event_columns
        self.case_columns = case_columns
        self.controlflow_columns = controlflow_columns    
        self.len_event = len(self.event_columns)
        self.len_case = len(self.case_columns)
        self.len_control = len(self.controlflow_columns)
        

        