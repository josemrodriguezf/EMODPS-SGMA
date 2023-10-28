#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:19:05 2022

@author: joserdgz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import json
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from Borg_results_vis_C import *
import pickle



path5 = './results_set.set'
results_p =  pd.read_csv(path5,header=None,sep=" ")

results = results_p
results =results.iloc[:,:-2]
results = results.loc[:, len(results.columns) - 5:len(results.columns)]
results = results.rename(
    columns={results.columns[-5]: "Average Revenue", 
             results.columns[-4]: "Average Groundwater Depth",
             results.columns[-3]: "5th Percentile Minimum Revenue",
             results.columns[-2]: "95th Percentile Maximum Depth Change",
             results.columns[-1]: "Reliance"})

index1 = results['Average Revenue'].idxmin() #solution that optimizes net revenues (max)
index2 = results['Average Groundwater Depth'].idxmin() #solution that optimizes groundwater depths (min)       
           
dy = results["Reliance"] 

list_3 =  np.abs(dy - (-0.6))
list_3 = list_3.sort_values()
list_3 = list_3.head(3)     
index3 =  results["Average Groundwater Depth"][results.index.isin(list_3.index)].idxmax()

# Paper example final
index1 = 295
index2 = 265
index3 = 210

valid = pd.read_csv('validation_results3.csv')
a = plot_3D(results,pelev=11,pazim=-50,save_path=None,index=[index1,index2,index3],insights=True,valid_df=None)
a

# resultsa =resultsa.iloc[:,:-1]

with open('robust_Borg_comp_dry.json') as json_file:
    data = json.load(json_file)
    

    
b = Borg_results_time(data, "0", "1" ,"2",color2="firebrick",color3="green",linewidth_p=3,labels=["RobustMinDepth","MaxRev","60%Rel"])
b


