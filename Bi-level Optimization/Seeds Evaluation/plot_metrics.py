#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 17:31:25 2022

@author: joserdgz
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


fig,ax = plt.subplots(figsize=(7.2,4.2), gridspec_kw={'hspace':0.25, 'wspace':0.3})
for seed in range(1,11):
    seed_results = pd.read_csv("seed"+str(seed)+".metrics",sep=" ")
    seed_results = seed_results.iloc[:245,:]
    ax.plot(seed_results["#Hypervolume"],alpha=0.7,label="Seed: "+str(seed),linewidth=2)
plt.xticks(ticks=(list(range(0,300,50))),labels=list(range(0,60000,10000)))
plt.xlabel("Number of Function Evaluations (NFEs)")
plt.ylabel("Hypervolume")
fig.legend(bbox_to_anchor=(0.89, 0.15), loc='lower right', borderaxespad=0,title="Borg Seeds")


