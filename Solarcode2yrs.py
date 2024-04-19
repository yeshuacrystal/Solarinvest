# This is a code for solar activity


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import glob
import statsmodels.api as sm
from sklearn.utils import shuffle
from calendar import isleap

import nolds

# This is a code to get all the files in  a folder

fl = ['20-21FP.xlsx','22-23FP.xlsx']
lf = []
for i in fl:
  print(i)
  dk = pd.read_excel(i,usecols=[4],header=None)
  lf.append(dk)

xx = pd.DataFrame(np.squeeze(np.concatenate(lf)),index=pd.date_range(start='1/1/2020 00:05:00',end='1/1/2024 00:05:00',freq='5min'),columns=['FP'])

xx['Year'] = xx.index.year
# ax = sns.boxplot(x='Year',y='FP',data=xx,showmeans=True,showfliers=False)
ux = []
for u,uu in enumerate(xx['Year'].unique()[:-1]):
  xc = xx[xx['Year']==uu]
  qstep = 8
  qs = np.arange(-8,8.1,2.0/qstep)
  scstep = 8
  scales = np.floor(2.0**np.arange(6,10,1.0/scstep)).astype('i4')

  # samp = nolds.sampen(xc['FP'].values)
  lya = nolds.lyap_r(xc['FP'].values,trajectory_len=9,fit='poly')
  #uy = nolds.corr_dim(xc['SYMH'].values,3,rvals=np.arange(0,10))
  
  # hur = nolds.hurst_rs(xc['SYMH'].values)
  
  print([u,uu,round(lya,2),round(xc['FP'].mean(),2)])
  ux.append([u,uu,round(lya,2),round(xc['FP'].mean(),2)])

  # print([u,uu,round(hur,2),round(xc['SYMH'].mean(),2)])
  # ux.append([u,uu,round(hur,2),round(xc['SYMH'].mean(),2)])