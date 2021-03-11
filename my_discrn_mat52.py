########################## Classical Discernibility Matrix: Reduct (works for 52 features A to Z+ a to z) ###############################
####################### Dr. Rajdeep Chatterjeem 24-01-21, version: 2.0 ###########################
import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import string

def my_discrn_mat52(df,nbin=3):
    
    r,c=df.shape
    
    if c<=26:
        oldf=df.columns   #check if actual column names are in 0 to ...
        newf=list(string.ascii_uppercase)[:c]
        df.rename(columns=dict(zip(oldf, newf)), inplace=True)
    elif c<=52:
        oldf=df.columns   #check if actual column names are in 0 to ...
        newf=list(string.ascii_uppercase+string.ascii_lowercase)[:c]
        df.rename(columns=dict(zip(oldf, newf)), inplace=True)
    
    # creating instance of labelencoder
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(df[newf[-1]])
    #print('Actual Class labels: ',list(labelencoder.classes_))

    # Assigning numerical values and storing in another column
    df[newf[-1]] = labelencoder.fit_transform(df[newf[-1]])
    #print('\nActual Dataset: \n',df)
    
    X=df.to_numpy()
    # transform the dataset with KBinsDiscretizer
    enc = KBinsDiscretizer(n_bins=3, encode='ordinal',strategy='uniform')
    X_binned = enc.fit_transform(X.tolist())
    X_binned=X_binned+1
    #print('\nDiscretized Dataset: \n', X_binned)
    
    r,c = X_binned.shape
    f = newf[:-1]
    D = []
  
    # Create DataFrame 
    #print('\nDiscernibilty Matrix Processing...')
    for i in range(r):
        for j in range(r):
            if i>j and X_binned[i][-1]!=X_binned[j][-1]:
                str1=''
                for k in range(c-1):
                    if X_binned[i][k]!=X_binned[j][k]:
                        if k==0 or str1=="":
                            str1 = str1+f[k]
                            
                        else:
                            str1 = str1+" "+f[k]
 
                D.append(str1)
                #print('{}-{}:{}'.format(i+1,j+1,str1))
                
            #elif i>j and X_binned[i][-1]==X_binned[j][-1]:
                #print('{}-{}:{}'.format(i+1,j+1,'\u03A6'))
                
    #print('\nValid entries in Discernibilty Matrix: ',D)
    
    l=len(D)
    core =[]
    for i in range(l):
        if len(D[i])==1 and not(any(D[i]==elem for elem in core)):
            core.append(D[i])
    
    print('\nCORE(s): ',core)
    
    D=sorted(D,reverse=True)
    t=len(core)
    reduct=[]
    if t!=0:
        for i in range(l):
            c=0
            for j in range(t):
                if D[i].find(core[j]) != -1 and len(D[i])>1:
                    c=c+1
            
            if c==t:
                reduct=D[i]
                break
    else:
        reduct=min(D, key = len)  
    
    print('\nREDUCT: ',reduct)
    rf=reduct.split(" ")
    rfb=[newf.index(i) for i in rf]
    print(rfb)
    
    rf.append(newf[-1])
    
    return df[rf], rfb