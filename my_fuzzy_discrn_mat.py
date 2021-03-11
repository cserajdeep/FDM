########################## Fuzzy Discernibility Matrix: Reduct ###############################
####################### Dr. Rajdeep Chatterjeem 25-01-21, version: 1.0 ###########################

import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

def gaussmf(x, mean, sigma):
    """
    Gaussian fuzzy membership function.

    Parameters
    ----------
    x : 1d array or iterable
        Independent variable.
    mean : float
        Gaussian parameter for center (mean) value.
    sigma : float
        Gaussian parameter for standard deviation.

    Returns
    -------
    y : 1d array
        Gaussian membership function for x

    """
    eps=1e-16
    
    return np.exp(-((x - mean) ** 2.) / (float(sigma) ** 2.+eps))

def fuzzy_distance10(x,y):
    eps = 1e-16
    d = 1-(sum(min(x,y))/(sum(min(x,y))+eps))
    
    return d

def getthefuzzyvalues(F,p,t,cls):
    
    temp=[]
    for i in range(cls):
        temp.append(F[p][((t)*cls)+i])
        
    return temp
    
def my_fuzzy_discrn_mat(df,best):
    
    r,c=df.shape
    dff=df.copy()
    if best<c:
        oldf=df.columns   #check if actual column names are in 0 to ...
        newf=list(range(c))
        df.rename(columns=dict(zip(oldf, newf)), inplace=True)
        dc=list(df[newf[-1]])
        #print('\nAfter changing the column names to 0 to ...\n',df)
        
        # creating instance of labelencoder
        labelencoder = preprocessing.LabelEncoder()
        labelencoder.fit(df[newf[-1]])
        #print(list(labelencoder.classes_))

        # Assigning numerical values and storing in another column
        df[newf[-1]] = labelencoder.fit_transform(df[newf[-1]])
        #df
        
        datasets = {}
        by_class = df.groupby(newf[-1])

        for groups, data in by_class:
            datasets[groups] = data
        
        #len(datasets)
        meand=[]
        stdd=[]
        for i in range(len(datasets)):
            #print(datasets[i])
            meand.append(list(datasets[i].mean(axis = 0)))
            #print('mean',meand)
            stdd.append(list(datasets[i].std(axis = 0)))
            #print('std',stdd)
            
        X=df.to_numpy()
        #r,c=X.shape

        #oldf=df.columns
        #newf=list(range(c))
        #df.rename(columns=dict(zip(oldf, newf)), inplace=True) #renaming column
        D=list(df[newf[-1]])

        labelencoder.fit(df[newf[-1]])
        #list(labelencoder.classes_)
        classes=len(list(labelencoder.classes_))
        FD=np.zeros((r,(c-1)*classes))

        '''for i in range(len(datasets)):
            oldf=datasets[i].columns
            newf=list(range(c))
            datasets[i].rename(columns=dict(zip(oldf, newf)), inplace=True) #renaming column'''
            #print(datasets[i])

        for j in range(c-1):
            for i in range(r):
                l=(j*classes)
                for k in range(classes):
                    g=gaussmf(X[i][j], np.array(meand[k][j]), np.array(stdd[k][j]))
                    #print(g)
                    FD[i][l]=g #float(str(round(g, 8)))
                    l = l+1
                    
                    
        D=np.array(D)
        D=D.reshape(D.shape[0],1)
        F=np.concatenate((FD, D), axis=1)
        
        # Create Fuzzyfied DataFrame 
        fdf = pd.DataFrame(F) 
  
        # Print the output. 
        #fdf 
        
        S=[]
        for i in range(r):
            for j in range(r):
                if i>j and F[i][-1]!=F[j][-1]:
                    dv=[]
                    for k in range(c-1):
                        x=getthefuzzyvalues(F,i,k,classes)
                        y=getthefuzzyvalues(F,j,k,classes)
                        dist=fuzzy_distance10(x,y)
                        #dist=float(str(round(dist, 4)))
             
                        dv.append(dist)
                
                    S.append(dv)

        #print('\nValid entries in Discernibilty Matrix: ',S)   
        
        A=np.mean(S,axis=0)
        ids=np.flip(np.argsort(A))
        reduct=ids[:best]
        idx=list(reduct)
        idx.append(newf[-1])
        df[newf[-1]]=dc
        rdf=df[idx]
        
        '''uf=[oldf[i] for i in range(len(oldf)) if i in idx]
        df.rename(columns=dict(zip(oldf, uf)), inplace=True)
        rdf=dff[uf].copy()'''
        
        return rdf, list(reduct)