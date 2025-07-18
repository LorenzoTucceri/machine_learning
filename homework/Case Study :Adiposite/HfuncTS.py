import math  
import pandas as pd
import numpy as np
import time
import datetime

# -------H-functions-----------------#
def H(df, X):
    # computes the joint entropy of X #
    X = list(X)
    SubTTS = df.loc[:,X].values
    SubTTS = [",".join(item) for item in SubTTS.astype(str)]
    AlphaX, Freq = np.unique(SubTTS, return_counts=True)
    total = float(sum(Freq))
    Prob = [Freq[i] / total for i in range(len(AlphaX))]
    return -sum(Prob[i] * math.log(Prob[i], 2) for i in range(len(AlphaX)))


def HH(df, X):
    # computes the exaustivity of X
    X = list(X)
    SubTTS = df.loc[:,X].values
    SubTTS = [",".join(item) for item in SubTTS.astype(str)]
    AlphaX, Freq = np.unique(SubTTS, return_counts=True)
    prodspaceX = 0
    for x in X: prodspaceX +=math.log(len(AlphaX), 2) 
   
    if prodspaceX == 0: return 0
    return H(df, X) / prodspaceX


def Hc(df, X, Y):
    # computes the conditional entropy H(X|Y)
    return H(df, X | Y) - H(df, Y)


def HHc(df, X, Y):
    # computes the normalized conditional entropy H(X|Y)
    if H(df, X) == 0: return 0
    return Hc(df, X, Y) / H(df, X)

#   I(X;Y) =  H(X) - H(X|Y)

def I(df, X, Y):
    # computes the mutual information of X and Y
    return H(df, X) - Hc(df, X, Y)


def II(df, X, Y):
    # computes the fraction of information that X  gives on Y
    if H(df, Y) == 0: return 0
    return (H(df, X) - Hc(df, X, Y)) / H(df, Y)

def Ic(df, X, Y, Z):
    # computes the info that X gives on Y given Z (Conditional Mutual Information)
    # I(X;Y|Z) =  H(X|Z) - H(X|Y, Z)
    return Hc(df, X, Z) - Hc(df, X, Y | Z)


def IIc(df, X, Y, Z):
    # computes the fraction of info that X gives on Y given Z
    if Hc(df, Y, Z) == 0: return 0
    return (Hc(df, X, Z) - Hc(df, X, Y | Z)) / Hc(df, Y, Z)

def IIcc(df, X, Y, Z):
    # computes the fraction of info that X gives on Y given Z
    if Hc(df, Y, Z) == 0: return 0
    return (HHc(df, X, Z) - HHc(df, X, Y | Z)) / HHc(df, Y, Z)


def Conf(df, X, Y):
    # computes the confidence [0,1]
    return (1 + II(df, X, Y) - HHc(df, Y, X)) / float(2)

#----------------------------------------------
def argmin1f3(df,XX,Y,Z,f):
    # One element x of XX that minimises f(x,Y,Z)
    X=XX.copy()
    xmin=X.pop()
    m=f(df,{xmin},Y,Z)
    while len(X)>0:
        x=X.pop()
        if f(df,{x},Y,Z)< m: m=f(df,{x},Y,Z); xmin=x
    return xmin

#------------------MinSuff----------------------------------
def argmax1f3(df,XX,Y,Z,f):
    # One element x of XX that maximises f(x,Y,Z)
    X=XX.copy()
    xmax=X.pop()
    m=f(df,{xmax},Y,Z)
    while len(X)>0:
        x=X.pop()
        if f(df,{x},Y,Z)> m: m=f(df,{x},Y,Z); xmax=x
    return xmax


def minsuffI(df, XX,Y):
    # Return a minimal sufficient subset M of XX for determining Y
    X=XX.copy()
    M=set()
    while Hc(df,Y,M)>0:
       if len(X) ==0: return set()
       i=argmax1f3(df,X,Y,M,Ic)
#       print(i, Hc(df,set([i]),M))
       M |= {i}
       X.remove(i)
    return easyoptimize(df,M)

def DOME_I(df, XX,Y):
    # Return the necessary subset D of XX for determining Y
    D=set()
    M=minsuffI(df,XX,Y)
    while len(M )> 0:
        x=M.pop()
        if ( II(df, XX-set([x]),set([x])) < 1): D |= {x}
            
#    if D==set(): return minsuffI(df,XX,Y)
    return D

    
def easyoptimize(df,SS):
    # eliminates "useless" elements in SS and returns a min suff set 
    S=SS.copy()
    restS=SS.copy()
    while len(S) >0 :
        s=S.pop()
        if Hc(df,{s},restS-{s})==0 : restS-={s}
    return restS


