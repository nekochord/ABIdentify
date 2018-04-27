# -*- coding: utf-8 -*-
import random
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import time
def matolist(M):
    k=np.size(M)
    answer=[0]*k
    for i in range(k):
        answer[i]=M[0,i]
    return answer
def addlist2(x1,x2):
    answer=[0]*len(x1)
    for i in range(len(x1)):
        a=np.matrix(x1[i])
        b=np.matrix(x2[i])
        c=a+b
        answer[i]=matolist(c)
    return answer
def sig(M):
    a=np.power(math.e,-M)
    b=1/(1+a)
    return b
def sigmoid(num):
    answer=1/(1+math.e**(-1*num))
    return answer
def numtable(num):
    a=[0]*2
    a[num]=1
    return a
def denumtable(lis):
    m=max(lis)
    return lis.index(m)
def binary(num,l):
    #回傳指定長度的二進位list
    answer=[0]*l
    k=bin(num)
    if num>0:
        for i in range(len(k)-2):
            answer[l-1-i]=int(k[len(k)-1-i])
    if num<0:
        for i in range(len(k)-3):
            answer[l-1-i]=int(k[len(k)-1-i])
        answer[0]=1
    return answer
def lossfun(a,b):
    answer=0
    for i in range(len(a)):
        answer=answer+(a[i]-b[i])**2
    return answer
class layer():
    def __init__(self,long,rank):
        self.W=[0]*long
        self.B=[0]*long
        for i in range(long):
            self.W[i]=[0]*rank
            self.B[i]=random.randint(-1,1)
        for i in range(long):
            for j in range(rank):
                self.W[i][j]=random.randint(-1,1)
class net():
    def __init__(self):
        self.brain=[layer(200,100),layer(100,200),layer(2,100)]
        self.sita=[[0]*200,[0]*100,[0]*2]
        self.tosita=[[0]*200,[0]*100,[0]*2]
        self.cal=[[0]*100,[0]*200,[0]*100,[0]*2]
        self.fix=0.001
    def test(self,d):
        self.cal[0]=d
        for i in range(len(self.brain)):
            self.cal[i+1]=self.active(self.cal[i],self.brain[i]) 
        return self.cal[len(self.cal)-1]
    def learn(self,x,y):
        self.test(x)
        self.calsita(y)
        #weight update
        for l in range(len(self.brain)):
            lay=self.brain[l]
            for j in range(len(lay.W)):
                if(self.sita[l][j]>0):
                    lay.B[j]=lay.B[j]-self.fix
                if(self.sita[l][j]<0):
                    lay.B[j]=lay.B[j]+self.fix
                for k in range(len(lay.W[j])):
                    if(self.cal[l][k]*self.sita[l][j]>0):
                        lay.W[j][k]=lay.W[j][k]-self.fix
                    if(self.cal[l][k]*self.sita[l][j]<0):
                        lay.W[j][k]=lay.W[j][k]+self.fix
    def learnmo(self,x,y):
        self.tosita=[[0]*200,[0]*100,[0]*2]
        for i in range(len(x)):
            self.test(x[i])
            self.calsita(y[i])
            self.tosita=addlist2(self.tosita,self.sita)
        #weight update
        for l in range(len(self.brain)):
            lay=self.brain[l]
            for j in range(len(lay.W)):
                if(self.tosita[l][j]>0):
                    lay.B[j]=lay.B[j]-self.fix
                if(self.tosita[l][j]<0):
                    lay.B[j]=lay.B[j]+self.fix
                for k in range(len(lay.W[j])):
                    if(self.cal[l][k]*self.tosita[l][j]>0):
                        lay.W[j][k]=lay.W[j][k]-self.fix
                    if(self.cal[l][k]*self.tosita[l][j]<0):
                        lay.W[j][k]=lay.W[j][k]+self.fix      
    def active(self,ca,lay):
        #check
        W=np.matrix(lay.W)
        B=np.matrix(lay.B)
        C=np.matrix(ca)
        a=W*np.transpose(C)
        a=np.transpose(a)+B
        a=sig(a)
        a=matolist(a)
        return a
    def calsita(self,y):
        T=np.matrix(y)
        F=self.cal[len(self.cal)-1]
        F=np.matrix(F)
        L=(F-T)
        #L=np.multiply((F-T),F)
        #L=np.multiply(L,(1-F))
        self.sita[len(self.sita)-1]=matolist(L)
        for i in range(1,len(self.sita)):
            k=len(self.sita)-1-i
            self.nextsita(k)
    def nextsita(self,k):
        L=np.matrix(self.sita[k+1])
        W=self.brain[k+1].W
        W=np.matrix(W)
        A=np.matrix(self.cal[k+1])
        a=L*W
        a=np.multiply(a,A)
        a=np.multiply(a,1-A)
        self.sita[k]=matolist(a)
    def savebrain(self,filename):
        try:
            fs=open('deep\\'+filename,'wb')
            pickle.dump(self.brain,fs)
            fs.close()
        except (EnvironmentError, pickle.PicklingError) as err:
            raise SaveError(str(err))
    def loadbrain(self,filename):
        try:
            fh=open('deep\\'+filename,'rb')
            self.brain=pickle.load(fh)
            fh.close()
        except (EnvironmentError, pickle.PicklingError) as err:
            raise LoadError(str(err))
#讀取學習資料
try:
    tk=open('deep\\learn','rb')
    D=pickle.load(tk)
    tk.close
except (EnvironmentError, pickle.PicklingError) as err:
    raise LoadError(str(err))
###學習
print(len(D))
check=D[20][0]
#A
first=net()
first.loadbrain('train0')
Y=[0]*2
Y[0]=[1,0]
Y[1]=[0,1]
plox=[0]*100
ploy=[0]*100
for i in range(100):
    plox[i]=i
    X=D[i]
    first.learnmo(X,Y)
    ploy[i]=lossfun(first.test(check),[1,0])
first.savebrain('train1')
plt.plot(plox,ploy)
