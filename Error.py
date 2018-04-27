# -*- coding: utf-8 -*-
import WriteIdentify as WI

#讀取資料
try:
    tk=open('deep\\test','rb')
    D=pickle.load(tk)
    tk.close
except (EnvironmentError, pickle.PicklingError) as err:
    raise LoadError(str(err))
first=WI.net()
first.loadbrain('train1')
total=0
acc=0
for i in range(len(D)):
    X=D[i]
    a=WI.denumtable(first.test(D[i][0]))
    if(a==[1,0]):
        acc=acc+1
    b=WI.denumtable(first.test(D[i][1]))
    if(b==[0,1]):
        acc=acc+1
    total=total+2
print('正確率:'+str(acc/total*100)+'%')