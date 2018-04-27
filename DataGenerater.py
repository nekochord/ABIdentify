# -*- coding: utf-8 -*-
from PIL import Image as im
from PIL import ImageStat as ims
import numpy as np
import pickle
def transform(mage):
    answer=[0]*10
    for i in range(10):
        answer[i]=[0]*10
    for i in range(10):
        for j in range(10):
            box=(16*j,16*i,16*j+16,16*i+16)
            region=mage.crop(box)
            a=ims.Stat(region)
            b=np.array(a._getmean())
            v=(b[0]+b[1]+b[2])/3
            if(v<230):
                answer[i][j]=1
    print(np.matrix(answer))
    r=[0]*100
    count=0
    for i in range(10):
        for j in range(10):
            r[count]=answer[i][j]
            count=count+1
    return r
def savelearn(number):
    D=[0]*number
    for i in range(number):
        D[i]=[0]*2
        print('imageid//A//'+str(i)+'.png')
        f=im.open('imageid//A//'+str(i)+'.png')
        g=transform(f)
        D[i][0]=g
        print('imageid//B//'+str(i)+'.png')
        f=im.open('imageid//B//'+str(i)+'.png')
        g=transform(f)
        D[i][1]=g
    try:
        fh=open('deep\\learn','wb')
        pickle.dump(D,fh)
        fh.close()
    except (EnvironmentError, pickle.PicklingError) as err:
        raise SaveError(str(err))
def savetest(number):
    D=[0]*number
    for i in range(number):
        D[i]=[0]*2
        print('imagetest//A//'+str(i)+'.png')
        f=im.open('imagetest//A//'+str(i)+'.png')
        g=transform(f)
        D[i][0]=g
        print('imagetest//B//'+str(i)+'.png')
        f=im.open('imagetest//B//'+str(i)+'.png')
        g=transform(f)
        D[i][1]=g
    try:
        fh=open('deep\\test','wb')
        pickle.dump(D,fh)
        fh.close()
    except (EnvironmentError, pickle.PicklingError) as err:
        raise SaveError(str(err))
savelearn(100)
savetest(25)