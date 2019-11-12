#import libraries
import time
startTime=time.perf_counter() #counter started

import sys
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras
from keras.layers import Dense,Activation,Input,Conv2D,Flatten,MaxPooling2D,Dropout,BatchNormalization
from keras.models import Sequential,Model
from keras.optimizers import Adamax
from keras import backend as k
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint



#initiate

resizeDim=32
figWidth = 20
heightPerRow=3

#process data

data_dir="H:\\Thesis\\Digit Recognition\\numta"

#path of every image for training
paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
paths_train_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))
paths_train_e=glob.glob(os.path.join(data_dir,'training-e','*.png'))

paths_train_all=paths_train_a+paths_train_b+paths_train_c+paths_train_d+paths_train_e

'''
#path of every image for testing
paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))

paths_test_all=paths_test_a + paths_test_b + paths_test_c + paths_test_d + paths_test_e + paths_test_f + paths_test_auga + paths_test_augc
'''

#paths of csv files for training label
label_train_a=os.path.join(data_dir,'training-a.csv')
label_train_b=os.path.join(data_dir,'training-b.csv')
label_train_c=os.path.join(data_dir,'training-c.csv')
label_train_d=os.path.join(data_dir,'training-d.csv')
label_train_e=os.path.join(data_dir,'training-e.csv')





'''
a=paths_train_a[0:10] #particular image directory

img=cv2.imread(a,cv2.IMREAD_GRAYSCALE) #image array(3D)

img=cv2.resize(img,(28,28),interpolation=cv2.INTER_AREA) #image array(1D)


plt.imshow(img,cmap='gray') #plot image
plt.show() #show image

img_lower=img[14:28] #lower half of the image
img_upper=img[:14] #upper half of the image
'''


############################### Function Area Starts ###############################

def getKey(path):
    key=path.split(sep=os.sep)[-1]
    return key


def countTime():
    curTime=time.perf_counter()
    temp=curTime-startTime
    if temp<60:
        temp=round(temp,2)
        s='Time elapsed: '+str(temp)+' sec'
        return s
    else:
        temp=temp/60
        temp=round(temp,2)
        s='Time elapsed: '+str(temp)+' min'
        return s


def getData(imgPath,pathLabel=None,resize=None):
    #imgPath = image location in the disk
    #pathLabel = csv file location for corresponding folder
    #resize = size of the image after resizing (n*n)
    
    x=[]
    for i,path in enumerate(imgPath):
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        if resize is not None:
            img=cv2.resize(img,(resize,resize),interpolation=cv2.INTER_AREA)
            
        gauBlur=cv2.GaussianBlur(img,(9,9),10.0)
        img=cv2.addWeighted(img,1.5,gauBlur,-0.5,0,img)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        x.append(img)

        #show off part starts
        if i==len(imgPath)-1:
            end='\n'
        else: end='\n'
        per=((i+1)/len(imgPath))*100

        temp=countTime()
        print('processed (%d of %d) --> %.2f'%(i+1,len(imgPath),per)+'%, '+temp)
        #show off part ends


    x=np.array(x)

    if pathLabel is None:
        return x
    else:
        df = pd.read_csv(pathLabel) 
        df=df.set_index('filename') 
        y_label=[df.loc[getKey(path)]['digit'] for path in  imgPath]
        y=to_categorical(y_label,10)
        return x,y


def imshowGroup(x,y,yPred=None,imgPerRow=5,phase='processed'):
    nSample=len(x)
    imgDim=x.shape[1]
    j=np.ceil(nSample/imgPerRow)
    fig=plt.figure(figsize=(figWidth,heightPerRow*j))

    for i,img in enumerate(x):
        plt.subplot(j,imgPerRow,i+1)
        plt.imshow(img,cmap='gray')

        if phase=='processed':
            plt.title(np.argmax(y[i]))
        if phase=='prediction':
            topN=3
            ind=np.argsort(yPred[i])[::-1]
            h=imgDim+4

            for k in range(topN):
                string='pred: {} ({:.0f}%)\n'.format(ind[k],yPred[i,ind[k]]*100)
                plt.text(imgDim/2, h, string, horizontalalignment='center',verticalalignment='center')
                h+=4
            if y is not None:
                plt.text(imgDim/2, -4, 'true label: {}'.format(np.argmax(y[i])), 
                         horizontalalignment='center',verticalalignment='center')
        plt.axis('off')
    plt.show()


def predictInput(imgPath="C:\\Users\\acer\\Desktop\\input.png",resize=32):
    img=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(resize,resize),interpolation=cv2.INTER_AREA)
    gauBlur=cv2.GaussianBlur(img,(9,9),10.0)
    img=cv2.addWeighted(img,1.5,gauBlur,-0.5,0,img)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)

    temp=img
    img=img.reshape(1,32,32,1)
    img=img.astype('float32')
    img=img/255

    p=model.predict(img)
    res=np.argmax(p)

    print(res)
    plt.imshow(temp,cmap='gray')
    plt.title(res)

    return p


############################# Function Area Ends ###############################


#Fetch training data

xTrainA,yTrainA=getData(paths_train_a,label_train_a,resize=resizeDim)
xTH:/Thesis/Digit Recognition/numta/training-b/b00005.pngrainB,yTrainB=getData(paths_train_b,label_train_b,resize=resizeDim)
xTrainC,yTrainC=getData(paths_train_c,label_train_c,resize=resizeDim)
xTrainD,yTrainD=getData(paths_train_d,label_train_d,resize=resizeDim)
xTrainE,yTrainE=getData(paths_train_e,label_train_e,resize=resizeDim)

xTrainAll=np.concatenate((xTrainA,xTrainB,xTrainC,xTrainD,xTrainE),axis=0)
yTrainAll=np.concatenate((yTrainA,yTrainB,yTrainC,yTrainD,yTrainE),axis=0)

#xTrainAll.shape,yTrainAll.shape #should be of same length


'''
#show some first data
plt.subplot(221)
plt.imshow(xTrainAll[0],cmap='gray')
plt.subplot(222)
plt.imshow(xTrainAll[1],cmap='gray')
pt.subplot(223)
plt.imshow(xTrainAll[2],cmap='gray')
plt.subplot(224)
plt.imshow(xTrainAll[3],cmap='gray')

plt.show()
'''


'''
#show histogram of a digit
img=xTrainAll[1]
hist=cv2.calcHist([img],[0],None,[256],[0,256])
plt.hist(img,ravel(),256,[0,256])

plt.show()
'''

'''
#Fetch test data
xTestA=getData(paths_test_a,resize=resizeDim)
xTestB=getData(paths_test_b,resize=resizeDim)
xTestC=getData(paths_test_c,resize=resizeDim)
xTestD=getData(paths_test_d,resize=resizeDim)
xTestE=getData(paths_test_e,resize=resizeDim)
xTestF=getData(paths_test_f,resize=resizeDim)
xTestAugA=getData(paths_test_auga,resize=resizeDim)
xTestAugC=getData(paths_test_augc,resize=resizeDim)

xTestAll=np.concatenate((xTestA,xTestB,xTestC,xTestD,xTestE,xTestF,xTestAugA,xTestAugC))
'''

xTrainAllShow=xTrainAll
yTrainAllShow=yTrainAll
#xTestAllShow=xTestAll


#convert image to 1 color channel
#Q-> Can we plt the image after converting it to one color channel?
#What an one color channel image actually is?

xTrainAll=xTrainAll.reshape(xTrainAll.shape[0],32,32,1)
xTrainAll=xTrainAll.astype('float32')

#xTestAll=xTestAll.reshape(xTestAll.shape[0],32,32,1)
#xTestAll=xTestAll.astype('float32')


#Normalize data
xTrainAll=xTrainAll/255
#xTestAll=xTestAll/255



#Differentiate test and validation data
indices=list(range(len(xTrainAll)))
np.random.seed(44)
np.random.shuffle(indices)

ind=int(len(indices)*0.80)

#Train data
xTrain=xTrainAll[indices[:ind]]
yTrain=yTrainAll[indices[:ind]]

#Validation data
xVal=xTrainAll[indices[ind:]]
yVal=yTrainAll[indices[ind:]]


#Build model
def myModel(imgSize=32,channel=1):
    model=Sequential()
    print(imgSize,channel)
    inputShape=(imgSize,imgSize,channel)

    model.add(Conv2D(32,(5,5),input_shape=inputShape,
                      activation='relu',padding='same'))
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    

    return model



midTime=time.perf_counter()

#Model Training
pathModel='new_model_filter.h5'
k.tensorflow_backend.clear_session()
model=myModel()
k.set_value(model.optimizer.lr,1e-3)

h=model.fit(x=xTrain,y=yTrain,batch_size=64,epochs=5,
            verbose=1,validation_data=(xVal,yVal),
            shuffle=True,
            callbacks=[ModelCheckpoint(filepath=pathModel),])



'''
predictions=model.predict(xTestAll)

sample=20
np.random.seed(44)
ind=np.random.randint(0,len(xTestAll),size=sample)


imshowGroup(x=xTestAllShow[ind],y=None,yPred=predictions[ind],phase='prediction')

labels=[np.argmax(pred) for pred in predictions]
keys=[getKey(path) for path in paths_test_all]
'''


#Ending time count
endTime=time.perf_counter()
print('\n\nProcess completed!!\nTotal time taken: %.2f min'%((endTime-startTime)/60))
