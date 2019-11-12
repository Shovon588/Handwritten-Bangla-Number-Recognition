from keras.models import load_model;
import cv2
import numpy as np
import matplotlib.pyplot as plt

model=load_model("H:\\Thesis\\Digit Recognition\\numta\\model_filter.h5");

def getNumbers(imgPath="C:\\Users\\acer\\Desktop\\input.png",resize=150):
    img=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(resize,resize),interpolation=cv2.INTER_AREA)
    gauBlur=cv2.GaussianBlur(img,(9,9),10.0)
    img=cv2.addWeighted(img,1.5,gauBlur,-0.5,0,img)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    img=np.transpose(img)
    img=img[::-1]

    number=[]

    left='';right=''
    for i in range(150):
        temp=sum(img[i])
        if temp==38250 and left=='':
            pass
        elif temp==38250 and left!='':
            right=i
        elif temp!=38250 and left=='':
            left=i
        else:
            pass

        if left!='' and right!='':
            if left-5>=0:left=left-5
            else:left=0

            if right+5<150:right=right+5
            else:right=149

            a=img[left:right]

            a=a[::-1]
            a=np.transpose(a)
            a=cv2.resize(a,(32,32),interpolation=cv2.INTER_AREA)

            number.append(a)
            left='';right=''
    number=number[::-1]
    return number

def show(path):
    plt.imshow(path,cmap='gray')
    plt.show()

def predict(numbers):
    result=''
    for i in range(len(numbers)):
        a=numbers[i]
        a=a.reshape(1,32,32,1)
        a=a.astype('float32')
        a=a/255
        p=model.predict(a)
        s=np.argmax(p)
        
        result+=str(s)
    print("The predicted number is: ",result)

numbers=getNumbers()
predict(numbers)


