import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import struct
import cv2

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    #以二進制方式打開文件 文件路徑是本方法的傳入參數
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

#讀取訓練數據集
image_train_int8,labels_train = load_mnist(r'D:\github\dn_nemo_class\Mnist',kind='train')

#讀取測試數據
image_test_int8,labels_test = load_mnist(r'D:\github\dn_nemo_class\Mnist',kind='t10k')

thresh = 50
image_train = (image_train_int8>=thresh)*1
image_test  = (image_test_int8>=thresh)*1

fig, ax = plt.subplots(
        nrows = 2,
        ncols = 5,
        sharex = True,
        sharey = True)

ax = ax.flatten()

for i in range(10):
    img = image_train[labels_train==i][0].reshape(28,28)
    ax[i].imshow(img,cmap = 'Greys',interpolation = 'nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()



from sklearn.ensemble import RandomForestClassifier
import datetime

'''
1.指定一个模型
2.给数据 根据输入数据 以选择的模型基础上进行训练
3.使用训练好的模型进行预测
'''
model=  RandomForestClassifier(n_estimators=50,n_jobs=-1)
starttime  = datetime.datetime.now()
model.fit(image_train,labels_train)
endtime = datetime.datetime.now()
timetrain = endtime-starttime
print(timetrain)



# 读取图片
im   = cv2.imread(r'D:\github\dn_nemo_class\4\1.png')

#将图片转为灰阶
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

#重置图片大小

im_final = cv2.resize(im_gray,(28,28))

thresg = 30
im_bi = (im_final>=thresh)*1
im_vec = im_bi.flatten()
im_vec = im_vec.reshape(1,-1)

 #使用训练好得模型进行预测
 
prediciton = model.predict(im_vec)
 
print ('the num of the img is :'+ str(prediciton))

cv2.imshow('666',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
1.读取图片
2.转化为图片格式
3.预处里
4.向量化
5.进模型
'''
from sklearn.externals import joblib
#指定名稱
filename = 'model.m'

#調用 DUMP 函數
joblib.dump(model,filename)

# 調用乙存儲模型
model = joblib.load('model.m')




















