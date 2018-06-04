import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import struct


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