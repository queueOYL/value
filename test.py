# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:26:35 2023

@author: queueOYL
"""
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# 导入模型
model = tf.keras.models.load_model('DenseModel.h5')

# 加载图片
img = cv.imread("pic/1.jpg")
# 图片灰度化
gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img_resize = cv.resize(gray_img,(28,28))
plt.imshow(img_resize)
plt.show()

res = model.predict(img_resize.reshape((1,28,28,1)))
print (res)
print (np.argmax(res))


