#  Preprocessing for all datasets

#Circular Crop
#Resize
#Gaussian Blur

### REQUIREMENTS
#install tensorflow 1.14.0
#install keras 2.4.0





# import libraries

import warnings
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pandas as pd
import scipy
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

print(tf.__version__)
print(cv2.__version__)

# Image size
im_size = 320
# Batch size
BATCH_SIZE = 32
# Bucket Number
bucket_num = 9



# Crop function: https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping

import os
import glob
import cv2
import numpy as np
def crop_image_from_gray(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def circle_crop_v2(img):
    """
    Create circular crop around image centre
    """
    img = cv2.imread(img)
    img = crop_image_from_gray(img)

    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)

    return img

def preprocess_image_old(image, desired_size=im_size):
    img = circle_crop_v2(image)  # takes path
    img = cv2.resize(img, (desired_size, desired_size))
    img = cv2.addWeighted(img,4,cv2.GaussianBlur(img, (0,0), desired_size/40) ,-4 ,128)#blend two images
    
    return img


ImgPath=r'D://Kolyah//Graduation Project//Preprocessing//Sample Images From Messidor2//20051020_43808_0100_PP.png'
processed=preprocess_image_old(ImgPath)