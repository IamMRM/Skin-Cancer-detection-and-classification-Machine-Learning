import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.stats import skew
from scipy.stats import kurtosis


file_name1 = "train_zoom/"
file_name2 = "train_seg/"


path = file_name1
file = []
for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        file.append(filename)
        continue        
    else:
        continue


def color_moments(img):  
    
    # Split the channels 
    one, two, three = cv2.split(img)
    
    # Initialize the color feature
    color_feature = []
    
    # The first central moment - average 
    one_mean = np.mean(one)  # np.sum(h)/float(N)
    two_mean = np.mean(two)  # np.sum(s)/float(N)
    three_mean = np.mean(three)  # np.sum(v)/float(N)
    color_feature.extend([one_mean, two_mean, three_mean])
    
    # The second central moment - standard deviation
    one_std = np.std(one) 
    two_std = np.std(two)  
    three_std = np.std(three)  
    color_feature.extend([one_std, two_std, three_std])
    
    # The third central moment - the third root of the skewness
    one_skewness = skew(one.reshape(-1))
    two_skewness = skew(two.reshape(-1))
    three_skewness = skew(three.reshape(-1))
    color_feature.extend([one_skewness, two_skewness, three_skewness])

    return color_feature

hist=[]
for i in tqdm(range(len(file))):
    path = file_name1+  file[i]
    path2 = file_name2 + file[i]
    image = cv2.imread(path,cv2.COLOR_BGR2RGB)
   
    mask = cv2.imread(path2,cv2.COLOR_BGR2RGB)
    mask = mask[:,:,1]
    image[mask == 0] = 0
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imago = image.copy()

    imgHSV = cv2.cvtColor(imago, cv2.COLOR_RGB2HSV)
    imgYCrCb = cv2.cvtColor(imago, cv2.COLOR_RGB2YCrCb)
    c_rgb = color_moments(imago, mask)
    c_HSV = color_moments(imgHSV, mask)
    c_YCrCb = color_moments(imgYCrCb, mask)
    hist.append(np.hstack((c_rgb, c_HSV,c_YCrCb)))


np.savez("train_cm", hist)

