import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import local_binary_pattern
import pickle

path = 'test'
file = []
for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        file.append(filename)
        continue        
    else:
        continue

def LBP(image, channel=3, P=12, R=2, bins=10):
    if (channel==3):
        lbp       = local_binary_pattern(image[:,:,0], P=P, R=R, method="uniform")
        lbp1, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        lbp       = local_binary_pattern(image[:,:,1], P=P, R=R, method="uniform")
        lbp2, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        lbp       = local_binary_pattern(image[:,:,2], P=P, R=R, method="uniform")
        lbp3, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        return lbp1, lbp2, lbp3
    
    elif (channel==1):
        lbp       = local_binary_pattern(image, P=P, R=R, method="uniform")
        lbp1, _  = np.histogram(lbp, density=True, bins=bins, range=(0,int(lbp.max()+1)))
        return lbp1

lbp_features = np.zeros((len(file), 90))

for i in tqdm(range(len(file))):
    file_name = 'test_zoom/'
    file_name2 = 'test_seg/'
    path = file_name +  file[i]
    path2 = file_name2 + file[i]
    image = cv2.imread(path,cv2.COLOR_BGR2RGB)
        
    img_RGB = image.copy()   
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_YCrCb = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    
    lbp_R, lbp_G, lbp_B = LBP(img_RGB)
    lbp_H, lbp_S, lbp_V = LBP(img_HSV)
    lbp_Y, lbp_Cr, lbp_Cb = LBP(img_YCrCb)
    
    LBP_features  = np.concatenate((lbp_R,lbp_G,lbp_B,lbp_H,lbp_S,lbp_V,lbp_Y,lbp_Cr,lbp_Cb),axis=0)
    
    for j in range(len(LBP_features)):
        lbp_features[i,j] = LBP_features[j]

with open('lbp_test.pkl','wb') as f:
    pickle.dump(lbp_features, f)

