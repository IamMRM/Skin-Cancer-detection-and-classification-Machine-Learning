import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import greycomatrix, greycoprops
import pickle

path = 'test'
file = []
for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        file.append(filename)
        continue        
    else:
        continue
        
print(len(file))

def glcm_features(image):

    one = image[:,:,0]
    two = image[:,:,1]
    three = image[:,:,2]

    diss_sim = []
    corr = []
    homogen = []
    energy = []
    contrast = []
    
    glcm_one = greycomatrix(one, distances=[15], angles=[0], levels=256,
                        symmetric=True, normed=True)
    glcm_two = greycomatrix(two, distances=[15], angles=[0], levels=256,
                        symmetric=True, normed=True)

    glcm_three = greycomatrix(three, distances=[15], angles=[0], levels=256,
                        symmetric=True, normed=True)
    
    diss_sim.append(greycoprops(glcm_one, 'dissimilarity')[0, 0]) #[0,0] to convert array to value
    diss_sim.append(greycoprops(glcm_two, 'dissimilarity')[0, 0]) #[0,0] to convert array to value
    diss_sim.append(greycoprops(glcm_three, 'dissimilarity')[0, 0]) #[0,0] to convert array to value


    corr.append(greycoprops(glcm_one, 'correlation')[0, 0])
    corr.append(greycoprops(glcm_two, 'correlation')[0, 0])
    corr.append(greycoprops(glcm_three, 'correlation')[0, 0])

    homogen.append(greycoprops(glcm_one, 'homogeneity')[0, 0])
    homogen.append(greycoprops(glcm_two, 'homogeneity')[0, 0])
    homogen.append(greycoprops(glcm_three, 'homogeneity')[0, 0])

    energy.append(greycoprops(glcm_one, 'energy')[0, 0])
    energy.append(greycoprops(glcm_two, 'energy')[0, 0])
    energy.append(greycoprops(glcm_three, 'energy')[0, 0])
    
    contrast.append(greycoprops(glcm_one, 'contrast')[0, 0])
    contrast.append(greycoprops(glcm_two, 'contrast')[0, 0])
    contrast.append(greycoprops(glcm_three, 'contrast')[0, 0])

    return diss_sim,corr, homogen, energy, contrast


glcm_features = np.zeros((len(file), 45))

for i in tqdm(range(len(file))):
    file_name = 'test_zoom/'
    file_name2 = 'test_seg/'
    path = file_name +  file[i]
    path2 = file_name2 + file[i]
    image = cv2.imread(path,cv2.COLOR_BGR2RGB)
        
    img_RGB = image.copy()   
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_YCrCb = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)
    
    diss_sim,corr, homogen, energy, contrast = glcm_features(img_RGB)
    diss_sim_hsv,corr_hsv, homogen_hsv, energy_hsv, contrast_hsv = glcm_features(img_HSV)
    diss_sim_y,corr_y, homogen_y, energy_y, contrast_y = glcm_features(img_YCrCb)
    
    GLCM_features  = diss_sim + corr + homogen +  energy + contrast + diss_sim_hsv + corr_hsv +  homogen_hsv + energy_hsv + contrast_hsv + diss_sim_y + corr_y + homogen_y + energy_y + contrast_y
    
    for j in range(len(GLCM_features)):
        glcm_features[i,j] = LBP_features[j]



with open('glcm_test.pkl','wb') as f:
    pickle.dump(glcm_features, f)

