#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 70,10
import helper
import pandas as pd
import csv
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets, metrics
from sklearn.model_selection import GridSearchCV

# In[2]:


file_name = "Code/train/les/"
file_name2 = "Code/train/ls_mask_segmentation/"
file_name3 = "Code/train/nv/"
file_name4 = "Code/train/nv_mask_segmentation/"
file_name5 = "Code/val/les/"
file_name6 = "Code/val/les_seg_val/"
file_name7 = "Code/val/nv/"
file_name8 = "Code/val/nv_seg_val/"


# In[51]:


file = helper.filee(file_name)
mask, img_rgb, image, output, grayScale = helper.loading(13,file,file_name,file_name2)
helper.plot(mask, img_rgb, image, output, grayScale)


# In[33]:


file = helper.filee(file_name)
mask1, img_rgb1, image1, output1, grayScale1 = helper.loading(113,file,file_name,file_name2)
helper.plot(mask1, img_rgb1, image1, output1, grayScale1)


# In[5]:


sift = cv2.xfeatures2d.SIFT_create()


# In[55]:


keypoints_1, descriptors_1 = sift.detectAndCompute(img_rgb,mask)
 
img_1 = cv2.drawKeypoints(image,keypoints_1,output)
plt.imshow(img_1)


# In[15]:


keypoints_2, descriptors_2 = sift.detectAndCompute(img_rgb1,mask1)

img_2 = cv2.drawKeypoints(grayScale1,keypoints_1,output)
plt.imshow(img_2)


# In[16]:


#feature matching
"""bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img_rgb, keypoints_1, img_rgb1, keypoints_2, matches[:50], img_rgb1, flags=2)
plt.imshow(img3),plt.show()"""


# In[29]:


dico=[]
keysize=[]
for i,nam in enumerate(file):
    mask, img_rgb, image, output, grayScale = helper.loading(i,file,file_name,file_name2)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_rgb,mask)
    if descriptors_1 is not None:
        """keysize.append(np.size(keypoints_1))
        for d in descriptors_1:
            dico.append(d)"""
    else:
        print(nam)


# In[41]:


file = helper.filee(file_name3)
dico2=[]
keysize2=[]
for i,nam in enumerate(file):
    mask, img_rgb, image, output, grayScale = helper.loading(i,file,file_name3,file_name4)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_rgb,mask)
    if descriptors_1 is not None:
        keysize2.append(np.size(keypoints_1))
        for d in descriptors_1:
            dico2.append(d)


# In[49]:


print(len(keysize2[::2]))


# In[50]:


np.savez("newlesSIFT", np.array(dico))
np.savez("newnvSIFT", np.array(dico2))
np.savez("keysizeSIFT", np.array(keysize))
np.savez("keysize2SIFT", np.array(keysize2[::2]))


# In[17]:


#FOR LOADING
"""dico= np.load("lesSIFT.npz")
print(dico['arr_0'].shape)
dico2 = np.load("nvSIFT.npz")
print(dico2['arr_0'].shape)
fini=list(dico['arr_0'])+list(dico2['arr_0'])"""


# In[51]:


print(len(dico))
print(len(dico2))
fini=dico+dico2
print(len(fini))


# In[53]:


noClasses = 2
k = np.size(noClasses)*10
batch_size = np.size(len(file)*2) * 3
print(batch_size)
print(k)
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(fini)


# In[54]:


kmeans.verbose = False
histo_list = []
file_name = "Code/train/les/"
file_name2 = "Code/train/ls_mask_segmentation/"
file_name3 = "Code/train/nv/"
file_name4 = "Code/train/nv_mask_segmentation/"
file = helper.filee(file_name)


# In[65]:


count=0
histo_list=[]
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysize[i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dico[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_list.append(histo)


# In[72]:


count=0
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysize2[::2][i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dico2[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_list.append(histo)


# In[73]:


X = np.array(histo_list)


# In[22]:


X= np.load("histSIFT.npz")
print(X['arr_0'].shape)


# In[74]:


one = [1]*2400
zero = [0]*2400
Y = zero+one
print(len(Y))


# In[194]:


mlp = MLPClassifier(verbose=True, max_iter=600000,hidden_layer_sizes=150)
mlp.fit(X,Y)

rnd = RandomForestClassifier(n_estimators = 100)
rnd.fit(X,Y)
rnd.score(X,Y)


# In[104]:


rnd.score(X_Val,Y_Val)


# In[165]:


print(mlp.score(X,Y))
print(mlp.score(X_Val,Y_Val))


# In[81]:


dico3=[]
keysize3=[]
file = helper.filee(file_name5)
for i,nam in enumerate(file):
    mask, img_rgb, image, output, grayScale = helper.loading(i,file,file_name5,file_name6)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_rgb,mask)
    if descriptors_1 is not None:
        keysize3.append(np.size(keypoints_1))
        for d in descriptors_1:
            dico3.append(d)


# In[82]:


file = helper.filee(file_name7)
dico4=[]
keysize4=[]
for i,nam in enumerate(file):
    mask, img_rgb, image, output, grayScale = helper.loading(i,file,file_name7,file_name8)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_rgb,mask)
    if descriptors_1 is not None:
        keysize4.append(np.size(keypoints_1))
        for d in descriptors_1:
            dico4.append(d)


# In[84]:


print(len(dico3))
print(len(keysize3))
print(len(dico4))
print(len(keysize4))


# In[85]:


count=0
file = helper.filee(file_name5)
histo_listval=[]
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysize3[i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dico3[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_listval.append(histo)


# In[86]:


count=0
file = helper.filee(file_name7)
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysize4[i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dico4[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_listval.append(histo)


# In[88]:


X_Val=np.array(histo_listval)
one = [1]*600
zero = [0]*600
Y_Val = zero+one
print(len(Y_Val))


# In[176]:

sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X)
X_Test = sc_X.transform(X_Val)
#(X_Val,Y_Val)
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, Y)

# Predicting the test set results

Y_Pred = classifier.predict(X_Val)

# Making the Confusion Matrix 

cm = confusion_matrix(Y_Val, Y_Pred)
print(accuracy_score(Y_Val, Y_Pred))


# In[178]:


y_score=rnd.predict_proba(X_Val)[:, 1]
print("Classification report for classifier %s:\n%s\n"
    % (rnd, metrics.classification_report(Y_Val, y_score>.5)))


# In[200]:




clf = GridSearchCV(SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False)

clf.fit(X,Y)


# In[186]:


df=pd.DataFrame(clf.cv_results_)
df

model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [20,40,60,70,90,100],
            'kernel': ['rbf','linear','poly']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [50,100,150,200]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'mlp':{
        'model':MLPClassifier(verbose=True, max_iter=600000),
        'params':{
            'activation':['identity', 'logistic', 'tanh', 'relu'],
            'solver':['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes':[100,150,200,250,300]
        }
    },
    'xgboost':{
        'model':XGBClassifier(verbosity=0,num_class=2),
        'params':{'booster':['gbtree','dart'],
        'eta':[0.3,0.5],
        'max_depth':[5,6],
        'objective':['multi:softmax','binary:logistic'],
        'n_estimators': [50,100,150,200]
        }
    }
}


# In[211]:


scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X, Y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[222]:


rnd = RandomForestClassifier(n_estimators = 50)
rnd.fit(X,Y)
rnd.score(X_Val,Y_Val)


# In[224]:


df


# In[227]:


from xgboost import XGBClassifier


# In[232]:


xgb_model = XGBClassifier(objective="multi:softmax", random_state=42,num_class=2)
xgb_model.fit(X, Y)


# In[233]:


xgb_model.score(X,Y)


# In[234]:


xgb_model.score(X_Val,Y_Val)


# In[238]:


clf = GridSearchCV(XGBClassifier(verbosity=0,num_class=2),{
    'booster':['gbtree','dart'],
    'eta':[0.3,0.5],
    'max_depth':[5,6],
    'objective':['multi:softmax','binary:logistic'],
    'n_estimators': [50,100,150,200,250]
},cv=5,return_train_score=False)

clf.fit(X,Y)


# In[239]:


clf.score(X,Y)

clf.score(X_Val,Y_Val)