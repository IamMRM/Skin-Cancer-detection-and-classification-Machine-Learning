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


# In[2]:


bcc = "Code/train_c2/bcc_zoom/"
bcc_seg = "Code/train_c2/bcc_seg/"
bkl = "Code/train_c2/bkl_zoom/"
bkl_seg = "Code/train_c2/bkl_seg/"
mel = "Code/train_c2/mel_zoom/"
mel_seg = "Code/train_c2/mel_seg/"
#########################################
bccVal = "Code/val_c2/bcc_zoom_val/"
bcc_segVal = "Code/val_c2/bcc_seg_val/"
bklVal = "Code/val_c2/bkl_zoom_val/"
bkl_segVal = "Code/val_c2/bkl_seg_val/"
melVal = "Code/val_c2/mel_zoom_val/"
mel_segVal = "Code/val_c2/mel_seg_val/"


# In[4]:


file = helper.filee(bklVal)
mask, img_rgb, image, output, grayScale = helper.loading(1,file,bklVal,bkl_segVal)
helper.plot(mask, img_rgb, image, output, grayScale)


# In[5]:


file = helper.filee(bccVal)
mask, img_rgb, image, output, grayScale = helper.loading(9,file,bccVal,bcc_segVal)
helper.plot(mask, img_rgb, image, output, grayScale)


# In[6]:


sift = cv2.xfeatures2d.SIFT_create()


# In[7]:


keypoints_1, descriptors_1 = sift.detectAndCompute(image,None)
 
img_1 = cv2.drawKeypoints(grayScale,keypoints_1,image)
plt.imshow(img_1)


# In[8]:


dicobcc=[]
keysizebcc=[]
file = helper.filee(bcc)
for i,nam in enumerate(file):
    __, __, image, __, __ = helper.loading(i,file,bcc,bcc_seg,zoom=0)
    keypoints_1, descriptors_1 = sift.detectAndCompute(image,None)
    if descriptors_1 is not None:
        keysizebcc.append(np.size(keypoints_1))
        for d in descriptors_1:
            dicobcc.append(d)


# In[9]:


print(len(keysizebcc))
print(len(dicobcc))
#np.savez("SIFTdicobcc", np.array(dicobcc))
#np.savez("SIFTkeysizebcc", np.array(keysizebcc))


# In[10]:


dicobkl=[]
keysizebkl=[]
file = helper.filee(bkl)
for i,nam in enumerate(file):
    __, __, image, __, __  = helper.loading(i,file,bkl,bkl_seg,zoom=0)
    keypoints_1, descriptors_1 = sift.detectAndCompute(image,None)
    if descriptors_1 is not None:
        keysizebkl.append(np.size(keypoints_1))
        for d in descriptors_1:
            dicobkl.append(d)


# In[11]:


print(len(keysizebkl))
print(len(dicobkl))
#np.savez("SIFTdicobkl", np.array(dicobkl))
#np.savez("SIFTkeysizebkl", np.array(keysizebkl))


# In[12]:


dicomel=[]
keysizemel=[]
file = helper.filee(mel)
for i,nam in enumerate(file):
    __, __, image, __, __ = helper.loading(i,file,mel,mel_seg,zoom=0)
    keypoints_1, descriptors_1 = sift.detectAndCompute(image,None)
    if descriptors_1 is not None:
        keysizemel.append(np.size(keypoints_1))
        for d in descriptors_1:
            dicomel.append(d)
print(len(keysizemel))
print(len(dicomel))
#np.savez("SIFTdicomel", np.array(dicomel))
#np.savez("SIFTkeysizemel", np.array(keysizemel))


# In[21]:


fini=dicobcc+dicobkl+dicomel
print(len(fini))


# In[22]:


#FOR LOADING
"""dico= np.load("lesSIFT.npz")
print(dico['arr_0'].shape)
dico2 = np.load("nvSIFT.npz")
print(dico2['arr_0'].shape)
fini=list(dico['arr_0'])+list(dico2['arr_0'])"""


# In[23]:


noClasses = 3
k = (noClasses)*10
batch_size = int(len(file)/3)
print(batch_size)
print(k)
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(fini)


# In[24]:


kmeans.verbose = False


# In[25]:


count=0
histo_list=[]
file = helper.filee(bcc)
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysizebcc[i]
        for d in range(nkp):
            idx = kmeans.predict([dicobcc[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_list.append(histo)


# In[26]:


count=0
file = helper.filee(bkl)
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysizebkl[i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dicobkl[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_list.append(histo)


# In[27]:


count=0
file = helper.filee(mel)
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysizemel[i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dicomel[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_list.append(histo)


# In[28]:


X = np.array(histo_list)
print(X.shape)
#np.savez("challenge2trainhistSIFT", X)


# In[30]:


"""X= np.load("histSIFT.npz")
print(X['arr_0'].shape)"""


# In[29]:


zero = [0]*400
one = [1]*800
two = [2]*800
Y = zero+one+two
print(len(Y))


# In[30]:


mlp = MLPClassifier(verbose=True, max_iter=600000,hidden_layer_sizes=150)
mlp.fit(X,Y)


# In[31]:


from sklearn.ensemble import RandomForestClassifier

rnd = RandomForestClassifier(n_estimators = 150)
rnd.fit(X,Y)
rnd.score(X,Y)


# In[41]:


rnd.score(X_Val,Y_Val)


# In[40]:


print(mlp.score(X,Y))
print(mlp.score(X_Val,Y_Val))


# In[33]:


dicobccVal=[]
keysizebccVal=[]
file = helper.filee(bccVal)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, __ = helper.loading(i,file,bccVal,bcc_segVal,zoom=0)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_rgb,mask)
    if descriptors_1 is not None:
        keysizebccVal.append(np.size(keypoints_1))
        for d in descriptors_1:
            dicobccVal.append(d)
print(len(dicobccVal))
print(len(keysizebccVal))
#np.savez("SIFTdicomel", np.array(dicomel))
#np.savez("SIFTkeysizemel", np.array(keysizemel))


# In[34]:


dicobklVal=[]
keysizebklVal=[]
file = helper.filee(bklVal)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, __ = helper.loading(i,file,bklVal,bkl_segVal,zoom=0)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_rgb,mask)
    if descriptors_1 is not None:
        keysizebklVal.append(np.size(keypoints_1))
        for d in descriptors_1:
            dicobklVal.append(d)
print(len(dicobklVal))
print(len(keysizebklVal))
#np.savez("SIFTdicomel", np.array(dicomel))
#np.savez("SIFTkeysizemel", np.array(keysizemel))


# In[35]:


dicomelVal=[]
keysizemelVal=[]
file = helper.filee(melVal)
for i,nam in enumerate(file):
    mask, img_rgb, __, __, __ = helper.loading(i,file,melVal,mel_segVal,zoom=0)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_rgb,mask)
    if descriptors_1 is not None:
        keysizemelVal.append(np.size(keypoints_1))
        for d in descriptors_1:
            dicomelVal.append(d)
print(len(dicomelVal))
print(len(keysizemelVal))
#np.savez("SIFTdicomel", np.array(dicomel))
#np.savez("SIFTkeysizemel", np.array(keysizemel))


# In[43]:


np.savez("dicobccVal", np.array(dicobccVal))
np.savez("keysizebccVal", np.array(keysizebccVal))
np.savez("dicobklVal", np.array(dicobklVal))
np.savez("keysizebklVal", np.array(keysizebklVal))
np.savez("dicomelVal", np.array(dicomelVal))
np.savez("keysizemelVal", np.array(keysizemelVal))


# In[38]:


#################NOW CALCULATING THE HISTOGRAMS#######################


# In[36]:


count=0
file = helper.filee(bccVal)
histo_listval=[]
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysizebccVal[i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dicobccVal[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_listval.append(histo)


# In[37]:


count=0
file = helper.filee(bklVal)
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysizebklVal[i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dicobklVal[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_listval.append(histo)


# In[38]:


count=0
file = helper.filee(melVal)
for i,__ in enumerate(file):
    histo = np.zeros(k)
    try:
        nkp = keysizemelVal[i]#np.size(keypoints_1)
        for d in range(nkp):
            idx = kmeans.predict([dicomelVal[count+d]])
            histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
        count+=nkp
    except:
        histo[0]+=1
        print("EXCEPT"+str(i))
    histo_listval.append(histo)


# In[39]:


X_Val=np.array(histo_listval)
print(X_Val.shape)
zero = [0]*100
one = [1]*200
two = [2]*200
Y_Val = zero+one+two
print(len(Y_Val))


# In[44]:


np.savez("challenge2valhistSIFT", X_Val)


# In[176]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X)
X_Test = sc_X.transform(X_Val)
#(X_Val,Y_Val)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X, Y)

# Predicting the test set results

Y_Pred = classifier.predict(X_Val)

# Making the Confusion Matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Val, Y_Pred)


# In[177]:


from sklearn.metrics import accuracy_score
from sklearn import svm, datasets, metrics
print(accuracy_score(Y_Val, Y_Pred))


# In[178]:


y_score=rnd.predict_proba(X_Val)[:, 1]
print("Classification report for classifier %s:\n%s\n"
    % (rnd, metrics.classification_report(Y_Val, y_score>.5)))


# In[200]:


from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False)

clf.fit(X,Y)


# In[186]:


df=pd.DataFrame(clf.cv_results_)
df


# In[187]:


df[['param_C','param_kernel','mean_test_score']]


# In[188]:


clf.best_score_


# In[190]:


clf.best_params_


# In[210]:


from sklearn.linear_model import LogisticRegression
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


# In[241]:


clf.best_score_


# In[240]:


clf.score(X_Val,Y_Val)


# In[242]:


clf.best_params_


# # EXTRA

# In[17]:
"""

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(image, keypoints_1, image1, keypoints_2, matches[:50], image1, flags=2)
plt.imshow(img3),plt.show()

"""