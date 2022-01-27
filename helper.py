import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

def filee(file_name):
    file = []
    for filename in os.listdir(file_name):
        if filename.endswith('.jpg'):
            file.append(filename)
    return file

def paddedzoom(img, zoomfactor=0.8):#it will clip the image if zoomfactor<1
    out  = np.zeros_like(img)
    zoomed = cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor)
    h, w, _ = img.shape
    zh, zw, _ = zoomed.shape
    if zoomfactor<1:    # zero padded
        out[int((h-zh)/2):int(-(h-zh)/2), int((w-zw)/2):int(-(w-zw)/2)] = zoomed
    else:               # clip out
        out = zoomed[int((zh-h)/2):int(-(zh-h)/2), int((zw-w)/2):int(-(zw-w)/2),:]
    return out

def plot(mask, img_rgb, image, output, grayScale):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,5,1)
    ax1.imshow(mask,cmap='gray')
    ax2 = fig.add_subplot(1,5,2)
    ax2.imshow(img_rgb)
    ax3 = fig.add_subplot(1,5,3)
    ax3.imshow(image)
    ax4 = fig.add_subplot(1,5,4)
    ax4.imshow(output)
    ax4 = fig.add_subplot(1,5,5)
    ax4.imshow(grayScale,cmap='gray')

def loading(img_no, file, file_name, file_name2, filterSize=(21, 21), zoom=1.4):
    path = file_name + file[img_no]
    path2 = file_name2 + file[img_no]
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if int(zoom) !=0:
        image = paddedzoom(image, zoom)
    mask = cv2.imread(path2, cv2.IMREAD_COLOR)
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#image[mask[:, :, 1] == 0] = 0
    kernel = cv2.getStructuringElement(1, filterSize)
    img_norm = cv2.normalize(src=grayScale, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    blackhat = cv2.morphologyEx(img_norm, cv2.MORPH_BLACKHAT, kernel)
    b_blackhat = cv2.GaussianBlur(blackhat, (13, 13), cv2.BORDER_DEFAULT)
    __, Opp_mask = cv2.threshold(b_blackhat, 10, 255, cv2.THRESH_BINARY)
    output = cv2.inpaint(image, Opp_mask, 1, cv2.INPAINT_NS)
    corrected_img=output.copy()
    output[mask[:, :, 1] == 0] = 0
    grayScale = cv2.inpaint(grayScale, Opp_mask, 1, cv2.INPAINT_NS)

    return mask[:,:,1], img_rgb, corrected_img, output, grayScale