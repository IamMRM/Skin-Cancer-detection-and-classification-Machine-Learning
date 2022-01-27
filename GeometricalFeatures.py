import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import imutils
import pickle

path = 'test'
file = []
for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        file.append(filename)
        continue        
    else:
        continue

def A_values(mask):
    """Calculate assymetry with respect to X and Y axis"""
    A1 = 0
    A2 = 0
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # fit ellipse and get ellipse properties
    for cnt in contours:
        
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        ellipse = cv2.fitEllipse(cnt)
        (xc, yc), (d1, d2), angle = ellipse
        
        # Rotate the mask so the major axis overlaps with the X axis
        rot_angle = 90 - angle
        if rot_angle < 0:
            rot_angle += 180
        
        rotated = imutils.rotate_bound(mask, rot_angle)
        
        # Center the mask
        mask_c = center_image(rotated)
        # ALONG X-axis
        flipX = np.flipud(mask_c)
        # ALONG Y-axis
        flipY = np.fliplr(mask_c)
        
        xL = (abs(mask_c - flipX))
        A1 += round(np.sum(xL)/np.sum(mask_c),2)

        yL = (abs(mask_c - flipY))
        A2 += round(np.sum(yL)/np.sum(mask_c),2)
    
    return A1, A2


def B_values(image, mask):
    """ As a result return Area to parameter Ratio, compactness index,
    perimeter multiplied by area"""
    B1 = 0
    B2 = 0
    B3 = 0
    
    ## find contours and sort them
    cnts, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(cnts, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        B1 += area/peri
        B2 += 4*np.pi*area/(peri**2)
        B3 += peri*area
    
    return B1, B2, B3

def center_image(image):
    height, width = image.shape
    wi=(width/2)
    he=(height/2)

    ret,thresh = cv2.threshold(image,95,255,0)

    M = cv2.moments(thresh)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    offsetX = (wi-cX)
    offsetY = (he-cY)
    T = np.float32([[1, 0, offsetX], [0, 1, offsetY]]) 
    centered_image = cv2.warpAffine(image, T, (width, height))

    return centered_image


def eccentricity_from_ellipse(contour):
    """Calculates the eccentricity fitting an ellipse from a contour"""

    (_, _), (MA, ma), angle = cv2.fitEllipse(contour)

    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a ** 2 - b ** 2) / a
    return ecc 


def D_values(mask):
    label_img = label(mask, connectivity = mask.ndim)
    props = measure.regionprops(mask , label_img, ["area", "major_axis_length", "minor_axis_length"])
    D1_p = np.sqrt(4*props[0].area/np.pi)
    D1_pp = (props[0].major_axis_length + props[0].minor_axis_length)/2
    D1 = (D1_p + D1_pp)/2
    D2 = props[0].major_axis_length - props[0].minor_axis_length
    return D1, D2


def color(image, mask):
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    area1 = np.zeros(mask.shape)
    area1[mask != 0] = 1
    area = []
    full_area = np.sum(mask)
    mask_w = cv2.inRange(norm_image, (0.8, 0.8, 0.8), (1, 1,1))
    mask_r = cv2.inRange(norm_image, (0.588, 0.2, 0.2), (1, 1,1))
    mask_lb = cv2.inRange(norm_image, (0.588, 0.2, 0), (0.94, 0.588, 0.382))
    mask_db = cv2.inRange(norm_image,  (0.243, 0 , 0), (0.56, 0.392, 0.392))
    mask_bg = cv2.inRange(norm_image, (0, 0.392, 0.490), (0.588, 0.588, 0.588))
    mask_b = cv2.inRange(norm_image, (0, 0, 0), (0.243, 0.243, 0.243))
    plt.imshow(mask_db)
    color_values = [mask_w, mask_r, mask_lb, mask_db, mask_bg, mask_b]
    c = 0

    for i in color_values:
        area2 = np.zeros(i.shape)
        area2[i != 0] = 1
        if np.sum(area2) > 0.05*full_area:
            # Get number of pixels present in the image
            c += 1
    return c

geometrica_features = np.zeros((len(file), 8))

for i in tqdm(range(1)):
    file_name = 'test_zoom/'
    file_name2 = 'test_seg/'
    path = file_name +  file[i]
    path2 = file_name2 + file[i]
    image = cv2.imread(path,cv2.COLOR_BGR2RGB)
    mask = cv2.imread(path2,cv2.COLOR_BGR2RGB)
    mask = mask[:,:,1]
    imago = image.copy()
    imago[mask == 0] = 0
    
    
    img_RGB = imago
    grayScale = cv2.cvtColor(imago, cv2.COLOR_BGR2GRAY)
    area = []
    list_of_data = []
    B1, B2, B3 = B_values(grayScale, mask)
    A1, A2 = A_values(mask)
    D1, D2 = D_values(mask)
    C = color(imago, mask)
    list_of_data = [A1, A2, B1, B2, B3/10**8, D1, D2, C]
    for j in range(len(list_of_data)):
        geometrica_features[i,j] = list_of_data[j]


with open('test_1_geom.pkl','wb') as f:
    pickle.dump(info2, f)