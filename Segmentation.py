import numpy as np
import cv2 
import os
from scipy import ndimage
from skimage.segmentation import morphological_chan_vese
from tqdm import tqdm


path = 'test'
file = []
for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        file.append(filename)
        continue        
    else:
        continue


def paddedzoom(img, zoomfactor=0.8):
    out  = np.zeros_like(img)
    zoomed = cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor)
    
    h, w, _ = img.shape
    zh, zw, _ = zoomed.shape
    
    if zoomfactor<1:    # zero padded
        out[(h-zh)/2:-(h-zh)/2, (w-zw)/2:-(w-zw)/2, :] = zoomed
    else:               # clip out
        out = zoomed[int((zh-h)/2):int(-(zh-h)/2), int((zw-w)/2):int(-(zw-w)/2), :]

    return out
    


def create_vignette_mask(h, w, radius):
    center = [int(w/2), int(h/2)]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask



def FolderCreater(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)



path = 'test_segmentation/'
path2 = 'test_zoomed/'

FolderCreater(path)
FolderCreater(path2)



file_name = open('test_seg.dat', 'w+')
file_name2 = open('test_zoom.dat', 'w+')

for i in tqdm(range(len(file))):
    path = 'test/' + file[i]
    image=cv2.imread(path,cv2.IMREAD_COLOR)
    n ,k, _ = np.shape(image) 

    vignette_effect = create_vignette_mask(n, k, 280)

    n = n - 1 
    k = k - 1
    if (image[0,0, :] < 30).any() and (image[0,k, :] < 30).any() and (image[n, 0, :] < 30).any() and (image[n,k, :] < 30).any():
        image = paddedzoom(image, 1.4)
    elif (image[0,0, :] < 30).any()  and (image[0,k, :] < 30).any()  and (image[n, 0, :] < 30).any() :
        image = paddedzoom(image, 1.4)
    elif (image[0,0, :] < 30).any()  and (image[0,k, :] < 30).any() and (image[n,k, :] < 30).any() :
        image = paddedzoom(image, 1.4)
    elif (image[0,k, :] < 30).any()  and (image[n, 0, :] < 30).any() and (image[n,k, :] < 30).any()  :
        image = paddedzoom(image, 1.4)
    else:
        image = paddedzoom(image, 1.2)

    
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    filterSize = (21, 21)
    kernel = cv2.getStructuringElement(1,filterSize) 
    img_norm = cv2.normalize(src=grayScale, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    # Get rid of Hair
    blackhat = cv2.morphologyEx(img_norm, cv2.MORPH_BLACKHAT, kernel)
    ret,mask = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    output = cv2.inpaint(grayScale, mask, 1, cv2.INPAINT_NS)
    output_color = cv2.inpaint(image, mask, 1, cv2.INPAINT_NS)

    # Otsu Thresholding
    (T, threshInv) = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Chan-Vase active contours

    mask = morphological_chan_vese(output, 5, init_level_set=threshInv, smoothing=10).astype(np.uint8)

    if mask[0,0] == 1 and mask[0,k] == 1 and mask[n, 0] == 1 and mask[n, k] == 1:
        mask = mask.copy() * vignette_effect
    elif mask[0,0] == 1 and mask[0,k] == 1 and mask[n,0] == 1:
        mask = mask.copy() * vignette_effect
    elif mask[0,0] == 1 and mask[0,k] == 1 and mask[n,k] == 1:
        mask = mask.copy() * vignette_effect
    elif mask[0, k] == 1 and mask[n, 0] == 1 and mask[n, k]== 1:
        mask = mask.copy() * vignette_effect

    mask = ndimage.binary_fill_holes(mask, structure=np.ones((5,5)))

    result_path = file_name + file[i]
    result_path2 = file_name2 + file[i]
    plt.imsave(result_path, mask)
    plt.imsave(result_path2, output_color)

