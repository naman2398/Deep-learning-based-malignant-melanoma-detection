import cv2
import pandas as pd
import numpy as np

IMG_DIM = 224

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
      
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
  #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
  #         print(img.shape)
        return img

def clahe_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # lab = img
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

def load_ben_color(image, sigmaX):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    # clahe = cv2.createCLAHE(clipLimit=0.0)
    # image[:,:,1] = clahe.apply(image[:,:,1])
    image = clahe_lab(image)
    image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    image=cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image

def bgrgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess(image, s):
    img = load_ben_color(image, sigmaX=s)
    return img


data = pd.read_csv("../trainLabels.csv")
train_img = np.zeros([data.shape[0], IMG_DIM, IMG_DIM, 3], dtype=np.uint8)

for i in range(data.shape[0]):
    img_name = data.iloc[i].image
    img_path = "../train/"+img_name+".jpeg"
    img = cv2.imread(img_path)
    if img is not None:
        cv2.imshow("xyz", preprocess(img, 2))
    train_img[i, :, :, :] = preprocess(img)