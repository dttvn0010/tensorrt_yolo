import cv2
import numpy as np

def preprocess(img, width, height):
        imgH, imgW, _ = img.shape
        dim = max(imgW, imgH)
        resizeH = (imgH * height) // dim
        resizeW = (imgW * width) // dim
        if (height - resizeH) % 2 == 1: resizeH -= 1
        if (width - resizeW) % 2 == 1: resizeW -= 1
        xOffset = (width - resizeW) // 2
        yOffset = (height - resizeH) // 2
        new_img = 128 * np.ones((height, width, 3), dtype='uint8')
        new_img[yOffset:height-yOffset,xOffset:width-xOffset] = cv2.resize(img, (resizeW, resizeH), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

def packImages(imgList, width, height):
    return cv2.dnn.blobFromImages(imgList, 1.0, (width, height), (0.0, 0.0, 0.0), False)
