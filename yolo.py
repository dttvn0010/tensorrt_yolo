import cv2
import numpy as np
from ctypes import *

lib = CDLL('./lib/libyolo.so')
lib.detect.argtypes = (POINTER(c_float), c_int, c_int, c_int)
lib.detect.restype = c_void_p

lib.free_mem.argtypes = c_void_p,
lib.free_mem.restype = None


class Model:
    def detect(self, data , imgW, imgH, batchSize):
        ptr = lib.detect(data, imgW, imgH, batchSize)
        result = cast(ptr, c_char_p).value
        lib.free_mem(ptr)
        result = result.decode().split('\n')
        bboxesList = []
        bboxes = []
        for line in result:
            if line.strip() == '':
                bboxesList.append(bboxes)
                bboxes = []
            else:
                label, prob, x1, y1, x2, y2 = line.split(',')
                bboxes.append([float(x1), float(y1), float(x2), float(y2), float(prob)])
        return bboxesList[:batchSize] 

import preprocess
import time
if __name__ == '__main__':        
    model = Model()
    batchSize = 4
    img = cv2.imread('images/test.jpg')
    height, width, _ = img.shape
    img2 = preprocess.preprocess(img, 448, 448)
    imgList = [img2] * 4
    blob = preprocess.packImages(imgList, 448, 448)
    data = blob.ctypes.data_as(POINTER(c_float))
    bboxesList = model.detect(data, width, height, batchSize)
    for r in bboxesList[0]: print(r)

    t = time.time()
    N = 100
    for _ in range(N): model.detect(data, width, height, batchSize)
    t = time.time() - t
    print('Mean inference time (ms):', 1000*t/N/4)



