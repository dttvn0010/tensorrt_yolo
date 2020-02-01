/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/
#include "trt_utils.h"
#include "yolo.h"
#include "yolov2.h"
#include <string.h>
#include <stdlib.h>

#define BATCH_SIZE 4
#define CFG_PATH "data/person-yolov2.cfg"
#define WEIGHT_PATH "data/person-yolov2.weights"
#define LABEL_PATH "data/person-labels.txt"
#define CALIB_TABLE_PATH "data/person-yolov2-calibration.table"
#define ENGINE_PATH "data/person-yolov2-kINT8-kGPU-batch4.engine"
#define CALIB_IMAGES_PATH "data/calibration_images.txt"
#define INPUT_BLOB_NAME "data"

#define PROB_THRESH 0.4
#define NMS_THRESH 0.4

YoloV2 model(BATCH_SIZE, 
    (NetworkInfo) {"yolov2", CFG_PATH, WEIGHT_PATH,
    LABEL_PATH, "kINT8", "kGPU",
    CALIB_TABLE_PATH,
    ENGINE_PATH, INPUT_BLOB_NAME}, 
    (InferParams){0,0, CALIB_IMAGES_PATH, "" , PROB_THRESH, NMS_THRESH});


extern "C" const char* detect(float* ptr, int imgWidth, int imgHeight, int batchSize)
{
    model.doInference((uchar*)ptr, batchSize);
    std::stringstream sstr;
    
    for (int i = 0; i < batchSize; ++i)
    {
        auto binfo = model.decodeDetections(i, imgHeight, imgWidth);
        auto remaining = nmsAllClasses(model.getNMSThresh(), binfo, model.getNumClasses());
        
        for (auto b : remaining)
        {
            sstr << b.label << "," << b.prob << "," << b.box.x1 << "," << b.box.y1 << "," << b.box.x2 << "," << b.box.y2 << std::endl;                 
        }

        sstr << std::endl;
    }
    std::string st = sstr.str();
    char* result = (char*) malloc(st.length() + 1);
    strcpy(result, st.data());
    return result;
}

extern "C" void free_mem(void* p) 
{
    free(p);
}
