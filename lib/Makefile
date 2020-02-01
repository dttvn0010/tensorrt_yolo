CPP_OBJS =  calibrator.o plugin_factory.o trt_utils.o yolo.o yolov2.o yolov3.o trt-yolo-app.o
CFLAGS = -fPIC  -I/usr/local/cuda/include -std=c++11 -Wall -O2
NVCC_FLAGS = -m64 --std c++11 -Xcompiler ,\"-Wall\",\"-O2\",\"-fPIC\" -DNVCC -I/usr/local/cuda/include 
CC = c++

hello:${CPP_OBJS} kernels
	${CC} -shared *.o  -o libyolo.so -L/usr/local/cuda/lib64 -lcudart -lcudnn -lcublas /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so -lstdc++fs -ldl

clean:
	rm -f *.o libyolo.so

.cpp.o:
	${CC} ${CFLAGS} -c $<
        
kernels:
	nvcc ${NVCC_FLAGS} -c kernels.cu 
