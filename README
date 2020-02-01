Original source : https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/tree/restructure/yolo

1. Install cuda-10:
   https://developer.nvidia.com/cuda-10.0-download-archive

2. Install cudnn for cuda-10:
   https://developer.nvidia.com/rdp/cudnn-download

3. Add /usr/local/cuda/bin/ to PATH and /usr/local/cuda/lib64 to LD_LIBRARY_PATH in .profile (or .bash_login, .bash_rc, ...)

4 Install tensorrt-7:
   https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-700/tensorrt-install-guide/index.html

5.Build library:
   cd lib
   make

6. Install opencv-python:
   pip/pip3 install -r opencv-python

7. Download weight file from https://drive.google.com/open?id=1MXVAaa5xON9_XC8zHzks7KrxuG-3wpRF , and copy to folder "data"

8. Run yolo test: (in the first run, tensorrt will take some time to build the engine)
   python/python3 yolo.py
