## Installation


### Requirements:
- PyTorch 1.3
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.2


### installation on Linux

```bash
# create a conda virtual environment
conda create --name STFT -y python=3.7
source activate STFT

# install the right pip and dependencies
conda install ipython pip
pip install ninja yacs cython matplotlib tqdm opencv-python scipy

# PyTorch installation with CUDA 10.0
conda install pytorch=1.3.0 torchvision cudatoolkit=10.0 -c pytorch

# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install STFT
git clone https://github.com/lingyunwu14/STFT.git
cd STFT
python setup.py build develop

pip install 'pillow<7.0.0'
pip install tensorboardX mmcv
```
