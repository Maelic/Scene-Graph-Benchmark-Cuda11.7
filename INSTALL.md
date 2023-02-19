## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- Python <= 3.8
- PyTorch >= 1.2 (Mine 1.13.1 (CUDA 11.7))
- torchvision >= 0.4
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly, make sure you have the latest conda
conda update --force conda

# create and activate env
conda create --name scene_graph_benchmark python=3.8
conda activate scene_graph_benchmark

# this installs the right conda dependencies for the fresh python
conda install ipython scipy h5py ninja yacs cython matplotlib tqdm pandas

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 11.7
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/Maelic/Scene-Graph-Benchmark.pytorch.git
cd scene-graph-benchmark

# some pip dependencies
pip install -r requirements.txt

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR