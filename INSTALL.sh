# Download VG images
cd datasets
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# unziping and merging
unzip images.zip
unzip images2.zip
mv VG_100K_2/* VG_100K/
rm -r VG_100K_2
rm images.zip
rm images2.zip

# Download VG annotations
# VG150 connected
mkdir -p VG150
cd VG150
wget -O VG150_connected.zip https://mycore.core-cloud.net/index.php/s/9Afn2UO8wVt4wov/download
unzip VG150_connected.zip
rm VG150_connected.zip

cd ../..

curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh -b -p $HOME/miniconda3
source ~/.bashrc

conda update --force conda
conda create --name scene_graph_benchmark python=3.8
conda activate scene_graph_benchmark
conda install ipython scipy h5py ninja yacs cython matplotlib tqdm pandas
conda remove yacs

pip install -r requirements.txt
python setup.py build develop
