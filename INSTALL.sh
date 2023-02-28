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

conda update --force conda
conda create --name scene_graph_benchmark python=3.9
conda activate scene_graph_benchmark
conda install ipython scipy h5py ninja yacs cython matplotlib tqdm pandas
conda remove yacs

cd ..
pip install -r requirements.txt
python setup.py build develop
