cd data_gen_scripts
# Download the expert policies for locomotion environments (not required for other environments).
wget https://rail.eecs.berkeley.edu/datasets/ogbench/experts.tar.gz
tar xf experts.tar.gz && rm experts.tar.gz
# Create a directory to save datasets.
mkdir -p data