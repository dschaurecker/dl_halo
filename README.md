# Superresolving Dark Matter Halos using Generative Deep Learning
This repo contains code to train a generating U-Net using a GAN method to predict halos from a low-res dm-only simulations.

The following will shortly explain the pre- and post-processing of data, while also outlining the basic training method:

* [Generating training data](#Generating-training-data)
* [Training and testing](#Training-and-testing)
* [Evaluation of statistics](#Evaluation-of-statistics)
    

## Generating training data

The ```/gen_train_data/``` folder contains the code and slurm script to divide the low- and high-res simulations into the desired training data cubes, 64x64x64 pixels + padding in this paper's case.
The particle position catalogs are painted onto a mesh of the desired size using the specified window function, eg. tsc (triangular shaped cloud).
Following that, the mesh is divided into the individual training cubes while padding them with their neighboring regions, before saving each cubeas a numpy array. The position of each cube inside the entire simulation can be inferred from its file name.
In order to reduce the amount of files stored per folder, the Jupyter notebook ```putDataInDirs.ipynb``` creates ```/0-9/``` subfolders and stores each training cube inside one of them.

More informaition on the arguments passed in the slurm script upon launching the process can be read at the beginning of the Python code.

## Training and testing

The ```/train_test/``` folder contains an adapted version of Yin Li's PyTorch [map2map](https://github.com/eelregit/map2map) repository, plus the slurm scripts for training (GPU) and testing (CPU). All used models, more details on the exact trainig/testing process and much more can be found here. 

Inside the ```args.py``` code all details and explanations on what arguments are possible to be passed to map2map can be found. Generally D and G will train for a given number of epochs while storing the generator's state and plotting some statistics for later evaluation after each epoch. The training is built to be possible on multiple node and multiple GPUs to reduce training times significantly. For AMD GPUs we had to use a beta version of PyTorch, which will most likely be outdated by now. The testing process happens only on CPUs as there are usually more of them available on any given cluster and each testing cube has to be passed through the generator only once.

## Evaluation of statistics

The ```/statistics/``` folder contains all the tools and scripts used to evaluate the trained net's performance statistically. 

```calc_cat.py``` produces a catalog from a given number of test cubes of the test region. This is done by first Poisson sampling all float values to integers and then randomly placing the given number of particles inside each voxel's volume.

These catalogs are then used to calculate the power spectra, the pdf and find FoF halos inside ```eval_cat.py```. The calculation of the two-point correlation function is computationally very expensive when using the LS estimator, thus it is done seperately on mutiple ranks inside the ```calc_2pcf.py``` script.

A lot of the calculations build on [nbodykit](https://github.com/bccp/nbodykit), a massively parallel large-scale structure toolkit, which is easy to use and reasonably fast.
