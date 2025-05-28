
## AGF code for IJCAI 25
## LRA Benchmark


We released the source code for LRA benchmark.

To prepare the datasets, one would need
```
tensorboard>=2.3.0, tensorflow>=2.3.1, tensorflow-datasets>=4.0.1
```

We upload ListOps datasets in `datasets/ folder. 

For other datasets, one would need to download the source code from [LRA repo](https://github.com/google-research/long-range-arena) and place `long-range-arena` folder in folder `LRA/datasets/` and also download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) released by LRA repo and place the unzipped folder in folder `datasets/`. The directory structure would be
```
datasets/long-range-arena
datasets/lra_release
```
Then, run `sh create_datasets.sh` and it will create train, dev, and test dataset pickle files for each task.

To run the LRA tasks, one would need
```
pytorch==1.7.1, transformers==3.3.1, performer-pytorch
```
To run a LRA experiment, run the following command in `code` folder
```
CUDA_VISIBLE_DEVICES=0 sh best_agf.sh
```
