## Setting up an environment

```sh 
conda create -n nvist python=3.9
conda activate nvist
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg lpips tensorboard torch_efficient_distloss
pip install easydict timm plyfile matplotlib kornia accelerate
```

This code is tested for Pytorch 2.1.2 with CUDA 11.8.
We use [Nerfies](https://github.com/google/nerfies) to retrieve COLMAP information.

```sh
pip install tensorflow pandas
pip install git+https://github.com/google/nerfies.git@v2
pip install "git+https://github.com/google/nerfies.git#egg=pycolmap&subdirectory=third_party/pycolmap"
```


## Dataset

Download MVImgNet dataset from this [official repository](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet).
Highly recommend to use [Tip](https://docs.google.com/document/d/1krVb4B3rZw-0FaBBPS7c3SJKfqq5AVYTs2HN2LnlBPQ/edit#heading=h.2ukfzxh5c9pq) provided by the authors.

For the paper, we use the subset of MVImgNet - 1.14M frames, 38K scenes of 177 categories for training, and for testing, a total of 13,228 frames from 447 scenes and 177 categories are used. 


### Pre-process the dataset

In this paper, we downsample the images by 12, and we use portrait images having 160x90 for training. 
We made a cache for the dataset. 

* Unzip files. In my case, unzip files in '../../data/mvimgnet' (data_dir)
* Downsample images by 12 

```sh
python -m preprocess.downsample_imgs --data_dir ../../data/mvimgnet 
```

* Retrieve camera poses : We retrieve camera poses, and boundary of point clouds from COLMAP results. We follow OpenCV convention.

```sh
python -m preprocess.read_colmap_results_mvimgnet
```

* Make Cache files 

```sh
python -m preprocess.make_cache --data_dir ../../data/mvimgnet
python -m preprocess.make_cache --data_dir ../../data/mvimgnet --split test
```

Please get in touch with [me](mailto:won.jang1108@gmail.com) if you need the processed dataset.
