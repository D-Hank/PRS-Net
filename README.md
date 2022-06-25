# PRS-Net

## Introduction
A pytorch implementation of PRS-Net, a research work on TVCG by Lin Gao et al.
Official link: http://geometrylearning.com/prs-net/

Author: D-Hank
Feel free to contact me at [daihankun19@mails.ucas.ac.cn](daihankun19@mails.ucas.ac.cn). To use my code, please cite the link of this repository.

## Dependencies

PCL 1.12.1: generate point cloud (poisson disk) {https://pointclouds.org/}

cuda_voxelizer: generate voxel (mesh split) {https://github.com/Forceflow/cuda_voxelizer}

Open3D: read pcd file in Python {http://www.open3d.org/}

Libigl: compute closest points on the mesh (barycentric coordinates) {https://github.com/libigl/libigl-python-bindings}

point_cloud_utils: read and write obj files {https://github.com/fwilliams/point-cloud-utils}

Mayavi / matplotlib: visualization {http://docs.enthought.com/mayavi/mayavi/}

**Note: ** The library cuda voxelizer should be localized for our model. We include the revised version in the directory `extern/`

## Directory

The project directory should be organized like this:
```

├── augment      # dataset after augmentation

├── data         # dataset after preprocess

├── shapenet     # original shapenet data

	├── 02691156

	└── ……

└── prsnet-repr  # main working directory

	├── checkpoint     # saved models

	└── ……
```

## Running Tips

Change your working directory to `prsnet-repr`. About 3 days and 80 GB free space are required. You can set the default options in `settings.py`.

To run this project from the start, first run `python augment.py` to generate augmented data. It takes one day to run on CPU.

Then run `python preprocess.py` to generate voxel, point cloud and closest points. We use 4 processes to run simultaneously, which takes around 2 days with CUDA acceleration.

Finally use `python main.py` to run the main program. It takes 0.5 hour to train.

If you'd like to use the pre-trained model in `checkpoint/`, then set `CONTINUE_` in `settings.py` to be True and run `main.py` directly.

## Results

For different categories in test set, we've achieved great results.
Reflective plane (with coordinate axes on the left-buttom corner):

<img src="teaser/a02691156_829108f586f9d0ac7f5c403400264eea_0.gif" width=20% /><img src="teaser/a02691156_17874281e56ff0fbfca1faa43bb6bc17_0.gif" width=20% /><img src="teaser/a02691156_fb06b00775efdc8e21b85e5214b0d6a7_0.gif" width=20% /><img src="teaser/a02747177_8b071aca0c2cc07c81faebbdea6bd9be_0.gif" width=20% /><img src="teaser/a02828884_133d46d90aa8e1742b76566a81e7d67e_0.gif" width=20% />
<img src="teaser/a02828884_cd052cd64a9f956428baa2ac864e8e40_0.gif" width=20% /><img src="teaser/a02876657_d3ed110edb3b8a4172639f272c5e060d_0.gif" width=20% /><img src="teaser/a02880940_a0ac0c76dbb4b7685430c0f7a585e679_0.gif" width=20% /><img src="teaser/a02958343_4aa7fc4a0c08be8c962283973ea6bbeb_0.gif" width=20% /><img src="teaser/a03046257_5437b68ddffc8f229e5629b793f22d35_0.gif" width=20% />
<img src="teaser/a03624134_a683ed081504a35e4a9a3a0b87d50a92_0.gif" width=20% /><img src="teaser/a03691459_85bbc49aa67149c531baa3c8ee4148cd_0.gif" width=20% /><img src="teaser/a03691459_403649d8cf6b019d5c01f9a624be205a_0.gif" width=20% /><img src="teaser/a04090263_9397161352dec4498bfbe54b5d01550_0.gif" width=20% /><img src="teaser/a04225987_ac2b6924a60a7a87aa4f69d519551495_0.gif" width=20% />
<img src="teaser/a04256520_3bde46b6d6fb84976193d9e76bb15876_0.gif" width=20% /><img src="teaser/a04256520_29bfdc7d14677c6b3d6d3c2fb78160fd_0.gif" width=20% /><img src="teaser/a04256520_79745b6df9447d3419abd93be2967664_0.gif" width=20% /><img src="teaser/a04256520_bdd7a0eb66e8884dad04591c8486ec0_0.gif" width=20% /><img src="teaser/a04256520_c983108db7fcfa3619fb4103277a6b93_0.gif" width=20% />
<img src="teaser/a04379243_290df469e3338a67c3bd24f986301745_0.gif" width=20% /><img src="teaser/a04401088_927b3450c8976f3393078ad6013586e7_0.gif" width=20% /><img src="teaser/a04468005_e5d292b873af06b24c7ef8f59a6ea23a_0.gif" width=20% /><img src="teaser/a04530566_ac5dad64a080899bba2dc6b0ec935a93_0.gif" width=20% /><img src="teaser/a04530566_d271233ccca1e7ee23a3427fc25942e0_0.gif" width=20% />

For generalized objects, the rotation axis:

<img src="teaser/a02828884_cd052cd64a9f956428baa2ac864e8e40_0_r.gif" width=20% /><img src="teaser/a02880940_a0ac0c76dbb4b7685430c0f7a585e679_0_r.gif" width=20% /><img src="teaser/a02933112_73c2405d760e35adf51f77a6d7299806_0_r.gif" width=20% /><img src="teaser/a03691459_23efeac8bd7132ffb96d0ef27244d1aa_0_r.gif" width=20% /><img src="teaser/a04379243_6af7f1e6035abb9570c2e04669f9304e_0_r.gif" width=20% />

## Limitations

- Position of the rotation axes

  Motivated by [YuzhuoChen99](https://github.com/YizhuoChen99/PRS-Net)'s implementation, the model can only predict rotation axes near the original point. Even for the already-normalized shapenet dataset, the rotational center is not always near the origin. Therefore the model performs not so well (sometimes disturbed).

  The solution is to introduce a shift vector or use the generalized 4×4 rotation matrix.

- Problems with axis-angle representation

  Basically, the network can learn to use tricks for better performance. That is, it can randomly pick three orthogonal axes and set the rotational angle to be 0 or 2π. Then both the distance and regularized loss will be relatively low. So sometimes the training rotation loss looks like:

  <img src="teaser/rotloss.jpg" width=400px />

  After training for a long time, the network can get lazy for rotation. But there seems no good way to solve this.

## Acknowledgement

Quaternion: tiny reimplementation of pytorch3D for Quaternion

Pairwise-Cosine: https://github.com/pytorch/pytorch/issues/11202

Reference: He Yue's implementation {https://github.com/hysssb/PRS_net}

Reference: official release {https://github.com/IGLICT/PRS-Net}
