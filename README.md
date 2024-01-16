# CVTNet
The code for our paper accepted by **IEEE Transactions on Industrial Informatics**:   
### CVTNet: A Cross-View Transformer Network for LiDAR-Based Place Recognition in Autonomous Driving Environments.

[[IEEE Xplore TII 2023](https://ieeexplore.ieee.org/document/10273716)] [[arXiv](https://arxiv.org/abs/2302.01665)] [[Supplementary Materials](https://ieeexplore.ieee.org/document/10273716/media#media)]

[Junyi Ma](https://github.com/BIT-MJY), Guangming Xiong, [Jingyi Xu](https://github.com/BIT-XJY), [Xieyuanli Chen*](https://github.com/Chen-Xieyuanli)

<img src="https://github.com/BIT-MJY/CVTNet/blob/main/motivation.png" width="70%"/>

CVTNet fuses the range image views (RIVs) and bird's eye views (BEVs) generated from LiDAR data to recognize previously visited places. RIVs and BEVs have the same shift for each yaw-angle rotation, which can be used to extract aligned features.

<img src="https://github.com/BIT-MJY/CVTNet/blob/main/corresponding_rotation.gif" width="70%"/>

## Table of Contents
1. [Publication](#Publication)
2. [Dependencies](#Dependencies)
3. [How to Use](##How-to-Use)
4. [TODO](#Miscs)
5. [Related Work](#Related-Work)
6. [License](#License)

## Publication

If you use the code in your work, please cite our [paper](https://arxiv.org/abs/2302.01665):

```
@ARTICLE{10273716,
  author={Ma, Junyi and Xiong, Guangming and Xu, Jingyi and Chen, Xieyuanli},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={CVTNet: A Cross-View Transformer Network for LiDAR-Based Place Recognition in Autonomous Driving Environments}, 
  year={2023},
  doi={10.1109/TII.2023.3313635}}
```

## Dependencies

Please refer to our [SeqOT repo](https://github.com/BIT-MJY/SeqOT).

## How to Use

We provide a training and test tutorial for NCLT sequences in this repository. Before any operation, please modify the [config file](https://github.com/BIT-MJY/CVTNet/tree/main/config) according to your setups.

### Data Preparation

* laser scans from NCLT dataset: [[2012-01-08](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-01-08_vel.tar.gz)]       [[2012-02-05](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2012-02-05_vel.tar.gz)]
* [pretrained model](https://drive.google.com/file/d/1iQEY-DMDxchQ2RjG4RPkQcQehb_fujvO/view?usp=share_link)
* [training indexes](https://drive.google.com/file/d/1jEcnuHjEi0wqe8GAoh6UTa4UXTu0sDPr/view)
* [ground truth](https://drive.google.com/file/d/13-tpLQiHK4krd-womDV6UPevvHUIFNyF/view?usp=share_link)

You need to generate RIVs and BEVs from raw LiDAR data by

```
cd tools
python ./gen_ri_bev.py
```

### Training

You can start the training process with

```
cd train
python ./train_cvtnet.py
```
Note that we only train our model using the oldest sequence of NCLT dataset (2012-01-08), to prove that our model works well for long time spans even if seeing limited data.  

### Test

You can test the PR performance of CVTNet by

```
cd test
python ./test_cvtnet_prepare.py
python ./cal_topn_recall.py
```

You can also test the yaw-rotation invariance of CVTNet by

```
cd test
python ./test_yaw_rotation_invariance.py
```

<img src="https://github.com/BIT-MJY/CVTNet/blob/main/yaw_rotation_invariance.gif" width="70%"/>

It can be seen that the global descriptors generated by CVTNet are not affected by yaw-angle rotation.

### C++ Implementation

We provide a toy example showing C++ implementation of CVTNet with libtorch. First, you need to generate the model file by

```
cd CVTNet_libtorch
python ./gen_libtorch_model.py
```

* Before building, make sure that [PCL](https://github.com/PointCloudLibrary/pcl) exists in your environment.
* Here we use [LibTorch for CUDA 11.3 (Pre-cxx11 ABI)](https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.11.0%2Bcu113.zip). Please modify the path of **Torch_DIR** in [CMakeLists.txt](https://github.com/BIT-MJY/CVTNet/blob/main/CVTNet_libtorch/ws/CMakeLists.txt). 
* For more details of LibTorch installation, please check this [website](https://pytorch.org/get-started/locally/).  

Then you can generate a descriptor of the provided 1.pcd by

```
cd ws
mkdir build
cd build
cmake ..
make -j6
./fast_cvtnet
```

## TODO
- [x] Release the preprocessing code and pretrained model
- [ ] Release sequence-enhanced CVTNet (SeqCVT)

## Related Work

Thanks for your interest in our previous OT series for LiDAR-based place recognition.

* [OverlapNet](https://github.com/PRBonn/OverlapNet): Loop Closing for 3D LiDAR-based SLAM

```
@inproceedings{chen2020rss, 
  author = {X. Chen and T. L\"abe and A. Milioto and T. R\"ohling and O. Vysotska and A. Haag and J. Behley and C. Stachniss},
  title  = {{OverlapNet: Loop Closing for LiDAR-based SLAM}},
  booktitle = {Proceedings of Robotics: Science and Systems (RSS)},
  year = {2020}
}
```
* [OverlapTransformer](https://github.com/haomo-ai/OverlapTransformer): An Efficient and Yaw-Angle-Invariant Transformer Network for LiDAR-Based Place Recognition

```
@ARTICLE{ma2022ral,
  author={Ma, Junyi and Zhang, Jun and Xu, Jintao and Ai, Rui and Gu, Weihao and Chen, Xieyuanli},
  journal={IEEE Robotics and Automation Letters}, 
  title={OverlapTransformer: An Efficient and Yaw-Angle-Invariant Transformer Network for LiDAR-Based Place Recognition}, 
  year={2022},
  volume={7},
  number={3},
  pages={6958-6965},
  doi={10.1109/LRA.2022.3178797}}
```
* [SeqOT](https://github.com/BIT-MJY/SeqOT): A Spatial-Temporal Transformer Network for Place Recognition Using Sequential LiDAR Data

```
@ARTICLE{ma2022tie,
  author={Ma, Junyi and Chen, Xieyuanli and Xu, Jingyi and Xiong, Guangming},
  journal={IEEE Transactions on Industrial Electronics}, 
  title={SeqOT: A Spatial-Temporal Transformer Network for Place Recognition Using Sequential LiDAR Data}, 
  year={2022},
  doi={10.1109/TIE.2022.3229385}}
```

## License

Copyright 2023, Junyi Ma, Guangming Xiong, Jingyi Xu, Xieyuanli Chen, Beijing Institute of Technology.

This project is free software made available under the MIT License. For more details see the LICENSE file.
