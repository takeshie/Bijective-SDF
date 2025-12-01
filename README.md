<p align="center">
<h1 align="center">Learning Bijective Surface Parameterization for Inferring Signed Distance
Functions from Sparse Point Clouds with Grid Deformation</h1>
  <p align="center">
    <a href="https://github.com/takeshie/"><strong>Takeshi Noda*</strong></a>
    ·
    <a href="https://github.com/chenchao15/"><strong>Chao Chen*</strong></a>
    ·
	<a href="https://junshengzhou.github.io/"><strong> junshengzhou</strong></a>
    ·
    <a href="https://weiqi-zhang.github.io/"><strong>Weiqi Zhang</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>
</p>

  <p align="center"><strong>(* Equal Contribution)</strong></p>
  <h2 align="center">CVPR 2025</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2503.23670">Paper</a> | <a href="https://github.com/takeshie/Bijective-SDF">Project Page</a></h3>
  <div align="center"></div>
</p>



## Overview

<p align="center">
  <img src="fig/main_pic.png" width="780" />
</p>

Overview of Our method. Given a sparse point cloud \( Q \), we first learn a mapping function \( \Phi \) to encode \( Q \) into a unit sphere parametric domain. We consider each point as center point and sample local patches on the parametric surface. Next, we learn the inverse mapping \( \Psi \) to predict the positions of these local patches in 3D space and integrate them to obtain \( S \). We leverage \( S \) as the supervision for the grid deformation network \( g \) and predict the signed distance field through the GDO optimization strategy. We further extract dense point cloud $\bar{V}$ from the implicit field and optimize the parameterized surface $S$.
## Related works

Please also check out the following works that inspire us a lot:

* [Chao Chen et al. - Unsupervised Inference of Signed Distance Functions from Single Sparse Point Clouds without Learning Priors (TPAMI 2024)](https://github.com/chenchao15/NeuralTPS)
* [Baorui Ma et al. - Neural-Pull: Learning Signed Distance Functions from Point Clouds by Learning to Pull Space onto Surfaces (ICML2021)](https://github.com/mabaorui/NeuralPull/)
* [TianChang Shen et al. - Flexible Isosurface Extraction for Gradient-Based Mesh Optimization (SIGGRAPH 2023)](https://github.com/nv-tlabs/FlexiCubes)

## Installation

Our code is implemented using Python 3.8, PyTorch 2.1.0, and CUDA 11.8. We provide the complete installation instructions below.

- Install python Dependencies
  
  ```
  conda env create -f env.yaml
  ```
- Compile C++ extensions
  
  ```
  cd extensions/chamfer_dist
  python setup.py install
  ```
- Compile PointNet extensions
  
  ```
  cd third_party/pointnet2
  python setup.py install
  ```
## Data Setup
For convenient batch training, we read the point clouds listed in **/data_list/** from the **/data/** and save the results in **/outs/**.

## Train

You can train our method to reconstruct surfaces from a single point cloud as:
```
python run.py --gpu <gpu> --conf confs/<your config> --filelist <your filelist>
```

## Use Your Own Data
```
We support both raw point clouds and mesh-sampled point clouds as inputs. Similarly, you can place the point cloud or mesh files in the **/data** directory. 
To sample point clouds at different resolutions, update the value of preenc_npoints in the configuration file.

```

## Citation
If you find our code or paper useful, please consider citing
```
@inproceedings{noda2025learning,
  title={Learning bijective surface parameterization for inferring signed distance functions from sparse point clouds with grid deformation},
  author={Noda, Takeshi and Chen, Chao and Zhou, Junsheng and Zhang, Weiqi and Liu, Yu-Shen and Han, Zhizhong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={22139--22149},
  year={2025}}
```