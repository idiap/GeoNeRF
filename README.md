> # [CVPR 2022] GeoNeRF: Generalizing NeRF with Geometry Priors <br>
> Mohammad Mahdi Johari, Yann Lepoittevin, François Fleuret <br>
> [Project Page](https://www.idiap.ch/paper/geonerf/) | [Paper](https://arxiv.org/abs/2111.13539)

This repository contains a PyTorch Lightning implementation of our paper, GeoNeRF: Generalizing NeRF with Geometry Priors.

## Installation

#### Tested on NVIDIA Tesla V100 and GeForce RTX 3090 GPUs with PyTorch 1.9 and PyTorch Lightning 1.3.7

To install the dependencies, in addition to PyTorch, run:

```
pip install -r requirements.txt
```

## Evaluation and Training
To reproduce our results, download pretrained weights from [here](https://drive.google.com/drive/folders/1ZtAc7VYvltcdodT_BrUrQ_4IAhz_L-Rf?usp=sharing) and put them in [pretrained_weights](./pretrained_weights) folder. Then, follow the instructions for each of the [LLFF (Real Forward-Facing)](#llff-real-forward-facing-dataset), [NeRF (Realistic Synthetic)](#nerf-realistic-synthetic-dataset), and [DTU](#dtu-dataset) datasets.

## LLFF (Real Forward-Facing) Dataset
Download `nerf_llff_data.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and set its path as `llff_path` in the [config_llff.txt](./configs/config_llff.txt) file.

For evaluating our generalizable model (`pretrained.ckpt` model in the [pretrained_weights](./pretrained_weights) folder), set the `scene` properly (e.g. fern) and set the number of source views to 9 (nb_views = 9) in the [config_llff.txt](./configs/config_llff.txt) file and run the following command:

```
python run_geo_nerf.py --config configs/config_llff.txt --eval
```

For fine-tuning on a specific scene, set nb_views = 7 and run the following command:

```
python run_geo_nerf.py --config configs/config_llff.txt
```

Once fine-tuning is finished, run the evaluation command with nb_views = 9 to get the final rendered results.

## NeRF (Realistic Synthetic) Dataset
Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and set its path as `nerf_path` in the [config_nerf.txt](configs/config_nerf.txt) file.

For evaluating our generalizable model (`pretrained.ckpt` model in the [pretrained_weights](./pretrained_weights) folder), set the `scene` properly (e.g. lego) and set the number of source views to 9 (nb_views = 9) in the [config_nerf.txt](configs/config_nerf.txt) file and run the following command:

```
python run_geo_nerf.py --config configs/config_nerf.txt --eval
```

For fine-tuning on a specific scene, set nb_views = 7 and run the following command:

```
python run_geo_nerf.py --config configs/config_nerf.txt
```

Once fine-tuning is finished, run the evaluation command with nb_views = 9 to get the final rendered results.

## DTU Dataset
 Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) 
and replace its `Depths` directory with [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repository](https://github.com/YoYo000/MVSNet), and set `dtu_pre_path` referring to this dataset in the [config_dtu.txt](configs/config_dtu.txt) file.

Then, download the original `Rectified` images from [DTU Website](https://roboimagedata.compute.dtu.dk/?page_id=36), and set `dtu_path` in the [config_dtu.txt](configs/config_dtu.txt) file accordingly.

For evaluating our generalizable model (`pretrained.ckpt` model in the [pretrained_weights](./pretrained_weights) folder), set the `scene` properly (e.g. scan21) and set the number of source views to 9 (nb_views = 9) in the [config_dtu.txt](./configs/config_dtu.txt) file and run the following command:

```
python run_geo_nerf.py --config configs/config_dtu.txt --eval
```

For fine-tuning on a specific scene, use the same nb_views = 9 and run the following command:

```
python run_geo_nerf.py --config configs/config_dtu.txt
```

Once fine-tuning is finished, run the evaluation command with nb_views = 9 to get the final rendered results.

### RGBD Compatible model
By adding `--use_depth` argument to the aforementioned commands, you can use our RGB compatible model on the DTU dataset and exploit the ground truth, low-resolution depths to help the rendering process. The pretrained weights for this model is `pretrained_w_depth.ckpt`.

## Training From Scratch
For training our model from scratch, first, prepare the following datasets:

* The original `Rectified` images from [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36). Set the corresponding path as  `dtu_path` in the [config_general.txt](configs/config_general.txt) file.

* The preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) 
with the replacement of its `Depths` directory with [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). Set the corresponding path as  `dtu_pre_path` in the [config_general.txt](configs/config_general.txt) file.

 * LLFF released scenes. Download [real_iconic_noface.zip](https://drive.google.com/drive/folders/1M-_Fdn4ajDa0CS8-iqejv0fQQeuonpKF) and remove the test scenes with the following command:
    ```
    unzip real_iconic_noface.zip
    cd real_iconic_noface/
    rm -rf data2_fernvlsb data2_hugetrike data2_trexsanta data3_orchid data5_leafscene data5_lotr data5_redflower
    ```
    Then, set the corresponding path as  `llff_path` in the [config_general.txt](configs/config_general.txt) file.

* Collected scenes from [IBRNet](https://github.com/googleinterns/IBRNet) ([Subset1](https://drive.google.com/file/d/1rkzl3ecL3H0Xxf5WTyc2Swv30RIyr1R_/view?usp=sharing) and [Subset2](https://drive.google.com/file/d/1Uxw0neyiIn3Ve8mpRsO6A06KfbqNrWuq/view?usp=sharing)). Set the corresponding paths as  `ibrnet1_path` and `ibrnet2_path` in the [config_general.txt](configs/config_general.txt) file.

Also, download `nerf_llff_data.zip` and `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) for validation and testing and set their corresponding paths as  `llff_test_path` and `nerf_path` in the [config_general.txt](configs/config_general.txt) file.

Once all the datasets are available, train the network from scratch with the following command:
```
python run_geo_nerf.py --config configs/config_general.txt
```
### Contact
You can contact the author through email: mohammad.johari At idiap.ch.

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{johari-et-al-2022,
  author = {Johari, M. and Lepoittevin, Y. and Fleuret, F.},
  title = {GeoNeRF: Generalizing NeRF with Geometry Priors},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```

### Acknowledgement
This work was supported by ams OSRAM.