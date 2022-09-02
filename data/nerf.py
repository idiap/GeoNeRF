# GeoNeRF is a generalizable NeRF model that renders novel views
# without requiring per-scene optimization. This software is the 
# implementation of the paper "GeoNeRF: Generalizing NeRF with 
# Geometry Priors" by Mohammad Mahdi Johari, Yann Lepoittevin,
# and Francois Fleuret.

# Copyright (c) 2022 ams International AG

# This file is part of GeoNeRF.
# GeoNeRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# GeoNeRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GeoNeRF. If not, see <http://www.gnu.org/licenses/>.

# This file incorporates work covered by the following copyright and  
# permission notice:

    # MIT License

    # Copyright (c) 2021 apchenstu

    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:

    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

from torch.utils.data import Dataset
from torchvision import transforms as T

import os
import json
import numpy as np
from PIL import Image

from utils.utils import get_nearest_pose_ids

class NeRF_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        nb_views,
        downSample=1.0,
        max_len=-1,
        scene="None",
    ):
        self.root_dir = root_dir
        self.split = split
        self.nb_views = nb_views
        self.scene = scene

        self.downsample = downSample
        self.max_len = max_len

        self.img_wh = (int(800 * self.downsample), int(800 * self.downsample))

        self.define_transforms()
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        self.build_metas()

    def define_transforms(self):
        self.transform = T.ToTensor()

        self.src_transform = T.Compose(
            [
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def build_metas(self):
        self.meta = {}
        with open(
            os.path.join(self.root_dir, self.scene, "transforms_train.json"), "r"
        ) as f:
            self.meta["train"] = json.load(f)

        with open(
            os.path.join(self.root_dir, self.scene, "transforms_test.json"), "r"
        ) as f:
            self.meta["val"] = json.load(f)

        w, h = self.img_wh

        # original focal length
        focal = 0.5 * 800 / np.tan(0.5 * self.meta["train"]["camera_angle_x"])

        # modify focal length to match size self.img_wh
        focal *= self.img_wh[0] / 800

        self.near_far = np.array([2.0, 6.0])

        self.image_paths = {"train": [], "val": []}
        self.c2ws = {"train": [], "val": []}
        self.w2cs = {"train": [], "val": []}
        self.intrinsics = {"train": [], "val": []}

        for frame in self.meta["train"]["frames"]:
            self.image_paths["train"].append(
                os.path.join(self.root_dir, self.scene, f"{frame['file_path']}.png")
            )

            c2w = np.array(frame["transform_matrix"]) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            self.c2ws["train"].append(c2w)
            self.w2cs["train"].append(w2c)

            intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
            self.intrinsics["train"].append(intrinsic.copy())

        self.c2ws["train"] = np.stack(self.c2ws["train"], axis=0)
        self.w2cs["train"] = np.stack(self.w2cs["train"], axis=0)
        self.intrinsics["train"] = np.stack(self.intrinsics["train"], axis=0)

        for frame in self.meta["val"]["frames"]:
            self.image_paths["val"].append(
                os.path.join(self.root_dir, self.scene, f"{frame['file_path']}.png")
            )

            c2w = np.array(frame["transform_matrix"]) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            self.c2ws["val"].append(c2w)
            self.w2cs["val"].append(w2c)

            intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
            self.intrinsics["val"].append(intrinsic.copy())

        self.c2ws["val"] = np.stack(self.c2ws["val"], axis=0)
        self.w2cs["val"] = np.stack(self.w2cs["val"], axis=0)
        self.intrinsics["val"] = np.stack(self.intrinsics["val"], axis=0)

    def __len__(self):
        return len(self.image_paths[self.split]) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        target_frame = self.meta[self.split]["frames"][idx]
        c2w = np.array(target_frame["transform_matrix"]) @ self.blender2opencv
        w2c = np.linalg.inv(c2w)

        if self.split == "train":
            src_views = get_nearest_pose_ids(
                c2w,
                ref_poses=self.c2ws["train"],
                num_select=self.nb_views + 1,
                angular_dist_method="dist",
            )[1:]
        else:
            src_views = get_nearest_pose_ids(
                c2w,
                ref_poses=self.c2ws["train"],
                num_select=self.nb_views,
                angular_dist_method="dist",
            )

        imgs, depths, depths_h, depths_aug = [], [], [], []
        intrinsics, w2cs, c2ws, near_fars = [], [], [], []
        affine_mats, affine_mats_inv = [], []

        w, h = self.img_wh

        for vid in src_views:
            img_filename = self.image_paths["train"][vid]
            img = Image.open(img_filename)
            if img.size != (w, h):
                img = img.resize((w, h), Image.BICUBIC)

            img = self.transform(img)
            img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
            imgs.append(self.src_transform(img))

            intrinsic = self.intrinsics["train"][vid]
            intrinsics.append(intrinsic)

            w2c = self.w2cs["train"][vid]
            w2cs.append(w2c)
            c2ws.append(self.c2ws["train"][vid])

            aff = []
            aff_inv = []
            for l in range(3):
                proj_mat_l = np.eye(4)
                intrinsic_temp = intrinsic.copy()
                intrinsic_temp[:2] = intrinsic_temp[:2] / (2**l)
                proj_mat_l[:3, :4] = intrinsic_temp @ w2c[:3, :4]
                aff.append(proj_mat_l.copy())
                aff_inv.append(np.linalg.inv(proj_mat_l))
            aff = np.stack(aff, axis=-1)
            aff_inv = np.stack(aff_inv, axis=-1)

            affine_mats.append(aff)
            affine_mats_inv.append(aff_inv)

            near_fars.append(self.near_far)

            depths_h.append(np.zeros([h, w]))
            depths.append(np.zeros([h // 4, w // 4]))
            depths_aug.append(np.zeros([h // 4, w // 4]))

        ## Adding target data
        img_filename = self.image_paths[self.split][idx]
        img = Image.open(img_filename)
        if img.size != (w, h):
            img = img.resize((w, h), Image.BICUBIC)

        img = self.transform(img)  # (4, h, w)
        img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
        imgs.append(self.src_transform(img))

        intrinsic = self.intrinsics[self.split][idx]
        intrinsics.append(intrinsic)

        w2c = self.w2cs[self.split][idx]
        w2cs.append(w2c)
        c2ws.append(self.c2ws[self.split][idx])

        near_fars.append(self.near_far)

        depths_h.append(np.zeros([h, w]))
        depths.append(np.zeros([h // 4, w // 4]))
        depths_aug.append(np.zeros([h // 4, w // 4]))

        ## Stacking
        imgs = np.stack(imgs)
        depths = np.stack(depths)
        depths_h = np.stack(depths_h)
        depths_aug = np.stack(depths_aug)
        affine_mats = np.stack(affine_mats)
        affine_mats_inv = np.stack(affine_mats_inv)
        intrinsics = np.stack(intrinsics)
        w2cs = np.stack(w2cs)
        c2ws = np.stack(c2ws)
        near_fars = np.stack(near_fars)

        closest_idxs = []
        for pose in c2ws[:-1]:
            closest_idxs.append(
                get_nearest_pose_ids(
                    pose, ref_poses=c2ws[:-1], num_select=5, angular_dist_method="dist"
                )
            )
        closest_idxs = np.stack(closest_idxs, axis=0)

        sample = {}
        sample["images"] = imgs
        sample["depths"] = depths
        sample["depths_h"] = depths_h
        sample["depths_aug"] = depths_aug
        sample["w2cs"] = w2cs.astype("float32")
        sample["c2ws"] = c2ws.astype("float32")
        sample["near_fars"] = near_fars
        sample["affine_mats"] = affine_mats
        sample["affine_mats_inv"] = affine_mats_inv
        sample["intrinsics"] = intrinsics.astype("float32")
        sample["closest_idxs"] = closest_idxs

        return sample
