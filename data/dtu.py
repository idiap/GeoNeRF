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
import cv2
import numpy as np
from PIL import Image

from utils.utils import read_pfm, get_nearest_pose_ids

class DTU_Dataset(Dataset):
    def __init__(
        self,
        original_root_dir,
        preprocessed_root_dir,
        split,
        nb_views,
        downSample=1.0,
        max_len=-1,
        scene="None",
    ):
        self.original_root_dir = original_root_dir
        self.preprocessed_root_dir = preprocessed_root_dir
        self.split = split
        self.scene = scene

        self.downSample = downSample
        self.scale_factor = 1.0 / 200
        self.interval_scale = 1.06
        self.max_len = max_len
        self.nb_views = nb_views

        self.build_metas()
        self.build_proj_mats()
        self.define_transforms()

    def define_transforms(self):
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def build_metas(self):
        self.metas = []
        with open(f"configs/lists/dtu_{self.split}_all.txt") as f:
            self.scans = [line.rstrip() for line in f.readlines()]
            if self.scene != "None":
                self.scans = [self.scene]

        # light conditions 2-5 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = (
            [3] if "train" != self.split or self.scene != "None" else range(2, 5)
        )

        self.id_list = []

        if self.split == "train":
            if self.scene == "None":
                pair_file = f"configs/lists/dtu_pairs.txt"
            else:
                pair_file = f"configs/lists/dtu_pairs_ft.txt"
        else:
            pair_file = f"configs/lists/dtu_pairs_val.txt"

        for scan in self.scans:
            with open(pair_file) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, ref_view, src_views)]
                        self.id_list.append([ref_view] + src_views)

        self.id_list = np.unique(self.id_list)
        self.build_remap()

    def build_proj_mats(self):
        near_fars, intrinsics, world2cams, cam2worlds = [], [], [], []
        for vid in self.id_list:
            proj_mat_filename = os.path.join(
                self.preprocessed_root_dir, f"Cameras/train/{vid:08d}_cam.txt"
            )
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            extrinsic[:3, 3] *= self.scale_factor

            intrinsic[:2] = intrinsic[:2] * self.downSample
            intrinsics += [intrinsic.copy()]

            near_fars += [near_far]
            world2cams += [extrinsic]
            cam2worlds += [np.linalg.inv(extrinsic)]

        self.near_fars, self.intrinsics = np.stack(near_fars), np.stack(intrinsics)
        self.world2cams, self.cam2worlds = np.stack(world2cams), np.stack(cam2worlds)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ")
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ")
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min, depth_interval = lines[11].split()
        depth_min = float(depth_min) * self.scale_factor
        depth_max = depth_min + float(depth_interval) * 192 * self.interval_scale * self.scale_factor

        intrinsics[0, 2] = intrinsics[0, 2] + 80.0 / 4.0
        intrinsics[1, 2] = intrinsics[1, 2] + 44.0 / 4.0
        intrinsics[:2] = intrinsics[:2]

        return intrinsics, extrinsics, [depth_min, depth_max]

    def read_depth(self, filename, far_bound, noisy_factor=1.0):
        depth_h = self.scale_factor * np.array(
            read_pfm(filename)[0], dtype=np.float32
        )
        depth_h = cv2.resize(
            depth_h, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST
        )

        depth_h = cv2.resize(
            depth_h,
            None,
            fx=self.downSample * noisy_factor,
            fy=self.downSample * noisy_factor,
            interpolation=cv2.INTER_NEAREST,
        )

        ## Exclude points beyond the bounds
        depth_h[depth_h > far_bound * 0.95] = 0.0

        depth = {}
        for l in range(3):
            depth[f"level_{l}"] = cv2.resize(
                depth_h,
                None,
                fx=1.0 / (2**l),
                fy=1.0 / (2**l),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.split == "train":
            cutout = np.ones_like(depth[f"level_2"])
            h0 = int(np.random.randint(0, high=cutout.shape[0] // 5, size=1))
            h1 = int(
                np.random.randint(
                    4 * cutout.shape[0] // 5, high=cutout.shape[0], size=1
                )
            )
            w0 = int(np.random.randint(0, high=cutout.shape[1] // 5, size=1))
            w1 = int(
                np.random.randint(
                    4 * cutout.shape[1] // 5, high=cutout.shape[1], size=1
                )
            )
            cutout[h0:h1, w0:w1] = 0
            depth_aug = depth[f"level_2"] * cutout
        else:
            depth_aug = depth[f"level_2"].copy()

        return depth, depth_h, depth_aug

    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype("int")
        for i, item in enumerate(self.id_list):
            self.remap[item] = i

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        if self.split == "train" and self.scene == "None":
            noisy_factor = float(np.random.choice([1.0, 0.5], 1))
            close_views = int(np.random.choice([3, 4, 5], 1))
        else:
            noisy_factor = 1.0
            close_views = 5

        scan, light_idx, target_view, src_views = self.metas[idx]
        view_ids = src_views[:self.nb_views] + [target_view]

        affine_mats, affine_mats_inv = [], []
        imgs, depths_h, depths_aug = [], [], []
        depths = {"level_0": [], "level_1": [], "level_2": []}
        intrinsics, w2cs, c2ws, near_fars = [], [], [], []

        for vid in view_ids:
            # Note that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(
                self.original_root_dir,
                f"Rectified/{scan}/rect_{vid + 1:03d}_{light_idx}_r5000.png",
            )
            depth_filename = os.path.join(
                self.preprocessed_root_dir, f"Depths/{scan}/depth_map_{vid:04d}.pfm"
            )
            img = Image.open(img_filename)
            img_wh = np.round(
                np.array(img.size) / 2.0 * self.downSample * noisy_factor
            ).astype("int")
            img = img.resize(img_wh, Image.BICUBIC)
            img = self.transform(img)
            imgs += [img]

            index_mat = self.remap[vid]

            intrinsic = self.intrinsics[index_mat].copy()
            intrinsic[:2] = intrinsic[:2] * noisy_factor
            intrinsics.append(intrinsic)

            w2c = self.world2cams[index_mat]
            w2cs.append(w2c)
            c2ws.append(self.cam2worlds[index_mat])

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

            near_far = self.near_fars[index_mat]

            depth, depth_h, depth_aug = self.read_depth(
                depth_filename, near_far[1], noisy_factor
            )

            depths["level_0"].append(depth["level_0"])
            depths["level_1"].append(depth["level_1"])
            depths["level_2"].append(depth["level_2"])
            depths_h.append(depth_h)
            depths_aug.append(depth_aug)

            near_fars.append(near_far)

        imgs = np.stack(imgs)
        depths_h, depths_aug = np.stack(depths_h), np.stack(depths_aug)
        depths["level_0"] = np.stack(depths["level_0"])
        depths["level_1"] = np.stack(depths["level_1"])
        depths["level_2"] = np.stack(depths["level_2"])
        affine_mats, affine_mats_inv = np.stack(affine_mats), np.stack(affine_mats_inv)
        intrinsics = np.stack(intrinsics)
        w2cs = np.stack(w2cs)
        c2ws = np.stack(c2ws)
        near_fars = np.stack(near_fars)

        closest_idxs = []
        for pose in c2ws[:-1]:
            closest_idxs.append(
                get_nearest_pose_ids(
                    pose,
                    ref_poses=c2ws[:-1],
                    num_select=close_views,
                    angular_dist_method="dist",
                )
            )
        closest_idxs = np.stack(closest_idxs, axis=0)

        sample = {}
        sample["images"] = imgs
        sample["depths"] = depths
        sample["depths_h"] = depths_h
        sample["depths_aug"] = depths_aug
        sample["w2cs"] = w2cs
        sample["c2ws"] = c2ws
        sample["near_fars"] = near_fars
        sample["intrinsics"] = intrinsics
        sample["affine_mats"] = affine_mats
        sample["affine_mats_inv"] = affine_mats_inv
        sample["closest_idxs"] = closest_idxs

        return sample
