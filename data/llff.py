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
import glob
import numpy as np
from PIL import Image

from utils.utils import get_nearest_pose_ids

def normalize(v):
    return v / np.linalg.norm(v)


def average_poses(poses):
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)

    # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_homo[:3] = pose_avg
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)

    # (N_images, 4, 4) homogeneous coordinate
    poses_homo = np.concatenate([poses, last_row], 1)

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo) @ blender2opencv


class LLFF_Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        nb_views,
        downSample=1.0,
        max_len=-1,
        scene="None",
        imgs_folder_name="images",
    ):
        self.root_dir = root_dir
        self.split = split
        self.nb_views = nb_views
        self.scene = scene
        self.imgs_folder_name = imgs_folder_name

        self.downsample = downSample
        self.max_len = max_len
        self.img_wh = (int(960 * self.downsample), int(720 * self.downsample))

        self.define_transforms()
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )

        self.build_metas()

    def define_transforms(self):
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def build_metas(self):
        if self.scene != "None":
            self.scans = [
                os.path.basename(scan_dir)
                for scan_dir in sorted(
                    glob.glob(os.path.join(self.root_dir, self.scene))
                )
            ]
        else:
            self.scans = [
                os.path.basename(scan_dir)
                for scan_dir in sorted(glob.glob(os.path.join(self.root_dir, "*")))
            ]

        self.meta = []
        self.image_paths = {}
        self.near_far = {}
        self.id_list = {}
        self.closest_idxs = {}
        self.c2ws = {}
        self.w2cs = {}
        self.intrinsics = {}
        self.affine_mats = {}
        self.affine_mats_inv = {}
        for scan in self.scans:
            self.image_paths[scan] = sorted(
                glob.glob(os.path.join(self.root_dir, scan, self.imgs_folder_name, "*"))
            )
            poses_bounds = np.load(
                os.path.join(self.root_dir, scan, "poses_bounds.npy")
            )  # (N_images, 17)
            poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
            bounds = poses_bounds[:, -2:]  # (N_images, 2)

            # Step 1: rescale focal length according to training resolution
            H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images

            focal = [focal * self.img_wh[0] / W, focal * self.img_wh[1] / H]

            # Step 2: correct poses
            poses = np.concatenate(
                [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1
            )
            poses, _ = center_poses(poses, self.blender2opencv)
            # poses = poses @ self.blender2opencv

            # Step 3: correct scale so that the nearest depth is at a little more than 1.0
            near_original = bounds.min()
            scale_factor = near_original * 0.75  # 0.75 is the default parameter
            bounds /= scale_factor
            poses[..., 3] /= scale_factor

            self.near_far[scan] = bounds.astype('float32')

            num_viewpoint = len(self.image_paths[scan])
            val_ids = [idx for idx in range(0, num_viewpoint, 8)]
            w, h = self.img_wh

            self.id_list[scan] = []
            self.closest_idxs[scan] = []
            self.c2ws[scan] = []
            self.w2cs[scan] = []
            self.intrinsics[scan] = []
            self.affine_mats[scan] = []
            self.affine_mats_inv[scan] = []
            for idx in range(num_viewpoint):
                if (
                    (self.split == "val" and idx in val_ids)
                    or (
                        self.split == "train"
                        and self.scene != "None"
                        and idx not in val_ids
                    )
                    or (self.split == "train" and self.scene == "None")
                ):
                    self.meta.append({"scan": scan, "target_idx": idx})

                view_ids = get_nearest_pose_ids(
                    poses[idx, :, :],
                    ref_poses=poses[..., :],
                    num_select=self.nb_views + 1,
                    angular_dist_method="dist",
                )

                self.id_list[scan].append(view_ids)

                closest_idxs = []
                source_views = view_ids[1:]
                for vid in source_views:
                    closest_idxs.append(
                        get_nearest_pose_ids(
                            poses[vid, :, :],
                            ref_poses=poses[source_views],
                            num_select=5,
                            angular_dist_method="dist",
                        )
                    )
                self.closest_idxs[scan].append(np.stack(closest_idxs, axis=0))

                c2w = np.eye(4).astype('float32')
                c2w[:3] = poses[idx]
                w2c = np.linalg.inv(c2w)
                self.c2ws[scan].append(c2w)
                self.w2cs[scan].append(w2c)

                intrinsic = np.array([[focal[0], 0, w / 2], [0, focal[1], h / 2], [0, 0, 1]]).astype('float32')
                self.intrinsics[scan].append(intrinsic)

    def __len__(self):
        return len(self.meta) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        if self.split == "train" and self.scene == "None":
            noisy_factor = float(np.random.choice([1.0, 0.75, 0.5], 1))
            close_views = int(np.random.choice([3, 4, 5], 1))
        else:
            noisy_factor = 1.0
            close_views = 5

        scan = self.meta[idx]["scan"]
        target_idx = self.meta[idx]["target_idx"]

        view_ids = self.id_list[scan][target_idx]
        target_view = view_ids[0]
        src_views = view_ids[1:]
        view_ids = [vid for vid in src_views] + [target_view]

        closest_idxs = self.closest_idxs[scan][target_idx][:, :close_views]

        imgs, depths, depths_h, depths_aug = [], [], [], []
        intrinsics, w2cs, c2ws, near_fars = [], [], [], []
        affine_mats, affine_mats_inv = [], []

        w, h = self.img_wh
        w, h = int(w * noisy_factor), int(h * noisy_factor)

        for vid in view_ids:
            img_filename = self.image_paths[scan][vid]
            img = Image.open(img_filename).convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h), Image.BICUBIC)
            img = self.transform(img)
            imgs.append(img)

            intrinsic = self.intrinsics[scan][vid].copy()
            intrinsic[:2] = intrinsic[:2] * noisy_factor
            intrinsics.append(intrinsic)

            w2c = self.w2cs[scan][vid]
            w2cs.append(w2c)
            c2ws.append(self.c2ws[scan][vid])

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

            near_fars.append(self.near_far[scan][vid])

            depths_h.append(np.zeros([h, w]))
            depths.append(np.zeros([h // 4, w // 4]))
            depths_aug.append(np.zeros([h // 4, w // 4]))

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

        sample = {}
        sample["images"] = imgs
        sample["depths"] = depths
        sample["depths_h"] = depths_h
        sample["depths_aug"] = depths_aug
        sample["w2cs"] = w2cs
        sample["c2ws"] = c2ws
        sample["near_fars"] = near_fars
        sample["affine_mats"] = affine_mats
        sample["affine_mats_inv"] = affine_mats_inv
        sample["intrinsics"] = intrinsics
        sample["closest_idxs"] = closest_idxs

        return sample
