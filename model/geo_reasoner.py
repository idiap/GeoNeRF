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

    # Copyright (c) 2020 AI葵

    # This file is part of CasMVSNet_pl.
    # CasMVSNet_pl is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License version 3 as
    # published by the Free Software Foundation.

    # CasMVSNet_pl is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    # GNU General Public License for more details.

    # You should have received a copy of the GNU General Public License
    # along with CasMVSNet_pl. If not, see <http://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.utils import homo_warp
from inplace_abn import InPlaceABN


def get_depth_values(current_depth, n_depths, depth_interval):
    depth_min = torch.clamp_min(current_depth - n_depths / 2 * depth_interval, 1e-7)
    depth_values = (
        depth_min
        + depth_interval
        * torch.arange(
            0, n_depths, device=current_depth.device, dtype=current_depth.dtype
        )[None, :, None, None]
    )
    return depth_values


class ConvBnReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        pad=1,
        norm_act=InPlaceABN,
    ):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        pad=1,
        norm_act=InPlaceABN,
    ):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class FeatureNet(nn.Module):
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act),
        )

        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
            ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
        )

        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
            ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
            ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
        )

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True) + y

    def forward(self, x, dummy=None):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x)  # (B, 8, H, W)
        conv1 = self.conv1(conv0)  # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1)  # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2)  # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))  # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0))  # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1)  # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0)  # (B, 8, H, W)

        feats = {"level_0": feat0, "level_1": feat1, "level_2": feat2}

        return feats


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(32),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 16, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(16),
        )

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(
                16, 8, 3, padding=1, output_padding=1, stride=2, bias=False
            ),
            norm_act(8),
        )

        self.br1 = ConvBnReLU3D(8, 8, norm_act=norm_act)
        self.br2 = ConvBnReLU3D(8, 8, norm_act=norm_act)

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        if x.shape[-2] % 8 != 0 or x.shape[-1] % 8 != 0:
            pad_h = 8 * (x.shape[-2] // 8 + 1) - x.shape[-2]
            pad_w = 8 * (x.shape[-1] // 8 + 1) - x.shape[-1]
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        else:
            pad_h = 0
            pad_w = 0

        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        ####################
        x1 = self.br1(x)
        with torch.enable_grad():
            x2 = self.br2(x)
        ####################
        p = self.prob(x1)

        if pad_h > 0 or pad_w > 0:
            x2 = x2[..., :-pad_h, :-pad_w]
            p = p[..., :-pad_h, :-pad_w]

        return x2, p


class CasMVSNet(nn.Module):
    def __init__(self, num_groups=8, norm_act=InPlaceABN, levels=3, use_depth=False):
        super(CasMVSNet, self).__init__()
        self.levels = levels  # 3 depth levels
        self.n_depths = [8, 32, 48]
        self.interval_ratios = [1, 2, 4]
        self.use_depth = use_depth

        self.G = num_groups  # number of groups in groupwise correlation
        self.feature = FeatureNet()

        for l in range(self.levels):
            if l == self.levels - 1 and self.use_depth:
                cost_reg_l = CostRegNet(self.G + 1, norm_act)
            else:
                cost_reg_l = CostRegNet(self.G, norm_act)

            setattr(self, f"cost_reg_{l}", cost_reg_l)

    def build_cost_volumes(self, feats, affine_mats, affine_mats_inv, depth_values, idx, spikes):
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]

        ref_feats, src_feats = feats[:, idx[0]], feats[:, idx[1:]]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)

        affine_mats_inv = affine_mats_inv[:, idx[0]]
        affine_mats = affine_mats[:, idx[1:]]
        affine_mats = affine_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        ref_volume = ref_volume.view(B, self.G, C // self.G, *ref_volume.shape[-3:])
        volume_sum = 0

        for i in range(len(idx) - 1):
            proj_mat = (affine_mats[i].double() @ affine_mats_inv.double()).float()[
                :, :3
            ]
            warped_volume, grid = homo_warp(src_feats[i], proj_mat, depth_values)

            warped_volume = warped_volume.view_as(ref_volume)
            volume_sum = volume_sum + warped_volume  # (B, G, C//G, D, h, w)

        volume = (volume_sum * ref_volume).mean(dim=2) / (V - 1)

        if spikes is None:
            output = volume
        else:
            output = torch.cat([volume, spikes], dim=1)

        return output

    def create_neural_volume(
        self,
        feats,
        affine_mats,
        affine_mats_inv,
        idx,
        init_depth_min,
        depth_interval,
        gt_depths,
    ):
        if feats["level_0"].shape[-1] >= 800:
            hres_input = True
        else:
            hres_input = False

        B, V = affine_mats.shape[:2]

        v_feat = {}
        depth_maps = {}
        depth_values = {}
        for l in reversed(range(self.levels)):  # (2, 1, 0)
            feats_l = feats[f"level_{l}"]  # (B*V, C, h, w)
            feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)
            h, w = feats_l.shape[-2:]
            depth_interval_l = depth_interval * self.interval_ratios[l]
            D = self.n_depths[l]
            if l == self.levels - 1:  # coarsest level
                depth_values_l = init_depth_min + depth_interval_l * torch.arange(
                    0, D, device=feats_l.device, dtype=feats_l.dtype
                )  # (D)
                depth_values_l = depth_values_l[None, :, None, None].expand(
                    -1, -1, h, w
                )

                if self.use_depth:
                    gt_mask = gt_depths > 0
                    sp_idx_float = (
                        gt_mask * (gt_depths - init_depth_min) / (depth_interval_l)
                    )[:, :, None]
                    spikes = (
                        torch.arange(D).view(1, 1, -1, 1, 1).cuda()
                        == sp_idx_float.floor().long()
                    ) * (1 - sp_idx_float.frac())
                    spikes = spikes + (
                        torch.arange(D).view(1, 1, -1, 1, 1).cuda()
                        == sp_idx_float.ceil().long()
                    ) * (sp_idx_float.frac())
                    spikes = (spikes * gt_mask[:, :, None]).float()
            else:
                depth_lm1 = depth_l.detach()  # the depth of previous level
                depth_lm1 = F.interpolate(
                    depth_lm1, scale_factor=2, mode="bilinear", align_corners=True
                )  # (B, 1, h, w)
                depth_values_l = get_depth_values(depth_lm1, D, depth_interval_l)

            affine_mats_l = affine_mats[..., l]
            affine_mats_inv_l = affine_mats_inv[..., l]

            if l == self.levels - 1 and self.use_depth:
                spikes_ = spikes
            else:
                spikes_ = None

            if hres_input:
                v_feat_l = checkpoint(
                    self.build_cost_volumes,
                    feats_l,
                    affine_mats_l,
                    affine_mats_inv_l,
                    depth_values_l,
                    idx,
                    spikes_,
                    preserve_rng_state=False,
                )
            else:
                v_feat_l = self.build_cost_volumes(
                    feats_l,
                    affine_mats_l,
                    affine_mats_inv_l,
                    depth_values_l,
                    idx,
                    spikes_,
                )

            cost_reg_l = getattr(self, f"cost_reg_{l}")
            v_feat_l, depth_prob = cost_reg_l(v_feat_l)  # (B, 1, D, h, w)

            depth_l = (F.softmax(depth_prob, dim=2) * depth_values_l[:, None]).sum(
                dim=2
            )

            v_feat[f"level_{l}"] = v_feat_l
            depth_maps[f"level_{l}"] = depth_l
            depth_values[f"level_{l}"] = depth_values_l

        return v_feat, depth_maps, depth_values

    def forward(
        self, imgs, affine_mats, affine_mats_inv, near_far, closest_idxs, gt_depths=None
    ):
        B, V, _, H, W = imgs.shape

        ## Feature Pyramid
        feats = self.feature(
            imgs.reshape(B * V, 3, H, W)
        )  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)
        feats_fpn = feats[f"level_0"].reshape(B, V, *feats[f"level_0"].shape[1:])

        feats_vol = {"level_0": [], "level_1": [], "level_2": []}
        depth_map = {"level_0": [], "level_1": [], "level_2": []}
        depth_values = {"level_0": [], "level_1": [], "level_2": []}
        ## Create cost volumes for each view
        for i in range(0, V):
            permuted_idx = torch.tensor(closest_idxs[0, i]).cuda()

            init_depth_min = near_far[0, i, 0]
            depth_interval = (
                (near_far[0, i, 1] - near_far[0, i, 0])
                / self.n_depths[-1]
                / self.interval_ratios[-1]
            )

            v_feat, d_map, d_values = self.create_neural_volume(
                feats,
                affine_mats,
                affine_mats_inv,
                idx=permuted_idx,
                init_depth_min=init_depth_min,
                depth_interval=depth_interval,
                gt_depths=gt_depths[:, i : i + 1],
            )

            for l in range(3):
                feats_vol[f"level_{l}"].append(v_feat[f"level_{l}"])
                depth_map[f"level_{l}"].append(d_map[f"level_{l}"])
                depth_values[f"level_{l}"].append(d_values[f"level_{l}"])

        for l in range(3):
            feats_vol[f"level_{l}"] = torch.stack(feats_vol[f"level_{l}"], dim=1)
            depth_map[f"level_{l}"] = torch.cat(depth_map[f"level_{l}"], dim=1)
            depth_values[f"level_{l}"] = torch.stack(depth_values[f"level_{l}"], dim=1)

        return feats_vol, feats_fpn, depth_map, depth_values
